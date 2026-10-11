"""Episode-local gait and heading state: reset, the step order, and how a respawn clears it.

Shared by both backends: imports only array_api_compat, numpy, the standard
library and other shared modules.

`reset_state` seeds the info leaves. A step is `pre_physics`, then the
backend's own physics, then `update`. `update`'s docstring is the one
statement of the step order. After it, each backend calls
compose.total_reward and signals.observe.

Some `info` leaves belong to one episode: the no-progress meter, the
per-foot gait trackers and the heading reference. Joystick.step cannot clear
them on its own. Truncation and the restore of `data` and `obs` happen in the
brax and playground wrappers after the env step, so the step that ends an
episode does not know the next one starts. The wrappers in envs/wrappers.py
therefore call `clear_done` after the step, and the Torch runner calls it at
its own respawn.

`reseed_flags` says which layers a config needs. `clear_done` applies them to
a batch whose leaves carry a leading world axis.
"""

from __future__ import annotations

import math

import array_api_compat

from humanoid_lab.envs import progress, push, symmetry
from humanoid_lab.envs.constants import xp_of
from humanoid_lab.rewards import compose

# Per-foot gait trackers the 'gait' layer restarts on respawn. reset() seeds
# every one of them to zeros (False for last_contact).
GAIT_TRACKER_KEYS = (
    "feet_air_time",
    "feet_contact_time",
    "swing_apex",
    "last_apex",
    "last_contact",
    "air_dur_ema",
    "stance_dur_ema",
)

# Layer order. make_wrap_env_fn stacks the wrappers in this order.
LAYERS = ("progress", "gait", "heading")


def gait_dur_ema_on(scales) -> bool:
    """Whether the per-foot swing/stance duration EMAs exist: either
    symmetry scale is nonzero. reset() seeds them and step() updates them
    only then, and the respawn clears them only then."""
    return bool(scales.get("gait_symmetry", 0.0) or scales.get("gait_symmetry_income", 0.0))


def reseed_flags(cfg) -> tuple:
    """The layers a respawn must clear for this env config, in LAYERS order.

    env_config may be partial, so every block is read with a None guard.
    """
    flags = []
    no_progress = cfg.get("no_progress")
    if no_progress is not None and no_progress.enable:
        flags.append("progress")
    reward = cfg.get("reward")
    if reward is not None and gait_dur_ema_on(reward.get("scales", {})):
        flags.append("gait")
    obs = cfg.get("obs")
    if obs is not None:
        names = list(obs.get("state", ())) + list(obs.get("privileged", ()))
        if "heading_err" in names:
            flags.append("heading")
    return tuple(flags)


def _where(done, new, old):
    xp = xp_of(done, old)
    mask = xp.reshape(done, tuple(done.shape) + (1,) * (len(old.shape) - len(done.shape)))
    return xp.where(mask, new, old)


def clear_done(info, done, flags, reset_values):
    """A new info dict with the flagged layers cleared for the done worlds.

    done is a bool array with a leading world axis. Every leaf of info has
    that axis first, and the mask broadcasts over the trailing axes.
    reset_values holds the fresh values the layers need: 'progress_ema' for
    'progress' and 'yaw_ref' for 'heading'. Other keys are not touched, and
    dtypes are kept.
    """
    info = dict(info)
    if "progress" in flags:
        steps = info["steps_since_cmd"]
        info["progress_ema"] = _where(done, reset_values["progress_ema"], info["progress_ema"])
        info["steps_since_cmd"] = _where(done, xp_of(steps).zeros_like(steps), steps)
    if "gait" in flags:
        for k in GAIT_TRACKER_KEYS:
            v = info[k]
            info[k] = _where(done, xp_of(v).zeros_like(v), v)
    if "heading" in flags:
        info["yaw_ref"] = _where(done, reset_values["yaw_ref"], info["yaw_ref"])
    return info


def reset_offset(pose_noise, reset_noise):
    """The qpos offset of the actuated joints at reset: pose_noise (uniform
    in [-1, 1]) times reset_noise (rad). reset_noise 0 gives the exact pose.

    It is separate from reset_state because the heading reference reads the
    data after this offset is applied and forward has run.
    """
    return pose_noise * reset_noise


def reset_state(command, consts, cfg, yaw=None, push_countdown=None, mirror_u=None):
    """The episode-local info leaves at reset, without 'rng'.

    command has shape batch + (3,). yaw is the base yaw after the reset pose,
    needed only when consts.heading_obs. push_countdown is the drawn first
    gap, given when the push schedule is ranged. mirror_u is the uniform draw
    behind the mirror flag, used when symmetry is on. Every gated key follows
    its config gate, so the info pytree never depends on anything else.
    """
    xp = xp_of(command)
    dev = array_api_compat.device(command)
    batch = tuple(command.shape[:-1])
    f32 = xp.float32

    def zeros(*tail):
        return xp.zeros(batch + tail, dtype=f32, device=dev)

    nu, nf = consts.action_size, consts.n_feet
    info = {
        "command": command,
        "last_action": zeros(nu),
        "last_last_action": zeros(nu),
        "last_torque": zeros(nu),
        "feet_air_time": zeros(nf),
        "feet_contact_time": zeros(nf),
        "swing_apex": zeros(nf),
        "last_apex": zeros(nf),
        "last_contact": xp.zeros(batch + (nf,), dtype=xp.bool, device=dev),
        "phase": zeros(),
        "step_count": xp.zeros(batch, dtype=xp.int32, device=dev),
        "steps_since_cmd": xp.zeros(batch, dtype=xp.int32, device=dev),
    }
    if consts.heading_obs:
        info["yaw_ref"] = yaw
    if consts.gait_dur_ema_on:
        info["air_dur_ema"] = zeros(nf)
        info["stance_dur_ema"] = zeros(nf)
    if push_countdown is not None:
        info["push_countdown"] = push_countdown
    if cfg.symmetry.enable:
        info["mirror"] = mirror_u < cfg.symmetry.mirror_prob
    if cfg.no_progress.enable:
        info["progress_ema"] = progress.cmd_speed(command)
    return info


def pre_physics(info, action, consts, cfg, push_draws):
    """The first half of a step, before physics: (info, action, ctrl, kick).

    A mirrored world's action is mapped back to the real frame first, so
    everything after it, and the returned action, are in the real frame. ctrl
    is the clipped motor target. When the push is on, the push countdown in
    info is advanced (ranged schedule only) and kick is the velocity to add
    to the base. kick is None when the push is off.
    """
    info = dict(info)
    if cfg.symmetry.enable:
        action = symmetry.mirror(action, info["mirror"], consts.act_perm, consts.act_sign)
    xp = xp_of(action)
    ctrl = xp.clip(
        consts.actuator_model.ctrl_from_action(action, consts.default_pose, consts.action_scale),
        consts.ctrl_lo,
        consts.ctrl_hi,
    )
    kick = None
    if cfg.push.enable:
        push_now, countdown = push.push_schedule(
            cfg.push,
            info["step_count"],
            info.get("push_countdown"),
            push_draws.get("next_countdown"),
        )
        if countdown is not None:
            info["push_countdown"] = countdown
        kick = push.push_kick(cfg.push, push_now, push_draws)
    return info, action, ctrl, kick


def update(info, r, action, consts, cfg, draws):
    """The second half of a step, after physics: (info, rewards, done, aux).

    r is signals.readings of the post-physics data. draws comes from the
    backend's step draws and holds 'command' (the draws behind the next
    command) and 'u_term' (the uniform behind the no-progress cut). info is
    copied and the inputs are never mutated. aux has 'fall' always, and 'cut'
    and 'progress_ratio' when the no-progress cut is on. Each backend calls
    compose.total_reward and signals.observe on the result.

    The step order, which every backend must keep:

    1. yaw_ref follows the base yaw while the command turns (|wz| above
       TURN_WZ_THRESHOLD), when the heading observation is on.
    2. contact is this step's foot contact and contact_filt adds last step's.
       first_contact is a foot that was airborne and is now down. swing_apex
       takes the running maximum of each airborne foot's clearance. On the
       step a foot lands contact_filt is already true, so this skips and the
       reward reads the peak the completed swing reached. All of it runs
       whatever the scales are: the trackers are info state, and their shape
       must not depend on the config.
    3. The rewards are computed. They read the trackers as they were before
       this step's update, so a landing pays the completed swing's air time.
    4. When the duration EMAs exist, each completed mode duration is folded
       into its foot's EMA on the event step: a landing closes a swing, a
       liftoff closes a stance. This runs after the rewards, which read the
       previous EMAs, and before step 6 erases the completed durations.
    5. last_apex records the landing foot's peak, and only then does the
       zeroing arm the next swing.
    6. The mode times, last_contact, the action history, last_torque and
       step_count advance.
    7. The gait clock advances, frozen while the command is below the
       deadband.
    8. steps_since_cmd counts this step.
    9. With the no-progress cut on, the meter scores the command that drove
       this step. This is after the increment of step 8 and before the
       resample of step 10. The cut only flips done. It attaches no reward
       term. Losing the rest of the episode is the whole penalty.
    10. A due command is resampled and steps_since_cmd restarts. With the cut
       on, progress_ema is reseeded to the new demand. done is fall | cut.
    """
    info = dict(info)
    xp = xp_of(r["contact"])
    dt = consts.dt

    # 1.
    if consts.heading_obs:
        turning = xp.abs(info["command"][..., 2]) > progress.TURN_WZ_THRESHOLD
        info["yaw_ref"] = xp.where(turning, r["yaw"], info["yaw_ref"])

    # 2.
    contact = r["contact"]
    contact_filt = contact | info["last_contact"]
    first_contact = (info["feet_air_time"] > 0) & contact_filt
    info["swing_apex"] = xp.where(
        ~contact_filt,
        xp.maximum(info["swing_apex"], r["foot_clearance"]),
        info["swing_apex"],
    )

    # 3.
    rewards, fall = compose.compute_rewards(r, info, action, first_contact, contact, consts, cfg)

    # 4.
    if consts.gait_dur_ema_on:
        alpha = cfg.reward.gait_symmetry_alpha
        lift_off = (info["feet_contact_time"] > 0) & ~contact_filt
        info["air_dur_ema"] = xp.where(
            first_contact,
            (1.0 - alpha) * info["air_dur_ema"] + alpha * info["feet_air_time"],
            info["air_dur_ema"],
        )
        info["stance_dur_ema"] = xp.where(
            lift_off,
            (1.0 - alpha) * info["stance_dur_ema"] + alpha * info["feet_contact_time"],
            info["stance_dur_ema"],
        )

    # 5.
    info["last_apex"] = xp.where(first_contact, info["swing_apex"], info["last_apex"])
    info["swing_apex"] = xp.where(contact_filt, 0.0, info["swing_apex"])

    # 6.
    info["feet_air_time"] = xp.where(contact_filt, 0.0, info["feet_air_time"] + dt)
    info["feet_contact_time"] = xp.where(contact_filt, info["feet_contact_time"] + dt, 0.0)
    info["last_contact"] = contact
    info["last_last_action"] = info["last_action"]
    info["last_action"] = action
    info["last_torque"] = r["actuator_force"]
    info["step_count"] = info["step_count"] + 1

    # 7.
    phase = info["phase"] + progress.phase_dt(
        info["command"], consts.cmd_vmax, cfg.gait.freq, dt, consts.deadband
    )
    info["phase"] = xp.fmod(phase + math.pi, 2 * math.pi) - math.pi

    # 8.
    info["steps_since_cmd"] = info["steps_since_cmd"] + 1

    # 9. The no-progress cut (see no_progress in default_config): an env that
    # keeps ignoring its command dies with a probability that grows with the
    # shortfall. rewards["termination"] stays fall-only.
    done = fall
    aux = {"fall": fall}
    if cfg.no_progress.enable:
        npg = cfg.no_progress
        cmd = info["command"]
        demand = progress.cmd_speed(cmd)
        served = progress.served(r["linvel"][..., :2], r["gyro"][..., 2], cmd)
        alpha = dt / npg.ema_sec
        info["progress_ema"] = (1.0 - alpha) * info["progress_ema"] + alpha * served
        progress_ratio = info["progress_ema"] / xp.clip(demand, 1e-6, None)
        hazard = progress.hazard(progress_ratio, npg.risk_below, npg.p_max)
        armed = progress.armed(demand, info["steps_since_cmd"], dt, npg.grace_sec, consts.deadband)
        cut = draws["u_term"] < xp.where(armed, hazard, 0.0)
        done = done | cut
        aux["cut"] = cut
        aux["progress_ratio"] = progress_ratio

    # 10. steps_since_cmd going back to 0 keeps the hazard at zero for
    # grace_sec while the robot turns the gait around.
    resample = info["steps_since_cmd"] >= cfg.command.resample_steps
    new_command = progress.command_from_draws(draws["command"], cfg.command)
    info["command"] = xp.where(resample[..., None], new_command, info["command"])
    info["steps_since_cmd"] = xp.where(resample, 0, info["steps_since_cmd"])
    if cfg.no_progress.enable:
        # A fresh command restarts the meter at ratio 1: the EMA is reseeded
        # to the new demand.
        #
        # This is the only reseed site inside the step. It is not the only
        # one that has to exist. `info` survives every respawn, because the
        # trainer's wrap_for_brax_training ends in
        # BraxAutoResetWrapper(full_reset=False), which restores data and obs
        # from the cached first state and leaves info alone. A cut env would
        # otherwise come back with the dead episode's shortfall and a
        # steps_since_cmd well past grace, and die again inside what should
        # be its grace window. envs/wrappers.py's ProgressReseedWrapper does
        # the respawn reseed through clear_done, and make_wrap_env_fn adds it
        # whenever the cut is on. Any other code that restarts an episode in
        # place owns the same reseed.
        info["progress_ema"] = xp.where(
            resample, progress.cmd_speed(info["command"]), info["progress_ema"]
        )

    return info, rewards, done, aux
