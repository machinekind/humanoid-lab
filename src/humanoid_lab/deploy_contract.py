"""The deployment contract (`policy_meta.json`) built from a live env.

The contract is everything a robot-side runtime needs to run a trained
policy, as RESOLVED numbers read off the env instance that defined
training. The deploy side re-derives nothing; it interprets this file:

    obs   = concat(obs_layout components, in order)
    act   = tanh(mlp(normalize(obs)))
    ctrl  = clip(anchor_ctrl + act * action_scale, ctrl_low, ctrl_high)

`anchor_ctrl` is the ctrl a zero action commands, which is the actuator
model's own answer: the reset keyframe's joint pose for a `pd` preset,
zero torque for an `ideal_torque` one. `default_pose` is a separate
field because it anchors the OBSERVATION (`joint_pos = q - default_pose`)
and does not move with the ctrl anchor. Neither is moved by
`real_pose_ref`, which relocates the reward/reset anchor only -- see
envs/base.py.

Every key in the run's env config is classified below as CONSUMED (its
value reaches the robot, directly or through a resolved field) or
TRAINING_ONLY (it provably cannot). `check_config_covered` refuses a
config carrying a key in neither set, and that refusal is the point: a
new env option with deploy implications gets a contract field and runtime
support before a policy trained with it can ship. Keys are full dotted
paths into `envs/joystick.py::default_config()`, one entry per leaf.

Three properties of the classification, each forced by this env:

- Classification is by LEAF PATH, not by top-level block, so a new key
  inside an already-classified block still fails the guard.
- A policy that observes its gait clock can still ship. This clock is a
  fixed antiphase two-foot clock with a speed-scaled frequency, a closed
  form of ctrl_dt, the command and `gait.freq`, so it ships in `gait_clock`
  and the runtime advances it (export/runtime.py). A speed-blended
  walk/trot clock would not be reproducible on the robot and would have to
  be refused.
- `real_pose_ref` is training-only. It leaves both the ctrl anchor and the
  obs anchor where they were, and this env has no height command, so no
  contract field moves with it.
"""

from __future__ import annotations

import jax
import jax.numpy as jp
import numpy as np

from humanoid_lab.envs import progress
from humanoid_lab.robot.presets import effective_gains

SCHEMA_VERSION = 1

# Observations a robot can actually produce. `gyro`/`gravity` come from the
# IMU, `joint_pos`/`joint_vel` from the encoders, `command` from the
# operator, `last_action` and `phase` from the runtime's own state. Anything
# else on the actor list (base height, world linvel, contacts, actuator
# force) is a simulator readout with no deploy-side source, and a policy
# that observes one cannot be exported.
DEPLOYABLE_OBS = frozenset(
    {"gyro", "gravity", "joint_pos", "joint_vel", "last_action", "command", "phase"}
)

# Keys whose values become contract fields, directly or through the env's
# resolved state (the compiled model, the ctrl bounds, the anchor).
CONSUMED_KEYS = frozenset(
    {
        # Control period. Sets the rate the runtime calls the policy at,
        # and the gait clock's per-step increment.
        "ctrl_dt",
        # Picks the keyframe that IS `default_pose`: the obs anchor
        # (joint_pos = q - default_pose) and, under a pd preset, the ctrl
        # anchor. Ships resolved in `default_pose` / `anchor_ctrl`.
        "reset_keyframe",
        # The ordered actor observation list. Resolves `obs_layout`, which
        # is the byte order of the vector the runtime feeds the network.
        "obs.state",
        # The trained command box. Ships as command_low/command_high, and
        # command.vx additionally sets the gait clock's speed normalizer
        # (`cmd_speed_max`).
        "command.vx",
        "command.vy",
        "command.wz",
        # Frequency band of the gait clock the actor observes as `phase`.
        # Ships in `gait_clock`; the runtime integrates the same clock.
        "gait.freq",
    }
)

# Keys that shape training only: physics stepping, episode structure,
# exploration noise, curricula, rewards, terminations. Nothing here changes
# what the robot does with the exported network.
TRAINING_ONLY_KEYS = frozenset(
    {
        # Physics substep and the warp buffer sizes. The robot has real
        # physics; `sim.backend`/`sim.num_envs` are host-side plumbing.
        "sim_dt",
        "sim.backend",
        "sim.naconmax_per_env",
        "sim.njmax",
        "sim.num_envs",
        # Episode structure. The robot has no episodes.
        "episode_length",
        # Reset-pose exploration noise: a training-time start-state spread,
        # not a control-loop quantity.
        "reset_noise",
        # Moves the `pose`/`stand_still` reward reference and the reset qpos
        # onto the settled pose. envs/base.py holds the ctrl anchor
        # (`_default_pose`) and the obs anchor fixed under it on purpose, so
        # no contract field moves.
        "real_pose_ref",
        # The critic's observation list. It exists so the value function can
        # see what the actor cannot; none of it is ever computed on the robot.
        "obs.privileged",
        # Sensor noise injected during training so the policy tolerates the
        # real thing. The real robot has the real thing.
        "obs_noise.gyro",
        "obs_noise.joint_pos",
        "obs_noise.joint_vel",
        # Random shoves for robustness. A training perturbation with no
        # control-loop counterpart.
        "push.enable",
        "push.interval_steps",
        "push.interval_steps_range",
        "push.vel",
        "push.vel_z",
        "push.ang_vel_rp",
        "push.ang_vel_yaw",
        # Command sampling curriculum: how often a new command is drawn and
        # which corners of the trained box get extra practice. Every draw
        # stays inside command_low/command_high (check_config_covered
        # enforces that below), so what the policy trained under is the box
        # the contract already ships, and none of these keys is readable
        # from the control loop.
        "command.resample_steps",
        "command.zero_prob",
        "command.pure_wz_prob",
        "command.pure_vy_prob",
        "command.pure_slow_prob",
        "command.slow_vx",
        "command.pure_fast_prob",
        "command.fast_vx",
        "command.pure_back_prob",
        "command.back_vx",
        # CaT-style early termination for an env that ignores its command: a
        # PPO training signal (future value zero), no reward term, nothing in
        # the control loop. The robot has no episode to cut.
        "no_progress.enable",
        "no_progress.grace_sec",
        "no_progress.ema_sec",
        "no_progress.risk_below",
        "no_progress.p_max",
        # Episode termination thresholds. A fall ends a training episode; it
        # does not change the mapping from observation to ctrl. A deploy-side
        # safety cutout is a robot-side decision, not this policy's.
        "fall.min_height",
        "fall.max_tilt_gz",
        # Gait shaping targets. `swing_height`, `duty` and `air_time_cap`
        # feed the reward's clearance/stance targets only -- the actor's
        # `phase` observation reads none of them (envs/joystick.py's
        # _leg_phases), so only gait.freq is consumed.
        "gait.swing_height",
        "gait.duty",
        "gait.air_time_cap",
    }
)

# (probability key, range key, command box axis) for each pure command draw.
# The draws are training-only on one condition: each redraws its axis INSIDE
# the box the contract ships. Setting a fast range above the box (a known
# way to pull a policy past a speed it deadlocks at) would make the reported
# box a lie, so the condition is checked rather than assumed.
#
# envs/joystick.py's check_pure_draw_ranges refuses that configuration at
# env construction, and export rebuilds the run's env through the same
# constructor, so the condition is enforced without a second copy here.


# Prefix rule: every `reward.*` leaf is training-only by construction.
# Rewards shape what the network learned; the network is what ships, and no
# reward knob is ever consumed by the runtime. A rule instead of one ledger
# line per scale, so adding a reward term is not a two-file edit.
TRAINING_ONLY_PREFIXES = ("reward.",)


def is_classified(path: str) -> bool:
    """Whether `path` is covered by the ledger or a prefix rule."""
    return (
        path in CONSUMED_KEYS
        or path in TRAINING_ONLY_KEYS
        or path.startswith(TRAINING_ONLY_PREFIXES)
    )


def _leaf_paths(config, prefix: str = "") -> set[str]:
    """Every dotted path to a non-mapping value in `config`."""
    paths: set[str] = set()
    for key, value in config.items():
        path = f"{prefix}{key}"
        if hasattr(value, "items"):
            paths |= _leaf_paths(value, path + ".")
        else:
            paths.add(path)
    return paths


def check_config_covered(env_config) -> None:
    """Refuse an env config carrying a key this module has not classified.

    `env_config` is a run.json `env_config` block or any partial config of
    the same shape (a Hydra `task.env` override, say): only keys that ARE
    present are checked, so a partial config passes as long as every key in
    it is classified.
    """
    unknown = sorted(p for p in _leaf_paths(env_config) if not is_classified(p))
    if unknown:
        raise ValueError(
            f"env config key(s) {unknown} are not classified in "
            "src/humanoid_lab/deploy_contract.py -- decide whether each one "
            "reaches the robot (add a contract field, and runtime support for "
            "it, then list it in CONSUMED_KEYS) or cannot (list it in "
            "TRAINING_ONLY_KEYS) before exporting a policy trained with it"
        )


def build_contract(env, run: dict, checkpoint: str = "") -> dict:
    """The `policy_meta.json` dict for a live `Joystick` env.

    `env` is the run's own env, built with its robot, preset and env
    config, so its resolved state IS the training-time state. `run` is the
    parsed run.json; `checkpoint` the checkpoint directory that ships with
    this contract.
    """
    check_config_covered(run.get("env_config") or {})

    task = run.get("task", "joystick")
    if task != "joystick":
        raise NotImplementedError(
            f"the deploy contract is defined for the joystick task only, got "
            f"'{task}' -- no runtime maps that task's actions to a robot"
        )

    names = list(env.actor_obs_names)
    undeployable = [n for n in names if n not in DEPLOYABLE_OBS]
    if undeployable:
        raise ValueError(
            f"actor observation(s) {undeployable} have no source on the robot "
            f"(deployable: {sorted(DEPLOYABLE_OBS)}) -- a policy that observes "
            "them cannot be exported"
        )

    # One reset gives a real catalog to read component widths from. mjx.forward
    # only; nothing is stepped.
    state = env.reset(jax.random.PRNGKey(0))
    catalog = env._obs_catalog(state.data, state.info)
    layout = [{"name": n, "size": int(np.asarray(catalog[n]).size)} for n in names]

    m = env.mj_model
    forcerange = np.asarray(m.actuator_forcerange, np.float64)
    c = env._config.command
    gait = env._config.gait
    hydra = run.get("hydra_config") or {}
    preset = ((hydra.get("actuators") or {}).get("name")) or ""
    # The clock offsets as the env itself wraps them, measured off
    # _leg_phases at clock zero rather than copied from the task module's
    # constant: the runtime applies the same wrap, so these are the numbers
    # its cos/sin sees.
    offsets = np.asarray(env._leg_phases({"phase": jp.array(0.0)}), np.float64)

    return {
        "schema_version": SCHEMA_VERSION,
        # -- provenance -----------------------------------------------------
        "run_name": run.get("run_name", ""),
        "task": task,
        "checkpoint": str(checkpoint),
        "robot": env.robot_spec.name,
        "preset": preset,
        "actuator_model": env._preset.model,
        # -- what the network reads -----------------------------------------
        "obs_layout": layout,
        "obs_size": int(sum(component["size"] for component in layout)),
        "action_size": int(env.action_size),
        # -- what the network drives ----------------------------------------
        # Actuator (and action, and joint_pos) column order.
        "joint_names": list(env.robot_spec.actuated_joints),
        # ctrl a zero action commands: the action space's centre.
        "anchor_ctrl": np.asarray(env._neutral_ctrl, np.float64).tolist(),
        # The observation anchor: joint_pos = qpos - default_pose. Equal to
        # anchor_ctrl under a pd preset and unrelated to it under an
        # ideal_torque one, where ctrl is a torque.
        "default_pose": np.asarray(env._default_pose, np.float64).tolist(),
        "action_scale": np.asarray(env._action_scale, np.float64).tolist(),
        "ctrl_low": np.asarray(env._ctrl_lo, np.float64).tolist(),
        "ctrl_high": np.asarray(env._ctrl_hi, np.float64).tolist(),
        # rad for a position servo, Nm for a direct-torque actuator.
        "ctrl_unit": "Nm" if env._preset.model == "ideal_torque" else "rad",
        "ctrl_dt": float(env._config.ctrl_dt),
        # -- what the plant has to be able to do ------------------------------
        # Per-actuator torque envelope and the gains the policy trained
        # against, both read back off the compiled model (effective_gains,
        # the same block run.json stamps). The driver on the
        # robot runs these numbers; the contract is what says which.
        "torque_low": forcerange[:, 0].tolist(),
        "torque_high": forcerange[:, 1].tolist(),
        "gains": effective_gains(
            m.actuator_gainprm,
            m.actuator_biasprm,
            env.robot_spec.actuated_joints,
            model=env._preset.model,
            preset=preset,
        ),
        # -- the operator's command box ---------------------------------------
        # (vx, vy, wz), the order the `command` observation is packed in.
        # Every dimension has a /cmd_vel source on the robot, so there is no
        # dimension the runtime has to fill in itself.
        "command_low": [float(c.vx[0]), float(c.vy[0]), float(c.wz[0])],
        "command_high": [float(c.vx[1]), float(c.vy[1]), float(c.wz[1])],
        # -- the clock the actor observes -------------------------------------
        # `phase` is on the actor list, so the runtime integrates the env's
        # own clock: per-foot offsets, a frequency lerped over the commanded
        # speed fraction, frozen below the deadband. envs/joystick.py's
        # _phase_dt / _leg_phases are the definition; these are its resolved
        # constants.
        "gait_clock": {
            "offsets": offsets.tolist(),
            "freq_low": float(gait.freq[0]),
            "freq_high": float(gait.freq[1]),
            # |v_xy| + turn_weight * |wz|, normalized by cmd_speed_max, is
            # the speed fraction the frequency lerps over. Shipped from the
            # env's own constants (envs/progress.py), never re-typed here.
            "turn_weight": progress.YAW_SPEED_WEIGHT,
            "speed_deadband": progress.SPEED_DEADBAND,
            "cmd_speed_max": float(env._cmd_vmax),
        },
    }
