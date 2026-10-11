"""Reward composition: the reward terms, their gates, the sum and the clip.

This is a shared module. At module scope it may import only array_api_compat,
numpy, the standard library and other shared modules. Never jax, torch,
mujoco, brax, ml_collections or any env.

`compute_rewards` is the body of Joystick._compute_rewards. `r` is the
`signals.readings` dict, `info` the env info dict, and `cfg` the whole env
config. Every array has leading batch dims (none for one world). Flags and
parameters under `cfg.reward` are read at call time. The style gates,
`orientation_tol` and `deadband` come from `consts`, fixed at construction.
`total_reward` is the scaled sum and the clip.
"""

from __future__ import annotations

import array_api_compat

from humanoid_lab.envs import progress, signals
from humanoid_lab.envs.constants import xp_of
from humanoid_lab.rewards import terms


def compute_rewards(r, info, action, first_contact, contact, consts, cfg):
    """(rewards dict, fall) for one step. Dict order is the order of the sum."""
    xp = xp_of(r["gyro"])
    cmd = info["command"]
    linvel = r["linvel"]
    gyro = r["gyro"]
    gravity = r["gravity"]
    rcfg = cfg.reward
    moving = progress.cmd_speed(cmd) > consts.deadband

    qpos_act = r["qpos_act"]
    qvel_act = r["qvel_act"]
    actuator_force = r["actuator_force"]

    foot_clearance = r["foot_clearance"]
    target_clearance, _stance_mask = progress.gait_targets(info["phase"], cfg.gait.duty, cfg.gait.swing_height)
    foot_vel = r["foot_linvel"]

    fall = signals.fall(r["base_height"], gravity[..., 2], cfg.fall.min_height, cfg.fall.max_tilt_gz)

    # Velocity tracking kernels: absolute or command-relative, optionally
    # blended with a wider far-field exponential, optionally gating each
    # other. Every switch below is config read here at call time under a
    # plain Python if, so all-off gives exactly the two legacy absolute
    # kernels.
    err_lin = terms.tracking_err_lin(cmd[..., :2], linvel[..., :2])
    err_ang = terms.tracking_err_ang(cmd[..., 2], gyro[..., 2])

    far_w = rcfg.get("tracking_far_weight", 0.0)

    def _far_blend(kernel, err_sq):
        """The far-field mix-in (see tracking_far_weight in default_config).
        far_w is plain config, so weight 0 returns the bare kernel untouched
        rather than a numerically-equal blend."""
        if not far_w:
            return kernel
        return terms.tracking_far_blend(kernel, err_sq, far_w, rcfg.tracking_far_sigma)

    if rcfg.get("tracking_relative", False):
        k_lin = terms.tracking_kernel(
            err_lin,
            terms.tracking_rel_sigma(
                xp.linalg.vector_norm(cmd[..., :2], axis=-1), rcfg.tracking_rel_sigma, rcfg.tracking_rel_floor_lin
            ),
        )
        k_ang = terms.tracking_kernel(
            err_ang,
            terms.tracking_rel_sigma(xp.abs(cmd[..., 2]), rcfg.tracking_rel_sigma, rcfg.tracking_rel_floor_ang),
        )
    else:
        k_lin = terms.tracking_kernel(err_lin, rcfg.tracking_sigma)
        k_ang = terms.tracking_kernel(err_ang, rcfg.tracking_sigma)
    # Both branches blend, and both pass the raw squared error: the far
    # kernel is absolute either way.
    k_lin = _far_blend(k_lin, err_lin)
    k_ang = _far_blend(k_ang, err_ang)
    if rcfg.get("tracking_product", False):
        # Gate each term by the other's kernel: tracking pays only for
        # tracking the whole command (see tracking_product in
        # default_config). The reassignment is simultaneous, so both
        # sides read the pre-product kernels and both come out equal to
        # the same product.
        k_lin, k_ang = k_lin * k_ang, k_ang * k_lin

    # Gait-shaping gate (see shaping_tracking_gate in default_config):
    # the positive gait terms follow the linear tracking kernel, after
    # the product gate when that is on. Gated set: feet_air_time,
    # feet_air_time_biped, feet_apex, and gait_symmetry_income and
    # feet_apex_min (below).
    # feet_phase stays ungated on purpose -- it is the clock-following
    # gradient, and it has to survive at zero tracking because stepping
    # is how tracking starts.
    shape_gate = k_lin if rcfg.get("shaping_tracking_gate", False) else 1.0

    # sin^2 of the tilt from vertical, less the tolerance cone (see
    # reward.orientation_tol_deg). The cone is a construct-time float, so
    # the default 0 leaves the legacy expression exactly as it was.
    orientation = terms.orientation(gravity[..., :2])
    if consts.orientation_tol:
        orientation = xp.clip(orientation - consts.orientation_tol, 0.0, None)

    batch = gyro.shape[:-1]
    device = array_api_compat.device(gyro)

    rewards = {
        "tracking_lin_vel": k_lin,
        "tracking_ang_vel": k_ang,
        "lin_vel_z": terms.lin_vel_z(linvel[..., 2]),
        "ang_vel_xy": terms.ang_vel_xy(gyro[..., :2]),
        "orientation": orientation,
        "torques": terms.torques(actuator_force),
        "torque_rate": terms.torque_rate(actuator_force, info["last_torque"]),
        "action_rate": terms.action_rate(action, info["last_action"]),
        "action_accel": terms.action_accel(action, info["last_action"], info["last_last_action"]),
        "energy": terms.energy(qvel_act, actuator_force),
        # Both pose terms score the deviation from pose_anchor, which is
        # default_pose itself unless real_pose_ref settled one (see
        # default_config). The anchor is a REWARD reference only: the
        # action still centers on default_pose.
        "pose": terms.pose(qpos_act, consts.pose_anchor, consts.pose_weight),
        "feet_air_time": terms.feet_air_time(info["feet_air_time"], first_contact, cfg.gait.air_time_cap)
        * moving
        * shape_gate,
        "feet_slip": terms.feet_slip(foot_vel[..., :2], contact) * moving,
        "feet_phase": terms.feet_phase(foot_clearance, target_clearance, rcfg.phase_sigma) * moving,
        "stand_still": terms.stand_still(
            qpos_act, consts.pose_anchor, qvel_act,
            rcfg.get("stand_still_vel_weight", 0.2),
        ) * (~moving),
        "termination": terms.termination(fall),
        "torque_limit": terms.torque_limit(actuator_force, consts.torque_cap, rcfg.torque_limit_frac),
        # Appended at the END on purpose: the scaled sum adds the terms in
        # dict order, so a key inserted mid-dict shifts every later float
        # addition and breaks the recorded goldens.
        #
        # feet_apex is in the shape_gate's set (see above): a tall swing
        # while the command goes unserved is the same stand-and-lift
        # income the gate exists to remove. feet_landing is NOT -- it is
        # a penalty, and gating a penalty on the tracking kernel would
        # relax it exactly when tracking is failing, which is when feet
        # are being slammed into the floor.
        "feet_apex": terms.feet_apex(info["swing_apex"], first_contact, rcfg.apex_target)
        * moving
        * shape_gate,
        "feet_landing": terms.feet_landing(foot_vel[..., 2], foot_clearance, rcfg.glide_height)
        * moving,
        # Ported robolab terms, after the pre-port set for the same
        # order reason as feet_apex.
        "pose_l1": terms.pose_l1(qpos_act, consts.pose_anchor, consts.pose_l1_weight),
        "joint_pos_limits": terms.joint_pos_limits(qpos_act, consts.soft_lo, consts.soft_hi),
        "joint_vel": terms.joint_vel(qvel_act),
        "joint_acc": terms.joint_acc(r["qacc_act"]),
        "upward": terms.upward(gravity[..., 2]),
        "feet_distance": terms.distance_band(r["feet_sep"], *rcfg.feet_distance_range)
        if "feet_sep" in r
        else xp.zeros(batch, dtype=xp.float32, device=device),
        "knee_distance": terms.distance_band(r["knee_sep"], *rcfg.knee_distance_range)
        if "knee_sep" in r
        else xp.zeros(batch, dtype=xp.float32, device=device),
        # Zero-command mask, like stand_still. Upstream gates on
        # |cmd| < 0.01; ~moving uses the shared deadband.
        "feet_contact_without_cmd": terms.feet_contact_without_cmd(contact, gravity[..., 2])
        * (~moving),
        # The mode times read the pre-update info on purpose: the
        # bookkeeping after the reward call (the landing-event
        # feet_air_time needs the completed swing's time) runs later, so this
        # term sees each dwell one control step late. A dt lag against a 0.4 s
        # clamp does not change the shape. Gated by `moving`, not
        # upstream's |cmd_xy| > 0.1: our pure-spin commands need steps
        # too. In the shape_gate set with the other positive gait terms,
        # since single-stance dwell is stand-and-lift income by
        # definition.
        "feet_air_time_biped": terms.feet_air_time_biped(
            info["feet_air_time"],
            info["feet_contact_time"],
            contact | info["last_contact"],
            rcfg.get("biped_air_time_threshold", 0.4),
        )
        * moving
        * shape_gate,
    }
    # Style terms, gated by consts (see Joystick.__init__): absent when the
    # gate is off, so the goldens' metric set and float-add order are
    # untouched.
    # knee_stance and gait_symmetry are penalties, so neither joins the
    # shape_gate set (the feet_landing rationale above);
    # gait_symmetry_income is a positive gait term and does, since a
    # symmetric single stance while the command goes unserved is
    # stand-and-lift income like feet_air_time_biped's. knee_stance is
    # masked by `moving`: at a zero command stand_still and pose_l1
    # anchor the knees on the home keyframe's 0.3 rad, and an unmasked
    # cone would fight that anchor into exactly the standing dither it
    # must not add.
    if consts.knee_stance_on:
        rewards["knee_stance"] = (
            terms.knee_stance(qpos_act[..., consts.knee_qidx], contact, rcfg.knee_stance_tol)
            * moving
        )
    if consts.gait_symmetry_on:
        rewards["gait_symmetry"] = (
            terms.gait_symmetry(
                info["air_dur_ema"],
                info["stance_dur_ema"],
                rcfg.gait_symmetry_floor,
                rcfg.get("gait_symmetry_cap", 1.0),
            )
            * moving
        )
    if consts.gait_symmetry_income_on:
        rewards["gait_symmetry_income"] = (
            terms.gait_symmetry_income(
                info["air_dur_ema"],
                info["stance_dur_ema"],
                contact,
                rcfg.gait_symmetry_floor,
                rcfg.get("gait_symmetry_cap", 1.0),
            )
            * moving
            * shape_gate
        )
    if consts.feet_apex_min_on:
        # In the shape_gate set for feet_apex's reason.
        rewards["feet_apex_min"] = (
            terms.feet_apex_min(info["swing_apex"], info["last_apex"], first_contact, rcfg.apex_target)
            * moving
            * shape_gate
        )
    return rewards, fall


def total_reward(rewards, scales, dt):
    """The scaled sum in dict order, times dt, clipped to +-100."""
    reward = sum(rewards[k] * scales[k] for k in rewards)
    xp = xp_of(reward)
    return xp.clip(reward * dt, -100.0, 100.0)
