"""Reward term library: one pure function per reward component.

Each function reads only generic arrays (sensor readings, qpos/qvel slices
already gathered by the caller, actions, precomputed spec-derived indices)
-- never a joint, site or body name -- so the same library works for any
robot/task that assembles its own inputs.

Absent on purpose, because they are quadruped-only: contact_match
(diagonal-pair matching), high_step, height_tracking (variable stand height
plus its height table), and the splay/knee terms. feet_phase's gait clock is
a biped two-foot antiphase clock; that clock lives in envs/progress.py, not
here -- this module only scores the resulting clearance error.

The two velocity-tracking terms are the one exception to one-function-per-
component: they split into a squared error, a width, and the exp kernel that
consumes both, because the env's tracking_relative and tracking_far_weight
branches need the error and the width separately (see envs/joystick.py).

Batch convention: component axes (joints, feet, xy) trail any leading world
axes, and each term returns one value per world: shape () for unbatched
inputs, (N,) for (N, k) inputs. Masking by task context (e.g. "* moving", "*
~moving") is the caller's job, not this library's: "moving" is a
joystick-task concept, not a property of the reward math itself.

The three landing terms (step_length, foot_reach, knee_extend) return a pair
(pay, frac). pay has one value per world. frac is the per-foot fraction
before the pair-min, shape (..., F). The caller records frac into its
land_frac state and passes the previous one back as last_frac.

This is a shared module: it imports only array_api_compat, numpy and other
shared modules, and takes its array namespace from the data arguments
(envs.constants.xp_of). The same code runs on JAX, Torch and NumPy. The op
forms below keep the JAX bits and also run under array_api_compat.torch:
  - reduce with axis=-1 and index with [..., k];
  - xp.clip(x, lo, None) / xp.clip(x, None, hi) replace maximum/minimum
    against a Python scalar (torch rejects a scalar there);
  - xp.flip(x, axis=-1) replaces [..., ::-1] (torch rejects negative steps);
  - xp.astype(x, dtype) replaces .astype;
  - config scalars stay Python floats.
"""

from __future__ import annotations

import math

from humanoid_lab.envs.constants import xp_of

# Floors on the landing targets, in metres. A zero or tiny commanded step
# would otherwise divide by zero or pay a full fraction for any step.
STEP_LENGTH_MIN_TARGET = 0.05
FOOT_REACH_MIN_TARGET = 0.02


def tracking_err_lin(cmd_xy, linvel_xy):
    """Squared planar velocity tracking error."""
    xp = xp_of(cmd_xy, linvel_xy)
    return xp.sum(xp.square(cmd_xy - linvel_xy), axis=-1)


def tracking_err_ang(cmd_wz, gyro_z):
    """Squared yaw rate tracking error."""
    xp = xp_of(cmd_wz, gyro_z)
    return xp.square(cmd_wz - gyro_z)


def tracking_kernel(err_sq, sigma):
    """exp(-err^2/sigma); 1.0 at perfect tracking. The error and the width
    are separate arguments because the caller chooses the width: a fixed
    tracking_sigma for the absolute kernel, tracking_rel_sigma() for the
    command-relative one."""
    xp = xp_of(err_sq)
    return xp.exp(-err_sq / sigma)


def tracking_rel_sigma(cmd_magnitude, rel_sigma: float, floor: float):
    """Command-relative kernel width: rel_sigma * max(|cmd|, floor)^2.

    Dividing the squared error by the squared command makes the kernel score
    the FRACTION of the command tracked, so 80% of target pays the same at
    any commanded speed. rel_sigma is therefore dimensionless. The floor
    keeps a small or zero command from sharpening the kernel to a point and
    dividing by zero.
    """
    xp = xp_of(cmd_magnitude)
    return rel_sigma * xp.square(xp.clip(cmd_magnitude, floor, None))


def tracking_far_blend(kernel, err_sq, weight: float, far_sigma: float):
    """Mix a wide exponential into a tracking kernel:
    (1-weight)*kernel + weight*exp(-err^2/far_sigma).

    exp(-err^2/sigma) is gradient-free once the error is a few sigma out, so
    a capability the policy never explored gets no pull toward the command at
    all. The wide second exponential keeps a usable gradient at range. Both
    exponentials peak at zero error, so the optimum and the [0, 1] bound are
    unchanged.

    err_sq is the raw squared error, never a relative one: the far kernel
    stays absolute in both branches, so a state far off the command sees the
    same pull at any commanded speed.
    """
    xp = xp_of(kernel, err_sq)
    return (1.0 - weight) * kernel + weight * xp.exp(-err_sq / far_sigma)


def lin_vel_z(linvel_z):
    """Penalize vertical bounce."""
    xp = xp_of(linvel_z)
    return xp.square(linvel_z)


def ang_vel_xy(gyro_xy):
    """Penalize roll/pitch angular velocity."""
    xp = xp_of(gyro_xy)
    return xp.sum(xp.square(gyro_xy), axis=-1)


def orientation(gravity_xy):
    """Penalize base tilt: nonzero xy of the body-frame gravity direction."""
    xp = xp_of(gravity_xy)
    return xp.sum(xp.square(gravity_xy), axis=-1)


def torques(actuator_force):
    """Sum of squared actuator torques."""
    xp = xp_of(actuator_force)
    return xp.sum(xp.square(actuator_force), axis=-1)


def torque_rate(actuator_force, last_torque):
    """Step-to-step torque change: penalizes bang-bang motor commands
    directly at the actuator (action_rate only sees the policy output, not
    what the PD loop does with it)."""
    xp = xp_of(actuator_force, last_torque)
    return xp.sum(xp.square(actuator_force - last_torque), axis=-1)


def action_rate(action, last_action):
    """Step-to-step change in the raw policy action."""
    xp = xp_of(action, last_action)
    return xp.sum(xp.square(action - last_action), axis=-1)


def action_accel(action, last_action, last_last_action):
    """Second-order (acceleration) change in the raw policy action."""
    xp = xp_of(action, last_action, last_last_action)
    return xp.sum(xp.square(action - 2 * last_action + last_last_action), axis=-1)


def energy(qvel_actuated, actuator_force):
    """Mechanical power cost: sum |tau * qvel| over actuated joints."""
    xp = xp_of(qvel_actuated, actuator_force)
    return xp.sum(xp.abs(qvel_actuated) * xp.abs(actuator_force), axis=-1)


def pose(qpos_actuated, default_pose, weights):
    """Weighted deviation from the default (home keyframe) pose."""
    xp = xp_of(qpos_actuated, default_pose, weights)
    return xp.sum(weights * xp.square(qpos_actuated - default_pose), axis=-1)


def feet_air_time(air_time, first_contact, air_time_cap: float = 0.0, min_air_time: float = 0.1):
    """Reward a swing phase lasting at least min_air_time, capped so an
    unbounded flight isn't its own reward-maximizing strategy."""
    xp = xp_of(air_time, first_contact)
    capped = xp.clip(air_time, None, air_time_cap) if air_time_cap > 0 else air_time
    return xp.sum((capped - min_air_time) * first_contact, axis=-1)


def feet_slip(foot_linvel_xy, contact):
    """Penalize horizontal foot speed while a foot is in contact (skating)."""
    xp = xp_of(foot_linvel_xy, contact)
    return xp.sum(xp.sum(xp.square(foot_linvel_xy), axis=-1) * contact, axis=-1)


def feet_apex(swing_apex, first_contact, apex_target: float):
    """Pay each completed swing for how close its PEAK clearance came to
    `apex_target`, once, on the step the foot lands.

    A duration-averaged clearance term such as feet_phase tolerates a long
    1.5-2 cm skim that collects nearly as much as a crisp arc, so the
    optimizer skims. Pricing the peak instead has measured 3 to 5 cm swings
    and 30 to 70% better grip. Clipped at the target: the term
    asks for an apex, it does not pay for exceeding it.

    `swing_apex` is the caller's running maximum over the swing (the env
    tracks it in its info dict); `first_contact` selects the feet whose swing
    ended this step.
    """
    xp = xp_of(swing_apex, first_contact)
    return xp.sum(xp.clip(swing_apex / apex_target, 0.0, 1.0) * first_contact, axis=-1)


def feet_apex_min(swing_apex, last_apex, first_contact, apex_target: float):
    """Pay each completed swing for the LOWER of its own peak clearance and
    the other foot's last completed peak, as a fraction of `apex_target`,
    once, on the step the foot lands. Two feet only.

    feet_apex pays each foot its own peak, so a gait that lifts one leg and
    shuffles the other collects half of the term with the lifting leg. Here
    a landing pays clip(min(swing_apex[i], last_apex[other]) / apex_target,
    0, 1): the lifting leg's landings earn only as much as the other leg's
    most recent swing reached, and a gait earns the full payout only when
    both legs reach the target. Clipped at the target, like feet_apex.

    `swing_apex` is the caller's running maximum over the current swing, as
    for feet_apex. `last_apex` holds each foot's peak over its last
    completed swing, 0 before that foot's first landing; the caller updates
    it after this read, so on a simultaneous landing each foot is paired
    with the other's previous swing. `first_contact` selects the feet whose
    swing ended this step.
    """
    xp = xp_of(swing_apex, last_apex, first_contact)
    if swing_apex.shape[-1] != 2:
        raise ValueError(f"feet_apex_min needs two feet, got {swing_apex.shape[-1]}")
    paired = xp.minimum(swing_apex, xp.flip(last_apex, axis=-1))
    return xp.sum(xp.clip(paired / apex_target, 0.0, 1.0) * first_contact, axis=-1)


def feet_landing(foot_vz, foot_clearance, glide_height: float):
    """Penalize downward foot speed, weighted by closeness to the floor:
    sum(min(foot_vz, 0)^2 * clip(1 - clearance/glide_height, 0, 1)).

    Measured BEFORE contact on purpose. A penalty read at contact under-reads
    hard strikes, because the solver has already absorbed the impact within
    the control step it becomes visible. Weighting by proximity instead makes
    the gradient read "decelerate as you approach": 1 at the floor, 0 at
    `glide_height` and above. Stance feet score ~0 (vz ~ 0), and a swing high
    above the floor scores 0 whatever its speed.

    The physical reference for touchdown softness is free fall over the glide
    band: sqrt(2*9.81*0.03) ~ 0.77 m/s at a 0.03 m band.
    """
    xp = xp_of(foot_vz, foot_clearance)
    return xp.sum(
        xp.square(xp.clip(foot_vz, None, 0.0))
        * xp.clip(1.0 - foot_clearance / glide_height, 0.0, 1.0),
        axis=-1,
    )


def feet_phase(foot_clearance, target_clearance, phase_sigma: float):
    """exp(-err^2/sigma) between actual and gait-clock-commanded foot
    clearance; 1.0 when every foot matches its swing/stance target exactly."""
    xp = xp_of(foot_clearance, target_clearance)
    err = xp.sum(xp.square(foot_clearance - target_clearance), axis=-1)
    return xp.exp(-err / phase_sigma)


def stand_still(qpos_actuated, default_pose, qvel_actuated, vel_weight: float = 0.2):
    """Position pull to the default pose plus velocity damping, for the
    zero-command (stand) case. L1 position alone causes bang-bang fidgeting
    around the anchor. `vel_weight` sets the damping share; only the ratio
    matters, since `scales.stand_still` prices the sum."""
    xp = xp_of(qpos_actuated, default_pose, qvel_actuated)
    return xp.sum(xp.abs(qpos_actuated - default_pose), axis=-1) + vel_weight * xp.sum(
        xp.abs(qvel_actuated), axis=-1
    )


def termination(fall):
    xp = xp_of(fall)
    return xp.astype(fall, xp.float32)


def torque_limit(actuator_force, torque_cap, frac: float):
    """Actuator-saturation hinge: 0 well inside the cap, positive above
    `frac` of it. A soft margin on top of the actuator's hard forcerange."""
    xp = xp_of(actuator_force, torque_cap)
    return xp.sum(xp.clip(xp.abs(actuator_force) - frac * torque_cap, 0.0, None), axis=-1)


# -- terms ported from robolab (tasks/direct/base/mdp/rewards.py, at the
# commit docs/plans/roboto-first-run.md pins). Same shapes, so the
# upstream weights transfer 1:1.


def pose_l1(qpos_actuated, default_pose, weights):
    """Weighted L1 deviation from the default pose (upstream
    joint_deviation_l1, its four groups as one weight vector)."""
    xp = xp_of(qpos_actuated, default_pose, weights)
    return xp.sum(weights * xp.abs(qpos_actuated - default_pose), axis=-1)


def joint_pos_limits(qpos_actuated, soft_lo, soft_hi):
    """L1 excursion outside the soft joint limits."""
    xp = xp_of(qpos_actuated, soft_lo, soft_hi)
    return xp.sum(
        xp.clip(soft_lo - qpos_actuated, 0.0, None) + xp.clip(qpos_actuated - soft_hi, 0.0, None),
        axis=-1,
    )


def joint_vel(qvel_actuated):
    """Sum of squared actuated-joint velocities (upstream joint_vel_l2)."""
    xp = xp_of(qvel_actuated)
    return xp.sum(xp.square(qvel_actuated), axis=-1)


def joint_acc(qacc_actuated):
    """Sum of squared actuated-joint accelerations (upstream joint_acc_l2)."""
    xp = xp_of(qacc_actuated)
    return xp.sum(xp.square(qacc_actuated), axis=-1)


def upward(gravity_z):
    """Uprightness reward (upstream upward): -gravity_body_z, ~1 upright."""
    return -gravity_z


def distance_band(separation, band_min: float, band_max: float):
    """Keep a separation inside [band_min, band_max] (upstream
    body_distance_y): 1.0 in the band, exp(-100 * excursion) outside.
    The 0.5 m clamp and the 100/m decay are upstream constants."""
    xp = xp_of(separation)
    d_min = xp.clip(separation - band_min, -0.5, 0.0)
    d_max = xp.clip(separation - band_max, 0.0, 0.5)
    return (xp.exp(-xp.abs(d_min) * 100.0) + xp.exp(-xp.abs(d_max) * 100.0)) / 2.0


def feet_contact_without_cmd(contact, gravity_z):
    """All feet planted, scaled by uprightness (upstream
    feet_contact_without_cmd). The zero-command mask is the caller's."""
    xp = xp_of(contact, gravity_z)
    upright = xp.clip(-gravity_z, 0.0, 0.7) / 0.7
    return xp.all(contact, axis=-1) * upright


def feet_air_time_biped(air_time, contact_time, in_contact, threshold: float):
    """Per-step single-stance reward (upstream feet_air_time_positive_biped):
    while exactly one foot is in contact, pay the smaller of the feet's
    current mode times (air time for the swing foot, contact time for the
    stance foot), clamped at `threshold`. Double support and flight pay zero.

    Unlike feet_air_time above, which pays once per completed swing at
    landing, this pays from the first instant a foot lifts, so a policy that
    has never made a step still sees a gradient toward making one. The
    zero-command mask is the caller's."""
    xp = xp_of(air_time, contact_time, in_contact)
    in_mode_time = xp.where(in_contact, contact_time, air_time)
    single_stance = xp.sum(xp.astype(in_contact, xp.int32), axis=-1) == 1
    return xp.clip(
        xp.min(xp.where(single_stance[..., None], in_mode_time, 0.0), axis=-1), None, threshold
    )


def knee_stance(knee_qpos, contact, tol: float):
    """Penalize knee flexion beyond `tol` while that leg's foot is in
    contact: sum(contact * max(|q_knee| - tol, 0)^2).

    None of the existing terms prices the stance leg's shape, so a
    permanently crouched stance is reward-neutral and the optimizer keeps
    it. Charging flexion only DURING contact leaves the swing leg free to
    bend as much as the step needs; the tolerance leaves the shock-absorbing
    flexion at touchdown unpriced and charges only the standing crouch.

    `knee_qpos` must be ordered to match `contact` (foot_sites order); the
    caller owns that alignment."""
    xp = xp_of(knee_qpos, contact)
    excess = xp.clip(xp.abs(knee_qpos) - tol, 0.0, None)
    return xp.sum(xp.square(excess) * contact, axis=-1)


def gait_symmetry(air_dur_ema, stance_dur_ema, floor: float, cap: float = 1.0):
    """Relative left-right asymmetry of the completed swing and stance
    durations: for each pair, ((d0 - d1) / max(mean(d), floor))^2, summed
    over the two pairs and clipped at `cap`.

    The velocity/gait terms price each step on its own, so a gait that
    limps -- one leg taking systematically longer swings than the other --
    pays the same as an even one. This term reads the caller's per-foot
    EMAs of completed swing and stance durations and charges their relative
    difference, so a 20% limp costs the same at any cadence.

    Each pair arms only once BOTH feet have a completed duration on record:
    the first step of an episode is one-legged by definition, and charging
    that transient would penalize starting to walk at all. The floor bounds
    the denominator away from zero for the freshly-armed case.

    The cap bounds the worst case, and it is what makes the term safe to
    arm at all: the run-5 gates measured that an UNCAPPED charge kills the
    gait before it forms. A first clumsy gait has near-maximal relative
    asymmetry (each pair saturates near (2d/d)^2 = 4, two pairs ~8), so at
    weight -2.0 the uncapped term taxed the fragile first-steps window at
    ~0.32/step while standing -- which never arms the term -- collected
    ~0.45/step of tracking for free. The optimizer took standing, twice
    (gates job-11, job-13). Capped, the exploration-phase fee is bounded
    at scale*cap per step while a settled limp (rel_sq well under the cap)
    still pays in proportion to its asymmetry."""

    xp = xp_of(air_dur_ema, stance_dur_ema)
    return xp.clip(
        _pair_rel_sq(air_dur_ema, floor)[0] + _pair_rel_sq(stance_dur_ema, floor)[0], None, cap
    )


def gait_symmetry_income(air_dur_ema, stance_dur_ema, in_contact, floor: float, cap: float = 1.0):
    """Positive twin of gait_symmetry: pay (1 - min(rel_sq_sum, cap) / cap)
    on every single-stance step once at least one duration pair is armed.

    rel_sq_sum is gait_symmetry's summed relative asymmetry over the same
    EMAs, with the same per-pair arming: an unarmed pair contributes 0 and
    does not count toward arming. An even gait earns 1 per single-stance
    step, a gait at or over the cap earns 0, and the pay falls linearly in
    rel_sq_sum between them.

    The penalty form charges asymmetry, so standing, which never arms it,
    is its cheapest state. The income form pays for symmetry, so standing
    earns nothing from it and a clumsy first gait costs nothing relative to
    standing. Double support and flight pay 0, so a policy that has armed
    the EMAs and then settles into standing on both feet stops collecting.

    One-legged standing is not excluded here: the EMAs update only on
    liftoff and landing, so a foot held in stance keeps the last recorded
    durations and keeps earning. The caller's command mask and the tracking
    terms are what price that state. Arming is per episode only if the
    caller restarts the EMAs on every respawn (in training,
    envs/wrappers.py's GaitReseedWrapper does); EMAs carried across a
    respawn would pay from the new episode's first single stance.

    `in_contact` is per-foot contact in the same foot order as the EMAs.
    `cap` must be positive: it is the divisor."""
    xp = xp_of(air_dur_ema, stance_dur_ema, in_contact)
    air_sq, air_armed = _pair_rel_sq(air_dur_ema, floor)
    stance_sq, stance_armed = _pair_rel_sq(stance_dur_ema, floor)
    rel_sq_sum = air_sq + stance_sq
    single_stance = xp.sum(xp.astype(in_contact, xp.int32), axis=-1) == 1
    armed_any = air_armed | stance_armed
    return (1.0 - xp.clip(rel_sq_sum, None, cap) / cap) * single_stance * armed_any


def _pair_rel_sq(d, floor: float):
    """((d0 - d1) / max(mean(d), floor))^2 for one left-right duration pair,
    and whether the pair is armed (both durations recorded). An unarmed pair
    returns 0."""
    xp = xp_of(d)
    armed = (d[..., 0] > 0.0) & (d[..., 1] > 0.0)
    mean = xp.clip(0.5 * (d[..., 0] + d[..., 1]), floor, None)
    return xp.square((d[..., 0] - d[..., 1]) / mean) * armed, armed


# -- landing and lean terms ported from the Torch+MJWarp runner
# (wtrain/wenv.py). Masking by the forward command (progress.fwd_mask) is the
# caller's job.


def _two_feet(name, x):
    if x.shape[-1] != 2:
        raise ValueError(f"{name} needs two feet, got {x.shape[-1]}")


def _pair_min(frac, pair_min: bool, last_frac):
    """frac, or the lower of frac and the OTHER foot's last_frac.

    The pair-min exists so one leg's good step cannot carry a shuffling
    partner. With it on, a landing pays no more than the other foot's last
    landing earned.
    """
    if not pair_min:
        return frac
    xp = xp_of(frac, last_frac)
    return xp.minimum(frac, xp.flip(last_frac, axis=-1))


def step_length(foot_x, first_contact, target, power: float = 1.0, pair_min: bool = False, last_frac=None):
    """Pay each landing for how far the landing foot is ahead of the other foot.

    `foot_x` is each foot's x in the base frame, shape (..., 2). `target` is
    the step length to aim for, one value per world. It is floored at
    STEP_LENGTH_MIN_TARGET. The per-foot fraction is clip(ahead / target, 0,
    1) ** power. A foot landing behind the other one earns 0. Power 1 pays the
    same per second at any cadence for a given speed. Power 2 makes the pay
    per second grow with the step itself.

    Returns (pay, frac). pay sums the paid fraction over the feet landing this
    step. frac is the fraction before the pair-min.
    """
    xp = xp_of(foot_x, first_contact, target, last_frac)
    _two_feet("step_length", foot_x)
    ahead = foot_x - xp.flip(foot_x, axis=-1)
    t = xp.clip(target, STEP_LENGTH_MIN_TARGET, None)
    frac = xp.clip(ahead / t[..., None], 0.0, 1.0) ** power
    paid = _pair_min(frac, pair_min, last_frac)
    return xp.sum(paid * first_contact, axis=-1), frac


def foot_reach(
    foot_x, first_contact, target, x_min: float = 0.0, power: float = 2.0, pair_min: bool = False, last_frac=None
):
    """Pay each landing for how far ahead of the base the foot lands.

    `foot_x` is each foot's x in the base frame, shape (..., 2). `target` is
    the reach to aim for, one value per world. It is floored at
    FOOT_REACH_MIN_TARGET. The per-foot fraction is clip((foot_x - x_min) /
    (target - x_min), 0, 1) ** power. A foot landing at x_min or further back
    earns 0.

    Returns (pay, frac), as step_length does. Only the pair-min needs two
    feet. The term refuses any other foot count anyway, like its caller.
    """
    xp = xp_of(foot_x, first_contact, target, last_frac)
    _two_feet("foot_reach", foot_x)
    t = xp.clip(target, FOOT_REACH_MIN_TARGET, None)
    frac = xp.clip((foot_x - x_min) / (t[..., None] - x_min), 0.0, 1.0) ** power
    paid = _pair_min(frac, pair_min, last_frac)
    return xp.sum(paid * first_contact, axis=-1), frac


def knee_extend(
    knee_q,
    foot_x,
    contact,
    first_contact,
    air_time,
    flex_lo: float,
    flex_hi: float,
    x_split: float = 0.0,
    td_min_air: float = 0.2,
    td_mult: float = 0.0,
    pair_min: bool = False,
    last_frac=None,
):
    """Pay an open knee in late swing, plus an optional bonus at touchdown.

    `knee_q` is each foot's knee angle, in foot order, shape (..., F). The
    extension is 1 at flex_lo or straighter and 0 at flex_hi. It is clipped to
    [0, 1]. Late swing means the foot is in the air and ahead of x_split in
    the base frame. Paying extension there makes the foot reach forward by
    opening the knee instead of driving the thigh.

    The touchdown bonus pays td_mult times the extension on the landing step.
    Only a real step earns it: a swing of at least td_min_air seconds that
    lands ahead of the base. Without that gate a foot tapping in place farmed
    the bonus.

    Returns (pay, td_ext). td_ext is the touchdown extension before the
    pair-min. A tap or a step landing behind records 0.
    """
    xp = xp_of(knee_q, foot_x, contact, first_contact, air_time, last_frac)
    if pair_min:
        _two_feet("knee_extend", knee_q)
    ext = xp.clip((flex_hi - knee_q) / (flex_hi - flex_lo), 0.0, 1.0)
    late = (~contact) & (foot_x > x_split)
    real_td = first_contact & (air_time >= td_min_air) & (foot_x > 0.0)
    td_ext = ext * real_td
    paid = _pair_min(td_ext, pair_min, last_frac)
    pay = xp.sum(ext * late, axis=-1) + td_mult * xp.sum(paid * first_contact, axis=-1)
    return pay, td_ext


def forward_lean(gravity_x, cmd_vx, max_deg: float, vx_max: float, sigma: float):
    """Pay a forward lean that grows with the forward command.

    `gravity_x` is the x of the body-frame gravity direction. It reads
    +sin(pitch) for a nose-down lean. The target is
    sin(radians(max_deg) * frac), with frac = clip(cmd_vx / vx_max, 0, 1).
    The pay is exp(-(gravity_x - target)^2 / sigma), 1 at the target.
    """
    xp = xp_of(gravity_x, cmd_vx)
    frac = xp.clip(cmd_vx / vx_max, 0.0, 1.0)
    target = xp.sin(math.radians(max_deg) * frac)
    return xp.exp(-xp.square(gravity_x - target) / sigma)
