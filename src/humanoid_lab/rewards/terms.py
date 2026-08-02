"""Reward term library: one pure function per reward component.

Each function reads only generic arrays (sensor readings, qpos/qvel slices
already gathered by the caller, actions, precomputed spec-derived indices)
-- never a joint, site or body name -- so the same library works for any
robot/task that assembles its own inputs.

Absent on purpose, because they are quadruped-only: contact_match
(diagonal-pair matching), high_step, height_tracking (variable stand height
plus its height table), and the splay/knee terms. feet_phase's gait clock is
a biped two-foot antiphase clock; that clock lives in envs/joystick.py, not
here -- this module only scores the resulting clearance error.

The two velocity-tracking terms are the one exception to one-function-per-
component: they split into a squared error, a width, and the exp kernel that
consumes both, because the env's tracking_relative and tracking_far_weight
branches need the error and the width separately (see envs/joystick.py).

Every term returns a scalar. Masking by task context (e.g. "* moving", "*
~moving") is the caller's job, not this library's: "moving" is a
joystick-task concept, not a property of the reward math itself.
"""

from __future__ import annotations

import jax.numpy as jp


def tracking_err_lin(cmd_xy, linvel_xy):
    """Squared planar velocity tracking error."""
    return jp.sum(jp.square(cmd_xy - linvel_xy))


def tracking_err_ang(cmd_wz, gyro_z):
    """Squared yaw rate tracking error."""
    return jp.square(cmd_wz - gyro_z)


def tracking_kernel(err_sq, sigma):
    """exp(-err^2/sigma); 1.0 at perfect tracking. The error and the width
    are separate arguments because the caller chooses the width: a fixed
    tracking_sigma for the absolute kernel, tracking_rel_sigma() for the
    command-relative one."""
    return jp.exp(-err_sq / sigma)


def tracking_rel_sigma(cmd_magnitude, rel_sigma: float, floor: float):
    """Command-relative kernel width: rel_sigma * max(|cmd|, floor)^2.

    Dividing the squared error by the squared command makes the kernel score
    the FRACTION of the command tracked, so 80% of target pays the same at
    any commanded speed. rel_sigma is therefore dimensionless. The floor
    keeps a small or zero command from sharpening the kernel to a point and
    dividing by zero.
    """
    return rel_sigma * jp.square(jp.maximum(cmd_magnitude, floor))


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
    return (1.0 - weight) * kernel + weight * jp.exp(-err_sq / far_sigma)


def lin_vel_z(linvel_z):
    """Penalize vertical bounce."""
    return jp.square(linvel_z)


def ang_vel_xy(gyro_xy):
    """Penalize roll/pitch angular velocity."""
    return jp.sum(jp.square(gyro_xy))


def orientation(gravity_xy):
    """Penalize base tilt: nonzero xy of the body-frame gravity direction."""
    return jp.sum(jp.square(gravity_xy))


def torques(actuator_force):
    """Sum of squared actuator torques."""
    return jp.sum(jp.square(actuator_force))


def torque_rate(actuator_force, last_torque):
    """Step-to-step torque change: penalizes bang-bang motor commands
    directly at the actuator (action_rate only sees the policy output, not
    what the PD loop does with it)."""
    return jp.sum(jp.square(actuator_force - last_torque))


def action_rate(action, last_action):
    """Step-to-step change in the raw policy action."""
    return jp.sum(jp.square(action - last_action))


def action_accel(action, last_action, last_last_action):
    """Second-order (acceleration) change in the raw policy action."""
    return jp.sum(jp.square(action - 2 * last_action + last_last_action))


def energy(qvel_actuated, actuator_force):
    """Mechanical power cost: sum |tau * qvel| over actuated joints."""
    return jp.sum(jp.abs(qvel_actuated) * jp.abs(actuator_force))


def pose(qpos_actuated, default_pose, weights):
    """Weighted deviation from the default (home keyframe) pose."""
    return jp.sum(weights * jp.square(qpos_actuated - default_pose))


def feet_air_time(air_time, first_contact, air_time_cap: float = 0.0, min_air_time: float = 0.1):
    """Reward a swing phase lasting at least min_air_time, capped so an
    unbounded flight isn't its own reward-maximizing strategy."""
    capped = jp.minimum(air_time, air_time_cap) if air_time_cap > 0 else air_time
    return jp.sum((capped - min_air_time) * first_contact)


def feet_slip(foot_linvel_xy, contact):
    """Penalize horizontal foot speed while a foot is in contact (skating)."""
    return jp.sum(jp.sum(jp.square(foot_linvel_xy), axis=-1) * contact)


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
    return jp.sum(jp.clip(swing_apex / apex_target, 0.0, 1.0) * first_contact)


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
    return jp.sum(
        jp.square(jp.clip(foot_vz, None, 0.0))
        * jp.clip(1.0 - foot_clearance / glide_height, 0.0, 1.0)
    )


def feet_phase(foot_clearance, target_clearance, phase_sigma: float):
    """exp(-err^2/sigma) between actual and gait-clock-commanded foot
    clearance; 1.0 when every foot matches its swing/stance target exactly."""
    err = jp.sum(jp.square(foot_clearance - target_clearance))
    return jp.exp(-err / phase_sigma)


def stand_still(qpos_actuated, default_pose, qvel_actuated, vel_weight: float = 0.2):
    """Position pull to the default pose plus velocity damping, for the
    zero-command (stand) case. L1 position alone causes bang-bang fidgeting
    around the anchor. `vel_weight` sets the damping share; only the ratio
    matters, since `scales.stand_still` prices the sum."""
    return jp.sum(jp.abs(qpos_actuated - default_pose)) + vel_weight * jp.sum(jp.abs(qvel_actuated))


def termination(fall):
    return fall.astype(jp.float32)


def torque_limit(actuator_force, torque_cap, frac: float):
    """Actuator-saturation hinge: 0 well inside the cap, positive above
    `frac` of it. A soft margin on top of the actuator's hard forcerange."""
    return jp.sum(jp.maximum(jp.abs(actuator_force) - frac * torque_cap, 0.0))
