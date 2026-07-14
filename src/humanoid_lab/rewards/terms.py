"""Reward term library: one pure function per reward component.

Ported from w01-tek's wojtek_rl/env.py `_get_reward`. Each function reads
only generic arrays (sensor readings, qpos/qvel slices already gathered by
the caller, actions, precomputed spec-derived indices) -- never a joint,
site or body name -- so the same library works for any robot/task that
assembles its own inputs.

Dropped (quadruped/w01-tek-only, do not port): contact_match (diagonal-pair
matching), high_step, height_tracking (variable stand height + its height
table), splay/knee terms. feet_phase's gait clock is a biped two-foot
antiphase clock instead of w01-tek's 4-leg walk/trot blend; that clock lives
in envs/joystick.py, not here -- this module only scores the resulting
clearance error.

Every term returns a scalar. Masking by task context (e.g. "* moving", "*
~moving") is the caller's job, not this library's: "moving" is a
joystick-task concept, not a property of the reward math itself.
"""

from __future__ import annotations

import jax.numpy as jp


def tracking_lin_vel(cmd_xy, linvel_xy, sigma: float):
    """exp(-err^2/sigma); 1.0 at perfect tracking."""
    return jp.exp(-jp.sum(jp.square(cmd_xy - linvel_xy)) / sigma)


def tracking_ang_vel(cmd_wz, gyro_z, sigma: float):
    """exp(-err^2/sigma); 1.0 at perfect tracking."""
    return jp.exp(-jp.square(cmd_wz - gyro_z) / sigma)


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


def feet_phase(foot_clearance, target_clearance, phase_sigma: float):
    """exp(-err^2/sigma) between actual and gait-clock-commanded foot
    clearance; 1.0 when every foot matches its swing/stance target exactly."""
    err = jp.sum(jp.square(foot_clearance - target_clearance))
    return jp.exp(-err / phase_sigma)


def stand_still(qpos_actuated, default_pose, qvel_actuated):
    """Position pull to the default pose plus velocity damping, for the
    zero-command (stand) case. L1 position alone causes bang-bang fidgeting
    around the anchor."""
    return jp.sum(jp.abs(qpos_actuated - default_pose)) + 0.2 * jp.sum(jp.abs(qvel_actuated))


def termination(fall):
    return fall.astype(jp.float32)


def torque_limit(actuator_force, torque_cap, frac: float):
    """Actuator-saturation hinge: 0 well inside the cap, positive above
    `frac` of it. A soft margin on top of the actuator's hard forcerange."""
    return jp.sum(jp.maximum(jp.abs(actuator_force) - frac * torque_cap, 0.0))
