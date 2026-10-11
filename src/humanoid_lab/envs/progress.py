"""Command, gait clock and no-progress math, CaT-style (arXiv 2403.18765).

This is a shared module. At module scope it may import only array_api_compat,
numpy, the standard library and other shared modules. Never jax or torch.

Pure functions of plain arrays -- no env, no model, no state -- so the math is
testable on its own (tests/unit/test_progress_clock.py, test_no_progress.py)
and every backend runs the same code. Each function works on a single item or
on any leading batch shape. The EMA that smooths `served`, the cut and the
done flag are in envs/gait_state.update, because those need per-episode state.
The backend supplies the uniform draw behind the cut. The RNG draws for the
command also live in the backend. This module only turns draws into a command
(`command_from_draws`).

Config scalars stay Python floats and stay in the order the env used, so the
float32 results match the original jnp code bit for bit.
"""

from __future__ import annotations

import math

import array_api_compat

from humanoid_lab.envs.constants import xp_of

# SPEED_DEADBAND is the default of the config key command.deadband. The env,
# the no-progress cut and the deploy contract use the resolved value of that
# key. The constant lives here because joystick imports this module.
#
# A command asking for no more than the deadband is a stand command: the
# gait clock freezes, the `moving` mask opens, and the no-progress cut
# never arms. Units are commanded speed (`Joystick._cmd_speed`).
SPEED_DEADBAND = 0.05

# Weight of |yaw rate| in commanded speed -- an effective turning radius in
# metres. Used identically by `_cmd_speed` (gait-clock speed scaling) and
# by `served` below, so served and demand blend the same way and their
# ratio is meaningful. Carried from the quadruped predecessor's 0.21 m
# leg; re-derive for a robot of this scale.
YAW_SPEED_WEIGHT = 0.3

# |commanded wz| above this counts as a turn for the heading reference
# yaw_ref. It is separate from the stand deadband.
TURN_WZ_THRESHOLD = 0.05


def served(linvel_xy, gyro_z, command):
    """Speed actually delivered in service of `command`.

    Body-frame planar velocity projected onto the commanded direction, plus
    the yaw rate toward the commanded turn. Projection, not magnitude: moving
    against the command reads negative, which is worse than standing still,
    and moving across it scores zero.

    `linvel_xy` and `gyro_z` are body-frame; `command` is the full
    `(vx, vy, wz)`. A zero linear command divides by a 1e-6 floor rather than
    by zero (nothing arms at that demand anyway).
    """
    xp = xp_of(linvel_xy, gyro_z, command)
    direction = xp.clip(xp.linalg.vector_norm(command[..., :2], axis=-1), 1e-6, None)
    return (
        xp.vecdot(linvel_xy, command[..., :2], axis=-1) / direction
        + YAW_SPEED_WEIGHT * gyro_z * xp.sign(command[..., 2])
    )


def hazard(progress_ratio, risk_below, p_max):
    """Per-step cut probability from smoothed progress as a fraction of demand.

    Zero at and above `risk_below`, ramping linearly to `p_max` at zero
    progress and staying there for negative progress. At `p_max` the expected
    survival of a dead stop is 1/p_max control steps.
    """
    xp = xp_of(progress_ratio)
    return p_max * xp.clip((risk_below - progress_ratio) / risk_below, 0.0, 1.0)


def armed(demand, steps_since_cmd, dt, grace_sec, deadband=SPEED_DEADBAND):
    """Whether the hazard applies at all this step.

    Two conditions: the command asks for real motion
    (`demand > deadband`), and the command has been standing for at
    least `grace_sec`. The grace window covers the reset transient and,
    since `steps_since_cmd` restarts on every resample, the time it takes
    to turn a gait around for a new command.
    """
    return (demand > deadband) & (steps_since_cmd * dt >= grace_sec)


# Per-foot gait clock offset, robot_spec.foot_sites order: antiphase (one
# foot swings while the other stances), the only sensible clock for a biped:
# a quadruped's walk/trot blend has no two-foot analogue.
PHASE_OFFSETS = (0.0, math.pi)


def cmd_speed(command):
    """Planar speed the gait clock should serve; turning counts too."""
    xp = xp_of(command)
    return xp.linalg.vector_norm(command[..., :2], axis=-1) + YAW_SPEED_WEIGHT * xp.abs(
        command[..., 2]
    )


def clock_freq(speed, vmax, freq):
    """Gait frequency in Hz: `freq[0]` at rest up to `freq[1]` at `vmax`."""
    xp = xp_of(speed)
    return freq[0] + (freq[1] - freq[0]) * xp.clip(speed / vmax, 0.0, 1.0)


def phase_dt(command, vmax, freq, dt, deadband):
    """Clock increment: speed-scaled; frozen when told to stand.

    `vmax` is the commanded speed at the top of the vx range, `freq` the
    (low, high) config pair, `dt` the control step.
    """
    xp = xp_of(command)
    speed = cmd_speed(command)
    return xp.where(speed > deadband, 2 * math.pi * dt * clock_freq(speed, vmax, freq), 0.0)


def leg_phases(phase):
    """Per-leg phase in [-pi, pi): the clock plus each leg's offset."""
    xp = xp_of(phase)
    offsets = xp.asarray(PHASE_OFFSETS, dtype=phase.dtype, device=array_api_compat.device(phase))
    p = phase[..., None] + offsets
    return xp.fmod(p + math.pi, 2 * math.pi) - math.pi


def gait_targets(phase, duty, swing_height):
    """(target foot clearance, stance mask) from the duty-aware clock."""
    xp = xp_of(phase)
    theta = xp.fmod(leg_phases(phase) + 2 * math.pi, 2 * math.pi) / (2 * math.pi)
    swing_frac = 1.0 - duty
    in_swing = theta < swing_frac
    rz = swing_height * xp.sin(math.pi * theta / swing_frac) * in_swing
    return rz, ~in_swing


# The pure command draws, in precedence order (a later draw overwrites an
# earlier one): (name, JAX fold-in index, probability key, range key). What
# they are for is in default_config's command block.
#
# The JAX backend keys each draw off `fold_in(rng, 0x100 + idx)`, an index of
# its own, rather than taking keys from the base split, so every draw owns an
# independent stream. Enabling one draw moves no other draw's samples, and
# enabling none leaves the base sampler untouched. The 0x100 offset is
# explained at jax_command_draws in joystick.py.
#
# The index table is fixed. Index 0 is reserved and 6 up are free for future
# draws. A new draw takes the next free index and never reuses one.
#
# Every draw is gated on its probability, pure_wz and pure_vy included. A draw
# that took its key out of the base split unconditionally would shift every
# other sample whenever it was switched off.
COMMAND_DRAWS = (
    ("pure_wz", 1, "pure_wz_prob", None),
    ("pure_vy", 2, "pure_vy_prob", None),
    ("pure_slow", 3, "pure_slow_prob", "slow_vx"),
    ("pure_fast", 4, "pure_fast_prob", "fast_vx"),
    ("pure_back", 5, "pure_back_prob", "back_vx"),
)


def command_from_draws(draws, command_cfg):
    """Turn the random draws into a (vx, vy, wz) command.

    `draws` maps vx, vy, wz and zero_on, plus `<name>_on` (and `<name>_vx` for
    the ranged draws) for each enabled pure draw, to arrays of one batch shape.
    Precedence is wz, vy, slow, fast, back, then standing still last. A draw
    whose probability key is absent or 0 is skipped, so it needs no draws.
    pure_wz keeps the drawn wz and pure_vy the drawn vy. The other axes are
    +0.0.
    """
    xp = xp_of(draws["vx"])
    vel = xp.stack([draws["vx"], draws["vy"], draws["wz"]], axis=-1)
    z = xp.zeros_like(draws["vx"])
    for name, _idx, prob_key, _range_key in COMMAND_DRAWS:
        if not command_cfg.get(prob_key, 0.0):
            continue
        if name == "pure_wz":
            clean = xp.stack([z, z, vel[..., 2]], axis=-1)
        elif name == "pure_vy":
            clean = xp.stack([z, vel[..., 1], z], axis=-1)
        else:
            clean = xp.stack([draws[f"{name}_vx"], z, z], axis=-1)
        vel = xp.where(draws[f"{name}_on"][..., None], clean, vel)
    return xp.where(draws["zero_on"][..., None], xp.zeros_like(vel), vel)
