"""No-progress termination math (port item 1.5), CaT-style (arXiv 2403.18765).

Three pure functions of plain arrays -- no env, no model, no state -- so the
model is testable on its own (tests/unit/test_no_progress.py) and the env only
has to wire them up. The EMA that smooths `served`, the bernoulli draw, and
the done flag live in the env, because those need per-episode state and an
RNG key.

Ported from w01-tek's wojtek_rl/env.py (the `no_progress` block of
`default_config` and the cut block in `step`), where the same math sits inline
in the env.
"""

from __future__ import annotations

import jax.numpy as jp

# A command asking for less than this much speed is a stand command, and
# standing is what it asked for: the cut never arms below it. Same threshold
# the joystick task's `moving` mask uses, in the same commanded-speed units
# (`Joystick._cmd_speed`: planar norm plus 0.3 * |yaw rate|).
MIN_DEMAND = 0.05

# Weight on the yaw rate in the served measure, matching the 0.3 that
# `_cmd_speed` weights the commanded yaw rate by, so served and demand are
# blended the same way and their ratio is meaningful.
_YAW_WEIGHT = 0.3


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
    command = jp.asarray(command)
    direction = jp.maximum(jp.linalg.norm(command[:2]), 1e-6)
    return (
        jp.dot(jp.asarray(linvel_xy), command[:2]) / direction
        + _YAW_WEIGHT * gyro_z * jp.sign(command[2])
    )


def hazard(progress_ratio, risk_below, p_max):
    """Per-step cut probability from smoothed progress as a fraction of demand.

    Zero at and above `risk_below`, ramping linearly to `p_max` at zero
    progress and staying there for negative progress. At `p_max` the expected
    survival of a dead stop is 1/p_max control steps.
    """
    return p_max * jp.clip((risk_below - progress_ratio) / risk_below, 0.0, 1.0)


def armed(demand, steps_since_cmd, dt, grace_sec):
    """Whether the hazard applies at all this step.

    Two conditions: the command asks for real motion (`demand > MIN_DEMAND`),
    and the command has been standing for at least `grace_sec`. The grace
    window covers the reset transient and, since `steps_since_cmd` restarts on
    every resample, the time it takes to turn a gait around for a new command.
    """
    return (demand > MIN_DEMAND) & (steps_since_cmd * dt >= grace_sec)
