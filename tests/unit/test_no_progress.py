"""Unit tests for the no-progress termination's math (envs/progress.py).

Port item 1.5 (see docs/port-details.md). The three pieces the env composes
-- the served-progress measure, the hazard ramp, and the arming predicate --
are pure functions of plain arrays, so they are tested here on synthetic
numbers without building a model or an env. What the env does with them (EMA
state, the bernoulli draw, the done flag, the metrics) is covered in
tests/integration/test_no_progress_env.py.
"""

import jax.numpy as jp
import pytest

from humanoid_lab.envs import progress

# w01-tek's defaults, the numbers the env ships with.
RISK_BELOW = 0.5
P_MAX = 0.02
GRACE_SEC = 2.0
DT = 0.02


# -- served progress -------------------------------------------------------


def test_served_is_the_commanded_speed_when_the_command_is_tracked_exactly():
    cmd = jp.array([0.8, 0.0, 0.0])
    assert progress.served(jp.array([0.8, 0.0]), 0.0, cmd) == pytest.approx(0.8)


def test_backward_motion_under_a_forward_command_is_negative():
    """The whole point of projecting onto the command instead of taking a
    magnitude: moving the wrong way is worse than standing still."""
    cmd = jp.array([0.8, 0.0, 0.0])
    standing = progress.served(jp.array([0.0, 0.0]), 0.0, cmd)
    backward = progress.served(jp.array([-0.4, 0.0]), 0.0, cmd)

    assert standing == pytest.approx(0.0)
    assert float(backward) < 0.0
    assert float(backward) < float(standing)


def test_sideways_motion_under_a_forward_command_scores_nothing():
    cmd = jp.array([0.8, 0.0, 0.0])
    assert progress.served(jp.array([0.0, 0.6]), 0.0, cmd) == pytest.approx(0.0)


def test_yaw_counts_toward_progress_in_the_commanded_direction_only():
    """Turning is blended in at 0.3, the same weight the commanded-speed
    blend uses, and signed by the commanded turn: spinning the wrong way
    reads negative."""
    spin_left = jp.array([0.0, 0.0, 0.6])
    assert progress.served(jp.array([0.0, 0.0]), 0.6, spin_left) == pytest.approx(0.18)
    assert progress.served(jp.array([0.0, 0.0]), -0.6, spin_left) == pytest.approx(-0.18)


def test_served_survives_a_zero_command_without_dividing_by_zero():
    cmd = jp.zeros(3)
    out = progress.served(jp.array([0.5, -0.2]), 0.3, cmd)
    assert bool(jp.isfinite(out))


# -- hazard ramp -----------------------------------------------------------


def test_hazard_is_zero_at_and_above_the_risk_threshold():
    assert progress.hazard(RISK_BELOW, RISK_BELOW, P_MAX) == pytest.approx(0.0)
    assert progress.hazard(1.0, RISK_BELOW, P_MAX) == pytest.approx(0.0)
    assert progress.hazard(5.0, RISK_BELOW, P_MAX) == pytest.approx(0.0)


def test_hazard_is_p_max_at_zero_progress():
    assert progress.hazard(0.0, RISK_BELOW, P_MAX) == pytest.approx(P_MAX)


def test_hazard_ramps_linearly_between_the_threshold_and_zero():
    assert progress.hazard(0.25, RISK_BELOW, P_MAX) == pytest.approx(0.5 * P_MAX)
    assert progress.hazard(0.4, RISK_BELOW, P_MAX) == pytest.approx(0.2 * P_MAX)


def test_hazard_saturates_at_p_max_for_negative_progress():
    """Moving against the command cannot buy more than the per-step cap."""
    assert progress.hazard(-3.0, RISK_BELOW, P_MAX) == pytest.approx(P_MAX)


# -- arming ----------------------------------------------------------------


def test_a_zero_command_never_arms():
    """Demand 0 is below the 0.05 threshold, so no amount of elapsed time
    arms the cut: standing still is what a zero command asks for."""
    assert not bool(progress.armed(0.0, 100000, DT, GRACE_SEC))


def test_a_near_zero_command_never_arms():
    assert not bool(progress.armed(0.04, 100000, DT, GRACE_SEC))


def test_a_real_command_arms_only_after_the_grace_window():
    steps = int(GRACE_SEC / DT)  # 100 steps of 0.02 s = 2.0 s
    assert not bool(progress.armed(0.8, steps - 1, DT, GRACE_SEC))
    assert bool(progress.armed(0.8, steps, DT, GRACE_SEC))


def test_arming_is_elementwise_over_a_batch():
    """The env runs this vmapped, so the predicate has to broadcast."""
    demand = jp.array([0.0, 0.8, 0.8])
    steps = jp.array([1000, 1, 1000])
    out = progress.armed(demand, steps, DT, GRACE_SEC)
    assert [bool(v) for v in out] == [False, False, True]
