"""Unit tests for the reward term library (rewards/terms.py): every term is a
pure function of generic arrays, so each is tested directly on synthetic
small arrays without building a model or an env.
"""

import jax.numpy as jp
import pytest

from humanoid_lab.rewards import terms


def _is_finite_scalar(x) -> bool:
    x = jp.asarray(x)
    return x.shape == () and bool(jp.all(jp.isfinite(x)))


def test_tracking_lin_vel_is_one_at_perfect_tracking():
    cmd = jp.array([0.3, -0.2])
    assert terms.tracking_lin_vel(cmd, cmd, sigma=0.25) == pytest.approx(1.0)


def test_tracking_lin_vel_decays_with_error():
    cmd = jp.array([0.3, -0.2])
    off = jp.array([0.0, 0.0])
    assert terms.tracking_lin_vel(cmd, off, sigma=0.25) < 1.0


def test_tracking_ang_vel_is_one_at_perfect_tracking():
    assert terms.tracking_ang_vel(0.4, 0.4, sigma=0.25) == pytest.approx(1.0)


def test_tracking_ang_vel_decays_with_error():
    assert terms.tracking_ang_vel(0.4, 0.0, sigma=0.25) < 1.0


def test_action_rate_zero_for_constant_action():
    a = jp.array([0.1, -0.2, 0.3])
    assert terms.action_rate(a, a) == pytest.approx(0.0)


def test_action_rate_nonzero_for_changing_action():
    a = jp.array([0.1, -0.2, 0.3])
    b = jp.array([0.0, 0.0, 0.0])
    assert terms.action_rate(a, b) > 0.0


def test_action_accel_zero_for_constant_action():
    a = jp.array([0.1, -0.2, 0.3])
    assert terms.action_accel(a, a, a) == pytest.approx(0.0)


def test_action_accel_nonzero_when_second_difference_is_nonzero():
    # A constant ramp (a - prev == prev - prev2) gives a zero second
    # difference by construction; use a non-collinear triple instead.
    a = jp.array([0.1, -0.2])
    prev = jp.array([0.0, 0.0])
    prev2 = jp.array([0.05, 0.1])
    assert terms.action_accel(a, prev, prev2) > 0.0


def test_feet_air_time_uses_min_air_time_offset():
    air_time = jp.array([0.3, 0.05])
    first_contact = jp.array([True, True])
    out = terms.feet_air_time(air_time, first_contact, air_time_cap=0.0, min_air_time=0.1)
    assert out == pytest.approx((0.3 - 0.1) + (0.05 - 0.1))


def test_feet_air_time_cap_bounds_reward():
    air_time = jp.array([10.0])
    first_contact = jp.array([True])
    capped = terms.feet_air_time(air_time, first_contact, air_time_cap=0.5, min_air_time=0.1)
    assert capped == pytest.approx(0.5 - 0.1)


def test_feet_phase_is_one_at_perfect_tracking():
    clearance = jp.array([0.01, 0.0])
    assert terms.feet_phase(clearance, clearance, phase_sigma=0.002) == pytest.approx(1.0)


def test_torque_limit_zero_within_cap():
    force = jp.array([1.0, -2.0])
    cap = jp.array([10.0, 10.0])
    assert terms.torque_limit(force, cap, frac=0.85) == pytest.approx(0.0)


def test_torque_limit_positive_above_cap():
    force = jp.array([9.0, -9.5])
    cap = jp.array([10.0, 10.0])
    assert terms.torque_limit(force, cap, frac=0.85) > 0.0


@pytest.mark.parametrize(
    "fn, args",
    [
        (terms.tracking_lin_vel, (jp.array([0.1, 0.2]), jp.array([0.0, 0.3]), 0.25)),
        (terms.tracking_ang_vel, (0.2, -0.1, 0.25)),
        (terms.lin_vel_z, (0.3,)),
        (terms.ang_vel_xy, (jp.array([0.1, -0.2]),)),
        (terms.orientation, (jp.array([0.05, -0.02]),)),
        (terms.torques, (jp.array([1.0, -2.0, 3.0]),)),
        (terms.torque_rate, (jp.array([1.0, -2.0]), jp.array([0.5, -1.0]))),
        (terms.action_rate, (jp.array([0.1, 0.2]), jp.array([0.0, 0.1]))),
        (
            terms.action_accel,
            (jp.array([0.1, 0.2]), jp.array([0.0, 0.1]), jp.array([-0.1, 0.0])),
        ),
        (terms.energy, (jp.array([1.0, -1.0]), jp.array([2.0, 3.0]))),
        (terms.pose, (jp.array([0.1, -0.1]), jp.array([0.0, 0.0]), jp.array([1.0, 0.5]))),
        (terms.feet_air_time, (jp.array([0.2, 0.0]), jp.array([True, False]))),
        (
            terms.feet_slip,
            (jp.array([[0.1, 0.0], [0.0, 0.0]]), jp.array([True, False])),
        ),
        (terms.feet_phase, (jp.array([0.01, 0.0]), jp.array([0.0, 0.0]), 0.002)),
        (
            terms.stand_still,
            (jp.array([0.1, -0.1]), jp.array([0.0, 0.0]), jp.array([0.01, -0.01])),
        ),
        (terms.termination, (jp.array(False),)),
        (terms.torque_limit, (jp.array([1.0, 2.0]), jp.array([10.0, 10.0]), 0.85)),
    ],
)
def test_every_term_returns_a_finite_scalar(fn, args):
    assert _is_finite_scalar(fn(*args))
