"""Build-order step 10 gate (part 1): eval/battery.py's pure metric
functions and scenario command builders, exercised on synthetic arrays --
no checkpoint, no env, no mujoco model. Mirrors sizing/report.py's own
test_sizing_report.py pattern (pure numpy in, dict/scalar out).
"""

from __future__ import annotations

import numpy as np
import pytest

from humanoid_lab.eval.battery import (
    antiphase_score,
    battery_scenarios,
    foot_slip,
    mech_power_mean,
    torque_saturation_fraction,
    vibration_index,
)

# -- vibration_index ---------------------------------------------------------

_DT = 1.0 / 50.0  # asimov ctrl_dt


def test_vibration_index_low_frequency_sine_scores_near_zero():
    t = np.arange(400) * _DT
    sig = np.sin(2 * np.pi * 1.0 * t)  # 1 Hz, well under the 5 Hz cutoff

    assert vibration_index(sig, _DT) < 0.05


def test_vibration_index_high_frequency_sine_scores_near_one():
    t = np.arange(400) * _DT
    sig = np.sin(2 * np.pi * 12.0 * t)  # 12 Hz, above the 5 Hz cutoff

    assert vibration_index(sig, _DT) > 0.95


def test_vibration_index_pools_across_joints():
    t = np.arange(300) * _DT
    low = np.sin(2 * np.pi * 1.0 * t)
    high = np.sin(2 * np.pi * 12.0 * t)
    sig = np.stack([low, high], axis=-1)  # (T, 2) -- two "joints"

    idx = vibration_index(sig, _DT)

    # pooled across a pure-low and pure-high joint: strictly between the
    # two single-joint extremes above.
    assert 0.1 < idx < 0.9


# -- foot_slip -----------------------------------------------------------


def test_foot_slip_zero_for_static_planted_feet():
    speed = np.zeros((100, 2))
    contact = np.ones((100, 2), dtype=bool)

    assert foot_slip(speed, contact) == pytest.approx(0.0)


def test_foot_slip_ignores_swinging_unplanted_foot():
    speed = np.zeros((100, 2))
    speed[:, 1] = 5.0  # swing foot moves fast -- not slip, it's airborne
    contact = np.zeros((100, 2), dtype=bool)
    contact[:, 0] = True  # only foot 0 is planted, and it's stationary

    assert foot_slip(speed, contact) == pytest.approx(0.0)


def test_foot_slip_no_planted_foot_returns_zero_not_nan():
    speed = np.full((50, 2), 3.0)
    contact = np.zeros((50, 2), dtype=bool)

    assert foot_slip(speed, contact) == pytest.approx(0.0)


def test_foot_slip_averages_only_planted_samples():
    speed = np.array([[1.0, 9.0], [3.0, 9.0]])
    contact = np.array([[True, False], [True, False]])

    assert foot_slip(speed, contact) == pytest.approx(2.0)  # mean(1, 3)


# -- antiphase_score -------------------------------------------------------


def test_antiphase_score_high_for_alternating_contacts():
    t = np.arange(200)
    left = (t % 20) < 10
    right = ~left  # perfectly alternating -- the expected walking gait

    assert antiphase_score(left, right) > 0.9


def test_antiphase_score_low_for_in_phase_contacts():
    t = np.arange(200)
    left = (t % 20) < 10
    right = left.copy()  # both feet up/down together

    assert antiphase_score(left, right) < 0.1


def test_antiphase_score_neutral_when_always_planted():
    left = np.ones(50, dtype=bool)
    right = np.ones(50, dtype=bool)

    assert antiphase_score(left, right) == pytest.approx(0.5)


# -- torque_saturation_fraction / mech_power_mean ---------------------------


def test_torque_saturation_fraction_counts_over_cap_samples():
    cap = np.array([10.0, 10.0])
    tau = np.array([[9.0, 9.0], [9.6, 5.0], [3.0, 3.0], [0.0, 9.51]])

    frac = torque_saturation_fraction(tau, cap, frac=0.95)

    # threshold = 9.5: row1 col0 (9.6) and row3 col1 (9.51) exceed it, 2/8 total.
    assert frac == pytest.approx(2 / 8)


def test_torque_saturation_fraction_none_when_cap_unknown():
    tau = np.zeros((10, 2))

    assert torque_saturation_fraction(tau, np.array([0.0, 0.0])) is None


def test_mech_power_mean_matches_hand_computation():
    tau = np.array([[2.0, -3.0], [4.0, 0.0]])
    omega = np.array([[1.0, 1.0], [0.5, 2.0]])

    # per-step total power: |2*1|+|-3*1|=5; |4*0.5|+|0*2|=2 -> mean 3.5
    assert mech_power_mean(tau, omega) == pytest.approx(3.5)


# -- scenario command builders -----------------------------------------------


def test_battery_scenarios_have_expected_names():
    scenarios = battery_scenarios(_DT)
    assert set(scenarios) == {"stand", "walk_ramp", "turn", "strafe", "walk_to_stop"}


def test_stand_is_always_zero():
    cmd_at, n = battery_scenarios(_DT)["stand"]
    for i in (0, n // 2, n - 1):
        cmd = np.asarray(cmd_at(i))
        assert cmd.shape == (3,)
        assert np.allclose(cmd, 0.0)


def test_walk_ramp_increases_monotonically_then_holds_at_target():
    cmd_at, n = battery_scenarios(_DT)["walk_ramp"]
    vx = np.array([float(np.asarray(cmd_at(i))[0]) for i in range(n)])

    assert vx[0] == pytest.approx(0.0)
    assert np.all(np.diff(vx) >= -1e-9)  # monotonically non-decreasing
    assert vx[-1] == pytest.approx(0.6)
    assert vx.max() <= 0.6 + 1e-9


def test_turn_commands_stay_within_asimov_envelope():
    cmd_at, _n = battery_scenarios(_DT)["turn"]
    cmd = np.asarray(cmd_at(0))

    assert cmd.shape == (3,)
    assert abs(cmd[0]) <= 0.8
    assert abs(cmd[2]) <= 0.6
    assert cmd[2] != 0.0  # turn commands nonzero yaw


def test_strafe_commands_vy_only():
    cmd_at, _n = battery_scenarios(_DT)["strafe"]
    cmd = np.asarray(cmd_at(0))

    assert cmd[0] == pytest.approx(0.0)
    assert cmd[1] != 0.0
    assert abs(cmd[1]) <= 0.6
    assert cmd[2] == pytest.approx(0.0)


def test_walk_to_stop_switches_to_zero_partway():
    cmd_at, n = battery_scenarios(_DT)["walk_to_stop"]
    vx = np.array([float(np.asarray(cmd_at(i))[0]) for i in range(n)])

    assert vx[0] == pytest.approx(0.5)
    assert vx[-1] == pytest.approx(0.0)
    assert (vx > 0).sum() < n  # a real switch happens before the end


def test_scenario_step_counts_scale_with_dt():
    n_fast = battery_scenarios(0.02)["stand"][1]
    n_slow = battery_scenarios(0.04)["stand"][1]

    assert n_fast == 2 * n_slow
