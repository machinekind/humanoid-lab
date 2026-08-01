"""Build-order step 10 gate (part 1): eval/battery.py's pure metric
functions and scenario command builders, exercised on synthetic arrays --
no checkpoint, no env, no mujoco model. Mirrors sizing/report.py's own
test_sizing_report.py pattern (pure numpy in, dict/scalar out).
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pytest

from humanoid_lab.eval import battery
from humanoid_lab.eval.battery import (
    SETTLE_STEPS,
    _measurement_env_overrides,
    armed_grid_flags,
    peak_over,
    antiphase_score,
    battery_scenarios,
    foot_slip,
    mech_power_mean,
    scenario_result,
    torque_saturation_fraction,
    tracking_error,
    vibration_index,
    yaw_progress_deg,
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
    assert set(scenarios) == {
        "stand", "walk_ramp", "turn", "strafe", "walk_to_stop",
        "spin_left", "spin_right",
    }


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


# -- spin probes -------------------------------------------------------------
#
# Chirality needs its own row per direction. A policy has shipped unable to
# spin right because every scenario that turned at all turned left.


def test_spin_left_holds_a_pure_positive_yaw_command():
    cmd_at, n = battery_scenarios(_DT)["spin_left"]

    for i in (0, n // 2, n - 1):
        cmd = np.asarray(cmd_at(i))
        assert cmd.shape == (3,)
        assert cmd[0] == pytest.approx(0.0)  # pure spin: no translation
        assert cmd[1] == pytest.approx(0.0)
        assert cmd[2] > 0.0  # + wz is CCW / left


def test_spin_right_is_the_exact_mirror_of_spin_left():
    left_at, n_left = battery_scenarios(_DT)["spin_left"]
    right_at, n_right = battery_scenarios(_DT)["spin_right"]

    assert n_left == n_right  # same duration, so the two rows compare directly
    for i in (0, n_left // 2, n_left - 1):
        assert np.allclose(np.asarray(right_at(i)), -np.asarray(left_at(i)))


def test_spin_commands_stay_inside_the_yaw_envelope():
    for name in ("spin_left", "spin_right"):
        cmd_at, n = battery_scenarios(_DT)[name]
        wz = np.array([float(np.asarray(cmd_at(i))[2]) for i in range(n)])

        assert np.all(np.abs(wz) <= 0.6)  # asimov's yaw command box
        assert np.all(wz == wz[0])  # held, not ramped


def test_spin_scenarios_are_six_seconds_at_any_dt():
    for name in ("spin_left", "spin_right"):
        assert battery_scenarios(0.02)[name][1] == 300
        assert battery_scenarios(0.04)[name][1] == 150


# -- yaw_progress_deg --------------------------------------------------------


def test_yaw_progress_integrates_the_post_settle_window():
    n = SETTLE_STEPS + 250
    wz = np.full(n, 0.5)  # rad/s, held

    # only the 250 post-settle steps count: 250 * 0.02 * 0.5 = 2.5 rad
    assert yaw_progress_deg(wz, _DT) == pytest.approx(np.degrees(2.5))


def test_yaw_progress_sign_follows_the_spin_direction():
    n = SETTLE_STEPS + 100
    left = yaw_progress_deg(np.full(n, 0.5), _DT)
    right = yaw_progress_deg(np.full(n, -0.5), _DT)

    assert left > 0.0
    assert right == pytest.approx(-left)


def test_yaw_progress_ignores_the_reset_transient():
    n = SETTLE_STEPS + 100
    quiet = np.zeros(n)
    quiet[SETTLE_STEPS:] = 0.5
    spiked = quiet.copy()
    spiked[:SETTLE_STEPS] = 40.0  # a violent reset transient, excluded

    assert yaw_progress_deg(spiked, _DT) == pytest.approx(yaw_progress_deg(quiet, _DT))


def test_yaw_progress_is_none_when_nothing_outlived_the_settle_window():
    assert yaw_progress_deg(np.full(SETTLE_STEPS, 0.5), _DT) is None
    assert yaw_progress_deg(np.zeros(0), _DT) is None


# -- scenario_result's spin fields -------------------------------------------


def _rec(n: int, wz: float = 0.0, cmd_wz: float = 0.0) -> dict:
    """A synthetic rollout record of `n` steps, shaped like rollout()'s."""
    cmd = np.zeros((n, 3))
    cmd[:, 2] = cmd_wz
    return {
        "cmd": cmd,
        "vx": np.zeros(n),
        "vy": np.zeros(n),
        "wz": np.full(n, wz),
        "height": np.full(n, 0.7),
        "qvel": np.zeros((n, 4)),
        "contact": np.ones((n, 2), dtype=bool),
        "foot_speed": np.zeros((n, 2)),
        "foot_clear": np.zeros((n, 2)),
        "foot_vz": np.zeros((n, 2)),
        "tau": np.zeros((n, 4)),
        "ctrl": np.zeros((n, 4)),
        "qpos": np.zeros((n, 4)),
    }


def test_scenario_result_reports_yaw_progress_per_direction():
    n = SETTLE_STEPS + 250
    left = scenario_result("spin_left", _rec(n, wz=0.5, cmd_wz=0.5), None, _DT, np.zeros(4), n)
    right = scenario_result("spin_right", _rec(n, wz=-0.5, cmd_wz=-0.5), None, _DT, np.zeros(4), n)

    assert left["yaw_progress_deg"] == pytest.approx(np.degrees(2.5), abs=1e-2)
    assert right["yaw_progress_deg"] == pytest.approx(-np.degrees(2.5), abs=1e-2)
    # the commanded sweep over the same window is the row's own denominator
    assert left["yaw_cmd_deg"] == pytest.approx(np.degrees(2.5), abs=1e-2)
    assert right["yaw_cmd_deg"] == pytest.approx(-np.degrees(2.5), abs=1e-2)


def test_scenario_result_marks_a_full_length_run_completed():
    n = SETTLE_STEPS + 100
    r = scenario_result("spin_left", _rec(n), None, _DT, np.zeros(4), n)

    assert r["completed"] is True
    assert r["fell"] is False


def test_scenario_result_marks_an_early_termination_incomplete():
    r = scenario_result("spin_left", _rec(40), 39, _DT, np.zeros(4), 300)

    assert r["completed"] is False
    assert r["fell"] is True
    assert r["fell_at"] == 39


def test_a_row_too_short_to_score_still_carries_fell_and_completed():
    r = scenario_result("spin_right", _rec(3), 2, _DT, np.zeros(4), 300)

    assert r["completed"] is False
    assert r["fell"] is True
    assert "yaw_progress_deg" not in r  # 3 steps measures nothing


def test_yaw_progress_is_null_when_the_row_never_outlived_the_settle_window():
    n = SETTLE_STEPS - 10
    r = scenario_result("spin_left", _rec(n, wz=0.5, cmd_wz=0.5), n - 1, _DT, np.zeros(4), 300)

    assert r["yaw_progress_deg"] is None
    assert r["yaw_cmd_deg"] is None


def test_the_existing_scenario_fields_keep_their_meaning():
    n = SETTLE_STEPS + 100
    r = scenario_result("stand", _rec(n), None, _DT, np.zeros(4), n)

    # The spin probes are additive: every older key is still there, unchanged.
    for key in (
        "fell", "fell_at", "steps", "vel_err_vx", "vel_err_vy", "vel_err_wz",
        "height_mean", "height_std", "vibration", "foot_slip",
        "torque_sat_frac", "mech_power_mean", "antiphase_score",
    ):
        assert key in r


# -- scenario_result's gait KPIs ---------------------------------------------


def test_scenario_result_carries_the_gait_kpis():
    n = SETTLE_STEPS + 100
    rec = _rec(n)
    rec["foot_clear"][SETTLE_STEPS + 10 : SETTLE_STEPS + 16, 0] = 0.04
    rec["foot_vz"][SETTLE_STEPS + 15, 0] = -0.2

    r = scenario_result("walk_ramp", rec, None, _DT, np.zeros(4), n)

    assert r["swings"] == 1
    assert r["swing_apex_med_m"] == pytest.approx(0.04)
    assert r["swing_apex_p90_m"] == pytest.approx(0.04)
    assert r["touchdown_v_med"] == pytest.approx(0.2)
    assert r["touchdown_softness_med"] is not None


def test_scenario_result_excludes_the_reset_transient_from_the_gait_kpis():
    n = SETTLE_STEPS + 100
    rec = _rec(n)
    rec["foot_clear"][5:20, 0] = 0.09  # a big lift inside the settle window

    r = scenario_result("stand", rec, None, _DT, np.zeros(4), n)

    assert r["swings"] == 0
    assert r["swing_apex_med_m"] is None


def test_a_scenario_that_never_lifted_a_foot_reports_zero_swings():
    n = SETTLE_STEPS + 100
    r = scenario_result("stand", _rec(n), None, _DT, np.zeros(4), n)

    assert r["swings"] == 0
    assert r["touchdown_softness_med"] is None


# -- tracking_error ----------------------------------------------------------
#
# The servo KPI: did the PD loop hold the setpoint the policy commanded?
# ctrl is the setpoint, qpos the angle reached, both over actuated joints.


def test_tracking_error_measures_the_gap_between_setpoint_and_angle():
    n = SETTLE_STEPS + 100
    ctrl = np.full((n, 3), 0.5)
    qpos = np.full((n, 3), 0.4)  # a flat 0.1 rad of sag on every joint

    err = tracking_error(ctrl, qpos, SETTLE_STEPS)

    assert err["rms"] == pytest.approx(0.1)
    assert err["p95"] == pytest.approx(0.1)


def test_tracking_error_is_zero_for_a_perfect_servo():
    q = np.linspace(0.0, 1.0, SETTLE_STEPS + 100)[:, None] * np.ones((1, 2))

    err = tracking_error(q, q, SETTLE_STEPS)

    assert err["rms"] == pytest.approx(0.0)
    assert err["p95"] == pytest.approx(0.0)


def test_tracking_error_excludes_the_reset_transient():
    n = SETTLE_STEPS + 100
    ctrl = np.full((n, 2), 0.5)
    qpos = np.full((n, 2), 0.5)
    settled = tracking_error(ctrl, qpos, SETTLE_STEPS)

    qpos[:SETTLE_STEPS] = -3.0  # the robot falling into its pose after reset

    assert tracking_error(ctrl, qpos, SETTLE_STEPS) == settled
    # ... and the same history scored from step 0 would have been dominated
    # by it, which is the whole reason for the window.
    assert tracking_error(ctrl, qpos, 0)["rms"] > 1.0


def test_tracking_error_p95_reads_the_worst_joints_not_the_mean():
    n = SETTLE_STEPS + 100
    ctrl = np.zeros((n, 10))
    qpos = np.zeros((n, 10))
    qpos[SETTLE_STEPS:, 0] = 0.4  # one joint lagging badly

    err = tracking_error(ctrl, qpos, SETTLE_STEPS)

    assert err["p95"] > err["rms"]


def test_tracking_error_is_null_when_nothing_outlived_the_settle_window():
    err = tracking_error(np.zeros((SETTLE_STEPS, 2)), np.zeros((SETTLE_STEPS, 2)), SETTLE_STEPS)

    assert err == {"rms": None, "p95": None}


def test_scenario_result_carries_the_tracking_error():
    n = SETTLE_STEPS + 100
    rec = _rec(n)
    rec["ctrl"][:] = 0.5
    rec["qpos"][:] = 0.4

    r = scenario_result("walk_ramp", rec, None, _DT, np.zeros(4), n)

    assert r["tracking_err_rms"] == pytest.approx(0.1)
    assert r["tracking_err_p95"] == pytest.approx(0.1)


def test_scenario_result_tracking_error_is_null_on_a_row_that_never_settled():
    n = SETTLE_STEPS - 10
    r = scenario_result("stand", _rec(n), n - 1, _DT, np.zeros(4), 300)

    assert r["tracking_err_rms"] is None
    assert r["tracking_err_p95"] is None


# -- the measurement env overrides -------------------------------------------
#
# Pure dict logic over run.json's recorded hydra_config, so it belongs in the
# fast suite: no env, no model.


def _run(env_block: dict) -> dict:
    return {"hydra_config": {"task": {"env": env_block}}}


def test_a_run_trained_with_the_no_progress_cut_measures_with_it_off():
    """The battery measures; it does not train. A probabilistic termination
    is not a fall, but rollout() reports every `done` as one, so a run that
    opted into the cut would have its scenarios scored as falls that never
    happened."""
    overrides = _measurement_env_overrides(_run({"no_progress": {"enable": True}}))

    assert overrides["no_progress"]["enable"] is False


def test_neutralising_the_cut_keeps_the_run_s_other_no_progress_settings():
    """Only `enable` is measurement-only. A run's re-derived grace/hazard
    numbers stay in the rebuilt config, so run.json still reconstructs the
    env the run trained with in every respect the battery does not need to
    change."""
    overrides = _measurement_env_overrides(
        _run({"no_progress": {"enable": True, "grace_sec": 3.5, "p_max": 0.05}})
    )

    assert overrides["no_progress"]["grace_sec"] == 3.5
    assert overrides["no_progress"]["p_max"] == 0.05


def test_the_cut_is_neutralised_even_when_the_run_never_mentioned_it():
    """A run.json from before the switch existed, or one that left it at the
    default, still gets the explicit off -- the battery's env does not depend
    on what the recorded config happened to contain."""
    overrides = _measurement_env_overrides(_run({}))

    assert overrides["no_progress"]["enable"] is False


def test_a_run_trained_with_the_mirror_augmentation_measures_with_it_off():
    """The deployment-frame rule. Mirror augmentation is a
    training-only stochastic augmentation: it flips half the envs into a
    left-right mirrored view of the world. A measurement has to describe the
    frame the robot will actually be deployed in, and a battery that drew the
    coin the other way would report a policy's spin_left as its spin_right."""
    overrides = _measurement_env_overrides(
        _run({"symmetry": {"enable": True, "mirror_prob": 0.5}})
    )

    assert overrides["symmetry"]["enable"] is False


def test_neutralising_the_mirror_keeps_the_run_s_other_symmetry_settings():
    """Only `enable` is measurement-only, same as the no-progress cut."""
    overrides = _measurement_env_overrides(_run({"symmetry": {"mirror_prob": 0.25}}))

    assert overrides["symmetry"]["mirror_prob"] == 0.25
    assert overrides["symmetry"]["enable"] is False


def test_the_mirror_is_neutralised_even_when_the_run_never_mentioned_it():
    overrides = _measurement_env_overrides(_run({}))

    assert overrides["symmetry"]["enable"] is False


def test_pushes_and_the_command_resample_are_still_neutralised():
    """The two older measurement-only changes, pinned alongside the new one."""
    overrides = _measurement_env_overrides(
        _run({"push": {"enable": True, "interval_range": [5, 10]}, "command": {"resample_steps": 500}})
    )

    assert overrides["push"]["enable"] is False
    assert overrides["push"]["interval_range"] == [5, 10]
    assert overrides["command"]["resample_steps"] == 10_000_000


def test_every_other_key_of_the_run_s_env_block_survives():
    """The rebuild is the run's own env, minus the measurement-only changes."""
    overrides = _measurement_env_overrides(
        _run({"real_pose_ref": True, "reward": {"scales": {"pose": -1.0}}})
    )

    assert overrides["real_pose_ref"] is True
    assert overrides["reward"] == {"scales": {"pose": -1.0}}


# -- the grid CLI ------------------------------------------------------------
#
# main() with run_battery stubbed out: the argparse wiring and the write
# target are pure plumbing, and pinning them costs no rollout. What the
# perturbations DO to the physics is tests/integration/test_grid_env.py.


@pytest.fixture
def stub_battery(monkeypatch):
    """Replace run_battery with a recorder, so main() can be driven without
    a checkpoint. Returns the dict its call kwargs land in."""
    seen = {}

    def fake_run_battery(run_dir, **kwargs):
        seen["run_dir"] = run_dir
        seen.update(kwargs)
        return {"run": "stub", "checkpoint": "0"}

    monkeypatch.setattr(battery, "run_battery", fake_run_battery)
    return seen


def _main(argv, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["battery", *argv])
    battery.main()


def test_out_writes_the_grid_cell_somewhere_else_and_leaves_battery_json_alone(
    tmp_path, monkeypatch, stub_battery
):
    """The canonical battery.json is the run's headline number table. A grid
    cell is a perturbed measurement and must never land on top of it."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    canonical = run_dir / "battery.json"
    canonical.write_text('{"run": "canonical"}')
    before_mtime = canonical.stat().st_mtime_ns

    cell = run_dir / "grid" / "battery_a1.58_lag5ms_envnone.json"
    _main(["--run", str(run_dir), "--alpha", "1.58", "--lag-tau", "0.005", "--out", str(cell)],
          monkeypatch)

    assert cell.exists()
    assert json.loads(canonical.read_text()) == {"run": "canonical"}
    assert canonical.stat().st_mtime_ns == before_mtime


def test_out_creates_the_grid_directory_it_was_pointed_at(tmp_path, monkeypatch, stub_battery):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    cell = run_dir / "grid" / "battery_a1_lag0ms_envnone.json"

    _main(["--run", str(run_dir), "--out", str(cell)], monkeypatch)

    assert cell.exists()


def test_without_out_the_battery_still_writes_the_canonical_file(
    tmp_path, monkeypatch, stub_battery
):
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    _main(["--run", str(run_dir)], monkeypatch)

    assert (run_dir / "battery.json").exists()


def test_the_defaults_are_the_unperturbed_battery(tmp_path, monkeypatch, stub_battery):
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    _main(["--run", str(run_dir)], monkeypatch)

    assert stub_battery["alpha"] == 1.0
    assert stub_battery["lag_tau"] == 0.0
    assert stub_battery["torque_envelope"] is None


def test_every_perturbation_flag_reaches_run_battery(tmp_path, monkeypatch, stub_battery):
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    _main(
        ["--run", str(run_dir), "--alpha", "1.58", "--lag-tau", "0.01",
         "--torque-envelope", "5,15", "--out", str(tmp_path / "cell.json")],
        monkeypatch,
    )

    assert stub_battery["alpha"] == 1.58
    assert stub_battery["lag_tau"] == 0.01
    assert stub_battery["torque_envelope"] == (5.0, 15.0)


def test_a_malformed_envelope_is_rejected_before_any_rollout(
    tmp_path, monkeypatch, stub_battery
):
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    with pytest.raises(SystemExit):
        _main(
            ["--run", str(run_dir), "--torque-envelope", "15,5",
             "--out", str(tmp_path / "cell.json")],
            monkeypatch,
        )

    assert stub_battery == {}


# -- a grid cell cannot land on the canonical path ---------------------------


@pytest.mark.parametrize(
    "flag",
    [["--alpha", "1.58"], ["--lag-tau", "0.005"], ["--torque-envelope", "5,15"]],
)
def test_a_perturbation_flag_without_out_refuses_to_run(
    flag, tmp_path, monkeypatch, stub_battery, capsys
):
    """A default --out would let a forgotten flag drop a perturbed
    measurement onto the run's canonical battery.json. The CLI refuses
    instead, and says why -- same shape as build_model.py's --set/--out."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    with pytest.raises(SystemExit):
        _main(["--run", str(run_dir), *flag], monkeypatch)

    err = capsys.readouterr().err
    assert flag[0] in err
    assert "requires --out" in err
    assert stub_battery == {}
    assert not (run_dir / "battery.json").exists()


def test_the_refusal_names_every_armed_flag(tmp_path, monkeypatch, stub_battery, capsys):
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    with pytest.raises(SystemExit):
        _main(
            ["--run", str(run_dir), "--alpha", "1.58", "--lag-tau", "0.005",
             "--torque-envelope", "5,15"],
            monkeypatch,
        )

    err = capsys.readouterr().err
    assert "--alpha" in err and "--lag-tau" in err and "--torque-envelope" in err


def test_the_unperturbed_battery_is_not_a_grid_cell():
    """The baseline cell is the native path, so plain defaults are the
    canonical battery and may write the canonical file."""
    assert armed_grid_flags(1.0, 0.0, None) == []
    assert armed_grid_flags(1.0, 0.0, "5,15") == ["--torque-envelope"]
    assert armed_grid_flags(0.8, 0.005, None) == ["--alpha", "--lag-tau"]


# -- contact budget peaks ----------------------------------------------------


def test_peak_over_scenarios_ignores_the_unmeasured_ones():
    """One number per budget over the whole battery. On the jax backend the
    row peak is None at every step, and a None must not read as a zero peak
    that would then be reported as a measurement."""
    assert peak_over([3, None, 11, 7]) == 11
    assert peak_over([None, None]) is None
    assert peak_over([]) is None


def test_peak_over_scenarios_returns_a_plain_int():
    """It lands in json.dumps via budget_report."""
    assert type(peak_over([np.int32(4), np.int64(9)])) is int
