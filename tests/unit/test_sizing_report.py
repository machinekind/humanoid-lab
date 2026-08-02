"""Build-order step 7 gate (part 2): sizing/report.py's pure reducers and
rendering, exercised on synthetic npz-shaped arrays -- no checkpoint, no
env, no mujoco model: pure numpy in, dict/str out.
"""

from __future__ import annotations

import numpy as np
import pytest

from humanoid_lab.sizing.report import (
    abs_percentiles,
    group_column_indices,
    group_metrics,
    match_catalog_to_groups,
    plot_torque_speed_scatter,
    render_percentile_table,
    render_report_markdown,
    save_scatter,
)

# A tiny synthetic two-group, four-joint rig: mirrors the shape of a real
# npz (joint_names/tau/omega) without needing a robot model or rollout.
JOINT_NAMES = ["left_knee_joint", "right_knee_joint", "left_hip_pitch_joint", "right_hip_pitch_joint"]
JOINT_GROUPS = {
    "knee": ["left_knee_joint", "right_knee_joint"],
    "hip_pitch": ["left_hip_pitch_joint", "right_hip_pitch_joint"],
}


def _synthetic_data(seed=0, t=500):
    rng = np.random.default_rng(seed)
    tau = rng.normal(loc=0.0, scale=20.0, size=(t, 4))
    omega = rng.normal(loc=0.0, scale=4.0, size=(t, 4))
    return tau, omega


def test_abs_percentiles_matches_numpy_by_hand():
    rng = np.random.default_rng(1)
    values = rng.normal(size=(1000,))

    got = abs_percentiles(values)

    a = np.abs(values)
    assert got["p50"] == pytest.approx(float(np.percentile(a, 50)))
    assert got["p90"] == pytest.approx(float(np.percentile(a, 90)))
    assert got["p95"] == pytest.approx(float(np.percentile(a, 95)))
    assert got["p99"] == pytest.approx(float(np.percentile(a, 99)))
    assert got["max"] == pytest.approx(float(a.max()))


def test_abs_percentiles_empty_is_all_none():
    got = abs_percentiles(np.zeros((0,)))
    assert got == {"p50": None, "p90": None, "p95": None, "p99": None, "max": None}


def test_group_column_indices_maps_joint_names_to_columns():
    cols = group_column_indices(JOINT_NAMES, JOINT_GROUPS)
    assert cols == {"knee": [0, 1], "hip_pitch": [2, 3]}


def test_group_column_indices_raises_on_unknown_joint():
    with pytest.raises(ValueError, match="ankle_joint"):
        group_column_indices(JOINT_NAMES, {"ankle": ["ankle_joint"]})


def test_group_metrics_percentiles_verified_by_hand():
    tau, omega = _synthetic_data()

    metrics = group_metrics(tau, omega, JOINT_NAMES, JOINT_GROUPS)

    assert set(metrics.keys()) == {"knee", "hip_pitch"}
    for group, cols in (("knee", [0, 1]), ("hip_pitch", [2, 3])):
        m = metrics[group]
        assert m["n_samples"] == tau.shape[0] * len(cols)

        expected_tau = np.abs(tau[:, cols]).ravel()
        assert m["tau"]["p50"] == pytest.approx(float(np.percentile(expected_tau, 50)))
        assert m["tau"]["p99"] == pytest.approx(float(np.percentile(expected_tau, 99)))
        assert m["tau"]["max"] == pytest.approx(float(expected_tau.max()))

        expected_omega = np.abs(omega[:, cols]).ravel()
        assert m["omega"]["p99"] == pytest.approx(float(np.percentile(expected_omega, 99)))

        expected_power = np.abs(tau[:, cols] * omega[:, cols]).ravel()
        assert m["mech_power"]["p50"] == pytest.approx(float(np.percentile(expected_power, 50)))
        assert m["mech_power"]["max"] == pytest.approx(float(expected_power.max()))


def test_match_catalog_to_groups_maps_via_joints_field():
    catalog = {
        "hip_pitch": {"model": "EC-A6416-P2-25", "joints": ["hip_pitch"], "tau_peak_nm": 120},
        "ankle": {"model": "EC-A4310-P2-36", "joints": ["ankle_pitch", "ankle_roll"], "tau_peak_nm": 36},
    }
    matches = match_catalog_to_groups(catalog, ["hip_pitch", "knee", "ankle_pitch", "ankle_roll"])

    assert matches["hip_pitch"]["model"] == "EC-A6416-P2-25"
    assert matches["ankle_pitch"]["model"] == "EC-A4310-P2-36"
    assert matches["ankle_roll"]["model"] == "EC-A4310-P2-36"
    assert matches["knee"] is None  # no catalog entry claims this group


def test_match_catalog_to_groups_ignores_entries_without_joints_field():
    # motors/mab.yaml shape: entries with no `joints` field (PLAN.md open gap).
    catalog = {"MA-p-100-30": {"torque_peak_nm": 150}}
    matches = match_catalog_to_groups(catalog, ["knee", "hip_pitch"])
    assert matches == {"knee": None, "hip_pitch": None}


def test_render_percentile_table_contains_p99_numbers():
    tau, omega = _synthetic_data()
    metrics = group_metrics(tau, omega, JOINT_NAMES, JOINT_GROUPS)

    table = render_percentile_table(metrics["knee"])

    assert "P99" in table
    assert f"{metrics['knee']['tau']['p99']:.3f}" in table
    assert f"{metrics['knee']['omega']['p99']:.3f}" in table
    assert f"{metrics['knee']['mech_power']['p99']:.3f}" in table


def test_render_report_markdown_contains_group_names_and_p99(tmp_path):
    tau, omega = _synthetic_data()
    metrics = group_metrics(tau, omega, JOINT_NAMES, JOINT_GROUPS)

    md = render_report_markdown(
        run_name="smoke_sizing_gate",
        preset_name="sizing_ideal",
        motors_name="encos",
        n_samples=tau.shape[0],
        metrics_by_group=metrics,
        joint_groups=JOINT_GROUPS,
        catalog_matches=None,
        png_name="sizing_scatter.png",
    )

    assert "## knee" in md
    assert "## hip_pitch" in md
    assert f"{metrics['knee']['tau']['p99']:.3f}" in md
    assert f"{metrics['hip_pitch']['tau']['p99']:.3f}" in md
    assert "sizing_scatter.png" in md
    assert "ENCOS" in md  # PLAN.md sanity-anchor caveat


def test_scatter_png_writes_nonzero_bytes(tmp_path):
    tau, omega = _synthetic_data(t=50)
    effort_limit = np.array([150.0, 150.0, 240.0, 240.0])
    velocity_limit = np.array([24.5, 24.5, 25.14, 25.14])
    catalog_matches = {
        "knee": {"model": "EC-A4315-P2-36", "tau_peak_nm": 75, "omega_max_rad_s": 12.25, "can_clamp_nm": 70},
        "hip_pitch": None,
    }

    fig = plot_torque_speed_scatter(
        tau, omega, JOINT_NAMES, JOINT_GROUPS, effort_limit, velocity_limit, catalog_matches
    )
    out = save_scatter(fig, tmp_path / "sizing_scatter.png")

    assert out.exists()
    assert out.stat().st_size > 0
