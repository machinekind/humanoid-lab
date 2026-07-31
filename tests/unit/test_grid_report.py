"""Tests for the robustness grid aggregator (port item 4.4): cell
discovery, the four gates, and the markdown table.

Model-free: every input here is a synthetic cell dict with the same shape
`eval/battery.py::run_battery` writes, so nothing has to roll a policy out
to exercise the scoring.
"""

from __future__ import annotations

import json

import pytest

from humanoid_lab.eval import grid_report


def _row(**over) -> dict:
    """One passing scenario row, in battery.json's own shape."""
    row = {
        "fell": False,
        "fell_at": None,
        "steps": 300,
        "completed": True,
        "vel_err_vx": 0.05,
        "vel_err_vy": 0.03,
        "vel_err_wz": 0.09,
        "vibration": 0.20,
        "torque_sat_frac": 0.01,
        "tracking_err_rms": 0.04,
    }
    row.update(over)
    return row


def _cell(**over) -> dict:
    """A whole passing cell: the battery's identification fields, two
    scenario rows, and the `contacts` block (a dict that is NOT a scenario
    and must not be gated)."""
    cell = {
        "run": "r",
        "checkpoint": "100",
        "alpha": 1.0,
        "lag_tau": 0.0,
        "torque_envelope": None,
        "stand": _row(),
        "walk_ramp": _row(),
        "contacts": {"backend": "jax", "nacon_max": 31, "nefc_max": None},
    }
    cell.update(over)
    return cell


# -- cell name parsing ------------------------------------------------------


def test_parse_cell_name_reads_every_axis():
    assert grid_report.parse_cell_name("battery_a1.58_lag5ms_env5-15.json") == (1.58, 5, "5-15")
    assert grid_report.parse_cell_name("battery_a1_lag0ms_envnone.json") == (1.0, 0, "none")


def test_parse_cell_name_rejects_a_foreign_filename():
    assert grid_report.parse_cell_name("battery.json") is None
    assert grid_report.parse_cell_name("eval_report.md") is None


def test_find_cells_reads_a_grid_directory(tmp_path):
    grid_dir = tmp_path / "grid"
    grid_dir.mkdir()
    for name in ("battery_a1_lag0ms_envnone.json", "battery_a1_lag5ms_envnone.json"):
        (grid_dir / name).write_text(json.dumps(_cell()))
    (grid_dir / "notes.txt").write_text("ignored")

    cells = grid_report.find_cells(tmp_path)
    assert set(cells) == {(1.0, 0, "none"), (1.0, 5, "none")}


def test_find_cells_is_empty_when_the_run_never_got_a_grid_pass(tmp_path):
    assert grid_report.find_cells(tmp_path) == {}


# -- scenario extraction ----------------------------------------------------


def test_scenario_rows_skips_the_contacts_block_and_the_scalars():
    rows = grid_report.scenario_rows(_cell())
    assert set(rows) == {"stand", "walk_ramp"}


# -- gates ------------------------------------------------------------------


def test_a_clean_cell_passes_every_gate():
    v = grid_report.gate_cell(_cell())
    assert v.verdict == "PASS"
    assert v.reasons == []
    assert v.mean_track_err_rms == pytest.approx(0.04)


def test_a_fall_fails_the_cell():
    v = grid_report.gate_cell(_cell(stand=_row(fell=True, fell_at=42)))
    assert v.verdict == "FAIL"
    assert any(r.startswith("fell:stand") for r in v.reasons)


def test_linear_velocity_error_at_the_limit_fails():
    """`< 0.20`, not `<=`: exactly the limit is a failure, matching
    w01-tek's own comparison."""
    v = grid_report.gate_cell(_cell(walk_ramp=_row(vel_err_vx=grid_report.VEL_ERR_LIMIT)))
    assert v.verdict == "FAIL"
    assert any("vel_err_vx:walk_ramp" in r for r in v.reasons)


def test_lateral_velocity_error_is_gated_too():
    v = grid_report.gate_cell(_cell(stand=_row(vel_err_vy=0.5)))
    assert any("vel_err_vy:stand" in r for r in v.reasons)


def test_yaw_error_is_reported_but_not_gated():
    """w01-tek's 0.20 limit is metres per second. There is no ported rad/s
    gate and this module does not invent one."""
    v = grid_report.gate_cell(_cell(stand=_row(vel_err_wz=99.0)))
    assert v.verdict == "PASS"
    assert not any("wz" in r for r in v.reasons)


def test_a_missing_velocity_error_fails_rather_than_passing_silently():
    v = grid_report.gate_cell(_cell(stand=_row(vel_err_vx=None)))
    assert v.verdict == "FAIL"
    assert any("vel_err_vx:stand=None" in r for r in v.reasons)


def test_saturation_at_the_limit_fails():
    v = grid_report.gate_cell(_cell(stand=_row(torque_sat_frac=grid_report.SATURATION_LIMIT)))
    assert v.verdict == "FAIL"
    assert any("saturation" in r for r in v.reasons)


def test_saturation_pools_the_worst_scenario():
    v = grid_report.gate_cell(_cell(walk_ramp=_row(torque_sat_frac=0.9)))
    assert any("0.9" in r for r in v.reasons)


def test_vibration_is_gated_against_a_reference_multiple():
    reference = {"stand": 0.20, "walk_ramp": 0.20}
    limit = grid_report.VIBRATION_MULT * 0.20
    ok = grid_report.gate_cell(_cell(stand=_row(vibration=limit - 1e-6)), reference)
    bad = grid_report.gate_cell(_cell(stand=_row(vibration=limit + 1e-6)), reference)
    assert ok.verdict == "PASS"
    assert bad.verdict == "FAIL"
    assert any("vibration:stand" in r for r in bad.reasons)


def test_vibration_is_skipped_visibly_when_there_is_no_reference():
    v = grid_report.gate_cell(_cell(stand=_row(vibration=99.0)))
    assert v.verdict == "PASS"
    assert "vibration" in v.skipped


def test_vibration_is_skipped_for_a_scenario_the_reference_does_not_cover():
    v = grid_report.gate_cell(_cell(stand=_row(vibration=99.0)), {"walk_ramp": 0.2})
    assert v.verdict == "PASS"
    assert "vibration:stand" in v.skipped


def test_mean_tracking_error_ignores_the_rows_that_measured_nothing():
    v = grid_report.gate_cell(
        _cell(stand=_row(tracking_err_rms=None), walk_ramp=_row(tracking_err_rms=0.10))
    )
    assert v.mean_track_err_rms == pytest.approx(0.10)


def test_mean_tracking_error_is_none_when_no_row_measured_one():
    v = grid_report.gate_cell(
        _cell(stand=_row(tracking_err_rms=None), walk_ramp=_row(tracking_err_rms=None))
    )
    assert v.mean_track_err_rms is None


# -- the baseline reference -------------------------------------------------


def test_the_baseline_cell_supplies_the_vibration_reference(tmp_path):
    grid_dir = tmp_path / "grid"
    grid_dir.mkdir()
    (grid_dir / "battery_a1_lag0ms_envnone.json").write_text(
        json.dumps(_cell(stand=_row(vibration=0.10), walk_ramp=_row(vibration=0.30)))
    )
    cells = grid_report.find_cells(tmp_path)
    assert grid_report.baseline_vibration(cells) == {"stand": 0.10, "walk_ramp": 0.30}


def test_no_baseline_cell_means_no_reference(tmp_path):
    grid_dir = tmp_path / "grid"
    grid_dir.mkdir()
    (grid_dir / "battery_a1.58_lag5ms_envnone.json").write_text(json.dumps(_cell()))
    assert grid_report.baseline_vibration(grid_report.find_cells(tmp_path)) is None


# -- report -----------------------------------------------------------------


def _write_grid(tmp_path, cells_by_name):
    run_dir = tmp_path / "run_a"
    (run_dir / "grid").mkdir(parents=True)
    for name, cell in cells_by_name.items():
        (run_dir / "grid" / name).write_text(json.dumps(cell))
    return run_dir


def test_a_mini_grid_renders_one_row_per_alpha_and_envelope(tmp_path):
    run_dir = _write_grid(
        tmp_path,
        {
            "battery_a1_lag0ms_envnone.json": _cell(),
            "battery_a1_lag5ms_envnone.json": _cell(),
            "battery_a1.58_lag5ms_envnone.json": _cell(stand=_row(fell=True, fell_at=7)),
        },
    )
    md = grid_report.render_markdown(*grid_report.build_grid([run_dir]))

    assert "| 0ms | 5ms |" in md
    assert "PASS" in md and "FAIL" in md
    assert "MISSING" in md  # alpha 1.58 has no 0 ms cell
    assert "fell:stand" in md  # the FAIL reasons are spelled out


def test_the_report_names_the_gates_it_could_not_apply(tmp_path):
    run_dir = _write_grid(tmp_path, {"battery_a1.58_lag5ms_envnone.json": _cell()})
    md = grid_report.render_markdown(*grid_report.build_grid([run_dir]))
    assert "vibration" in md
    assert "not applied" in md


def test_an_empty_grid_reports_that_rather_than_crashing(tmp_path):
    run_dir = tmp_path / "run_a"
    run_dir.mkdir()
    md = grid_report.render_markdown(*grid_report.build_grid([run_dir]))
    assert "No grid cells" in md


def test_the_report_marks_the_gates_as_unre_derived(tmp_path):
    """The brief mandates re-derivation for a biped. The numbers ship with
    that caveat attached to every report, not only to the source."""
    run_dir = _write_grid(tmp_path, {"battery_a1_lag0ms_envnone.json": _cell()})
    md = grid_report.render_markdown(*grid_report.build_grid([run_dir]))
    assert "quadruped" in md
    assert "re-derive" in md
