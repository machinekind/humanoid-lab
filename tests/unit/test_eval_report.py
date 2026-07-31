"""Build-order step 10 gate (part 2): eval/report.py's markdown rendering
from a synthetic battery.json-shaped dict -- no checkpoint, no battery
rollout, no env needed. Mirrors test_sizing_report.py's pattern.
"""

from __future__ import annotations

import json

import pytest

from humanoid_lab.eval.report import (
    _HEIGHT_STD_ATTENTION,
    _TORQUE_SAT_ATTENTION,
    _VEL_ERR_ATTENTION,
    _VIBRATION_ATTENTION,
    build_report,
    render_markdown,
    scenario_flags,
)

# Synthetic battery.json shape, mirroring eval/battery.py's run_battery()
# output: {"run", "checkpoint", "timestamp", <scenario>: {...}, ...}.
GOOD_ROW = {
    "fell": False, "fell_at": None, "steps": 300,
    "vel_err_vx": 0.05, "vel_err_vy": 0.02, "vel_err_wz": 0.01,
    "height_mean": 0.72, "height_std": 0.01,
    "vibration": 0.15, "foot_slip": 0.01,
    "torque_sat_frac": 0.0, "mech_power_mean": 40.0,
    "antiphase_score": 0.9,
}

BAD_ROW = {
    "fell": True, "fell_at": 42, "steps": 43,
    "vel_err_vx": 0.9, "vel_err_vy": 0.4, "vel_err_wz": 0.7,
    "height_mean": 0.3, "height_std": 0.2,
    "vibration": 0.97, "foot_slip": 0.5,
    "torque_sat_frac": 0.3, "mech_power_mean": 500.0,
    "antiphase_score": 0.1,
}

BATTERY = {
    "run": "smoke_test_run",
    "checkpoint": "000000102400",
    "timestamp": "2026-07-15T00:00:00",
    "stand": GOOD_ROW,
    "walk_ramp": BAD_ROW,
    "turn": GOOD_ROW,
    "strafe": GOOD_ROW,
    "walk_to_stop": GOOD_ROW,
}


def test_render_markdown_contains_scenario_names():
    md = render_markdown(BATTERY)
    for name in ("stand", "walk_ramp", "turn", "strafe", "walk_to_stop"):
        assert name in md


def test_render_markdown_contains_run_and_checkpoint():
    md = render_markdown(BATTERY)
    assert "smoke_test_run" in md
    assert "000000102400" in md


def test_scenario_flags_empty_for_good_row():
    assert scenario_flags("walk_ramp", GOOD_ROW) == []


def test_scenario_flags_nonempty_for_bad_row():
    flags = scenario_flags("walk_ramp", BAD_ROW)
    assert flags
    assert any("vibration" in f for f in flags)


def test_scenario_flags_any_fall_in_stand_is_attention():
    row = {**GOOD_ROW, "fell": True, "fell_at": 5}
    flags = scenario_flags("stand", row)
    assert any("stand" in f for f in flags)


def test_scenario_flags_fall_in_moving_scenario_still_reported():
    row = {**GOOD_ROW, "fell": True, "fell_at": 5}
    flags = scenario_flags("walk_ramp", row)
    assert any("fell at step 5" in f for f in flags)


def test_render_markdown_has_pass_and_attention_lines():
    md = render_markdown(BATTERY)
    assert "PASS" in md
    assert "ATTENTION" in md

    walk_ramp_line = next(line for line in md.splitlines() if line.startswith("- **walk_ramp"))
    assert "ATTENTION" in walk_ramp_line

    stand_line = next(line for line in md.splitlines() if line.startswith("- stand"))
    assert "PASS" in stand_line


def test_render_markdown_documents_thresholds():
    md = render_markdown(BATTERY)
    assert str(_VIBRATION_ATTENTION) in md
    assert str(_VEL_ERR_ATTENTION) in md
    assert str(_TORQUE_SAT_ATTENTION) in md
    assert str(_HEIGHT_STD_ATTENTION) in md


def test_build_report_reads_battery_json(tmp_path):
    run_dir = tmp_path / "runs" / "fake_run"
    run_dir.mkdir(parents=True)
    (run_dir / "battery.json").write_text(json.dumps(BATTERY))

    md = build_report(run_dir)

    assert "smoke_test_run" in md
    assert "stand" in md


def test_build_report_missing_battery_json_raises(tmp_path):
    run_dir = tmp_path / "runs" / "no_battery"
    run_dir.mkdir(parents=True)

    with pytest.raises(FileNotFoundError):
        build_report(run_dir)
