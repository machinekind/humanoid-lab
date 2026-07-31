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


# -- the contacts block ------------------------------------------------------

_CONTACTS = {
    "backend": "warp",
    "nacon_max": 31,
    "naconmax_per_env": 224,
    "num_envs": 1,
    "pool": 224,
    "overflow": False,
    "nefc_max": 160,
    "njmax": 1120,
    "rows_overflow": False,
}


def test_the_contacts_block_is_not_rendered_as_a_scenario():
    """battery.json carries a `contacts` block alongside the scenarios. It has
    none of a scenario's fields, so a renderer that mistook it for one would
    emit a row of dashes and a bogus PASS line for it."""
    md = render_markdown({**BATTERY, "contacts": _CONTACTS})

    assert "| contacts |" not in md
    assert "- contacts: PASS" not in md


def test_the_contacts_block_is_reported_with_its_budgets():
    md = render_markdown({**BATTERY, "contacts": _CONTACTS})

    assert "31" in md and "224" in md and "1120" in md


def test_an_overflowed_budget_is_flagged():
    """Warp drops both overflows silently, so the report is the only place a
    reader can find out that the numbers above it were measured on a
    simulation missing contacts."""
    over = {**_CONTACTS, "nacon_max": 224, "overflow": True}
    md = render_markdown({**BATTERY, "contacts": over})

    assert "ATTENTION" in md
    assert "naconmax_per_env" in md


def test_a_battery_without_a_contacts_block_still_renders():
    """battery.json files written before the block existed."""
    md = render_markdown(BATTERY)
    assert "# Eval report" in md


# -- the spin probes (port item 4.1) -----------------------------------------

_SPIN_LEFT = {
    **GOOD_ROW, "completed": True,
    "yaw_progress_deg": 138.4, "yaw_cmd_deg": 143.2,
}
_SPIN_RIGHT = {
    **GOOD_ROW, "completed": False, "fell": True, "fell_at": 120,
    "yaw_progress_deg": -11.7, "yaw_cmd_deg": -143.2,
}
_SPIN_BATTERY = {**BATTERY, "spin_left": _SPIN_LEFT, "spin_right": _SPIN_RIGHT}


def _section(md: str, heading: str) -> list[str]:
    """The lines of one `## ...` section of the rendered report."""
    lines = md.splitlines()
    rest = lines[lines.index(heading) + 1 :]
    end = next((i for i, line in enumerate(rest) if line.startswith("## ")), len(rest))
    return rest[:end]


def test_the_spin_section_reports_each_direction_separately():
    """The whole point of the pair: a chirality bug is invisible in an
    average over both directions."""
    section = _section(render_markdown(_SPIN_BATTERY), "## Spin probes")

    left = next(line for line in section if line.startswith("| spin_left |"))
    right = next(line for line in section if line.startswith("| spin_right |"))
    assert "138.4" in left and "143.2" in left
    assert "-11.7" in right and "-143.2" in right


def test_the_spin_section_reports_yaw_asked_for_next_to_yaw_delivered():
    section = _section(render_markdown(_SPIN_BATTERY), "## Spin probes")
    header = next(line for line in section if line.startswith("| scenario |"))

    assert "yaw_progress_deg" in header and "yaw_cmd_deg" in header


def test_a_battery_without_spin_rows_renders_no_spin_section():
    """battery.json files written before port item 4.1."""
    md = render_markdown(BATTERY)
    assert "## Spin probes" not in md


# -- the gait KPIs (port item 4.2) -------------------------------------------

_WALKED = {
    **GOOD_ROW,
    "swings": 14, "swing_apex_med_m": 0.0312, "swing_apex_p90_m": 0.0455,
    "touchdown_v_med": 0.221, "touchdown_softness_med": 0.283,
}
_STOOD = {
    **GOOD_ROW,
    "swings": 0, "swing_apex_med_m": None, "swing_apex_p90_m": None,
    "touchdown_v_med": None, "touchdown_softness_med": None,
}
_GAIT_BATTERY = {**BATTERY, "stand": _STOOD, "walk_ramp": _WALKED}


def test_the_gait_section_shows_the_apex_and_touchdown_numbers():
    section = _section(render_markdown(_GAIT_BATTERY), "## Gait KPIs")
    row = next(line for line in section if line.startswith("| walk_ramp |"))

    assert "0.031" in row  # median apex
    assert "0.046" in row or "0.045" in row  # p90 apex
    assert "0.28" in row  # touchdown softness


def test_the_gait_section_shows_the_swing_count_behind_each_median():
    """A median over three swings and one over ninety are not the same
    reading, and only the count says which one this is."""
    section = _section(render_markdown(_GAIT_BATTERY), "## Gait KPIs")
    header = next(line for line in section if line.startswith("| scenario |"))
    row = next(line for line in section if line.startswith("| walk_ramp |"))

    assert "swings" in header
    assert "14" in row


def test_a_scenario_that_never_swung_renders_dashes_not_zeros():
    section = _section(render_markdown(_GAIT_BATTERY), "## Gait KPIs")
    row = next(line for line in section if line.startswith("| stand |"))

    assert "| 0 |" in row  # zero swings, stated
    assert "0.000" not in row  # but no apex or speed invented for them


def test_a_battery_without_gait_kpis_renders_no_gait_section():
    """battery.json files written before port item 4.2."""
    assert "## Gait KPIs" not in render_markdown(BATTERY)


# -- the servo tracking error (port item 4.3) --------------------------------

_TRACKED = {**GOOD_ROW, "tracking_err_rms": 0.0412, "tracking_err_p95": 0.0930}
_UNTRACKED = {**GOOD_ROW, "tracking_err_rms": None, "tracking_err_p95": None}
_TRACK_BATTERY = {**BATTERY, "walk_ramp": _TRACKED, "stand": _UNTRACKED}


def test_the_servo_section_shows_rms_and_p95_per_scenario():
    section = _section(render_markdown(_TRACK_BATTERY), "## Servo tracking")
    header = next(line for line in section if line.startswith("| scenario |"))
    row = next(line for line in section if line.startswith("| walk_ramp |"))

    assert "tracking_err_rms" in header and "tracking_err_p95" in header
    assert "0.0412" in row and "0.0930" in row


def test_a_scenario_that_never_settled_renders_a_dash():
    section = _section(render_markdown(_TRACK_BATTERY), "## Servo tracking")
    row = next(line for line in section if line.startswith("| stand |"))

    assert "| - |" in row
    assert "0.0000" not in row


def test_a_battery_without_a_tracking_error_renders_no_servo_section():
    """battery.json files written before port item 4.3."""
    assert "## Servo tracking" not in render_markdown(BATTERY)
