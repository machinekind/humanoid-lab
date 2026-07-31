"""Evaluation report: consolidates battery.json into a human-readable
markdown table plus a per-scenario PASS/ATTENTION line.

Run: ./run.sh report --run runs/<name>

Reads `<run>/battery.json` (written by eval/battery.py's run_battery) and
writes `<run>/eval_report.md`. Pure post-processing of an already-computed
battery result -- no env, checkpoint or jax needed here, so `report.py`
imports none of them (unlike battery.py/video.py).

Torque percentile analysis (p50/p90/p99/max, torque-by-speed binning,
motor-catalog overlay) is sizing/report.py's job and stays there -- this
module does not duplicate it. run.sh's `report` verb calls sizing-report
too, as a separate decoupled invocation, when sizing_data.npz exists for
the run (see run.sh's `report` case and its usage comment).

PASS/ATTENTION thresholds are loose and undocumented against any trained
policy -- they exist so a human skimming eval_report.md gets pointed at
the right row, not as a certified quality bar. Every threshold is a module
constant below, with its rationale, and is also printed at the bottom of
every rendered report.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Loose PASS/ATTENTION thresholds. `stand` additionally flags ANY fall
# (falling at zero command is always a problem); the other scenarios only
# use the thresholds below, because an early/undertrained checkpoint
# (e.g. this step's 100k-step smoke gate) is EXPECTED to fall or track
# badly while moving -- that's reported honestly, not hidden.
_VIBRATION_ATTENTION = 0.5  # training skill's smoothness gate (see battery.py's
# vibration_index docstring): fbb_v2 scored 0.972 buzzing, 0.168 after the fix.
_VEL_ERR_ATTENTION = 0.3  # mean |cmd - achieved| per axis, m/s or rad/s
_TORQUE_SAT_ATTENTION = 0.05  # fraction of (step, joint) samples over 0.95*cap
_HEIGHT_STD_ATTENTION = 0.05  # m; base-height std, flags bouncing/instability

# battery.json keys that are not scenarios. `contacts` is the warp budget
# block (see sim_budget.budget_report); it gets its own section below.
_META_KEYS = ("run", "checkpoint", "timestamp", "contacts")

# The per-direction spin rows (eval/battery.py's battery_scenarios), listed
# in the order the section renders them.
_SPIN_SCENARIOS = ("spin_left", "spin_right")


def _fmt(v, nd: int = 3) -> str:
    if v is None:
        return "-"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def scenario_flags(name: str, r: dict) -> list[str]:
    """List of human-readable reasons `r` (one battery.json scenario
    entry) needs ATTENTION; empty if it passes every loose threshold."""
    flags = []
    if r.get("fell"):
        if name == "stand":
            flags.append("fell while standing (zero command)")
        else:
            flags.append(f"fell at step {r.get('fell_at')}")

    v = r.get("vibration")
    if v is not None and v > _VIBRATION_ATTENTION:
        flags.append(f"vibration {v:.2f} > {_VIBRATION_ATTENTION}")

    for axis in ("vx", "vy", "wz"):
        e = r.get(f"vel_err_{axis}")
        if e is not None and abs(e) > _VEL_ERR_ATTENTION:
            flags.append(f"vel_err_{axis} {e:.2f} > {_VEL_ERR_ATTENTION}")

    sat = r.get("torque_sat_frac")
    if sat is not None and sat > _TORQUE_SAT_ATTENTION:
        flags.append(f"torque_sat_frac {sat:.2f} > {_TORQUE_SAT_ATTENTION}")

    hstd = r.get("height_std")
    if hstd is not None and hstd > _HEIGHT_STD_ATTENTION:
        flags.append(f"height_std {hstd:.3f} > {_HEIGHT_STD_ATTENTION}")

    return flags


def render_markdown(battery: dict) -> str:
    scenario_names = [k for k in battery if k not in _META_KEYS]

    lines = [
        f"# Eval report: {battery.get('run', '?')}",
        "",
        f"- checkpoint: {battery.get('checkpoint', '?')}",
        f"- generated: {battery.get('timestamp', '?')}",
        "",
        "## Battery",
        "",
        "| scenario | fell | fell_at | steps | vel_err (vx, vy, wz) | height (mean / std) "
        "| vibration | foot_slip | torque_sat | mech_power | antiphase |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name in scenario_names:
        r = battery[name]
        vel_err = (
            f"{_fmt(r.get('vel_err_vx'))}, {_fmt(r.get('vel_err_vy'))}, {_fmt(r.get('vel_err_wz'))}"
        )
        height = f"{_fmt(r.get('height_mean'))} / {_fmt(r.get('height_std'))}"
        lines.append(
            f"| {name} | {_fmt(r.get('fell'))} | {_fmt(r.get('fell_at'))} | {r.get('steps', '-')} | "
            f"{vel_err} | {height} | {_fmt(r.get('vibration'))} | {_fmt(r.get('foot_slip'), 4)} | "
            f"{_fmt(r.get('torque_sat_frac'), 4)} | {_fmt(r.get('mech_power_mean'))} | "
            f"{_fmt(r.get('antiphase_score'))} |"
        )

    # Spin probes (port item 4.1), their own section because the pair only
    # means something read side by side: a policy that turns 140 degrees one
    # way and 12 the other averages to a healthy-looking 76. Every scenario
    # row in battery.json carries the two yaw fields; these are the two rows
    # that exist to be read that way.
    spin_names = [n for n in scenario_names if n in _SPIN_SCENARIOS]
    if spin_names:
        lines += [
            "",
            "## Spin probes",
            "",
            "| scenario | yaw_progress_deg | yaw_cmd_deg | completed | fell |",
            "|---|---|---|---|---|",
        ]
        for name in spin_names:
            r = battery[name]
            lines.append(
                f"| {name} | {_fmt(r.get('yaw_progress_deg'), 1)} | "
                f"{_fmt(r.get('yaw_cmd_deg'), 1)} | {_fmt(r.get('completed'))} | "
                f"{_fmt(r.get('fell'))} |"
            )
        lines += [
            "",
            "Yaw is the body gyro's z channel integrated over the post-settle "
            "window, signed: positive is left (CCW). A right-hand row reading "
            "positive turned the wrong way; one near zero did not turn.",
        ]

    # Gait KPIs (port item 4.2). Raw metrics with no threshold attached, so
    # they get a table and no PASS/ATTENTION line: what a healthy apex is
    # depends on the robot's leg, and this repo has measured neither. The
    # swing count travels with the medians because a median over three
    # swings and one over ninety are not the same reading.
    if any("swings" in battery[n] for n in scenario_names):
        lines += [
            "",
            "## Gait KPIs",
            "",
            "| scenario | swings | swing_apex_med_m | swing_apex_p90_m "
            "| touchdown_v_med | touchdown_softness_med |",
            "|---|---|---|---|---|---|",
        ]
        for name in scenario_names:
            r = battery[name]
            if "swings" not in r:
                continue
            lines.append(
                f"| {name} | {r.get('swings')} | {_fmt(r.get('swing_apex_med_m'), 4)} | "
                f"{_fmt(r.get('swing_apex_p90_m'), 4)} | {_fmt(r.get('touchdown_v_med'))} | "
                f"{_fmt(r.get('touchdown_softness_med'))} |"
            )
        lines += [
            "",
            "A swing is a contiguous run of foot clearance over 5 mm, measured "
            "after the settle window. `touchdown_softness_med` is the touchdown "
            "speed over the free-fall speed from that swing's own apex: 1.0 is a "
            "foot dropped like a brick, lower is a foot flown in. A `-` is a "
            "median with no swings behind it, not a zero.",
        ]

    contacts = battery.get("contacts")
    if contacts:
        lines += [
            "",
            "## Contact budgets",
            "",
            f"- backend: {contacts.get('backend', '?')}",
            f"- contacts: peak {_fmt(contacts.get('nacon_max'))} of "
            f"{contacts.get('naconmax_per_env', '?')} per env "
            f"(pool {contacts.get('pool', '?')} over {contacts.get('num_envs', '?')} envs)",
            f"- constraint rows: peak {_fmt(contacts.get('nefc_max'))} of "
            f"{contacts.get('njmax', '?')} per world",
        ]
        if contacts.get("nacon_max") is None or contacts.get("nefc_max") is None:
            lines.append(
                "- a `-` peak was not measured: the jax backend has no live counter "
                "for it and no budget to overflow"
            )

    lines += ["", "## Attention", ""]
    if contacts and contacts.get("overflow"):
        lines.append(
            f"- **contacts: ATTENTION** -- the contact pool overflowed "
            f"({contacts.get('nacon_max')} >= naconmax_per_env "
            f"{contacts.get('naconmax_per_env')}). Warp drops the overflow silently, "
            "so every number above was measured on a simulation missing contacts. "
            "Raise the budget and re-run."
        )
    if contacts and contacts.get("rows_overflow"):
        lines.append(
            f"- **contacts: ATTENTION** -- the constraint rows overflowed "
            f"({contacts.get('nefc_max')} >= njmax {contacts.get('njmax')}). Rows past "
            "njmax apply no force, with no warning anywhere. Raise the budget and "
            "re-run."
        )
    for name in scenario_names:
        flags = scenario_flags(name, battery[name])
        if flags:
            lines.append(f"- **{name}: ATTENTION** -- " + "; ".join(flags))
        else:
            lines.append(f"- {name}: PASS")

    lines += [
        "",
        "## Thresholds (loose, see module docstring)",
        "",
        f"- vibration index > {_VIBRATION_ATTENTION} -> attention "
        "(training skill's smoothness gate)",
        f"- |vel tracking error| > {_VEL_ERR_ATTENTION} (m/s or rad/s) -> attention",
        f"- torque saturation fraction > {_TORQUE_SAT_ATTENTION} -> attention",
        f"- base height std > {_HEIGHT_STD_ATTENTION} m -> attention",
        "- any fall in `stand` (zero command) -> attention; falls in the moving "
        "scenarios are reported but not flagged on their own (expected for an "
        "undertrained checkpoint)",
        "",
    ]
    return "\n".join(lines)


def build_report(run_dir: Path) -> str:
    battery_path = run_dir / "battery.json"
    if not battery_path.exists():
        raise FileNotFoundError(
            f"{battery_path} not found -- run `./run.sh battery --run {run_dir}` first"
        )
    battery = json.loads(battery_path.read_text())
    return render_markdown(battery)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--out", default=None, type=Path)
    args = ap.parse_args()

    md = build_report(args.run)
    out = args.out or (args.run / "eval_report.md")
    out.write_text(md)
    print(md)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
