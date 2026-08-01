"""Aggregate the robustness grid's per-cell battery JSONs into one markdown
comparison table.

Run: `./run.sh grid-report --runs runs/<name> [runs/<other> ...] [--out FILE]`

Reads `<run>/grid/battery_a<alpha>_lag<ms>ms_env<tag>.json` for every run
listed -- the files `./run.sh battery --alpha/--lag-tau/--torque-envelope
--out` writes (see `eval/grid.py::cell_name`, the single writer of that
convention; `parse_cell_name` below is its single reader, and
`tests/unit/test_grid.py` round-trips them). Each cell is scored
independently against the four gates below, and a missing cell -- its run
crashed, or was never submitted -- prints as MISSING rather than crashing
the report. A cell that falls over under a harsh perturbation is an expected
outcome of this probe, not a bug.

Two choices here, both about not inventing numbers:

- The vibration reference is the grid's OWN baseline cell (alpha 1.0, lag 0,
  envelope none), because this repo has no keeper policy to hardcode a
  per-scenario table from. With no baseline cell present the vibration gate
  is not applied, and the report says so per cell.
- There is no "stiffest surviving run" summary. That needs a scalar
  stiffness stamp to rank by, and `run.json`'s `actuator_gains.kp` is
  per-actuator.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

# -- gates -------------------------------------------------------------------
#
# QUADRUPED GATES, RE-DERIVE FOR A BIPED BEFORE TRUSTING.
#
# These four numbers were derived on a 0.21 m four-bar quadruped that is
# statically stable standing still. Nothing about them has been re-derived
# for a biped, whose velocity error, joint-velocity spectrum and torque
# headroom all live on different scales. They are here so the grid produces
# a verdict on day one, and every report carries the caveat. Re-derive each
# one against a keeper before a PASS here is allowed to mean anything.

#: Mean |commanded - achieved| linear velocity per axis, m/s. Gates
#: `vel_err_overall` and the strafe scenario's `vy_err`.
VEL_ERR_LIMIT = 0.20

#: Multiple of the reference cell's per-scenario vibration index a cell may
#: reach. The reference is the grid's own baseline cell (see the module
#: docstring).
VIBRATION_MULT = 1.3

#: Fraction of (step, joint) samples over 95% of the torque cap, pooled over
#: every scenario.
SATURATION_LIMIT = 0.05

#: The cell whose numbers the vibration gate is measured against: no
#: perturbation on any axis.
BASELINE_CELL = (1.0, 0, "none")

# `<tag>` carries no underscore by construction (see grid.envelope_tag:
# "none" or "OMEGA_B-OMEGA_0"), so `[^_]+` cannot run past the `.json` this
# pattern anchors on.
CELL_RE = re.compile(r"^battery_a([0-9.]+)_lag(\d+)ms_env([^_]+)\.json$")


@dataclass
class CellVerdict:
    """One cell's score. `skipped` names the gates that could not be
    applied to it -- a gate that silently does not run would read as a gate
    that passed."""

    verdict: str
    mean_track_err_rms: float | None
    reasons: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)


# -- cell discovery ----------------------------------------------------------


def parse_cell_name(name: str):
    """`(alpha, lag_ms, env_tag)` for a grid cell filename, or None if the
    name is not one. `eval/grid.py::cell_name` is the writer."""
    m = CELL_RE.match(name)
    if not m:
        return None
    return float(m.group(1)), int(m.group(2)), m.group(3)


def find_cells(run_dir: Path) -> dict:
    """`{(alpha, lag_ms, env_tag): Path}` for every grid cell under
    `<run_dir>/grid`. Empty dict if the run never got a grid pass."""
    grid_dir = Path(run_dir) / "grid"
    if not grid_dir.is_dir():
        return {}
    out = {}
    for path in sorted(grid_dir.glob("battery_a*.json")):
        key = parse_cell_name(path.name)
        if key is not None:
            out[key] = path
    return out


def scenario_rows(battery: dict) -> dict:
    """The cell's scenario entries, `{name: row}`.

    Identified structurally (a dict carrying `fell`) rather than from a
    hardcoded name list: the battery's scenario set is
    `eval/battery.py::battery_scenarios`, it has grown twice during this
    port, and a report that gated a stale list would quietly stop scoring
    the new rows. The `contacts` block is a dict too, and is excluded by the
    same test.
    """
    return {k: v for k, v in battery.items() if isinstance(v, dict) and "fell" in v}


def baseline_vibration(cells: dict):
    """`{scenario: vibration}` from the grid's unperturbed cell, or None if
    it is not present. This is the vibration gate's reference."""
    path = cells.get(BASELINE_CELL)
    if path is None:
        return None
    rows = scenario_rows(json.loads(Path(path).read_text()))
    return {name: row.get("vibration") for name, row in rows.items()}


# -- gating ------------------------------------------------------------------


def gate_cell(battery: dict, reference_vibration: dict | None = None) -> CellVerdict:
    """Score one cell against the four gates.

    1. No scenario fell.
    2. Every scenario's mean linear velocity error is under `VEL_ERR_LIMIT`
       on both axes. The YAW error is reported by the battery but NOT gated:
       `VEL_ERR_LIMIT` is metres per second, no rad/s limit has been
       derived, and inventing one would bless a number nobody measured.
    3. Every scenario's vibration index is at most `VIBRATION_MULT` times
       the reference cell's number for that same scenario. Skipped, visibly,
       where no reference covers the scenario.
    4. The worst torque-saturation fraction over every scenario is under
       `SATURATION_LIMIT`.

    A missing metric fails its gate rather than passing silently: a row too
    short to score (`eval/battery.py::scenario_result` returns early under
    10 steps) carries no velocity error at all, and reading that as zero
    error would turn the harshest cells into the cleanest ones.

    `mean_track_err_rms` is the mean servo error over the scenarios that
    measured one, reported even on a FAIL cell -- it is usually the context
    for why some other gate went.
    """
    rows = scenario_rows(battery)
    reasons: list[str] = []
    skipped: list[str] = []

    for name, row in sorted(rows.items()):
        if row.get("fell"):
            reasons.append(f"fell:{name}@{row.get('fell_at')}")

    for name, row in sorted(rows.items()):
        for axis in ("vel_err_vx", "vel_err_vy"):
            v = row.get(axis)
            if v is None or abs(v) >= VEL_ERR_LIMIT:
                reasons.append(f"{axis}:{name}={v}")

    if reference_vibration is None:
        skipped.append("vibration")
    else:
        for name, row in sorted(rows.items()):
            ref = reference_vibration.get(name)
            if ref is None:
                skipped.append(f"vibration:{name}")
                continue
            v = row.get("vibration")
            limit = VIBRATION_MULT * ref
            if v is None or v > limit:
                reasons.append(f"vibration:{name}={v}>{limit:.4f}")

    sats = [row.get("torque_sat_frac") for row in rows.values()]
    measured = [s for s in sats if s is not None]
    if measured:
        worst = max(measured)
        if worst >= SATURATION_LIMIT:
            reasons.append(f"saturation={worst}>={SATURATION_LIMIT}")
    else:
        skipped.append("saturation")

    errs = [row.get("tracking_err_rms") for row in rows.values()]
    errs = [e for e in errs if e is not None]
    mean_err = sum(errs) / len(errs) if errs else None

    return CellVerdict(
        verdict="FAIL" if reasons else "PASS",
        mean_track_err_rms=mean_err,
        reasons=reasons,
        skipped=skipped,
    )


def build_grid(run_dirs):
    """`{run_name: {(alpha, lag_ms, env_tag): CellVerdict}}` for every run
    directory listed. Each run's vibration reference is its OWN baseline
    cell: two runs are different policies, and one's smoothness is not the
    other's bar."""
    grid = {}
    for run_dir in run_dirs:
        run_dir = Path(run_dir)
        cells = find_cells(run_dir)
        reference = baseline_vibration(cells)
        grid[run_dir.name] = {
            key: gate_cell(json.loads(path.read_text()), reference)
            for key, path in cells.items()
        }
    return grid


# -- rendering ---------------------------------------------------------------


def _env_sort_key(tag: str):
    """"none" first, then numerically by `(omega_b, omega_0)`. Lexical order
    would put "15-28" before "5-10"."""
    if tag == "none":
        return (0, 0.0, 0.0)
    try:
        omega_b, omega_0 = tag.split("-")
        return (1, float(omega_b), float(omega_0))
    except ValueError:
        return (2, 0.0, 0.0)  # malformed tag: sort last, do not crash the report


_HEADER = [
    "# Robustness grid report",
    "",
    "Eval-only sim2real plant perturbations. `alpha` is a Kt "
    "miscalibration (the built model's PD gains and torque cap scaled "
    "together); `lag` is the actuator-bandwidth first-order torque lag, in "
    "milliseconds; `envelope` is the speed-dependent DRIVING-torque cap, "
    "`none` or `OMEGA_B-OMEGA_0` rad/s. See `eval/grid.py` and "
    "docs/configuration.md's \"Robustness grid (eval-only)\".",
    "",
    "The alpha 1.0 / lag 0 / envelope none cell is the BASELINE: it takes "
    "the native rollout path, not the explicit-PD one. Every other cell "
    "with a lag or an envelope takes the explicit-PD path. The two agree "
    "within a measured tolerance as the lag goes to zero (see "
    "`tests/integration/test_grid_env.py`), which is what makes the rows "
    "comparable -- they are not the same code.",
    "",
    "Row = one run x alpha x envelope; columns = lags. Cell = mean "
    "`tracking_err_rms` over the run's battery scenarios, then PASS/FAIL "
    f"against four gates: no falls; linear velocity error < {VEL_ERR_LIMIT} "
    f"m/s per axis; vibration <= {VIBRATION_MULT}x the baseline cell's own "
    f"number per scenario; worst torque saturation < {SATURATION_LIMIT}. "
    "MISSING = the cell's battery run crashed or was never submitted.",
    "",
    "**These are quadruped gates. Re-derive them for a biped before "
    "trusting a PASS.** They were tuned on a statically stable 0.21 m "
    "four-bar quadruped; nothing here has been re-derived. See "
    "`eval/grid_report.py`'s gate constants.",
    "",
]


def render_markdown(grid: dict) -> str:
    all_keys = sorted({k for row in grid.values() for k in row})
    lags = sorted({lag for _, lag, _ in all_keys})
    alphas = sorted({a for a, _, _ in all_keys})
    envs = sorted({e for _, _, e in all_keys}, key=_env_sort_key)

    lines = list(_HEADER)
    if not all_keys:
        lines.append("No grid cells found under any listed run's `grid/` directory.")
        return "\n".join(lines) + "\n"

    lines += [
        "| run | alpha | envelope | " + " | ".join(f"{lag}ms" for lag in lags) + " |",
        "|" + "---|" * (3 + len(lags)),
    ]
    for run_name, row in grid.items():
        for alpha in alphas:
            for env in envs:
                cells = []
                for lag in lags:
                    cell = row.get((alpha, lag, env))
                    if cell is None:
                        cells.append("MISSING")
                        continue
                    err = (
                        f"{cell.mean_track_err_rms:.4f}"
                        if cell.mean_track_err_rms is not None
                        else "-"
                    )
                    cells.append(f"{err} {cell.verdict}")
                lines.append(
                    f"| {run_name} | {alpha:g} | {env} | " + " | ".join(cells) + " |"
                )

    failures = [
        (run_name, key, cell)
        for run_name, row in grid.items()
        for key, cell in sorted(row.items())
        if cell.verdict == "FAIL"
    ]
    if failures:
        lines += ["", "## Why each FAIL cell failed", ""]
        for run_name, (alpha, lag, env), cell in failures:
            lines.append(
                f"- `{run_name}` alpha={alpha:g} lag={lag}ms env={env}: "
                + ", ".join(cell.reasons)
            )

    skipped = sorted(
        {s for row in grid.values() for cell in row.values() for s in cell.skipped}
    )
    if skipped:
        lines += [
            "",
            "## Gates not applied",
            "",
            "A gate with no data behind it is reported here, not counted as a pass.",
            "",
        ]
        for name in skipped:
            lines.append(f"- `{name}`")
        if any(s.startswith("vibration") for s in skipped):
            lines.append(
                "- the vibration reference is this run's own alpha 1.0 / lag 0 / "
                "envelope none cell; run that cell to enable the gate."
            )

    lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--runs", nargs="+", required=True, type=Path,
        help="run directories, e.g. runs/<name> -- each is read at <run>/grid/",
    )
    ap.add_argument("--out", default=None, type=Path)
    args = ap.parse_args()

    md = render_markdown(build_grid(args.runs))
    out = args.out or (args.runs[0] / "grid_report.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md)
    print(md)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
