"""Render a courses.json document: the eval report's courses section and the
console table.

Pure: a parsed schema-1 dict in, text out. It imports no other humanoid_lab
module and does not import jax. eval/report.py can therefore render the
section without an env or a backend.
"""

from __future__ import annotations

import math

# Height and grip stay in the score's min but cannot bind while the other
# axes are healthy (see scoring.py), so the section names them as
# diagnostics.
DIAGNOSTIC_AXES = ("height", "grip")


def _fmt(value, nd: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{nd}f}" if math.isfinite(value) else str(value)
    return str(value)


def _signed(value, nd: int = 3) -> str:
    return "-" if value is None else f"{value:+.{nd}f}"


def _error_cell(row: dict) -> str:
    raw = row.get("raw_median") or {}
    if row.get("kind") == "spin":
        return f"wz {_fmt(raw.get('wz_err_rms'))} rad/s"
    return f"xte {_fmt(raw.get('xte_rms_m'), 4)} m"


def _seed_line(seed: dict) -> str:
    who = f"seed {seed.get('seed')}"
    what = seed.get("outcome")
    if what in ("fell", "settle_fell"):
        return f"{who} {what} at step {seed.get('fell_at')}"
    return f"{who} {what} after {seed.get('steps')} steps"


def render_markdown(courses_json: dict, battery_checkpoint: str | None = None) -> str:
    """The `## Courses` section of eval_report.md.

    `battery_checkpoint` is battery.json's `checkpoint`. When it differs
    from the courses checkpoint the section says so, so a report never joins
    two checkpoints silently.
    """
    doc = courses_json
    cat = doc.get("catalogue") or {}
    ckpt = doc.get("checkpoint")
    lines = [
        "## Courses",
        "",
        (
            f"- catalogue: version {cat.get('version', '?')}, "
            f"fingerprint {str(cat.get('fingerprint', '?'))[:12]}, "
            f"params {cat.get('params_source', '?')}"
        ),
        f"- robot: {doc.get('robot', '?')}, ground: {doc.get('ground_class', '?')}",
        f"- checkpoint: {ckpt} (step {doc.get('checkpoint_step', '?')})",
        (
            f"- seeds: {doc.get('seeds', '?')} from seed {doc.get('seed_base', '?')}, "
            f"canonical: {_fmt(doc.get('canonical'))}"
        ),
    ]
    if doc.get("env_overrides"):
        lines.append(f"- env_overrides: {doc['env_overrides']}")
    if battery_checkpoint is not None and battery_checkpoint != ckpt:
        lines.append(
            f"- **checkpoint mismatch**: courses measured {ckpt}, battery.json "
            f"measured {battery_checkpoint}. The two sections describe different "
            "checkpoints."
        )
    for warning in doc.get("warnings") or []:
        lines.append(f"- warning: {warning}")
    lines += [
        "",
        (
            "A seed scores the weakest of its axes when it completed the course, "
            "else 0. Median and worst are over the seeds. "
            f"{' and '.join(DIAGNOSTIC_AXES).capitalize()} are diagnostics: they stay "
            "in the min but cannot bind while the other axes are healthy. "
            "Compare a row across policies, never across rows. "
            "A difference below twice the row's noise band in docs/configuration.md is noise."
        ),
        "",
        (
            "| course | median | worst | binding | falls | done | error | speed ratio "
            "| unicycle tracking | Δ median | slip ratio |"
        ),
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    courses = doc.get("courses") or {}
    for name, row in courses.items():
        seeds = row.get("seeds", "?")
        raw = row.get("raw_median") or {}
        unicycle = row.get("perfect_unicycle") or {}
        vs = row.get("vs_baseline") or {}
        speed_ratio = raw.get("speed_ratio") if row.get("kind") == "path" else None
        lines.append(
            f"| {name} | {_fmt(row.get('score_median'))} | {_fmt(row.get('score_worst'))} "
            f"| {_fmt(row.get('binding'))} | {row.get('falls', '?')}/{seeds} "
            f"| {row.get('completed', '?')}/{seeds} | {_error_cell(row)} "
            f"| {_fmt(speed_ratio, 2)} | {_fmt(unicycle.get('tracking'), 1)} "
            f"| {_signed(vs.get('d_score_median'))} | {_fmt(vs.get('slip_ratio'), 2)} |"
        )
    lines += [
        "",
        (
            "`unicycle tracking` is what a unicycle that executes the follower's "
            "commands exactly scores on tracking. It is not a ceiling: a robot "
            "whose yaw lags can exceed it. `Δ median` is the row's median minus its "
            "baseline's. `slip ratio` is the row's median slip distance over its "
            "baseline's."
        ),
    ]

    incomplete = {
        name: row for name, row in courses.items()
        if row.get("completed") != row.get("seeds")
    }
    if incomplete:
        lines += ["", "### Seeds that did not complete", ""]
        for name, row in incomplete.items():
            missed = [s for s in row.get("per_seed") or [] if s.get("outcome") != "completed"]
            lines.append(f"- {name}: " + "; ".join(_seed_line(s) for s in missed))
    lines.append("")
    return "\n".join(lines)


def console_table(courses_json: dict) -> str:
    """One line per course in catalogue order, for the terminal."""
    head = (
        f"{'course':<22} {'median':>7} {'worst':>7} {'binding':<11} "
        f"{'done':>5} {'falls':>5}"
    )
    lines = [head, "-" * len(head)]
    for name, row in (courses_json.get("courses") or {}).items():
        seeds = row.get("seeds", "?")
        lines.append(
            f"{name:<22} {_fmt(row.get('score_median')):>7} "
            f"{_fmt(row.get('score_worst')):>7} {_fmt(row.get('binding')):<11} "
            f"{str(row.get('completed', '?')) + '/' + str(seeds):>5} "
            f"{str(row.get('falls', '?')) + '/' + str(seeds):>5}"
        )
    return "\n".join(lines)
