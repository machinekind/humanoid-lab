"""Sizing report: per-joint-group torque/speed/power percentiles and a
torque-speed scatter, from a sizing/collect.py rollout (build order step 7,
PLAN.md "First experiments" #2/#3).

Run:
    python -m humanoid_lab.sizing.report --run runs/<name> [--motors encos]

Reads `<run>/sizing_data.npz` (written by sizing/collect.py: tau [T,12],
omega [T,12], joint_names [12], effort_limit [12], velocity_limit [12]) and
`<run>/run.json` (for the actuator preset name and robot dir -- used only
to load robot.yaml's joint_groups, not to rebuild an env or load a
checkpoint: this module needs neither). Writes `<run>/sizing_report.md` and
`<run>/sizing_scatter.png`.

-- pure reducer functions (abs_percentiles, group_column_indices,
group_metrics, match_catalog_to_groups, render_percentile_table,
render_report_markdown) take plain numpy arrays / dicts in, dict/str out --
no npz file, matplotlib or robot model needed, so these are unit-tested
directly in test_sizing_report.py. plot_torque_speed_scatter is the one
function that touches matplotlib.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from humanoid_lab import paths

_PERCENTILES = (50, 90, 95, 99)


# -- pure reducers ---------------------------------------------------------


def abs_percentiles(values) -> dict:
    """P50/P90/P95/P99/max of |values|, flattened over every axis.

    Empty input -> every field None ("no samples" rather than raising).
    """
    a = np.abs(np.asarray(values, dtype=float)).ravel()
    if a.size == 0:
        return {"p50": None, "p90": None, "p95": None, "p99": None, "max": None}
    out = {f"p{p}": float(np.percentile(a, p)) for p in _PERCENTILES}
    out["max"] = float(a.max())
    return out


def group_column_indices(joint_names, joint_groups: dict) -> dict:
    """group_name -> [column indices into a [T,12]-shaped tau/omega array],
    from `joint_names` (npz order) and `joint_groups` (RobotSpec.joint_groups
    -- robot.yaml's group -> [left_joint, right_joint] map).
    """
    name_to_idx = {str(n): i for i, n in enumerate(joint_names)}
    out = {}
    for group, joints in joint_groups.items():
        missing = [j for j in joints if j not in name_to_idx]
        if missing:
            raise ValueError(
                f"joint_groups['{group}'] references joint(s) {missing} not present in "
                f"this rollout's joint_names {sorted(name_to_idx)}"
            )
        out[group] = [name_to_idx[j] for j in joints]
    return out


def group_metrics(tau, omega, joint_names, joint_groups: dict) -> dict:
    """group_name -> {"tau", "omega", "mech_power": abs_percentiles(...),
    "n_samples": pooled sample count (steps * joints in the group)}.

    mech_power is |tau*omega| per (step, joint) sample, element-wise --
    NOT the env's sizing/mech_power metric (a per-step scalar summed over
    every joint);
    this report reduces the raw per-joint rollout, not the env's own
    running metrics, so it can be re-sliced per joint group.
    """
    cols = group_column_indices(joint_names, joint_groups)
    tau = np.asarray(tau, dtype=float)
    omega = np.asarray(omega, dtype=float)
    power = np.abs(tau * omega)

    out = {}
    for group, idx in cols.items():
        t, o, p = tau[:, idx], omega[:, idx], power[:, idx]
        out[group] = {
            "tau": abs_percentiles(t),
            "omega": abs_percentiles(o),
            "mech_power": abs_percentiles(p),
            "n_samples": int(t.size),
        }
    return out


def load_motor_catalog(name: str) -> dict:
    path = paths.REPO_ROOT / "motors" / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"motor catalog not found at {path}")
    with path.open() as f:
        return yaml.safe_load(f) or {}


def match_catalog_to_groups(catalog: dict, group_names) -> dict:
    """robot.yaml joint_groups name -> the motor catalog entry whose
    `joints` field lists it (None if no catalog entry claims that group).

    A catalog entry's `joints` list may name more than one robot.yaml group
    (motors/encos.yaml's "ankle" entry: joints: [ankle_pitch, ankle_roll] --
    one physical ENCOS ankle motor pair drives both roles, PLAN.md "v1
    armatures": "two motors drive a parallel ankle, modeled as serial
    pitch/roll joints"), so both groups resolve to the same entry.
    """
    out = {g: None for g in group_names}
    for entry in catalog.values():
        if not isinstance(entry, dict) or "joints" not in entry:
            continue  # motors/mab.yaml: no per-entry joints field yet (PLAN.md open gaps)
        for group in entry["joints"]:
            if group in out and out[group] is None:
                out[group] = entry
    return out


# -- markdown rendering -----------------------------------------------------


def _fmt(v, nd=3):
    return "-" if v is None else f"{v:.{nd}f}"


def render_percentile_table(metrics: dict) -> str:
    """One group's {"tau", "omega", "mech_power"} percentiles -> a markdown
    table, one row per quantity."""
    rows = [
        ("abs(tau) (Nm)", metrics["tau"]),
        ("abs(omega) (rad/s)", metrics["omega"]),
        ("mech power (W)", metrics["mech_power"]),
    ]
    lines = ["| metric | P50 | P90 | P95 | P99 | max |", "|---|---|---|---|---|---|"]
    for label, m in rows:
        lines.append(
            f"| {label} | {_fmt(m['p50'])} | {_fmt(m['p90'])} | {_fmt(m['p95'])} | "
            f"{_fmt(m['p99'])} | {_fmt(m['max'])} |"
        )
    return "\n".join(lines)


def render_report_markdown(
    run_name: str,
    preset_name: str,
    motors_name: str | None,
    n_samples: int,
    metrics_by_group: dict,
    joint_groups: dict,
    catalog_matches: dict | None,
    png_name: str,
    n_dropped: int = 0,
) -> str:
    lines = [
        f"# Sizing report: {run_name}",
        "",
        f"- actuator preset: `{preset_name}`",
        f"- motor catalog overlay: `{motors_name}`" if motors_name else "- motor catalog overlay: none",
        f"- samples (control steps kept): {n_samples}",
        "",
        f"![torque-speed scatter]({png_name})",
        "",
    ]
    for group in joint_groups:
        m = metrics_by_group[group]
        lines.append(f"## {group}")
        lines.append("")
        lines.append(f"n = {m['n_samples']} (pooled over {len(joint_groups[group])} joint(s) x steps)")
        lines.append("")
        lines.append(render_percentile_table(m))
        entry = (catalog_matches or {}).get(group)
        if entry:
            bits = []
            if "tau_peak_nm" in entry:
                bits.append(f"tau_peak={entry['tau_peak_nm']} Nm")
            if "can_clamp_nm" in entry:
                bits.append(f"CAN clamp={entry['can_clamp_nm']} Nm")
            if "omega_max_rad_s" in entry:
                bits.append(f"omega_max={entry['omega_max_rad_s']} rad/s")
            if bits:
                lines.append("")
                lines.append(f"catalog overlay (`{entry.get('model', motors_name)}`): " + ", ".join(bits))
        lines.append("")

    lines += [
        "## Caveats",
        "",
        f"- preset: `{preset_name}` -- these are the caps a sizing_ideal-style preset ran "
        "under (generous headroom by design), not a real motor's limits.",
        f"- {n_samples} control steps kept for this rollout; per-group sample counts above "
        "are steps x joints-in-group (both left/right).",
        f"- {n_dropped} fall-transition step(s) dropped from the percentiles (episode-"
        "termination samples measure impact, not gait demand). A policy that never falls "
        "drops zero.",
        "- PLAN.md \"First experiments\" #3: the ENCOS v0 leg-motor table (35 kg, 1.2 m mass "
        "class) is a sanity anchor, not a target -- results far outside it mean the sizing "
        "setup is wrong, not that motor selection is done.",
    ]
    return "\n".join(lines)


# -- scatter figure (matplotlib; the only non-pure-numpy piece) ------------


def _downsample_idx(n: int, max_points: int, rng: np.random.Generator) -> np.ndarray:
    if n <= max_points:
        return np.arange(n)
    return rng.choice(n, size=max_points, replace=False)


def plot_torque_speed_scatter(
    tau,
    omega,
    joint_names,
    joint_groups: dict,
    effort_limit,
    velocity_limit,
    catalog_matches: dict | None = None,
    max_points: int = 3000,
    seed: int = 0,
):
    """One subplot per joint group: |omega| (x) vs |tau| (y), downsampled
    scatter, overlaid with the preset's effort/velocity limit box and (if
    `catalog_matches` names an entry for that group) the motor catalog's
    tau_peak/omega_max lines plus its CAN clamp line when present.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = group_column_indices(joint_names, joint_groups)
    groups = list(joint_groups)
    tau = np.asarray(tau, dtype=float)
    omega = np.asarray(omega, dtype=float)
    effort_limit = np.asarray(effort_limit, dtype=float)
    velocity_limit = np.asarray(velocity_limit, dtype=float)

    ncols = 3
    nrows = -(-len(groups) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows), squeeze=False)
    rng = np.random.default_rng(seed)

    for i, group in enumerate(groups):
        ax = axes[i // ncols][i % ncols]
        idx = cols[group]
        t = np.abs(tau[:, idx]).ravel()
        o = np.abs(omega[:, idx]).ravel()
        sel = _downsample_idx(t.size, max_points, rng)
        ax.scatter(o[sel], t[sel], s=4, alpha=0.3, color="tab:blue", label="rollout samples")

        ax.axhline(effort_limit[idx[0]], color="black", linestyle="--", linewidth=1, label="preset effort_limit")
        ax.axvline(velocity_limit[idx[0]], color="black", linestyle=":", linewidth=1, label="preset velocity_limit")

        entry = (catalog_matches or {}).get(group)
        if entry:
            if "tau_peak_nm" in entry:
                ax.axhline(entry["tau_peak_nm"], color="tab:red", linestyle="--", linewidth=1, label="catalog tau_peak")
            if "omega_max_rad_s" in entry:
                ax.axvline(entry["omega_max_rad_s"], color="tab:red", linestyle=":", linewidth=1, label="catalog omega_max")
            if "can_clamp_nm" in entry:
                ax.axhline(entry["can_clamp_nm"], color="tab:orange", linestyle="-.", linewidth=1, label="CAN clamp")

        ax.set_title(group)
        ax.set_xlabel("|omega| (rad/s)")
        ax.set_ylabel("|tau| (Nm)")

    for j in range(len(groups), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    # Pooled across every subplot (not just the first): groups differ in
    # whether a catalog entry matched, so the "catalog tau_peak"/"omega_max"/
    # "CAN clamp" labels may only exist on some axes.
    by_label = {}
    for row in axes:
        for ax in row:
            ax_handles, ax_labels = ax.get_legend_handles_labels()
            by_label.update(zip(ax_labels, ax_handles))
    if by_label:
        fig.legend(by_label.values(), by_label.keys(), loc="lower center", ncol=len(by_label), fontsize=8)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return fig


def save_scatter(fig, out_path: Path) -> Path:
    import matplotlib.pyplot as plt

    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


# -- report assembly (needs sizing_data.npz + run.json) --------------------


def build_report(run_dir: Path, motors_name: str | None = "encos") -> tuple[str, Path]:
    npz_path = run_dir / "sizing_data.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"{npz_path} not found -- run `python -m humanoid_lab.sizing.collect --run {run_dir}` first"
        )
    data = np.load(npz_path)
    tau, omega = data["tau"], data["omega"]
    # Drop fall-transition samples: collect.py keeps the step on which an
    # episode terminated, and impact/post-gait torques from a falling robot
    # do not belong in a motor-sizing percentile. The done mask is saved for
    # exactly this filter. A converged policy never falls, so on a good run
    # this drops nothing.
    if "done" in data:
        alive = ~data["done"].astype(bool)
        n_dropped = int((~alive).sum())
        tau, omega = tau[alive], omega[alive]
    else:
        n_dropped = 0
    joint_names = [str(x) for x in data["joint_names"]]
    effort_limit, velocity_limit = data["effort_limit"], data["velocity_limit"]

    run_json_path = run_dir / "run.json"
    if not run_json_path.exists():
        raise FileNotFoundError(f"{run_json_path} not found -- is {run_dir} a run.sh train/smoke output dir?")
    run = json.loads(run_json_path.read_text())
    hydra = run.get("hydra_config", {})
    preset_name = hydra.get("actuators", {}).get("name", "?")
    robot_rel = hydra.get("robot", {}).get("dir")
    if not robot_rel:
        raise ValueError(
            f"{run_json_path} records no robot.dir; refusing to guess which "
            "robot produced this run"
        )
    robot_dir = paths.REPO_ROOT / robot_rel

    from humanoid_lab.robot.spec import load_robot_spec

    robot_spec = load_robot_spec(robot_dir)
    joint_groups = robot_spec.joint_groups

    metrics_by_group = group_metrics(tau, omega, joint_names, joint_groups)

    catalog_matches = None
    if motors_name:
        catalog = load_motor_catalog(motors_name)
        catalog_matches = match_catalog_to_groups(catalog, joint_groups.keys())

    png_path = run_dir / "sizing_scatter.png"
    fig = plot_torque_speed_scatter(
        tau, omega, joint_names, joint_groups, effort_limit, velocity_limit, catalog_matches
    )
    save_scatter(fig, png_path)

    md = render_report_markdown(
        run_name=run.get("run_name", run_dir.name),
        preset_name=preset_name,
        motors_name=motors_name,
        n_samples=int(tau.shape[0]),
        metrics_by_group=metrics_by_group,
        joint_groups=joint_groups,
        catalog_matches=catalog_matches,
        png_name=png_path.name,
        n_dropped=n_dropped,
    )
    (run_dir / "sizing_report.md").write_text(md)
    return md, png_path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--motors", default="encos", help="motors/<name>.yaml catalog to overlay (empty string to disable)")
    args = ap.parse_args()

    motors_name = args.motors or None
    _md, png_path = build_report(args.run, motors_name)
    print(f"sizing report: {args.run / 'sizing_report.md'}  {png_path}")


if __name__ == "__main__":
    main()
