"""Turn a recorded course lane into a score.

Each axis divides a physical reference by a measured error, so 1.0 means the
error is as large as the reference and higher is better:

    tracking    stance half-width / RMS cross-track error
    speed       mean commanded vx / RMS (commanded vx - forward speed),
                over steps where motion was commanded
    height      nominal height / RMS (base height - nominal height)
    grip        base distance / foot slip distance
    smoothness  1 / vibration_index (the fraction of joint-velocity power above 5 Hz)

Spin rows replace tracking, speed and grip with

    rotation    |wz command| / RMS (wz command - gyro z)
    drift       stance half-width / largest planar drift from the settled pose

A seed's score is the weakest scored axis when the course completed, else 0.

Height and grip are diagnostics. On a record that never fell, every step
stands above fall.min_height. The height axis then stays above
h_nom / (h_nom - fall.min_height). With fall.min_height 0.45 m that floor is
2.5 on Roboto and 3.4 on Asimov. A trained Roboto policy read grip 10.7-34.9
on straight_10m and circle_r2 (4 seeds each), from nominal foot friction
down to mu 0.2. On straight_slow it read 2.9-4.7 (2 seeds), where the
policy stalled and speed or tracking bound first. Neither axis binds while
the others are healthy.

numpy, plus the battery's own reducers (vibration_index, yaw_progress_deg)
and eval/gait.py's gait KPIs, so the two tools read a record the same way.
perfect_unicycle_entry runs the jitted follower (follower.perfect_unicycle).
"""

from __future__ import annotations

import json
import math
import os
import secrets
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from humanoid_lab.envs.progress import SPEED_DEADBAND
from humanoid_lab.eval.battery import vibration_index, yaw_progress_deg
from humanoid_lab.eval.courses import follower
from humanoid_lab.eval.courses.spec import (
    MIN_MOVING_SEC,
    MIN_SCORE_SEC,
    SUBSCORE_CAP,
    VIBRATION_CUTOFF_HZ,
    Course,
    CourseParams,
    PathCourse,
    SpinCourse,
    budget_steps,
    seconds_to_steps,
)
from humanoid_lab.eval.gait import gait_metrics

# A seed's outcome, decided in this order: the first that applies wins.
OUTCOMES = ("nonfinite", "settle_fell", "fell", "completed", "timed_out")


def outcome(*, nonfinite: bool, settle_fell: bool, fell: bool, reached: bool) -> str:
    """The lane's stop flags as one of OUTCOMES."""
    if nonfinite:
        return "nonfinite"
    if settle_fell:
        return "settle_fell"
    if fell:
        return "fell"
    return "completed" if reached else "timed_out"


def seed_out(out, seed: int) -> dict:
    """A lane's stop flags as the mapping seed_result reads, in plain Python
    values.

    `out` is any object with the attributes `steps`, `nonfinite`,
    `settle_fell`, `fell`, `reached`, `settle_height` and `yaw_rad`, as
    lane.LaneOut has them. `fell_at` is 0 on a settle fall. On a course fall
    it is `steps - 1`, the index of the step that tripped `done`. Otherwise
    it is None."""
    steps = int(out.steps)
    result = outcome(
        nonfinite=bool(out.nonfinite),
        settle_fell=bool(out.settle_fell),
        fell=bool(out.fell),
        reached=bool(out.reached),
    )
    fell_at = {"settle_fell": 0, "fell": steps - 1}.get(result)
    return {
        "seed": seed,
        "outcome": result,
        "steps": steps,
        "fell_at": fell_at,
        "settle_height_m": float(out.settle_height),
        "yaw_rad": float(out.yaw_rad),
    }


def _ratio(reference, error) -> float:
    """`reference / error`, capped at SUBSCORE_CAP and rounded to 3 decimals.

    A non-finite input scores 0.0. The test comes first because Python's
    builtins would score a NaN error as perfect: max(nan, 1e-9) is nan and
    min(1000.0, nan) is 1000.0.
    """
    reference, error = float(reference), float(error)
    if not (math.isfinite(reference) and math.isfinite(error)):
        return 0.0
    return round(min(SUBSCORE_CAP, reference / max(error, 1e-9)), 3)


def _rms(x) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.sqrt(np.mean(np.square(x)))) if x.size else float("nan")


def _trim(rec: Mapping, n: int) -> dict:
    return {k: np.asarray(v)[:n] for k, v in rec.items()}


def _round(value, nd: int):
    return None if value is None else round(float(value), nd)


def _seed_header(course: Course, out: Mapping, n: int, progress: float) -> dict:
    label = "progress_rad" if isinstance(course, SpinCourse) else "progress_m"
    return {
        "seed": out.get("seed"),
        "outcome": out["outcome"],
        "completed": out["outcome"] == "completed",
        "fell_at": out.get("fell_at"),
        "steps": n,
        label: round(float(progress), 3),
        "score": 0.0,
        "subscores": None,
        "raw": None,
        "gait": None,
        "binding": None,
    }


def _gated(out: Mapping, n: int, dt: float) -> bool:
    # Too short for an RMS or a spectrum, or physics that went non-finite:
    # no sub-scores, so such a seed cannot move a median.
    return out["outcome"] == "nonfinite" or n < seconds_to_steps(MIN_SCORE_SEC, dt)


def _finish(res: dict, subs: dict, course: Course) -> dict:
    """Unscored axes become None; score and binding read the rest."""
    for axis in course.unscored:
        subs[axis] = None
    scored = {k: v for k, v in subs.items() if v is not None}
    res["subscores"] = subs
    res["binding"] = min(scored, key=scored.get)
    res["score"] = round(min(scored.values()), 3) if res["completed"] else 0.0
    return res


def path_seed_result(rec: Mapping, out: Mapping, dt: float, course: PathCourse,
                     params: CourseParams) -> dict:
    """One (path row, seed) entry.

    `rec` holds the lane's per-step record; only its first `out["steps"]`
    steps are read. `out` holds the lane's `outcome` (OUTCOMES), `steps`,
    `fell_at`, `settle_height_m` and `seed`. A seed that fell or timed out
    keeps its sub-scores, so a failure stays diagnosable; its score is 0.
    """
    n = int(out["steps"])
    rec = _trim(rec, n)
    progress = float(rec["s"][n - 1]) if n else 0.0
    res = _seed_header(course, out, n, progress)
    if _gated(out, n, dt):
        return res

    xte = np.asarray(rec["xte"], dtype=float)
    cmd_vx = np.asarray(rec["cmd"], dtype=float)[:, 0]
    v_fwd = np.asarray(rec["v_fwd"], dtype=float)
    h = np.asarray(rec["h"], dtype=float)
    distance = float(np.sum(rec["v_planar"]) * dt)
    slip = float(np.sum(np.asarray(rec["foot_speed"], dtype=float)
                        * np.asarray(rec["contact"], dtype=float)) * dt)
    vibration = vibration_index(rec["qvel_act"], dt, cutoff_hz=VIBRATION_CUTOFF_HZ)

    # Speed is scored only where the follower asked for motion. The spin
    # branch commands vx = 0, and charging a pivot with speed error would
    # price a corner twice.
    moving = cmd_vx > SPEED_DEADBAND
    if moving.sum() >= seconds_to_steps(MIN_MOVING_SEC, dt):
        speed_err = _rms(cmd_vx[moving] - v_fwd[moving])
        speed_cmd = float(cmd_vx[moving].mean())
        speed_ratio = float(v_fwd[moving].mean()) / speed_cmd
        speed_axis = _ratio(speed_cmd, speed_err)
    else:
        speed_err = speed_cmd = speed_ratio = None
        speed_axis = SUBSCORE_CAP

    xte_rms = _rms(xte)
    height_err = _rms(h - params.nominal_height_m)
    res["raw"] = {
        "xte_rms_m": round(xte_rms, 4),
        "xte_p95_m": round(float(np.percentile(np.abs(xte), 95)), 4),
        "speed_err_rms": _round(speed_err, 4),
        "speed_cmd_mean": _round(speed_cmd, 4),
        # Delivered over commanded forward speed. The speed axis divides by
        # the command, so a steady 23% shortfall still scores 1 / 0.23 = 4.3.
        # A trained Roboto policy that delivered 77% on straight_fast scored
        # 3.6. The ratio is reported beside the axis so a shortfall shows.
        "speed_ratio": _round(speed_ratio, 3),
        "t_over_ideal": round(n * dt / course.ideal_sec, 3),
        "height_err_rms_m": round(height_err, 4),
        "settle_height_m": _round(out.get("settle_height_m"), 4),
        "base_distance_m": round(distance, 3),
        "slip_distance_m": round(slip, 3),
        "vibration": round(vibration, 4),
        "duration_s": round(n * dt, 2),
    }
    # The record starts after the settle, so nothing is dropped. Reported,
    # never scored.
    res["gait"] = gait_metrics(rec, settle_steps=0)
    subs = {
        "tracking": _ratio(params.stance_halfwidth_m, xte_rms),
        "speed": speed_axis,
        "height": _ratio(params.nominal_height_m, height_err),
        "grip": _ratio(distance, slip),
        "smoothness": _ratio(1.0, vibration),
    }
    return _finish(res, subs, course)


def spin_seed_result(rec: Mapping, out: Mapping, dt: float, course: SpinCourse,
                     params: CourseParams) -> dict:
    """One (spin row, seed) entry.

    `out` is as for path_seed_result plus `yaw_rad`, the yaw progress in the
    spin's direction that gated completion. The rotation error runs over
    every recorded step, the spin-up from stand included: executing the
    spin-up is part of the row.
    """
    n = int(out["steps"])
    rec = _trim(rec, n)
    res = _seed_header(course, out, n, out.get("yaw_rad", 0.0) if n else 0.0)
    if _gated(out, n, dt):
        return res

    wz_cmd = float(course.wz)
    gyro_z = np.asarray(rec["gyro_z"], dtype=float)
    h = np.asarray(rec["h"], dtype=float)
    xy = np.stack([np.asarray(rec["x"], dtype=float), np.asarray(rec["y"], dtype=float)], -1)
    # rec's first pose is the post-settle pose the course started from.
    drift = float(np.max(np.linalg.norm(xy - xy[0], axis=1)))
    wz_err = _rms(wz_cmd - gyro_z)
    height_err = _rms(h - params.nominal_height_m)
    vibration = vibration_index(rec["qvel_act"], dt, cutoff_hz=VIBRATION_CUTOFF_HZ)
    res["raw"] = {
        "wz_cmd": round(wz_cmd, 4),
        "wz_err_rms": round(wz_err, 4),
        "drift_max_m": round(drift, 4),
        "yaw_rad": _round(out.get("yaw_rad"), 3),
        "yaw_progress_deg": _round(yaw_progress_deg(gyro_z, dt, settle=0), 2),
        "height_err_rms_m": round(height_err, 4),
        "vibration": round(vibration, 4),
        "duration_s": round(n * dt, 2),
    }
    subs = {
        # |wz| normalizes, so a right spin cannot score negative.
        "rotation": _ratio(abs(wz_cmd), wz_err),
        "drift": _ratio(params.stance_halfwidth_m, drift),
        "height": _ratio(params.nominal_height_m, height_err),
        "smoothness": _ratio(1.0, vibration),
    }
    return _finish(res, subs, course)


def seed_result(rec: Mapping, out: Mapping, dt: float, course: Course,
                params: CourseParams) -> dict:
    fn = spin_seed_result if isinstance(course, SpinCourse) else path_seed_result
    return fn(rec, out, dt, course, params)


def _median(values, nd: int):
    vals = [float(v) for v in values if v is not None]
    return round(float(np.median(vals)), nd) if vals else None


def _median_block(dicts: list[dict], nd: int) -> dict | None:
    if not dicts:
        return None
    keys = list(dicts[0])
    return {k: _median([d.get(k) for d in dicts], nd) for k in keys}


def aggregate(seeds: list[dict]) -> dict:
    """Median and worst over a row's seeds, the failure counts, and the
    per-axis medians over the seeds that have sub-scores (None skipped)."""
    scores = [s["score"] for s in seeds]
    scored = [s for s in seeds if s.get("subscores") is not None]
    sub_median = _median_block([s["subscores"] for s in scored], 3)
    live = {k: v for k, v in (sub_median or {}).items() if v is not None}
    return {
        "score_median": round(float(np.median(scores)), 3),
        "score_worst": round(float(np.min(scores)), 3),
        "seeds": len(seeds),
        "completed": sum(s["outcome"] == "completed" for s in seeds),
        # Settle falls included.
        "falls": sum(s["outcome"] in ("fell", "settle_fell") for s in seeds),
        "timeouts": sum(s["outcome"] == "timed_out" for s in seeds),
        "nonfinite": sum(s["outcome"] == "nonfinite" for s in seeds),
        "subscore_median": sub_median,
        "raw_median": _median_block([s["raw"] for s in scored], 4),
        "gait_median": _median_block([s["gait"] for s in scored if s.get("gait")], 4),
        "binding": min(live, key=live.get) if live else None,
        "per_seed": seeds,
    }


def summary(rows: dict[str, dict], catalogue: Mapping[str, Course]) -> dict:
    """Lane counts over every row, and each row's `vs_baseline` filled in.

    `rows` maps a row name to its aggregate(); it is updated in place.
    Pairs compare medians, never seeds: a row and its baseline share their
    reset states, but contact chaos decorrelates the two rollouts within a
    few hundred steps.
    """
    for name, row in rows.items():
        course = catalogue[name]
        if course.baseline is None:
            row["vs_baseline"] = None
            continue
        base = rows.get(course.baseline)
        vs = {"baseline": course.baseline, "d_score_median": None, "slip_ratio": None}
        if base is not None:
            vs["d_score_median"] = round(row["score_median"] - base["score_median"], 3)
            if course.friction is not None:
                slip = (row.get("raw_median") or {}).get("slip_distance_m")
                base_slip = (base.get("raw_median") or {}).get("slip_distance_m")
                if slip is not None and base_slip:
                    vs["slip_ratio"] = round(slip / base_slip, 3)
        row["vs_baseline"] = vs
    return {
        "lanes": sum(r["seeds"] for r in rows.values()),
        "lanes_completed": sum(r["completed"] for r in rows.values()),
        "lanes_fell": sum(r["falls"] for r in rows.values()),
        "lanes_timed_out": sum(r["timeouts"] for r in rows.values()),
        "lanes_nonfinite": sum(r["nonfinite"] for r in rows.values()),
        "rows_all_completed": sum(r["completed"] == r["seeds"] for r in rows.values()),
    }


def perfect_unicycle_entry(course: PathCourse, params: CourseParams, dt: float,
                           n_points: int) -> dict:
    """A row's `perfect_unicycle` block: what a unicycle that executes the
    follower's commands exactly earns on it (follower.perfect_unicycle).

    It is not a ceiling. A robot whose yaw lags can exceed it, because pure
    pursuit cuts curves to the inside and the lag pushes back out.
    """
    u = follower.perfect_unicycle(
        follower.pack_path(course, n_points), params.yaw_cap, budget_steps(course, dt), dt
    )
    xte = float(u["xte_rms_m"])
    return {
        "xte_rms_m": round(xte, 6),
        "tracking": _ratio(params.stance_halfwidth_m, xte),
        "t_over_ideal": round(int(u["steps"]) * dt / course.ideal_sec, 3),
        "completed": bool(u["completed"]),
    }


def write_json(path: Path, doc: dict) -> None:
    """Write `doc` atomically. The text goes to a temp file in the same
    directory, is synced to disk, and then replaces `path`, so a crash or a
    power loss leaves the previous file or the new one, never a partial one.
    The file mode follows the umask, as a plain open's does. A NaN or inf
    fails the write (allow_nan=False) instead of producing invalid JSON."""
    path = Path(path)
    text = json.dumps(doc, indent=2, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
    # O_EXCL: a name collision raises here, before the cleanup below could
    # delete a file this call did not create.
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
