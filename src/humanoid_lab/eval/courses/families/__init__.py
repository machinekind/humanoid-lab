"""Course families, one module each, and the catalogue built from them.

Each module exposes `FAMILY` and `courses(p) -> list[Course]`. The order of
FAMILY_MODULES[ground_class] is the catalogue and report order. Add a row by
appending it to its family's list, or add a module and list it here. Names
are unique within a ground class, and every `baseline` names a row of the
catalogue.

The catalogue's identity is `catalogue_fingerprint`: a sha256 over every row
of one ground class plus every frozen constant and derived parameter.
tests/unit/test_courses.py pins it per robot, so any change fails loudly.
Bump CATALOGUE_VERSION when an existing row's meaning changes (its spec, a
shared constant, a normalizer, a follower constant). Adding a row changes
the fingerprint and not the version; a row's name plus its `spec_hash` is
its identity.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Mapping

import numpy as np

from humanoid_lab.envs import progress
from humanoid_lab.eval import battery
from humanoid_lab.eval.courses import follower, geometry, spec
from humanoid_lab.eval.courses.families import (
    disturbance,
    floor,
    geometry_paths,
    speed,
    spin,
)
from humanoid_lab.eval.courses.spec import Course, CourseParams, PathCourse

CATALOGUE_VERSION = 1

FAMILY_MODULES = {"flat": (geometry_paths, speed, floor, disturbance, spin)}

# Floats are rounded to this many decimals before hashing, so a last-ulp
# libm difference between platforms cannot move a fingerprint.
HASH_DECIMALS = 9


def catalogue(p: CourseParams, ground_class: str = "flat") -> dict[str, Course]:
    """name -> course for one ground class, in family order."""
    if ground_class not in FAMILY_MODULES:
        raise ValueError(f"unknown ground class {ground_class!r}; known: {sorted(FAMILY_MODULES)}")
    out: dict[str, Course] = {}
    for mod in FAMILY_MODULES[ground_class]:
        for course in mod.courses(p):
            if course.name in out:
                raise ValueError(f"duplicate course name {course.name!r} in {ground_class!r}")
            out[course.name] = course
    # A row of another class may name a flat baseline; the report joins the
    # two files by row name.
    known = set(out) if ground_class == "flat" else set(out) | set(catalogue(p, "flat"))
    for course in out.values():
        if course.baseline is not None and course.baseline not in known:
            raise ValueError(f"{course.name}: baseline {course.baseline!r} is not a course")
    return out


def frozen_constants() -> dict:
    """Every robot-agnostic constant a score depends on, by block."""
    return {
        "follower": {
            "lookahead_m": follower.LOOKAHEAD_M,
            "k_yaw_spin": follower.K_YAW_SPIN,
            "spin_enter_rad": follower.SPIN_ENTER_RAD,
            "spin_exit_rad": follower.SPIN_EXIT_RAD,
            "goal_radius_m": follower.GOAL_RADIUS_M,
            "goal_min_progress_m": follower.GOAL_MIN_PROGRESS_M,
            "resample_ds_m": follower.RESAMPLE_DS_M,
            "progress_window_m": follower.PROGRESS_WINDOW_M,
            "holonomic": False,
        },
        "protocol": {
            "settle_s": battery.SETTLE_SEC,
            "lead_in_m": geometry.LEAD_IN_M,
            "time_factor": spec.TIME_FACTOR,
            "slack_s": spec.SLACK_SEC,
            "max_course_s": spec.MAX_COURSE_SEC,
            "min_score_s": spec.MIN_SCORE_SEC,
            "min_moving_s": spec.MIN_MOVING_SEC,
            "moving_threshold": progress.SPEED_DEADBAND,
            "yaw_speed_weight": progress.YAW_SPEED_WEIGHT,
            "spin_min_rad_s": spec.SPIN_MIN_RAD_S,
            "vibration_cutoff_hz": spec.VIBRATION_CUTOFF_HZ,
            "subscore_cap": spec.SUBSCORE_CAP,
            "slippery_mu": spec.SLIPPERY_MU,
            "push_at_m": spec.PUSH_AT_M,
            "canonical_seeds": spec.SEEDS,
        },
        "derivation": {
            "v_nom_frac": spec.V_NOM_FRAC,
            "v_slow_frac": spec.V_SLOW_FRAC,
            "v_fast_frac": spec.V_FAST_FRAC,
            "speed_step_fracs": spec.SPEED_STEP_FRACS,
            "yaw_cap_frac": spec.YAW_CAP_FRAC,
            "spin_fracs": spec.SPIN_FRACS,
            "r_tight_demand": spec.R_TIGHT_DEMAND,
            "r_tight_min_m": spec.R_TIGHT_MIN_M,
            "slalom_amplitude_m": spec.SLALOM_AMPLITUDE_M,
            "slalom_demand": spec.SLALOM_DEMAND,
            "slalom_wavelengths": spec.SLALOM_WAVELENGTHS,
        },
    }


def params_record(p: CourseParams) -> dict:
    """The params as plain JSON values. `source` is provenance, kept out of
    the fingerprint: equal numbers measured or pinned are the same catalogue."""
    rec = {f.name: getattr(p, f.name) for f in dataclasses.fields(p)}
    rec["speed_steps"] = list(p.speed_steps)
    rec["obs_noise"] = dict(p.obs_noise)
    return rec


def course_spec(course: Course) -> dict:
    """Every field that defines a row, as plain JSON values."""
    common = {
        "name": course.name,
        "family": course.family,
        "isolates": course.isolates,
        "baseline": course.baseline,
        "kind": spec.kind(course),
        "friction": course.friction,
        "ground": course.ground,
        "anchor": course.anchor,
        "origin": course.origin,
        "unscored": sorted(course.unscored),
    }
    if isinstance(course, PathCourse):
        return {
            **common,
            "waypoints": course.waypoints,
            "speeds": list(course.speeds),
            "push_at_m": course.push_at_m,
            "push_vel": course.push_vel,
        }
    return {**common, "wz": course.wz, "turns": course.turns}


def _canonical(obj):
    if obj is None or isinstance(obj, str):
        return obj
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        # + 0.0 turns a rounded -0.0 into 0.0, which json would print apart.
        return round(float(obj), HASH_DECIMALS) + 0.0
    if isinstance(obj, np.ndarray):
        return [_canonical(v) for v in obj.tolist()]
    if isinstance(obj, Mapping):
        return {str(k): _canonical(v) for k, v in obj.items()}
    if isinstance(obj, (frozenset, set)):
        return sorted(_canonical(v) for v in obj)
    if isinstance(obj, (list, tuple)):
        return [_canonical(v) for v in obj]
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        fields = {f.name: getattr(obj, f.name) for f in dataclasses.fields(obj)}
        return {"type": type(obj).__name__, **_canonical(fields)}
    raise TypeError(f"no canonical form for {type(obj).__name__}: {obj!r}")


def _sha256(obj) -> str:
    text = json.dumps(_canonical(obj), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()


def spec_hash(course: Course) -> str:
    """sha256 of one row's definition."""
    return _sha256(course_spec(course))


def catalogue_fingerprint(p: CourseParams, ground_class: str = "flat") -> str:
    """sha256 of one ground class's rows, the robot's params and every frozen
    constant. Rows of another class cannot move it."""
    params = params_record(p)
    del params["source"]
    return _sha256({
        "ground_class": ground_class,
        "rows": [course_spec(c) for c in catalogue(p, ground_class).values()],
        "params": params,
        "constants": frozen_constants(),
    })


def describe(course: Course, dt: float) -> dict:
    """The catalogue-side fields of a row's courses.json entry.

    `friction` is the row's own override (None keeps the model's); the
    runner records the effective value.
    """
    is_path = isinstance(course, PathCourse)
    return {
        "family": course.family,
        "isolates": course.isolates,
        "baseline": course.baseline,
        "kind": spec.kind(course),
        "spec_hash": spec_hash(course),
        "ground": course.ground,
        "anchor": course.anchor,
        "origin": None if course.origin is None else list(course.origin),
        "unscored": sorted(course.unscored),
        "friction": course.friction,
        "push_at_m": course.push_at_m if is_path else None,
        "push_vel": course.push_vel if is_path else None,
        "geometry": dict(course.geometry) if is_path else None,
        "speeds": list(course.speeds) if is_path else None,
        "wz": None if is_path else float(course.wz),
        "length_m": round(course.length_m, 3) if is_path else None,
        "turn_rad": None if is_path else round(course.turn_rad, 4),
        "ideal_s": round(course.ideal_sec, 3),
        "budget_steps": spec.budget_steps(course, dt),
    }
