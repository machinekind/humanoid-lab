"""Course benchmark core: geometry, the derived catalogue, the frozen
follower, scoring and the courses report section.

Model-free. The follower runs eagerly on packed path arrays and only the
perfect unicycle is jitted. A course score is compared across months of
runs, so the robot inputs, the derived params, the follower constants and
the catalogue fingerprint are asserted explicitly: a failure here is the
alarm when one of them moves.
"""

from __future__ import annotations

import inspect
import itertools
import json
import math
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from humanoid_lab import paths
from humanoid_lab.envs import progress
from humanoid_lab.envs.joystick import default_config
from humanoid_lab.eval import battery
from humanoid_lab.eval.courses import families, follower, scoring, spec
from humanoid_lab.eval.courses.families import (
    CATALOGUE_VERSION,
    catalogue,
    catalogue_fingerprint,
    describe,
    geometry_paths,
    spec_hash,
)
from humanoid_lab.eval.courses.geometry import (
    LEAD_IN_M,
    arc,
    circle,
    join,
    lead_in,
    line,
    rounded_square,
    sine_slalom,
)
from humanoid_lab.eval.courses.model_ids import friction_geom_ids
from humanoid_lab.eval.courses.report import console_table, render_markdown
from humanoid_lab.eval.courses.scoring import (
    _ratio,
    aggregate,
    path_seed_result,
    perfect_unicycle_entry,
    spin_seed_result,
    summary,
    write_json,
)
from humanoid_lab.eval.courses.spec import (
    PATH_AXES,
    ROBOT_INPUTS,
    SPIN_AXES,
    PathCourse,
    SpinCourse,
    budget_steps,
    params_for,
)
from humanoid_lab.eval.gait import gait_metrics

ROBOTS = ("roboto_origin", "asimov_v1")
PARAMS = {r: params_for(r) for r in ROBOTS}
CAT = {r: catalogue(PARAMS[r]) for r in ROBOTS}
ROBOTO = PARAMS["roboto_origin"]
_DT = 0.02  # ctrl_dt on both robots

TWENTY = [
    "straight_10m", "arc_r3_90deg", "circle_r2", "circle_tight", "figure_eight_r15",
    "square_3m", "slalom_05m", "u_turn",
    "straight_slow", "straight_fast", "circle_r2_fast", "speed_steps_straight",
    "straight_slippery", "circle_r2_slippery",
    "straight_push", "straight_push_fast",
    "spin_left", "spin_right", "spin_slow", "spin_fast",
]

# Per row: (length m, ideal s, budget steps at dt 0.02, perfect-unicycle
# cross-track RMS m, perfect-unicycle steps to the goal). The last two were
# computed by a float64 numpy implementation of the same follower driving a
# unicycle from the course origin at dt 0.02; every row completes.
TABLE = {
    "roboto_origin": {
        "straight_10m": (10.000, 20.00, 2600, 0.0, 976),
        "arc_r3_90deg": (5.712, 11.42, 1528, 0.002154, 548),
        "circle_r2": (13.565, 27.13, 3491, 0.001903, 1339),
        "circle_tight": (5.712, 11.42, 1528, 0.009170, 567),
        "figure_eight_r15": (19.846, 39.69, 5062, 0.003946, 1977),
        "square_3m": (12.461, 24.92, 3215, 0.012560, 1239),
        "slalom_05m": (14.582, 29.16, 3746, 0.012221, 1441),
        "u_turn": (9.356, 18.71, 2439, 0.010037, 923),
        "straight_slow": (10.000, 50.00, 6350, 0.0, 2438),
        "straight_fast": (10.000, 11.11, 1489, 0.0, 542),
        "circle_r2_fast": (13.565, 15.07, 1984, 0.002385, 744),
        "speed_steps_straight": (10.000, 16.35, 2144, 0.0, 800),
        "straight_slippery": (10.000, 20.00, 2600, 0.0, 976),
        "circle_r2_slippery": (13.565, 27.13, 3491, 0.001903, 1339),
        "straight_push": (10.000, 20.00, 2600, 0.0, 976),
        "straight_push_fast": (10.000, 11.11, 1489, 0.0, 542),
        "spin_left": (None, 6.25, 882, None, None),
        "spin_right": (None, 6.25, 882, None, None),
        "spin_slow": (None, 12.51, 1663, None, None),
        "spin_fast": (None, 4.17, 621, None, None),
    },
    "asimov_v1": {
        "straight_10m": (10.000, 25.00, 3225, 0.0, 1219),
        "arc_r3_90deg": (5.712, 14.28, 1885, 0.002099, 685),
        "circle_r2": (13.565, 33.91, 4339, 0.001798, 1673),
        "circle_tight": (8.853, 22.13, 2867, 0.003192, 1087),
        "figure_eight_r15": (19.846, 49.62, 6302, 0.003908, 2471),
        "square_3m": (12.102, 30.26, 3882, 0.006492, 1489),
        "slalom_05m": (19.432, 48.58, 6173, 0.006023, 2405),
        "u_turn": (10.926, 27.32, 3514, 0.004118, 1340),
        "straight_slow": (10.000, 62.50, 7912, 0.0, 3047),
        "straight_fast": (10.000, 13.89, 1836, 0.0, 678),
        "circle_r2_fast": (13.565, 18.84, 2455, 0.002155, 930),
        "speed_steps_straight": (10.000, 20.44, 2655, 0.0, 1000),
        "straight_slippery": (10.000, 25.00, 3225, 0.0, 1219),
        "circle_r2_slippery": (13.565, 33.91, 4339, 0.001798, 1673),
        "straight_push": (10.000, 25.00, 3225, 0.0, 1219),
        "straight_push_fast": (10.000, 13.89, 1836, 0.0, 678),
        "spin_left": (None, 16.36, 2145, None, None),
        "spin_right": (None, 16.36, 2145, None, None),
        "spin_slow": (None, 18.85, 2456, None, None),
        "spin_fast": (None, 10.91, 1464, None, None),
    },
}

# The alarm. A change here means every recorded courses.json of that robot
# is stale. Decide whether the change alters an existing row's meaning; if it
# does, bump CATALOGUE_VERSION as well.
FINGERPRINTS = {
    "roboto_origin": "75b79fd2ea4355f13c55aa6008fff2bdcdc3fcb2a35332eccea7b81443453853",
    "asimov_v1": "afeed9067b1b72baba89cc4a4600a10933d323d95fcff9ea1908ce400d006c3b",
}

# The catalogue-side keys of a courses.json row. ground, anchor, origin and
# unscored are the slots a placed course fills.
ROW_KEYS = {
    "family", "isolates", "baseline", "kind", "spec_hash", "ground", "anchor", "origin",
    "unscored", "friction", "push_at_m", "push_vel", "geometry", "speeds", "wz", "length_m",
    "turn_rad", "ideal_s", "budget_steps",
}


def _overlay(robot):
    return yaml.safe_load((paths.CONFIGS_DIR / "robot" / f"{robot}.yaml").read_text())


def _overlay_names():
    return sorted(p.stem for p in (paths.CONFIGS_DIR / "robot").glob("*.yaml"))


def _paths(robot):
    return {n: c for n, c in CAT[robot].items() if isinstance(c, PathCourse)}


def _spins(robot):
    return {n: c for n, c in CAT[robot].items() if isinstance(c, SpinCourse)}


# -- geometry ----------------------------------------------------------------


def test_line_runs_from_its_start_along_its_heading():
    assert np.allclose(line(10.0), [[0, 0], [10, 0]])
    assert np.allclose(line(2.0, start=(1.0, 1.0), heading=math.pi / 2), [[1, 1], [1, 3]])


def test_arc_leaves_tangent_to_its_heading_and_turns_left_for_a_positive_sweep():
    a = arc(2.0, math.pi / 2, n=64)
    assert np.allclose(a[0], [0.0, 0.0])
    assert np.allclose(a[-1], [2.0, 2.0], atol=1e-9)
    assert a[1][0] > 0 and abs(a[1][1]) < abs(a[1][0])
    assert np.all(np.diff(a[:, 1]) >= -1e-12)


def test_arc_with_a_negative_sweep_turns_right():
    assert np.allclose(arc(2.0, -math.pi / 2, n=64)[-1], [2.0, -2.0], atol=1e-9)


def test_a_full_circle_arc_closes_on_itself():
    a = arc(1.5, 2 * math.pi, n=96)
    assert np.allclose(a[-1], a[0], atol=1e-9)
    c = circle(2.0)
    assert np.allclose(c[0], [0.0, 0.0])
    assert np.allclose(c[-1], [LEAD_IN_M, 0.0], atol=1e-9)


def test_join_drops_shared_endpoints_and_keeps_disjoint_ones():
    j = join(line(3.0), line(2.0, start=(3.0, 0.0)))
    assert np.allclose(j, [[0, 0], [3, 0], [5, 0]])
    assert len(lead_in(line(2.0))) == 3
    gap = join(line(1.0), line(1.0, start=(2.0, 0.0)))
    assert len(gap) == 4 and np.allclose(gap, [[0, 0], [1, 0], [2, 0], [3, 0]])


def test_sine_slalom_amplitude_and_endpoints():
    s = sine_slalom(9.0, 0.5, 3.0, n=181)
    assert np.allclose(s[0], [0.0, 0.0])
    assert s[-1][0] == pytest.approx(9.0)
    assert np.abs(s[:, 1]).max() == pytest.approx(0.5, abs=1e-3)


@pytest.mark.parametrize("r", [0.75, 1.25])
def test_rounded_square_is_tangent_continuous_ends_at_r_and_has_its_length(r):
    sq = rounded_square(3.0, r)
    assert np.allclose(sq[0], [0.0, 0.0])
    assert np.allclose(sq[-1], [r, 0.0], atol=1e-9)
    heading = np.unwrap(np.arctan2(np.diff(sq[:, 1]), np.diff(sq[:, 0])))
    # No corner: consecutive chords turn by at most one arc step (90 deg over
    # 23 chords), and the whole square turns once.
    assert np.abs(np.diff(heading)).max() < (math.pi / 2) / 23 + 1e-9
    assert heading[-1] - heading[0] == pytest.approx(2 * math.pi, abs=0.07)
    exact = 12.0 - 7.0 * r + 2.0 * math.pi * r
    fine = rounded_square(3.0, r, n=4001)
    assert np.linalg.norm(np.diff(fine, axis=0), axis=1).sum() == pytest.approx(exact, abs=1e-6)
    assert np.linalg.norm(np.diff(sq, axis=0), axis=1).sum() == pytest.approx(exact, abs=2e-3)


def test_rounded_square_refuses_corners_that_leave_no_edge():
    for r in (1.5, 2.0):
        with pytest.raises(ValueError, match=f"side 3.0 m .* radius {r} m"):
            rounded_square(3.0, r)


# -- robot inputs and derivation ---------------------------------------------


@pytest.mark.parametrize("robot", ROBOTS)
def test_pinned_command_box_matches_robot_overlays(robot):
    box = _overlay(robot)["task"]["env"]["command"]
    inputs = ROBOT_INPUTS[robot]
    assert inputs.vx_max == box["vx"][1]
    assert inputs.wz_max == min(-box["wz"][0], box["wz"][1])


@pytest.mark.parametrize("robot", ROBOTS)
def test_pinned_push_vel_matches_training_push(robot):
    push = _overlay(robot)["task"]["env"].get("push") or {}
    trained = push.get("vel", default_config().push.vel)
    assert ROBOT_INPUTS[robot].push_vel == trained


@pytest.mark.parametrize("robot", ROBOTS)
def test_pinned_obs_noise_matches_overlay_over_task_defaults(robot):
    task = yaml.safe_load((paths.CONFIGS_DIR / "task" / "joystick.yaml").read_text())
    noise = {**task["env"]["obs_noise"], **(_overlay(robot)["task"]["env"].get("obs_noise") or {})}
    assert dict(ROBOT_INPUTS[robot].obs_noise) == noise


def test_every_robot_overlay_has_pinned_inputs():
    names = {_overlay(n)["robot"]["name"] for n in _overlay_names()}
    assert names <= set(ROBOT_INPUTS)
    assert set(ROBOTS) <= names


def test_derived_params_are_the_documented_fractions():
    documented = {
        "roboto_origin": {
            "v_nom": 0.5, "v_slow": 0.2, "v_fast": 0.9, "speed_steps": (0.5, 0.9, 0.5, 0.7),
            "yaw_cap": 1.256, "spin_nom": 1.005, "spin_slow": 0.502, "spin_fast": 1.507,
            "r_tight": 0.75, "slalom_wavelength": 3.964, "push_vel": 0.5,
        },
        "asimov_v1": {
            "v_nom": 0.4, "v_slow": 0.16, "v_fast": 0.72, "speed_steps": (0.4, 0.72, 0.4, 0.56),
            "yaw_cap": 0.48, "spin_nom": 0.384, "spin_slow": 0.333, "spin_fast": 0.576,
            "r_tight": 1.25, "slalom_wavelength": 5.736, "push_vel": 0.4,
        },
    }
    for robot, want in documented.items():
        p = PARAMS[robot]
        assert p.source == "pinned" and p.robot == robot
        assert p.slalom_amplitude == 0.5
        assert p.stance_halfwidth_m == ROBOT_INPUTS[robot].stance_halfwidth_m
        assert p.nominal_height_m == ROBOT_INPUTS[robot].nominal_height_m
        for key, value in want.items():
            assert getattr(p, key) == pytest.approx(value, abs=5e-4), (robot, key)


@pytest.mark.parametrize("robot", ROBOTS)
def test_every_commanded_speed_and_rate_sits_inside_the_robot_box(robot):
    box = _overlay(robot)["task"]["env"]["command"]
    p = PARAMS[robot]
    for name, c in _paths(robot).items():
        assert np.all(c.segment_speeds < box["vx"][1]), name
        assert np.all(c.segment_speeds > max(box["vx"][0], 0.0)), name
    # The follower may command either sign of its cap.
    assert box["wz"][0] < -p.yaw_cap < 0.0 < p.yaw_cap < box["wz"][1]
    for name, c in _spins(robot).items():
        assert box["wz"][0] < c.wz < box["wz"][1], name


@pytest.mark.parametrize("robot", ROBOTS)
def test_every_commanded_speed_clears_twice_the_stand_threshold(robot):
    floor = 2 * progress.SPEED_DEADBAND - 1e-12
    for name, c in _paths(robot).items():
        assert c.segment_speeds.min() >= floor, name
    for name, c in _spins(robot).items():
        assert progress.YAW_SPEED_WEIGHT * abs(c.wz) >= floor, name
    assert spec.SPIN_MIN_RAD_S == pytest.approx(0.3333, abs=1e-4)
    # The floor binds on Asimov's slow spin: 0.4 of its cap is 0.192 rad/s.
    assert PARAMS["asimov_v1"].spin_slow == spec.SPIN_MIN_RAD_S


def test_tight_radius_follows_the_cap_with_its_floor():
    for robot in ROBOTS:
        p = PARAMS[robot]
        by_demand = 1.5 * p.v_nom / p.yaw_cap
        assert p.r_tight == pytest.approx(max(by_demand, spec.R_TIGHT_MIN_M))
    # Roboto's demand radius is 0.597 m, so the 0.75 m floor binds there.
    assert PARAMS["roboto_origin"].r_tight == spec.R_TIGHT_MIN_M
    assert PARAMS["asimov_v1"].r_tight == pytest.approx(1.25)
    # A robot with a weak yaw gets a wider turn, never a harder demand.
    weak = spec.derive("weak", spec.RobotInputs(1.0, 0.3, 0.1, 0.7, 0.4, {"gyro": 0.01}))
    assert weak.r_tight * weak.yaw_cap / weak.v_nom == pytest.approx(1.5)


def test_slalom_peak_yaw_demand_is_half_the_cap():
    for robot in ROBOTS:
        p = PARAMS[robot]
        peak_curvature = p.slalom_amplitude * (2 * math.pi / p.slalom_wavelength) ** 2
        assert p.v_nom * peak_curvature == pytest.approx(0.5 * p.yaw_cap)
        c = CAT[robot]["slalom_05m"]
        assert c.waypoints[-1][0] == pytest.approx(LEAD_IN_M + 3 * p.slalom_wavelength)
        assert np.abs(c.waypoints[:, 1]).max() == pytest.approx(0.5, abs=1e-3)


def test_unknown_robot_has_no_pinned_params():
    assert params_for("no_such_robot") is None
    measured = spec.derive("other", ROBOT_INPUTS["roboto_origin"], source="measured")
    assert measured.source == "measured"


# -- catalogue -----------------------------------------------------------------


@pytest.mark.parametrize("robot", ROBOTS)
def test_catalogue_is_the_documented_twenty_in_family_order(robot):
    assert list(CAT[robot]) == TWENTY
    families_seen = [c.family for c in CAT[robot].values()]
    assert families_seen == sorted(families_seen, key=families_seen.index)
    assert list(dict.fromkeys(families_seen)) == [
        "geometry", "speed", "floor", "disturbance", "spin"
    ]


@pytest.mark.parametrize("robot", ROBOTS)
def test_every_path_course_starts_at_the_course_origin(robot):
    for name, c in _paths(robot).items():
        assert np.allclose(c.waypoints[0], [0.0, 0.0]), name


@pytest.mark.parametrize("robot", ROBOTS)
def test_segment_speeds_broadcast_or_match_the_segment_count(robot):
    for name, c in _paths(robot).items():
        assert len(c.segment_speeds) == len(c.waypoints) - 1, name
        assert len(c.speeds) in (1, len(c.waypoints) - 1), name


@pytest.mark.parametrize("robot", ROBOTS)
def test_speed_steps_is_the_only_varying_command_and_never_below_v_nom(robot):
    varying = {n for n, c in _paths(robot).items() if len(np.unique(c.segment_speeds)) > 1}
    assert varying == {"speed_steps_straight"}
    p = PARAMS[robot]
    steps = CAT[robot]["speed_steps_straight"]
    assert np.allclose(steps.waypoints[:, 0], [0.0, 2.5, 5.0, 7.5, 10.0])
    # The recorded block length is the hashed path's.
    assert np.allclose(np.diff(steps.waypoints[:, 0]), describe(steps, _DT)["geometry"]["block_m"])
    assert np.allclose(steps.segment_speeds, [f * ROBOT_INPUTS[robot].vx_max
                                              for f in (0.5, 0.9, 0.5, 0.7)])
    assert steps.segment_speeds.min() >= p.v_nom - 1e-12


@pytest.mark.parametrize("robot", ROBOTS)
def test_only_the_floor_rows_change_friction(robot):
    friction = {n: c.friction for n, c in CAT[robot].items() if c.friction is not None}
    assert friction == {"straight_slippery": 0.25, "circle_r2_slippery": 0.25}
    assert {CAT[robot][n].family for n in friction} == {"floor"}


@pytest.mark.parametrize("robot", ROBOTS)
def test_only_the_push_rows_kick_at_5_m(robot):
    pushed = {n: (c.push_at_m, c.push_vel) for n, c in _paths(robot).items()
              if c.push_at_m is not None}
    vel = ROBOT_INPUTS[robot].push_vel
    assert pushed == {"straight_push": (5.0, vel), "straight_push_fast": (5.0, vel)}


def _differences(course, base):
    """The variables `course` changes against `base`."""
    out = set()
    if isinstance(course, SpinCourse):
        if np.sign(course.wz) != np.sign(base.wz):
            out.add("spin sign")
        if abs(course.wz) != abs(base.wz):
            out.add("spin rate")
        return out
    # Compared on the densified path: speed_steps_straight cuts the same
    # line into four segments.
    pts, spd = follower.densify(course.waypoints, course.segment_speeds)
    base_pts, base_spd = follower.densify(base.waypoints, base.segment_speeds)
    same_path = pts.shape == base_pts.shape and np.allclose(pts, base_pts, atol=1e-9)
    if not same_path:
        out.add("geometry")
    if not (same_path and np.array_equal(spd, base_spd)):
        out.add("speed")
    if course.friction != base.friction:
        out.add("mu")
    if (course.push_at_m, course.push_vel) != (base.push_at_m, base.push_vel):
        out.add("kick")
    return out


@pytest.mark.parametrize("robot", ROBOTS)
def test_each_row_differs_from_its_baseline_in_exactly_its_variable(robot):
    expected = {
        "straight_slow": "speed", "straight_fast": "speed", "circle_r2_fast": "speed",
        "speed_steps_straight": "speed", "straight_slippery": "mu", "circle_r2_slippery": "mu",
        "straight_push": "kick", "straight_push_fast": "kick",
        "spin_right": "spin sign", "spin_slow": "spin rate", "spin_fast": "spin rate",
    }
    cat = CAT[robot]
    with_baseline = {n: c for n, c in cat.items() if c.baseline is not None}
    assert set(with_baseline) == set(expected)
    for name, c in with_baseline.items():
        base = cat[c.baseline]
        assert base.baseline is None or name == "straight_push_fast", name
        assert _differences(c, base) == {expected[name]}, name
        # A row on its baseline's path records every shape parameter the
        # baseline records. It may add fields of its own command (block_m).
        if isinstance(c, PathCourse):
            assert base.geometry.items() <= c.geometry.items(), name
    assert cat["straight_push_fast"].baseline == "straight_fast"
    assert cat["circle_r2_slippery"].baseline == "circle_r2"


@pytest.mark.parametrize("robot", ROBOTS)
def test_spin_directions_and_rates(robot):
    spins = _spins(robot)
    p = PARAMS[robot]
    assert spins["spin_left"].wz == p.spin_nom > 0
    assert spins["spin_right"].wz == -spins["spin_left"].wz
    assert 0 < spins["spin_slow"].wz < spins["spin_left"].wz < spins["spin_fast"].wz
    assert all(c.turns == 1.0 and c.turn_rad == pytest.approx(2 * math.pi)
               for c in spins.values())


def test_an_invalid_path_row_is_rejected_with_its_name_and_reason():
    with pytest.raises(ValueError, match="bad_row: 3 speeds for 2 segments"):
        PathCourse("bad_row", "test", "", np.array([[0, 0], [1, 0], [2, 0]]), (0.5, 0.5, 0.5))
    with pytest.raises(ValueError, match="bad_kick: push_at_m and push_vel come together"):
        PathCourse("bad_kick", "test", "", line(1.0), (0.5,), push_at_m=0.5)
    with pytest.raises(ValueError, match=r"bad_axis: unscored \['rotation'\] are not axes"):
        PathCourse("bad_axis", "test", "", line(1.0), (0.5,), unscored={"rotation"})


@pytest.mark.parametrize("robot", ROBOTS)
def test_flat_rows_carry_no_ground_no_origin_and_start_anchors(robot):
    for name, c in CAT[robot].items():
        assert (c.ground, c.anchor, c.origin, c.unscored) == (None, "start", None, frozenset())
        assert spec.ground_class(c) == "flat", name
        # A subset check: a row may gain keys without a schema bump, but no
        # key may go missing.
        d = describe(c, _DT)
        assert ROW_KEYS <= set(d), name
        slots = (d["ground"], d["anchor"], d["origin"], d["unscored"])
        assert slots == (None, "start", None, []), name
    assert spec.GROUND_CLASSES == {"flat": "courses.json"}


def test_a_non_flat_ground_is_rejected():
    for kwargs in ({"ground": "arena_1"}, {"anchor": "world"}, {"origin": (1.0, 2.0, 0.0)}):
        with pytest.raises(ValueError, match="placed"):
            PathCourse("placed", "test", "", line(1.0), (0.5,), **kwargs)
        with pytest.raises(ValueError, match="placed"):
            SpinCourse("placed", "test", "", wz=0.5, **kwargs)
    with pytest.raises(ValueError, match="unknown ground class"):
        catalogue(ROBOTO, "terrain")


def test_flat_fingerprint_ignores_other_ground_classes(monkeypatch):
    before = catalogue_fingerprint(ROBOTO)
    stub = SimpleNamespace(
        FAMILY="stub",
        courses=lambda p: [PathCourse("stub_row", "stub", "stub", line(3.0), (p.v_nom,),
                                      baseline="straight_10m")],
    )
    monkeypatch.setitem(families.FAMILY_MODULES, "terrain", (stub,))
    assert catalogue_fingerprint(ROBOTO, "flat") == before
    assert list(catalogue(ROBOTO, "terrain")) == ["stub_row"]
    assert catalogue_fingerprint(ROBOTO, "terrain") != before


def test_catalogue_rejects_duplicate_names_and_missing_baselines(monkeypatch):
    def family(*rows):
        return SimpleNamespace(FAMILY="stub", courses=lambda p: list(rows))

    row = PathCourse("row", "stub", "", line(1.0), (0.5,))
    monkeypatch.setitem(families.FAMILY_MODULES, "flat", (family(row, row),))
    with pytest.raises(ValueError, match="duplicate course name 'row'"):
        catalogue(ROBOTO)
    orphan = PathCourse("orphan", "stub", "", line(1.0), (0.5,), baseline="nowhere")
    monkeypatch.setitem(families.FAMILY_MODULES, "flat", (family(orphan),))
    with pytest.raises(ValueError, match="orphan: baseline 'nowhere'"):
        catalogue(ROBOTO)


def test_spec_hash_ignores_last_ulp_noise_and_moves_with_the_row():
    base = CAT["roboto_origin"]["circle_r2"]
    noisy = PathCourse(base.name, base.family, base.isolates, base.waypoints + 1e-15,
                       base.speeds, baseline=base.baseline)
    moved = PathCourse(base.name, base.family, base.isolates, base.waypoints * 1.000001,
                       base.speeds, baseline=base.baseline)
    assert spec_hash(noisy) == spec_hash(base)
    assert spec_hash(moved) != spec_hash(base)
    hashes = [spec_hash(c) for c in CAT["roboto_origin"].values()]
    # Equal definitions hash equal; every catalogue row is distinct.
    assert len(set(hashes)) == len(hashes)


@pytest.mark.parametrize("robot", ROBOTS)
def test_catalogue_fingerprint_and_version_are_pinned(robot):
    assert CATALOGUE_VERSION == 1
    assert catalogue_fingerprint(PARAMS[robot]) == FINGERPRINTS[robot]
    # The source is provenance, not meaning.
    measured = spec.derive(robot, ROBOT_INPUTS[robot], source="measured")
    assert catalogue_fingerprint(measured) == FINGERPRINTS[robot]


def test_describe_gives_the_catalogue_fields_of_a_courses_json_row():
    cat = CAT["roboto_origin"]
    path = describe(cat["straight_push"], _DT)
    assert path["kind"] == "path" and path["wz"] is None and path["turn_rad"] is None
    assert (path["push_at_m"], path["push_vel"], path["budget_steps"]) == (5.0, 0.5, 2600)
    assert path["spec_hash"] == spec_hash(cat["straight_push"])
    # The row's own friction override. The runner records the effective value.
    assert path["friction"] is None
    assert describe(cat["straight_slippery"], _DT)["friction"] == 0.25
    spin = describe(cat["spin_right"], _DT)
    assert spin["kind"] == "spin" and spin["length_m"] is None and spin["speeds"] is None
    assert spin["turn_rad"] == pytest.approx(2 * math.pi, abs=1e-4)
    assert spin["wz"] < 0 and spin["baseline"] == "spin_left"
    json.dumps(path, allow_nan=False)
    json.dumps(spin, allow_nan=False)


# -- budget --------------------------------------------------------------------


def test_roboto_straight_10m_budget_is_2600_steps():
    # 10 m at 0.5 m/s is 20 s ideal; 2.5 x 20 + 2 = 52 s = 2600 steps.
    assert budget_steps(CAT["roboto_origin"]["straight_10m"], _DT) == 2600


@pytest.mark.parametrize("robot", ROBOTS)
def test_every_row_has_its_documented_length_ideal_time_and_budget_under_the_ceiling(robot):
    for name, c in CAT[robot].items():
        length, ideal, budget, _, _ = TABLE[robot][name]
        if length is not None:
            assert c.length_m == pytest.approx(length, abs=5e-4), name
        assert c.ideal_sec == pytest.approx(ideal, abs=5e-3), name
        assert budget_steps(c, _DT) == budget, name
        assert spec.TIME_FACTOR * c.ideal_sec + spec.SLACK_SEC < spec.MAX_COURSE_SEC, name


def test_budget_never_exceeds_the_ceiling():
    # The ceiling binds only for rows outside the flat catalogue. 100 m at
    # 0.5 m/s is 200 s ideal, 502 s before the clamp.
    long = PathCourse("long", "test", "", line(100.0), (0.5,))
    assert spec.TIME_FACTOR * long.ideal_sec + spec.SLACK_SEC > spec.MAX_COURSE_SEC
    assert spec.budget_sec(long) == spec.MAX_COURSE_SEC
    assert budget_steps(long, _DT) == round(spec.MAX_COURSE_SEC / _DT)


def test_spin_budget_scales_with_rate_and_ignores_direction():
    spins = _spins("roboto_origin")
    left = budget_steps(spins["spin_left"], _DT)
    assert left == round((2.5 * 2 * math.pi / ROBOTO.spin_nom + 2.0) / _DT)
    assert budget_steps(spins["spin_right"], _DT) == left
    assert budget_steps(spins["spin_slow"], _DT) > left > budget_steps(spins["spin_fast"], _DT)


# -- follower ------------------------------------------------------------------

_CAP = ROBOTO.yaw_cap


def _straight(speed=0.5, length=10.0):
    course = PathCourse("probe", "test", "", line(length), (speed,))
    return course, follower.pack_path(course, follower.padded_points([course]))


def _step(path, x, y, yaw, state=None, cap=_CAP):
    state = follower.initial_state() if state is None else state
    return follower.follower_step(path, state, x, y, yaw, cap)


def _walk(path, upto_s, stride=follower.WINDOW_PTS // 2):
    """Advance the progress index along the path up to arclength `upto_s`,
    `stride` points per step. The default is half the progress window. A pose
    teleported to the goal is not representative: the window caps each move
    at WINDOW_PTS points."""
    pts, cum = np.asarray(path.pts), np.asarray(path.cum)
    state = follower.initial_state()
    for k in range(0, int(np.searchsorted(cum, upto_s)), stride):
        state = _step(path, float(pts[k][0]), float(pts[k][1]), 0.0, state).state
    return state


def test_padding_keeps_the_window_and_lookahead_inside_the_array():
    course, path = _straight()
    dense = follower.dense_points(course)
    assert path.pts.shape == (dense + follower.PAD_PTS, 2)
    assert follower.PAD_PTS == follower.WINDOW_PTS + follower.LOOK_PTS + 1
    # Past the end, argmin picks the real last point, not a padded copy.
    state = _walk(path, 10.0)
    for _ in range(3):
        state = _step(path, 10.5, 0.0, 0.0, state).state
    assert int(state.i) == dense - 1
    assert int(state.i) + follower.WINDOW_PTS + 1 + follower.LOOK_PTS <= path.pts.shape[0]
    with pytest.raises(ValueError, match="probe"):
        follower.pack_path(course, path.pts.shape[0] - 1)


def test_on_path_and_on_heading_commands_straight_ahead():
    _, path = _straight()
    fo = _step(path, 1.0, 0.0, 0.0)
    assert float(fo.cmd[0]) == pytest.approx(0.5, abs=1e-6)
    assert float(fo.cmd[1]) == 0.0  # never strafes
    assert float(fo.cmd[2]) == pytest.approx(0.0, abs=1e-6)
    assert float(fo.xte) == pytest.approx(0.0, abs=1e-7)
    assert float(fo.s) == pytest.approx(1.0, abs=0.02)
    assert not bool(fo.reached)


def test_cross_track_error_is_positive_to_the_left():
    _, path = _straight()
    assert float(_step(path, 1.0, 0.3, 0.0).xte) == pytest.approx(0.3, abs=1e-6)
    assert float(_step(path, 1.0, -0.3, 0.0).xte) == pytest.approx(-0.3, abs=1e-6)


def test_an_offset_left_of_the_path_steers_right():
    _, path = _straight()
    assert float(_step(path, 1.0, 0.2, 0.0).cmd[2]) < 0.0


def test_heading_error_past_60_deg_spins_in_place_toward_the_path():
    _, path = _straight()
    fo = _step(path, 1.0, 0.0, math.pi / 2)
    assert bool(fo.state.spinning)
    assert float(fo.cmd[0]) == 0.0 and float(fo.cmd[1]) == 0.0
    # Facing left of the path, it turns right; facing right, left. Both at
    # the cap: K_YAW_SPIN pi/2 is above it.
    assert float(fo.cmd[2]) == pytest.approx(-_CAP, abs=1e-5)
    assert float(_step(path, 1.0, 0.0, -math.pi / 2).cmd[2]) == pytest.approx(_CAP, abs=1e-5)
    # With the clip out of the way, the spin command is K_YAW_SPIN alpha.
    wide = _step(path, 1.0, 0.0, math.pi / 2, cap=10.0)
    assert float(wide.cmd[2]) == pytest.approx(follower.K_YAW_SPIN * -math.pi / 2, abs=1e-5)


def test_the_spin_branch_turns_a_unicycle_onto_a_path_behind_its_heading():
    # The path heads +y and the unicycle starts heading +x, 90 deg off. A
    # spin of the wrong sign would settle facing away and time out.
    course = PathCourse("sideways", "test", "", line(5.0, heading=math.pi / 2), (0.5,))
    u = follower.perfect_unicycle(follower.pack_path(course, follower.padded_points([course])),
                                  _CAP, budget_steps(course, _DT), _DT)
    assert bool(u["completed"]) and int(u["spin_steps"]) > 0


@pytest.mark.parametrize("robot", ROBOTS)
def test_the_yaw_command_never_exceeds_the_cap(robot):
    cap = PARAMS[robot].yaw_cap
    course = PathCourse("tight", "test", "", arc(0.4, math.pi), (1.0,))
    path = follower.pack_path(course, follower.padded_points([course]))
    for yaw in np.linspace(-math.pi, math.pi, 25):
        for spinning in (False, True):
            state = follower.FollowerState(i=follower.initial_state().i,
                                           spinning=np.bool_(spinning))
            fo = _step(path, 0.1, 0.0, float(yaw), state, cap=cap)
            assert abs(float(fo.cmd[2])) <= cap + 1e-5


def test_heading_error_slows_the_robot():
    _, path = _straight()
    straight = float(_step(path, 1.0, 0.0, 0.0).cmd[0])
    skewed = float(_step(path, 1.0, 0.0, 0.5).cmd[0])
    assert 0.0 < skewed < straight


def test_reaching_the_final_point_completes():
    _, path = _straight()
    x = 10.0 - follower.GOAL_RADIUS_M / 2
    state = _walk(path, x)
    assert bool(_step(path, x, 0.0, 0.0, state).reached)


def test_just_outside_the_goal_radius_does_not_complete():
    _, path = _straight()
    x = 10.0 - follower.GOAL_RADIUS_M * 1.5
    state = _walk(path, x)
    assert not bool(_step(path, x, 0.0, 0.0, state).reached)


@pytest.mark.parametrize("name", ["circle_r2", "circle_tight", "figure_eight_r15", "square_3m"])
def test_a_closed_course_does_not_complete_at_its_start(name):
    course = CAT["roboto_origin"][name]
    path = follower.pack_path(course, follower.padded_points([course]))
    end = np.asarray(path.pts)[-1]
    assert np.linalg.norm(end - course.waypoints[0]) < 1.0 + ROBOTO.r_tight + 1e-6
    assert not bool(_step(path, float(end[0]), float(end[1]), 0.0).reached)


def test_a_closed_course_completes_after_its_loop():
    course = CAT["roboto_origin"]["circle_r2"]
    path = follower.pack_path(course, follower.padded_points([course]))
    state = _walk(path, course.length_m + 1.0)
    end = np.asarray(path.pts)[-1]
    assert bool(_step(path, float(end[0]), float(end[1]), 0.0, state).reached)


def test_progress_is_monotone_through_the_figure_eight_crossing():
    course = CAT["roboto_origin"]["figure_eight_r15"]
    path = follower.pack_path(course, follower.padded_points([course]))
    pts = np.asarray(path.pts)
    state, seen = follower.initial_state(), []
    for k in range(0, follower.dense_points(course), 5):
        fo = _step(path, float(pts[k][0]), float(pts[k][1]), 0.0, state)
        state = fo.state
        seen.append(float(fo.s))
    assert all(b >= a for a, b in itertools.pairwise(seen))
    # The crossing at (1, 0) is passed after one loop; progress stays past it.
    assert seen[-1] == pytest.approx(course.length_m, abs=0.11)


def test_layout_places_the_course_in_the_anchor_frame():
    _, path = _straight(length=3.0)
    world = follower.layout(path, 5.0, -2.0, math.pi / 2)
    dense = follower.dense_points(PathCourse("p", "t", "", line(3.0), (0.5,)))
    pts = np.asarray(world.pts)
    assert np.allclose(pts[0], [5.0, -2.0], atol=1e-6)
    assert np.allclose(pts[dense - 1], [5.0, 1.0], atol=1e-5)
    assert np.allclose(np.asarray(world.tan)[0], [0.0, 1.0], atol=1e-6)
    fo = _step(world, 5.0, -1.0, math.pi / 2)
    assert float(fo.cmd[0]) == pytest.approx(0.5, abs=1e-5)
    assert float(fo.xte) == pytest.approx(0.0, abs=1e-5)
    assert float(fo.s) == pytest.approx(1.0, abs=0.02)


def test_the_lookahead_wz_is_exact_at_a_45_deg_geometry():
    # On the start point's normal, LOOKAHEAD_M to the left: the lookahead
    # point is 45 deg to the right of a robot heading +x.
    _, path = _straight()
    fo = _step(path, 0.0, follower.LOOKAHEAD_M, 0.0)
    alpha = -math.pi / 4
    vx = 0.5 * math.cos(alpha)
    want = float(np.clip(2.0 * vx * math.sin(alpha) / follower.LOOKAHEAD_M, -_CAP, _CAP))
    assert float(fo.cmd[0]) == pytest.approx(vx, abs=1e-6)
    assert float(fo.cmd[2]) == pytest.approx(want, abs=1e-6)


def test_spin_hysteresis_cannot_chatter():
    _, path = _straight()
    fo = _step(path, 1.0, 0.0, math.pi / 2)
    assert bool(fo.state.spinning) and float(fo.cmd[0]) == 0.0
    # 40 deg sits between the exit (20) and enter (60) angles: still spinning.
    # K_YAW_SPIN x 40 deg is under Roboto's cap, so the command is unclipped.
    fo = _step(path, 1.0, 0.0, math.radians(40.0), fo.state)
    assert bool(fo.state.spinning) and float(fo.cmd[0]) == 0.0
    assert float(fo.cmd[2]) == pytest.approx(follower.K_YAW_SPIN * -math.radians(40.0), abs=1e-5)
    fo = _step(path, 1.0, 0.0, math.radians(10.0), fo.state)
    assert not bool(fo.state.spinning) and float(fo.cmd[0]) > 0.0


def test_walking_does_not_spin_below_the_enter_angle():
    _, path = _straight()
    fo = _step(path, 1.0, 0.0, math.radians(40.0))
    assert not bool(fo.state.spinning) and float(fo.cmd[0]) > 0.0


def test_the_follower_constants_are_frozen():
    assert follower.LOOKAHEAD_M == 0.40
    assert follower.SPIN_ENTER_RAD == pytest.approx(math.radians(60.0), abs=0.01)
    assert follower.SPIN_EXIT_RAD == pytest.approx(math.radians(20.0), abs=0.01)
    assert follower.K_YAW_SPIN == 1.5
    assert follower.GOAL_RADIUS_M == 0.25
    assert follower.GOAL_MIN_PROGRESS_M == 0.80
    assert follower.RESAMPLE_DS_M == 0.02
    assert follower.PROGRESS_WINDOW_M == 1.0
    assert (follower.WINDOW_PTS, follower.LOOK_PTS) == (50, 20)
    assert LEAD_IN_M == 1.0


def test_quat_to_yaw_and_wrap_angle():
    for yaw in (-3.0, -1.0, 0.0, 0.5, 3.0):
        q = np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])
        assert float(follower.quat_to_yaw(q)) == pytest.approx(yaw, abs=1e-6)
    assert float(follower.wrap_angle(3 * math.pi / 2)) == pytest.approx(-math.pi / 2, abs=1e-6)
    assert float(follower.wrap_angle(-3 * math.pi / 2)) == pytest.approx(math.pi / 2, abs=1e-6)


# -- perfect unicycle ----------------------------------------------------------


@pytest.fixture(scope="module")
def unicycle():
    out = {}
    for robot in ROBOTS:
        p, cat = PARAMS[robot], CAT[robot]
        n_points = follower.padded_points(cat.values())
        for name, c in _paths(robot).items():
            u = follower.perfect_unicycle(follower.pack_path(c, n_points), p.yaw_cap,
                                          budget_steps(c, _DT), _DT)
            out[robot, name] = {k: v.item() for k, v in u.items()}
            out[robot, name]["entry"] = perfect_unicycle_entry(c, p, _DT, n_points)
    return out


def test_the_perfect_unicycle_completes_every_path_row(unicycle):
    for (robot, name), u in unicycle.items():
        assert u["completed"] and u["entry"]["completed"], (robot, name)
        # Pure pursuit never needs the spin branch or the full cap here; it
        # engages only when a real robot lags.
        assert u["spin_steps"] == 0, (robot, name)
        assert u["peak_wz"] < PARAMS[robot].yaw_cap, (robot, name)


def test_the_perfect_unicycle_matches_the_float64_numbers(unicycle):
    for (robot, name), u in unicycle.items():
        _, ideal, _, xte, steps = TABLE[robot][name]
        assert u["xte_rms_m"] == pytest.approx(xte, abs=1e-4), (robot, name)
        assert abs(u["steps"] - steps) <= 1, (robot, name)
        assert u["entry"]["t_over_ideal"] == pytest.approx(steps * _DT / ideal, abs=2e-3)
        tracking = min(spec.SUBSCORE_CAP, PARAMS[robot].stance_halfwidth_m / max(xte, 1e-9))
        assert u["entry"]["tracking"] == pytest.approx(tracking, rel=0.01), (robot, name)


def test_every_path_row_leaves_the_follower_error_4x_below_the_normalizer(unicycle):
    for (robot, name), u in unicycle.items():
        assert u["entry"]["tracking"] >= spec.MIN_PERFECT_TRACKING, (robot, name)


@pytest.mark.parametrize("vx_max, wz_max", [(1.0, 0.6), (0.8, 0.45)])
def test_a_wide_tight_radius_grows_the_square_and_keeps_its_corners(vx_max, wz_max):
    # A measured robot whose r_tight passes 1.25 m. The corners keep r_tight,
    # so they ask for R_TIGHT_DEMAND of the cap, and the side grows to leave
    # SQUARE_MIN_EDGE_M of straight per edge.
    inputs = spec.RobotInputs(vx_max, wz_max, 0.1, 0.7, 0.4, {"gyro": 0.01})
    p = spec.derive("weak", inputs, source="measured")
    sq = catalogue(p)["square_3m"]
    side, r = sq.geometry["side_m"], sq.geometry["corner_radius_m"]
    assert r == p.r_tight > 1.5
    assert side == pytest.approx(2 * r + geometry_paths.SQUARE_MIN_EDGE_M) and side > 3.0
    heading = np.unwrap(np.arctan2(np.diff(sq.waypoints[:, 1]), np.diff(sq.waypoints[:, 0])))
    assert np.abs(np.diff(heading)).max() < (math.pi / 2) / 23 + 1e-9
    exact = 4.0 * side - 7.0 * r + 2.0 * math.pi * r
    fine = rounded_square(side, r, n=4001)
    assert np.linalg.norm(np.diff(fine, axis=0), axis=1).sum() == pytest.approx(exact, abs=1e-6)
    assert sq.length_m == pytest.approx(LEAD_IN_M + exact, abs=5e-3)
    entry = perfect_unicycle_entry(sq, p, _DT, follower.padded_points([sq]))
    assert entry["completed"] and entry["tracking"] >= spec.MIN_PERFECT_TRACKING
    # Both pinned robots keep the 3 m side, Asimov's r_tight of 1.25 m plus
    # one ulp included.
    for robot in ROBOTS:
        assert CAT[robot]["square_3m"].geometry["side_m"] == geometry_paths.SQUARE_SIDE_M == 3.0


# -- model ids -----------------------------------------------------------------


def test_friction_geoms_are_the_colliding_world_geoms_and_every_foot():
    # geom 0: the floor; 1: a world geom with collisions off; 2-3: world
    # geoms that collide one way each; 4-9: body geoms, the feet among them.
    bodyid = np.array([0, 0, 0, 0, 1, 2, 3, 3, 4, 4])
    contype = np.array([1, 0, 1, 0, 1, 1, 1, 1, 1, 1])
    conaffinity = np.array([1, 0, 0, 2, 1, 1, 1, 1, 1, 1])
    ids = friction_geom_ids(bodyid, contype, conaffinity, np.array([9, 6, 7, 6]))
    assert ids.tolist() == [0, 2, 3, 6, 7, 9]
    assert ids.dtype == np.int32
    assert friction_geom_ids(bodyid, contype, conaffinity, 3).tolist() == [0, 2, 3]


# -- scoring -------------------------------------------------------------------

_CAT_R = CAT["roboto_origin"]
_STRAIGHT = _CAT_R["straight_10m"]


def _rec(n=600, xte=0.0, v_err=0.0, h_err=0.0, slip=0.0, cmd_v=0.5):
    """A synthetic path record at hand-set error levels. qvel is a 1 Hz
    sine, so the vibration fraction is ~0 and smoothness sits at the cap.
    Foot 0 is planted throughout and slides at `slip` m/s. Foot 1 is
    airborne throughout at 1 m/s; an airborne foot's speed is not slip."""
    t = np.arange(n) * _DT
    return {
        "x": cmd_v * t, "y": np.zeros(n), "yaw": np.zeros(n),
        "s": cmd_v * t, "xte": np.full(n, xte), "spinning": np.zeros(n, bool),
        "cmd": np.stack([np.full(n, cmd_v), np.zeros(n), np.zeros(n)], -1),
        "v_fwd": np.full(n, cmd_v - v_err), "v_planar": np.full(n, cmd_v),
        "h": np.full(n, ROBOTO.nominal_height_m + h_err), "gyro_z": np.zeros(n),
        "qvel_act": np.sin(2 * np.pi * 1.0 * t)[:, None] * np.ones((1, 12)),
        "foot_speed": np.stack([np.full(n, slip), np.full(n, 1.0)], -1),
        "contact": np.stack([np.ones(n, bool), np.zeros(n, bool)], -1),
        "foot_clear": np.zeros((n, 2)), "foot_vz": np.zeros((n, 2)),
    }


def _out(n=600, outcome="completed", fell_at=None, seed=0):
    return {"seed": seed, "outcome": outcome, "steps": n, "fell_at": fell_at,
            "settle_height_m": 0.7465}


def _path(rec=None, out=None, course=_STRAIGHT):
    rec = _rec() if rec is None else rec
    out = _out(len(rec["s"])) if out is None else out
    return path_seed_result(rec, out, _DT, course, ROBOTO)


def test_tracking_is_stance_halfwidth_over_rms_cross_track():
    assert _path(_rec(xte=ROBOTO.stance_halfwidth_m))["subscores"]["tracking"] == 1.0
    tenth = _path(_rec(xte=ROBOTO.stance_halfwidth_m / 10))["subscores"]["tracking"]
    assert tenth == pytest.approx(10.0, rel=1e-3)


def test_height_is_nominal_height_over_rms_height_error():
    r = _path(_rec(h_err=ROBOTO.nominal_height_m))
    assert r["subscores"]["height"] == 1.0
    assert r["raw"]["height_err_rms_m"] == pytest.approx(ROBOTO.nominal_height_m, abs=1e-4)


def test_speed_is_commanded_speed_over_rms_speed_error():
    assert _path(_rec(cmd_v=0.5, v_err=0.5))["subscores"]["speed"] == 1.0
    assert _path(_rec(cmd_v=0.5, v_err=0.05))["subscores"]["speed"] == pytest.approx(10.0)


def test_tracking_reduces_a_varying_cross_track_error_by_rms():
    # A ramp to the stance half-width a that alternates sides every step.
    # Its RMS is about a / sqrt(3), its mean |x| about a / 2, its p95 about
    # 0.95 a and its max a, so only the RMS reducer gives tracking sqrt(3).
    # The alternating sign rules out a signed mean.
    a = ROBOTO.stance_halfwidth_m
    k = np.arange(600)
    rec = _rec()
    rec["xte"] = a * (k + 1) / 600 * np.where(k % 2, 1.0, -1.0)
    r = _path(rec)
    assert r["raw"]["xte_rms_m"] == pytest.approx(a / math.sqrt(3), rel=2e-3)
    assert r["subscores"]["tracking"] == pytest.approx(math.sqrt(3), rel=2e-3)
    assert r["raw"]["xte_p95_m"] == pytest.approx(0.95 * a, rel=1e-3)
    # A one-sided drift: the median |x| reads 0.05 and the signed p95 -0.005.
    rec["xte"] = -np.linspace(0.0, 0.1, 600)
    p95 = float(np.percentile(np.abs(rec["xte"]), 95))
    assert p95 == pytest.approx(0.095, abs=1e-6)
    assert _path(rec)["raw"]["xte_p95_m"] == pytest.approx(p95, abs=1e-4)


def test_speed_references_the_mean_command_over_the_moving_steps():
    # A pivot at vx 0, then 0.5 and 0.9 m/s, delivered 0.1 m/s short. The
    # mean moving command is 0.7; a max would read 0.9 and score 9.0.
    rec = _rec()
    cmd = np.concatenate([np.zeros(200), np.full(200, 0.5), np.full(200, 0.9)])
    rec["cmd"][:, 0] = cmd
    rec["v_fwd"] = np.where(cmd > 0.0, cmd - 0.1, 0.0)
    r = _path(rec)
    assert r["raw"]["speed_cmd_mean"] == pytest.approx(0.7)
    assert r["raw"]["speed_err_rms"] == pytest.approx(0.1)
    assert r["subscores"]["speed"] == pytest.approx(7.0)
    assert r["raw"]["speed_ratio"] == pytest.approx(0.6 / 0.7, abs=1e-3)


def test_grip_is_one_when_the_feet_slide_as_far_as_the_body_moves():
    r = _path(_rec(cmd_v=0.5, slip=0.5))
    assert r["subscores"]["grip"] == 1.0
    assert r["raw"]["slip_distance_m"] == r["raw"]["base_distance_m"] == pytest.approx(6.0)


def test_slip_counts_each_foot_only_on_the_steps_it_is_in_contact():
    # The feet alternate stance step by step. The stance foot slides at
    # 0.25 m/s and the swing foot moves at 1 m/s. A mask per step (any foot
    # down), foot 0's contact applied to both feet, or no mask at all would
    # read 15.0, 7.5 or 15.0 m of slip.
    rec = _rec()
    even = np.arange(600) % 2 == 0
    rec["contact"] = np.stack([even, ~even], -1)
    rec["foot_speed"] = np.where(rec["contact"], 0.25, 1.0)
    r = _path(rec)
    assert r["raw"]["base_distance_m"] == pytest.approx(6.0)
    assert r["raw"]["slip_distance_m"] == pytest.approx(3.0)
    assert r["subscores"]["grip"] == 2.0


def test_smoothness_is_one_over_the_battery_vibration_index():
    rec = _rec()
    t = np.arange(600) * _DT
    rec["qvel_act"] = np.stack([np.sin(2 * np.pi * 1.0 * t), np.sin(2 * np.pi * 8.0 * t)], -1)
    r = _path(rec)
    vib = battery.vibration_index(rec["qvel_act"], _DT)
    assert r["raw"]["vibration"] == pytest.approx(vib, abs=1e-4)
    assert r["subscores"]["smoothness"] == _ratio(1.0, vib)
    cutoff = inspect.signature(battery.vibration_index).parameters["cutoff_hz"].default
    assert spec.VIBRATION_CUTOFF_HZ == cutoff


def test_subscores_are_capped_only_for_json():
    r = _path()
    assert set(r["subscores"].values()) == {spec.SUBSCORE_CAP}
    assert r["score"] == spec.SUBSCORE_CAP
    json.dumps(r, allow_nan=False)


def test_the_score_is_the_weakest_axis():
    r = _path(_rec(xte=ROBOTO.stance_halfwidth_m / 2, v_err=0.1))
    assert r["binding"] == "tracking"
    assert r["score"] == r["subscores"]["tracking"] == 2.0
    assert r["score"] < min(v for k, v in r["subscores"].items() if k != "tracking")


def test_an_unscored_axis_is_null_and_out_of_the_min():
    course = PathCourse("lumpy", "test", "", line(10.0), (0.5,), unscored={"tracking"})
    r = _path(_rec(xte=ROBOTO.stance_halfwidth_m, v_err=0.1), course=course)
    assert r["subscores"]["tracking"] is None
    assert r["binding"] == "speed"
    assert r["score"] == r["subscores"]["speed"] == 5.0
    agg = aggregate([r, r])
    assert agg["subscore_median"]["tracking"] is None and agg["binding"] == "speed"


def test_a_fall_scores_zero_and_keeps_its_subscores():
    # The record runs past the lane's last step; nothing after it is read.
    rec = _rec(n=600)
    rec["xte"][300:] = 1.0
    r = _path(rec, _out(n=300, outcome="fell", fell_at=299))
    assert r["score"] == 0.0 and r["fell_at"] == 299 and not r["completed"]
    assert r["steps"] == 300
    assert r["progress_m"] == pytest.approx(rec["s"][299], abs=1e-3)
    assert r["subscores"]["tracking"] == spec.SUBSCORE_CAP
    assert r["raw"]["xte_rms_m"] == 0.0


def test_a_timeout_scores_zero():
    r = _path(_rec(n=400), _out(n=400, outcome="timed_out"))
    assert r["score"] == 0.0 and r["fell_at"] is None and r["subscores"] is not None


def test_a_nan_error_scores_zero_never_the_cap():
    # Python alone would score it perfect: max(nan, 1e-9) is nan and
    # min(1000.0, nan) is 1000.0.
    assert min(spec.SUBSCORE_CAP, 1.0 / max(float("nan"), 1e-9)) == spec.SUBSCORE_CAP
    assert _ratio(1.0, float("nan")) == 0.0
    assert _ratio(1.0, float("inf")) == 0.0
    assert _ratio(float("nan"), 1.0) == 0.0
    r = _path(_rec(xte=float("nan")))
    assert r["subscores"]["tracking"] == 0.0 and r["score"] == 0.0


def test_a_nonfinite_lane_scores_zero_with_null_subscores_and_leaves_the_median():
    good = [_path(_rec(xte=x), _out(seed=k)) for k, x in enumerate((0.01, 0.02, 0.04))]
    broken = _path(_rec(n=500, xte=float("nan")), _out(n=500, outcome="nonfinite", seed=3))
    assert broken["score"] == 0.0 and broken["subscores"] is None and broken["raw"] is None
    assert aggregate(good + [broken])["subscore_median"] == aggregate(good)["subscore_median"]


def test_a_short_record_has_no_subscores():
    r = _path(_rec(n=49), _out(n=49, outcome="fell", fell_at=48))
    assert r["subscores"] is None and r["score"] == 0.0
    assert _path(_rec(n=50), _out(n=50))["subscores"] is not None


def test_a_settle_fall_is_fell_at_zero_and_zero_steps():
    empty = {k: v[:0] for k, v in _rec().items()}
    r = _path(empty, _out(n=0, outcome="settle_fell", fell_at=0))
    assert (r["outcome"], r["fell_at"], r["steps"], r["progress_m"]) == ("settle_fell", 0, 0, 0.0)
    assert r["score"] == 0.0 and r["subscores"] is None
    assert scoring.outcome(nonfinite=False, settle_fell=True, fell=True, reached=False) == \
        "settle_fell"


def test_outcomes_are_decided_in_order():
    def decide(**flags):
        base = {"nonfinite": False, "settle_fell": False, "fell": False, "reached": False}
        return scoring.outcome(**{**base, **flags})

    assert decide(nonfinite=True, settle_fell=True, fell=True, reached=True) == "nonfinite"
    assert decide(fell=True, reached=True) == "fell"
    assert decide(reached=True) == "completed"
    assert decide() == "timed_out"
    assert scoring.OUTCOMES == ("nonfinite", "settle_fell", "fell", "completed", "timed_out")


def test_a_lanes_stop_flags_become_its_seed_entry():
    def seed(steps, **flags):
        base = {"nonfinite": False, "settle_fell": False, "fell": False, "reached": False}
        # numpy scalars, as a lane's outputs read on the host.
        out = SimpleNamespace(
            steps=np.int32(steps), settle_height=np.float32(0.75), yaw_rad=np.float32(0.5),
            **{k: np.bool_(v) for k, v in {**base, **flags}.items()},
        )
        return scoring.seed_out(out, 3)

    def stop(steps, **flags):
        s = seed(steps, **flags)
        return s["outcome"], s["steps"], s["fell_at"]

    # A course fall stopped on the step that tripped `done`, the last counted one.
    assert stop(25, fell=True) == ("fell", 25, 24)
    assert stop(0, settle_fell=True, fell=True) == ("settle_fell", 0, 0)
    assert stop(0, nonfinite=True) == ("nonfinite", 0, None)
    assert stop(7, nonfinite=True, fell=True) == ("nonfinite", 7, None)
    assert stop(40, reached=True) == ("completed", 40, None)
    assert stop(100) == ("timed_out", 100, None)
    # Plain Python values, ready for the JSON writer.
    assert seed(100) == {"seed": 3, "outcome": "timed_out", "steps": 100, "fell_at": None,
                         "settle_height_m": 0.75, "yaw_rad": 0.5}
    assert {type(v) for v in seed(100).values()} == {int, str, type(None), float}


def test_speed_counts_only_where_motion_was_commanded():
    rec = _rec()
    rec["cmd"][:300, 0] = 0.0  # a pivot: the spin branch commands vx = 0
    rec["v_fwd"] = np.zeros(600)  # never moves forward
    r = _path(rec)
    assert r["subscores"]["speed"] == 1.0
    assert r["raw"]["speed_cmd_mean"] == 0.5
    # Under 0.2 s of commanded motion the axis has nothing to measure.
    rec["cmd"][:, 0] = 0.0
    rec["cmd"][:9, 0] = 0.5
    r = _path(rec)
    assert r["subscores"]["speed"] == spec.SUBSCORE_CAP
    assert r["raw"]["speed_err_rms"] is None and r["raw"]["speed_ratio"] is None


def test_speed_ratio_is_delivered_over_commanded():
    r = _path(_rec(cmd_v=0.9, v_err=0.2))
    assert r["raw"]["speed_ratio"] == pytest.approx(0.7 / 0.9, abs=1e-3)
    assert r["subscores"]["speed"] == pytest.approx(4.5)
    assert r["raw"]["t_over_ideal"] == pytest.approx(600 * _DT / _STRAIGHT.ideal_sec)


def test_a_20_hz_buzz_binds_smoothness():
    rec = _rec(xte=0.01)
    t = np.arange(600) * _DT
    rec["qvel_act"] = np.sin(2 * np.pi * 20.0 * t)[:, None] * np.ones((1, 12))
    r = _path(rec)
    assert r["binding"] == "smoothness" and r["subscores"]["smoothness"] < 2.0


def _spin_rec(n=600, wz_cmd=1.0, wz_err=0.0, drift=0.0, h_err=0.0):
    """A synthetic spin record. The base starts at a settled pose off the
    world origin, moves `drift` m diagonally out and comes back, so only the
    largest distance from the first pose reads `drift`."""
    t = np.arange(n) * _DT
    bump = np.concatenate([np.linspace(0.0, 1.0, n // 2), np.linspace(1.0, 0.0, n - n // 2)])
    return {
        "x": 0.3 + drift / math.sqrt(2) * bump,
        "y": -0.2 + drift / math.sqrt(2) * bump,
        "yaw": np.zeros(n),
        "s": np.zeros(n), "xte": np.zeros(n), "spinning": np.zeros(n, bool),
        "cmd": np.tile([0.0, 0.0, wz_cmd], (n, 1)),
        "gyro_z": np.full(n, wz_cmd - math.copysign(wz_err, wz_cmd)),
        "h": np.full(n, ROBOTO.nominal_height_m + h_err),
        "qvel_act": np.sin(2 * np.pi * 1.0 * t)[:, None] * np.ones((1, 12)),
    }


def _spin(course, rec, n=600, outcome="completed", yaw_rad=2 * math.pi):
    out = {"seed": 0, "outcome": outcome, "steps": n, "fell_at": None,
           "settle_height_m": 0.7465, "yaw_rad": yaw_rad}
    return spin_seed_result(rec, out, _DT, course, ROBOTO)


def test_spin_axes_score_a_right_spin_by_magnitude():
    left, right = _CAT_R["spin_left"], _CAT_R["spin_right"]
    r = _spin(left, _spin_rec(wz_cmd=left.wz, wz_err=left.wz))
    assert set(r["subscores"]) == set(SPIN_AXES)
    assert r["subscores"]["rotation"] == 1.0
    # A completed right spin made +2 pi of progress in its own direction.
    r = _spin(right, _spin_rec(wz_cmd=right.wz, wz_err=abs(right.wz) / 10), yaw_rad=2 * math.pi)
    assert r["subscores"]["rotation"] == pytest.approx(10.0)
    assert r["score"] > 0 and r["progress_rad"] == pytest.approx(2 * math.pi, abs=1e-3)
    assert "progress_m" not in r and r["gait"] is None
    r = _spin(left, _spin_rec(wz_cmd=left.wz, drift=ROBOTO.stance_halfwidth_m))
    assert r["subscores"]["drift"] == 1.0 and r["binding"] == "drift"
    assert r["raw"]["drift_max_m"] == pytest.approx(ROBOTO.stance_halfwidth_m)
    r = _spin(left, _spin_rec(wz_cmd=left.wz, h_err=ROBOTO.nominal_height_m))
    assert r["subscores"]["height"] == 1.0 and r["binding"] == "height"
    assert r["raw"]["height_err_rms_m"] == pytest.approx(ROBOTO.nominal_height_m, abs=1e-4)
    r = _spin(left, _spin_rec(wz_cmd=left.wz, wz_err=left.wz), outcome="timed_out", yaw_rad=0.1)
    assert r["score"] == 0.0 and r["subscores"]["rotation"] == 1.0


def test_spin_yaw_progress_uses_the_battery_reducer():
    course = _CAT_R["spin_left"]
    rec = _spin_rec(wz_cmd=course.wz, wz_err=0.1)
    r = _spin(course, rec)
    want = battery.yaw_progress_deg(rec["gyro_z"], _DT, settle=0)
    assert r["raw"]["yaw_progress_deg"] == pytest.approx(want, abs=0.01)
    assert r["raw"]["yaw_rad"] == pytest.approx(2 * math.pi, abs=1e-3)


def test_gait_kpis_are_reported_not_scored():
    rec = _rec(xte=0.02)
    swing = np.concatenate([np.zeros(10), np.linspace(0.0, 0.05, 10), np.linspace(0.05, 0.0, 10)])
    # Each foot lifts off right after sample 0 and swings 20 times.
    rec["foot_clear"] = np.tile(np.roll(np.resize(swing, 600), -10)[:, None], (1, 2))
    rec["foot_vz"] = np.gradient(rec["foot_clear"], axis=0) / _DT
    with_swings = _path(rec)
    flat = _path(_rec(xte=0.02))
    # The record starts after the settle, so the gait KPIs drop nothing.
    # Dropping even one step would start each foot mid-swing and lose that
    # swing.
    assert with_swings["gait"] == gait_metrics(rec, settle_steps=0)
    assert with_swings["gait"]["swings"] == 40
    assert gait_metrics(rec, settle_steps=1)["swings"] == 38
    assert flat["gait"]["swings"] == 0 and flat["gait"]["swing_apex_med_m"] is None
    assert set(with_swings["subscores"]) == set(PATH_AXES)
    assert with_swings["score"] == flat["score"]


def test_vs_baseline_carries_the_median_difference_and_the_slip_ratio():
    dry = aggregate([_path(_rec(xte=0.02, slip=0.01), _out(seed=k)) for k in range(3)])
    wet = aggregate([_path(_rec(xte=0.04, slip=0.03), _out(seed=k)) for k in range(3)])
    push = aggregate([_path(_rec(xte=0.05), _out(seed=k)) for k in range(3)])
    rows = {"straight_10m": dry, "straight_slippery": wet, "straight_push": push,
            "straight_push_fast": push}
    s = summary(rows, _CAT_R)
    assert rows["straight_10m"]["vs_baseline"] is None
    vs = rows["straight_slippery"]["vs_baseline"]
    assert vs["baseline"] == "straight_10m"
    assert vs["d_score_median"] == pytest.approx(wet["score_median"] - dry["score_median"])
    assert vs["slip_ratio"] == pytest.approx(3.0)
    assert rows["straight_push"]["vs_baseline"]["slip_ratio"] is None
    # straight_fast was not run, so there is nothing to difference against.
    assert rows["straight_push_fast"]["vs_baseline"] == {
        "baseline": "straight_fast", "d_score_median": None, "slip_ratio": None}
    assert s == {"lanes": 12, "lanes_completed": 12, "lanes_fell": 0, "lanes_timed_out": 0,
                 "lanes_nonfinite": 0, "rows_all_completed": 4}


def test_summary_counts_lanes_by_outcome_across_rows():
    # Every counter gets a different value, so a swapped counter fails. A
    # settle fall counts as a fall. Three rows have a completed lane and only
    # one completed every lane.
    lane = {
        "completed": _path(_rec(xte=0.02)),
        "fell": _path(_rec(n=300), _out(n=300, outcome="fell", fell_at=299)),
        "settle_fell": _path({k: v[:0] for k, v in _rec().items()},
                             _out(n=0, outcome="settle_fell", fell_at=0)),
        "timed_out": _path(_rec(n=400), _out(n=400, outcome="timed_out")),
        "nonfinite": _path(_rec(n=100), _out(n=100, outcome="nonfinite")),
    }
    outcomes = {
        "straight_10m": ["completed", "fell", "settle_fell", "fell"],
        "straight_slippery": ["completed", "timed_out", "timed_out", "nonfinite", "completed"],
        "straight_push": ["completed"] * 3,
    }
    rows = {name: aggregate([lane[o] for o in seeds]) for name, seeds in outcomes.items()}
    assert summary(rows, _CAT_R) == {
        "lanes": 12, "lanes_completed": 6, "lanes_fell": 3, "lanes_timed_out": 2,
        "lanes_nonfinite": 1, "rows_all_completed": 1,
    }


def test_the_json_writer_refuses_nan_and_keeps_the_previous_file(tmp_path):
    out = tmp_path / "courses.json"
    write_json(out, {"score": 1.0})
    with pytest.raises(ValueError):
        write_json(out, {"score": float("nan")})
    assert json.loads(out.read_text()) == {"score": 1.0}
    assert [p.name for p in tmp_path.iterdir()] == ["courses.json"]


def test_the_json_writer_removes_its_temp_file_when_the_swap_fails(tmp_path, monkeypatch):
    out = tmp_path / "courses.json"
    write_json(out, {"score": 1.0})

    def fail(*args, **kwargs):
        raise OSError("replace failed")

    monkeypatch.setattr(scoring.os, "replace", fail)
    with pytest.raises(OSError, match="replace failed"):
        write_json(out, {"score": 2.0})
    assert json.loads(out.read_text()) == {"score": 1.0}
    # iterdir lists dotfiles, so a leftover temp file would show here.
    assert [p.name for p in tmp_path.iterdir()] == ["courses.json"]


def test_the_json_writer_gives_the_mode_a_plain_write_gives(tmp_path):
    # Another account reading a shared runs/ tree sees courses.json as it
    # sees any other file the run wrote.
    sibling = tmp_path / "sibling.json"
    sibling.write_text("{}\n")
    mode = sibling.stat().st_mode & 0o777
    out = tmp_path / "courses.json"
    write_json(out, {"score": 1.0})
    assert out.stat().st_mode & 0o777 == mode
    # A rewrite replaces the file and keeps the mode.
    write_json(out, {"score": 2.0})
    assert out.stat().st_mode & 0o777 == mode
    assert json.loads(out.read_text()) == {"score": 2.0}


# -- aggregate -----------------------------------------------------------------


def test_aggregate_reports_median_and_worst():
    seeds = [_path(_rec(xte=x), _out(seed=k)) for k, x in enumerate((0.01, 0.02, 0.02, 0.5))]
    agg = aggregate(seeds)
    assert agg["seeds"] == 4 and agg["completed"] == 4 and agg["falls"] == 0
    assert agg["score_worst"] == min(s["score"] for s in seeds) < agg["score_median"]
    assert agg["score_median"] == pytest.approx(np.median([s["score"] for s in seeds]), abs=1e-3)
    assert agg["binding"] == "tracking" and len(agg["per_seed"]) == 4


def test_aggregate_counts_falls_timeouts_and_nonfinite_lanes():
    good = _path(_rec(xte=0.02))
    seeds = [
        good, good,
        _path(_rec(n=300), _out(n=300, outcome="fell", fell_at=299)),
        _path({k: v[:0] for k, v in _rec().items()}, _out(n=0, outcome="settle_fell", fell_at=0)),
        _path(_rec(n=400), _out(n=400, outcome="timed_out")),
        _path(_rec(n=100), _out(n=100, outcome="nonfinite")),
    ]
    agg = aggregate(seeds)
    assert (agg["completed"], agg["falls"], agg["timeouts"], agg["nonfinite"]) == (2, 2, 1, 1)
    assert agg["score_worst"] == 0.0


def test_aggregate_survives_seeds_without_subscores():
    dead = _path({k: v[:0] for k, v in _rec().items()}, _out(n=0, outcome="settle_fell",
                                                              fell_at=0))
    agg = aggregate([dead, dead])
    assert agg["score_median"] == 0.0
    assert agg["subscore_median"] is None and agg["raw_median"] is None
    assert agg["gait_median"] is None and agg["binding"] is None


def test_aggregate_skips_none_gait_medians_and_raw_values():
    rec = _rec(xte=0.02)
    swing = np.concatenate([np.zeros(10), np.linspace(0.0, 0.05, 10), np.linspace(0.05, 0.0, 10)])
    rec["foot_clear"] = np.tile(np.resize(swing, 600)[:, None], (1, 2))
    with_swings = _path(rec)
    without = _path(_rec(xte=0.02))
    agg = aggregate([with_swings, without, without])
    assert agg["gait_median"]["swing_apex_med_m"] == with_swings["gait"]["swing_apex_med_m"]
    assert agg["gait_median"]["swings"] == 0.0
    stalled = _rec()
    stalled["cmd"][:, 0] = 0.0
    agg = aggregate([_path(stalled), _path(_rec(v_err=0.1))])
    assert agg["raw_median"]["speed_err_rms"] == pytest.approx(0.1, abs=1e-4)


# -- report --------------------------------------------------------------------


def _doc(checkpoint="000786432000"):
    """A schema-1 courses.json built from the real producers on synthetic
    records: three rows, four seeds, one fall."""
    p, cat = ROBOTO, _CAT_R
    seeds = {
        "straight_10m": [_path(_rec(xte=0.03, slip=0.01), _out(seed=k)) for k in range(4)],
        "straight_slippery": [_path(_rec(xte=0.04, slip=0.02), _out(seed=k)) for k in range(3)]
        + [_path(_rec(n=300), _out(n=300, outcome="fell", fell_at=290, seed=3))],
        "spin_left": [_spin(cat["spin_left"], _spin_rec(wz_cmd=cat["spin_left"].wz, wz_err=0.2))
                      for _ in range(4)],
    }
    rows = {}
    for name, per_seed in seeds.items():
        rows[name] = {**describe(cat[name], _DT), **aggregate(per_seed)}
        rows[name]["perfect_unicycle"] = (
            {"xte_rms_m": 0.0, "tracking": 1000.0, "t_over_ideal": 0.975}
            if rows[name]["kind"] == "path" else None
        )
    doc = {
        "schema": spec.SCHEMA_VERSION, "ground_class": "flat", "run": "probe_run",
        "run_status": None, "checkpoint": checkpoint, "checkpoint_step": int(checkpoint),
        "robot": p.robot, "seeds": 4, "seed_base": 0, "canonical": True,
        "env_overrides": None,
        "catalogue": {"version": CATALOGUE_VERSION, "fingerprint": catalogue_fingerprint(p),
                      "params_source": "pinned"},
        "warnings": ["straight_fast: vx 0.90 m/s outside the run's vx box [-0.6, 0.8]"],
        "courses": rows,
    }
    doc["summary"] = summary(rows, cat)
    return json.loads(json.dumps(doc, allow_nan=False))


def test_the_courses_section_renders_from_a_schema_1_dict():
    md = render_markdown(_doc())
    assert md.startswith("## Courses")
    for name in ("straight_10m", "straight_slippery", "spin_left"):
        assert f"| {name} |" in md
    assert "version 1" in md and "roboto_origin" in md and "000786432000" in md
    assert "warning: straight_fast" in md
    assert "Compare a row across policies, never across rows." in md
    assert "A difference below the noise band in docs/configuration.md is noise." in md
    assert "Height and grip are diagnostics" in md
    assert "seed 3 fell at step 290" in md
    slippery = next(line for line in md.splitlines() if line.startswith("| straight_slippery |"))
    assert "1/4 | 3/4" in slippery and "2.00" in slippery  # falls, done, slip ratio


def test_console_rows_come_in_catalogue_order():
    doc = _doc()
    lines = console_table(doc).splitlines()[2:]
    assert [line.split()[0] for line in lines] == list(doc["courses"])
    assert list(doc["courses"]) == [n for n in TWENTY if n in doc["courses"]]


def test_a_checkpoint_different_from_the_battery_is_flagged():
    doc = _doc()
    assert "checkpoint mismatch" in render_markdown(doc, battery_checkpoint="000100000000")
    assert "checkpoint mismatch" not in render_markdown(doc, battery_checkpoint="000786432000")
    assert "checkpoint mismatch" not in render_markdown(doc)


def test_courses_report_imports_no_jax():
    code = (
        "import sys, humanoid_lab.eval.courses.report\n"
        "print(sorted(m for m in sys.modules if m == 'jax' or m.startswith('jax.')))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"
