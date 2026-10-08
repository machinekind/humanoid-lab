"""The terrain scan without a model: the course rules, the reduction, the
Wilson interval, the gate, the physics verdict, the keys, the batched
layout, partial scans, the arena checks, parking, the runner's accounting,
the CLI flags and refusals, and the run.sh verb.

Nothing is built. The runner rolls a scripted world, and the scan's tests
use namespace envs and a fake runner.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jp
import numpy as np
import pytest
from flax import struct
from mujoco_playground._src import mjx_env

from humanoid_lab import fd_capture, paths
from humanoid_lab.envs.terrain_joystick import default_config as terrain_default_config
from humanoid_lab.eval import terrain_scan as scan
from humanoid_lab.eval import terrain_suite as ts
from humanoid_lab.terrain import TYPES, fingerprint
from humanoid_lab.terrain.config import CPU_ARENA, arena_for, params_from_config

SETTLE = 3
R_OUT = 1.75
D0 = 0.03


def _cpu_arena():
    return arena_for(params_from_config(CPU_ARENA))


def _tiny_suite(**kw):
    """Three CPU-arena cells, 2 headings x 1 offset x 2 draws, both speeds."""
    params = params_from_config(CPU_ARENA)
    cells = ts.build_cells(params)[:3]
    fields = {
        "robot": "roboto_origin",
        "version": 1,
        "arena": params,
        "fingerprint": fingerprint(arena_for(params)),
        "cells": cells,
        "headings": 2,
        "offsets": (0.03,),
        "draws": 2,
        "settle_steps": SETTLE,
    }
    return ts.Suite(**{**fields, **kw})


def roll(distances, dones=None, deadline=100, settle=SETTLE):
    """Drive `track` over a (T, N) distance history. Returns the outcome."""
    distances = np.asarray(distances, float)
    steps, n = distances.shape
    dones = np.zeros((steps, n), bool) if dones is None else np.asarray(dones, bool)
    deadline = np.broadcast_to(np.asarray(deadline), (n,))
    o = scan.start_outcome(n)
    for i in range(steps):
        o, _, _ = scan.track(i, o, distances[i], dones[i], R_OUT, D0, deadline, settle)
    return o


def as_out(o: scan.Outcome, **metrics) -> dict:
    n = len(o.finished)
    out = {k: np.asarray(v) for k, v in o._asdict().items()}
    for key in ("saturation", "track_err", "clearance"):
        out[key] = np.asarray(metrics.get(key, np.zeros(n)), float)
    out["nonfinite"] = np.zeros(n, bool)
    return out


# -- course rules ------------------------------------------------------------------


def test_crossing_is_chebyshev_not_euclidean():
    centre = np.array([2.0, -1.0])
    # Euclidean 2.12 m from the centre, Chebyshev 1.5 m: not across.
    diagonal = centre + np.array([1.5, 1.5])
    # Chebyshev 1.76 m along an axis: across.
    axis = centre + np.array([1.76, 0.2])
    d = scan.tile_distance(np.stack([diagonal, axis]), centre)
    np.testing.assert_allclose(d, [1.5, 1.76])
    o = roll([[0.0, 0.0]] * SETTLE + [d])
    assert o.finished.tolist() == [False, True]


def test_standing_still_earns_nothing():
    o = roll(np.full((60, 1), D0), deadline=50)
    assert not o.finished[0] and not o.fell[0]
    assert o.progress[0] == 0.0
    # It ran to its deadline and was measured after the settle.
    assert o.steps[0] == 50 and o.measured[0] == 50 - SETTLE
    r = scan.reduce_runs(as_out(o))
    assert (r.passed, r.timeouts, r.falls, r.progress_mean) == (0, 1, 0, 0.0)


def test_a_crossing_in_the_settle_does_not_count():
    """The base must reach r_out after the settle. A run already past it at
    the settle's end finishes on its first measured step."""
    o = roll([[2.0]] * SETTLE)
    assert not o.finished[0]
    o = roll([[2.0]] * (SETTLE + 1))
    assert o.finished[0] and o.steps[0] == SETTLE + 1


def test_a_finished_run_never_unfinishes_and_stops_counting():
    walk = np.concatenate([np.full(SETTLE, D0), np.linspace(D0, 1.8, 10), np.zeros(20)])[:, None]
    o = roll(walk)
    assert o.finished[0]
    # It stopped on the step it crossed.
    first = SETTLE + int(np.argmax(walk[SETTLE:, 0] >= R_OUT))
    assert o.steps[0] == first + 1 and o.measured[0] == first + 1 - SETTLE
    assert o.progress[0] == 1.0


def test_still_running_stops_three_ways_with_a_per_run_deadline():
    finished = np.array([False, True, False, False])
    fell = np.array([False, False, True, False])
    deadline = np.array([10, 10, 10, 5])
    assert scan.still_running(4, finished, fell, deadline).tolist() == [True, False, False, True]
    assert scan.still_running(5, finished, fell, deadline).tolist() == [True, False, False, False]
    assert scan.still_running(10, finished, fell, deadline).tolist() == [False] * 4


def test_a_fall_after_finishing_is_not_recorded():
    d = [[D0]] * SETTLE + [[1.8]] + [[1.0]] * 5
    dones = [[False]] * (SETTLE + 1) + [[True]] * 5
    o = roll(d, dones)
    assert o.finished[0] and not o.fell[0]
    assert scan.reduce_runs(as_out(o)).passed == 1


def test_a_fall_past_the_deadline_is_not_recorded():
    o = roll(np.full((20, 1), D0), np.r_[np.zeros(10, bool), np.ones(10, bool)][:, None], deadline=10)
    assert not o.fell[0]
    assert scan.reduce_runs(as_out(o)).timeouts == 1


def test_falls_in_the_settle_are_counted_separately():
    d = np.full((10, 3), D0)
    dones = np.zeros((10, 3), bool)
    dones[1:, 0] = True  # in the settle
    dones[SETTLE + 2 :, 1] = True  # after it
    o = roll(d, dones, deadline=10)
    assert o.fell.tolist() == [True, True, False]
    assert o.fell_in_settle.tolist() == [True, False, False]
    # The settle fall stopped before any measured step.
    assert o.measured.tolist() == [0, 3, 10 - SETTLE]
    r = scan.reduce_runs(as_out(o))
    assert (r.falls, r.falls_in_settle, r.timeouts, r.measured) == (2, 1, 1, 2)


def test_progress_is_the_furthest_measured_fraction_of_the_crossing():
    # Backwards first, then to 0.89 m, then back toward the centre.
    d = [[1.0]] * SETTLE + [[0.0], [0.89], [0.5]]
    o = roll(d)
    # The settle's 1.0 m is not measured.
    assert o.progress[0] == pytest.approx((0.89 - D0) / (R_OUT - D0))


# -- reduction ----------------------------------------------------------------------


def test_reduction_averages_only_measured_runs():
    o = scan.Outcome(
        finished=np.array([True, False, False]),
        fell=np.array([False, True, True]),
        fell_in_settle=np.array([False, False, True]),
        steps=np.array([90, 40, 2]),
        measured=np.array([40, 37, 0]),
        progress=np.array([1.0, 0.5, 0.0]),
    )
    r = scan.reduce_runs(as_out(o, saturation=[0.1, 0.3, 0.0], track_err=[0.2, 0.4, 9.0], clearance=[0.02, 0.04, 9.0]))
    assert (r.passed, r.of, r.falls, r.falls_in_settle, r.timeouts) == (1, 3, 2, 1, 0)
    assert r.measured == 2 and r.steps_max == 90
    assert r.progress_mean == pytest.approx(0.75)
    assert r.saturation == pytest.approx(0.2)
    assert r.track_err == pytest.approx(0.3)
    assert r.clearance == pytest.approx(0.03)
    assert r.rate == pytest.approx(1 / 3)


def test_a_cell_where_every_run_died_in_the_settle_reports_zero_not_nan():
    o = scan.Outcome(
        finished=np.zeros(4, bool),
        fell=np.ones(4, bool),
        fell_in_settle=np.ones(4, bool),
        steps=np.array([1, 2, 2, 3]),
        measured=np.zeros(4, int),
        progress=np.zeros(4),
    )
    r = scan.reduce_runs(as_out(o, track_err=np.full(4, np.nan)))
    assert (r.measured, r.track_err, r.saturation, r.clearance, r.progress_mean) == (0, 0.0, 0.0, 0.0, 0.0)
    entry = scan.result_entry(r)
    assert not any(isinstance(v, float) and math.isnan(v) for v in entry.values())


def test_wilson_interval():
    lo, hi = scan.wilson(8, 10)
    assert (lo, hi) == (pytest.approx(0.4902, abs=1e-4), pytest.approx(0.9433, abs=1e-4))
    assert scan.wilson(0, 64)[0] == 0.0
    assert scan.wilson(64, 64)[1] == 1.0
    for passed in (0, 1, 39, 52, 61, 64):
        lo, hi = scan.wilson(passed, 64)
        assert 0.0 <= lo <= passed / 64 <= hi <= 1.0
    # Symmetric about one half.
    a, b = scan.wilson(20, 64)
    c, d = scan.wilson(44, 64)
    assert a == pytest.approx(1 - d) and b == pytest.approx(1 - c)


# -- gate ----------------------------------------------------------------------------


def _entry(threshold, passed, provenance="provisional"):
    """A report cell entry: `passed` maps each speed key to a pass count."""
    return {
        "type": "wave",
        "threshold": threshold,
        "provenance": provenance,
        **{speed: {"passed": n, "of": 64, "ci95": [0.0, 1.0]} for speed, n in passed.items()},
    }


@pytest.mark.parametrize(
    "cells, expected, clean, verdict, n_failures",
    [
        # Exactly the threshold passes.
        ({"a": _entry(61, {"0.3": 61, "0.6": 64})}, 2, True, "pass", 0),
        # One below fails.
        ({"a": _entry(61, {"0.3": 60, "0.6": 64})}, 2, True, "fail", 1),
        # Fewer pairs than a full scan gates.
        ({"a": _entry(61, {"0.3": 64})}, 2, True, "incomplete", 0),
        # A failure outranks an incomplete scan.
        ({"a": _entry(61, {"0.3": 3})}, 2, True, "fail", 1),
        # A dirty scan keeps its numbers, and its verdict is invalid.
        ({"a": _entry(61, {"0.3": 64, "0.6": 64})}, 2, False, "invalid", 0),
        # Tracked cells are never gated.
        (
            {"a": _entry(52, {"0.3": 52, "0.6": 52}), "b": _entry(None, {"0.3": 0, "0.6": 0}, "tracked")},
            2, True, "pass", 0,
        ),
    ],
)
def test_absolute_gate(cells, expected, clean, verdict, n_failures):
    gate = scan.absolute_gate(cells, expected, clean)
    assert gate["verdict"] == verdict
    assert gate["expected"] == expected
    assert len(gate["failures"]) == n_failures
    assert gate["checked"] == sum(len(scan.speed_entries(e)) for e in cells.values() if e["threshold"])
    for f in gate["failures"]:
        assert f["passed"] < f["threshold"] and f["provenance"] == "provisional"


def test_gated_pairs_count_the_full_suite():
    # 12 gated cells at 2 speeds.
    assert scan.gated_pairs(ts.ROBOTO_SUITE) == 24
    assert scan.gated_pairs(_tiny_suite()) == 0


def _fills(pool=None, rows=None) -> dict:
    """The fields of a pool_report entry that physics_verdict reads."""
    return {"fill_pool": pool, "fill_rows": rows}


CLEAN_FILLS = {"0.3": _fills(0.5, 0.5), "0.6": _fills(0.999, 0.999)}
NO_NONFINITE = {"0.3": 0, "0.6": 0}


@pytest.mark.parametrize("kind", fd_capture.GATING)
def test_a_gating_warp_message_makes_the_scan_dirty(kind):
    clean, warnings = scan.physics_verdict("warp", {kind: 1}, CLEAN_FILLS, NO_NONFINITE)
    assert clean is False
    assert len(warnings) == 1 and kind in warnings[0]


def test_physics_verdict_reads_fills_and_nonfinite_runs():
    # The EPA horizon is reported and never gates. A fill below 1 is clean.
    assert scan.physics_verdict("warp", {"epa_horizon": 5}, CLEAN_FILLS, NO_NONFINITE) == (True, [])
    for full in (_fills(1.0, 0.5), _fills(0.5, 1.0)):
        clean, warnings = scan.physics_verdict("warp", None, {**CLEAN_FILLS, "0.6": full}, NO_NONFINITE)
        assert clean is False
        assert len(warnings) == 1 and "speed 0.6" in warnings[0]
    # A fill nothing measured is not a fill.
    assert scan.physics_verdict("warp", None, {"0.3": _fills()}, {"0.3": 0}) == (True, [])
    clean, warnings = scan.physics_verdict("warp", None, CLEAN_FILLS, {"0.3": 1, "0.6": 0})
    assert clean is False
    assert len(warnings) == 1 and "0.3" in warnings[0] and "0.6" not in warnings[0]


def test_physics_verdict_on_jax_covers_nonfinite_runs_only():
    clean, warnings = scan.physics_verdict("jax", None, {"0.3": _fills()}, {"0.3": 0})
    assert clean is True
    assert len(warnings) == 1 and "jax backend" in warnings[0]
    clean, _ = scan.physics_verdict("jax", None, {"0.3": _fills()}, {"0.3": 2})
    assert clean is False


# -- keys ----------------------------------------------------------------------------


def test_cell_keys_are_per_cell_streams_and_seed_0_is_the_default():
    a, b = ts.ROBOTO_SUITE.cells[:2]
    key = scan.cell_key(a)
    expected = jax.random.PRNGKey(a.row * 1000 + TYPES.index(a.terrain_type))
    np.testing.assert_array_equal(key, expected)
    np.testing.assert_array_equal(scan.cell_key(a, 0), key)
    assert not np.array_equal(scan.cell_key(b), key)
    assert not np.array_equal(scan.cell_key(a, 1), key)
    assert not np.array_equal(scan.cell_key(a, 1), scan.cell_key(a, 2))
    np.testing.assert_array_equal(scan.cell_key(a, 3), jax.random.fold_in(key, 3))


def test_batched_reset_keys_equal_each_cells_own_split():
    cells = ts.ROBOTO_SUITE.cells[:3]
    runs = ts.ROBOTO_SUITE.runs_per_cell
    keys = scan.batch_reset_keys(jp.stack([scan.cell_key(c, 5) for c in cells]), runs)
    assert keys.shape == (3 * runs, 2)
    for c, cell in enumerate(cells):
        own = jax.random.split(scan.cell_key(cell, 5), runs)
        np.testing.assert_array_equal(keys[c * runs : (c + 1) * runs], own)


# -- spawn table ---------------------------------------------------------------------


def test_spawn_table_is_cell_major_and_starts_on_the_pads():
    suite = ts.ROBOTO_SUITE
    spec = ts.eval_arena(suite).spec
    cells = suite.cells[::7]
    runs = suite.runs_per_cell
    table = scan.spawn_table(spec, suite, cells, 0.02, suite.speeds)
    n = runs * len(cells)
    assert table["start"].shape == (n, 2)
    assert table["cell"].tolist() == np.repeat(np.arange(len(cells)), runs).tolist()
    course = ts.course(suite)
    for c, cell in enumerate(cells):
        tile = ts.cell_tile(spec, cell)
        part = slice(c * runs, (c + 1) * runs)
        np.testing.assert_allclose(table["centre"][part], np.tile(tile.origin[:2], (runs, 1)), atol=1e-6)
        assert set(table["row"][part]) == {cell.row}
        assert set(table["type"][part]) == {TYPES.index(cell.terrain_type)}
        offset = table["start"][part] - table["centre"][part]
        yaw = np.array([r.yaw for r in course])
        np.testing.assert_allclose(table["yaw"][part], yaw, atol=1e-6)
        np.testing.assert_allclose(
            offset, np.array([r.offset for r in course])[:, None] * np.stack([np.cos(yaw), np.sin(yaw)], -1),
            atol=1e-6,
        )
        # Every start lies on the pad, the footprint a cell inside it.
        assert np.abs(offset).max() + suite.footprint_reach <= tile.pad_radius - spec.params.cell_size
        r = ts.r_out(tile.feature_radius, spec.params.tile_size, suite.footprint_reach)
        np.testing.assert_allclose(table["r_out"][part], r, rtol=1e-6)
        np.testing.assert_allclose(table["d0"][part], np.abs(offset).max(-1), atol=1e-6)
        for s in suite.speeds:
            want = ts.run_deadlines(suite, r, s, 0.02)
            assert table["deadline"][scan.speed_key(s)][part].tolist() == list(want)


# -- runner pieces -------------------------------------------------------------------


def test_parked_worlds_sit_above_the_arena():
    arena = _cpu_arena()
    top = scan.arena_top(arena)
    hf = arena.spec.hfield
    assert top >= float(arena.lookup.max()) and top >= hf.pos_z + hf.elevation_z and top >= 0.0
    # The CPU arena's highest point is its stair summit.
    assert top == pytest.approx(0.425, abs=1e-6)

    rng = np.random.default_rng(0)
    nq, nv, b, z0 = 12, 11, 2, 0.7
    reset = rng.normal(size=nq).astype(np.float32)
    xy = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], np.float32)
    base_z = top + scan.PARK_LIFT + z0
    pose = scan.park_pose(reset, b, xy, base_z)
    np.testing.assert_array_equal(pose, scan.park_pose(jp.asarray(reset), b, jp.asarray(xy), base_z, xp=jp))
    qpos = rng.normal(size=(3, nq)).astype(np.float32)
    qvel = rng.normal(size=(3, nv)).astype(np.float32)
    parked = np.array([True, False, True])
    q, v = scan.park(qpos, qvel, parked, pose)
    for w in (0, 2):
        np.testing.assert_array_equal(q[w, :b], reset[:b])
        np.testing.assert_array_equal(q[w, b : b + 2], xy[w])
        # The soles sit PARK_LIFT above the arena's top at the reset pose.
        assert q[w, b + 2] - z0 == pytest.approx(top + scan.PARK_LIFT)
        np.testing.assert_array_equal(q[w, b + 3 :], reset[b + 3 :])
        assert not v[w].any()
    np.testing.assert_array_equal(q[1], qpos[1])
    np.testing.assert_array_equal(v[1], qvel[1])


# -- the runner on a scripted world ---------------------------------------------------


@struct.dataclass
class _Impl:
    nefc: jax.Array
    nacon: jax.Array
    ncollision: jax.Array


@struct.dataclass
class _Data:
    qpos: jax.Array
    qvel: jax.Array
    actuator_force: jax.Array
    time: jax.Array
    _impl: _Impl


class _RailEnv:
    """A base on a rail along x. After step k, world w's base sits `path[k,
    w]` from its tile centre, reports done `dones[k, w]` and has a non-finite
    height where `nan[k, w]`. The warp counters after step k are
    `counters[k]`. The step ignores the pose it is given, so a parked world
    follows its script too. One of four actuators pushes past 95% of its
    20 N cap, body vx is 0.5 and the feet clear 2 and 4 cm."""

    dt = 0.02
    _base_qadr = 0
    _z0 = 0.5

    def __init__(self, path, dones, nan, counters):
        self._path = jp.asarray(path, jp.float32)
        self._dones = jp.asarray(dones, jp.float32)
        self._nan = jp.asarray(nan)
        self._counters = jp.asarray(counters, jp.int32)
        self._arena = _cpu_arena()
        self._reset_qpos = jp.zeros(3)
        self.mj_model = SimpleNamespace(actuator_forcerange=np.array([[-20.0, 20.0]] * 4))

    def reset(self, start_xy, centre, w):
        data = _Data(
            qpos=jp.append(start_xy, 1.0),
            qvel=jp.zeros(3),
            actuator_force=jp.array([19.5, 1.0, -1.0, 0.0]),
            time=jp.zeros(()),
            _impl=_Impl(*jp.zeros(3, jp.int32)),
        )
        info = {"command": jp.zeros(3), "k": jp.zeros((), jp.int32), "w": w, "centre": centre}
        return mjx_env.State(data, {"state": jp.zeros(1)}, jp.zeros(()), jp.zeros(()), {}, info)

    def step(self, state, action):
        k, w, centre = state.info["k"], state.info["w"], state.info["centre"]
        z = jp.where(self._nan[k, w], jp.nan, 1.0)
        qpos = jp.stack([centre[0] + self._path[k, w], centre[1], z])
        data = state.data.replace(qpos=qpos, time=state.data.time + self.dt, _impl=_Impl(*self._counters[k]))
        return state.replace(data=data, done=self._dones[k, w], info={**state.info, "k": k + 1})

    def _local_linvel(self, data):
        return jp.array([0.5, 0.0, 0.0])

    def _foot_clearance(self, data):
        return jp.array([0.02, 0.04])


def test_the_runner_stops_measures_and_parks_each_run_on_its_own(monkeypatch):
    """Five scripted worlds, SETTLE 3, a budget of 12 and deadlines 8, 10,
    10, 8 and 10. World 0 stands, goes non-finite on step 5 while it runs,
    and reports done from step 8, past its deadline. World 1 reaches r_out
    on step 5 and reports done from step 6, after its finish. World 2 walks
    to 0.5 m and falls on step 6. World 3 falls on step 1, in the settle,
    and goes non-finite on step 4, once stopped. World 4 walks to 0.9 m and
    runs to its deadline. Each counter peaks once. Steps 10 and 11 never
    run, and their counters read 99."""
    steps, n = 12, 5
    path = np.full((steps, n), D0)
    path[SETTLE:, 1] = [0.6, 1.2, R_OUT + 0.05] + [1.0] * (steps - SETTLE - 3)
    path[SETTLE:, 2] = 0.5
    path[SETTLE:, 4] = 0.9
    dones = np.zeros((steps, n), bool)
    dones[8:, 0] = dones[6:, 1] = dones[6:, 2] = dones[1:, 3] = True
    nan = np.zeros((steps, n), bool)
    nan[5, 0] = nan[4, 3] = True
    counters = np.ones((steps, 3), np.int32)
    counters[2, 0], counters[6, 1], counters[4, 2] = 9, 40, 16
    counters[10:] = 99
    env = _RailEnv(path, dones, nan, counters)
    monkeypatch.setattr(scan, "scan_reset", lambda env_, key, xy, yaw, centre, row, ttype: env_.reset(xy, centre, row))

    centre = np.array([10.0, -5.0], np.float32)
    start = np.tile(centre + np.array([D0, 0.0], np.float32), (n, 1))
    table = {
        # The scripted reset reads the row as the world index.
        "row": jp.arange(n, dtype=jp.int32),
        "type": jp.zeros(n, jp.int32),
        "centre": jp.tile(jp.asarray(centre), (n, 1)),
        "start": jp.asarray(start),
        "yaw": jp.zeros(n),
        "r_out": jp.full(n, R_OUT),
        "d0": jp.full(n, D0),
    }
    deadline = jp.array([8, 10, 10, 8, 10], jp.int32)

    def inference(obs, key):
        return jp.zeros((obs["state"].shape[0], 4)), {}

    runner = scan.make_batch_runner(env, _tiny_suite(), inference)
    keys = jax.random.split(jax.random.PRNGKey(0), n)
    out = jax.tree.map(np.asarray, runner(keys, table, 0.6, deadline, budget=steps))
    runs = out["runs"]

    # The loop ends when the last run stops, not the first.
    assert int(out["iterations"]) == 10
    assert runs["finished"].tolist() == [False, True, False, False, False]
    assert runs["fell"].tolist() == [False, False, True, True, False]
    assert runs["fell_in_settle"].tolist() == [False, False, False, True, False]
    assert runs["steps"].tolist() == [8, 6, 7, 2, 10]
    assert runs["measured"].tolist() == [5, 3, 4, 0, 7]
    gain = [(d - D0) / (R_OUT - D0) for d in (0.5, 0.9)]
    np.testing.assert_allclose(runs["progress"], [0.0, 1.0, gain[0], 0.0, gain[1]], rtol=1e-6)
    # Each metric is a mean over the measured steps alone. World 3 has none.
    measured = np.array([1.0, 1.0, 1.0, 0.0, 1.0])
    np.testing.assert_allclose(runs["saturation"], 0.25 * measured, rtol=1e-6)
    np.testing.assert_allclose(runs["track_err"], 0.1 * measured, rtol=1e-5)
    np.testing.assert_allclose(runs["clearance"], 0.03 * measured, rtol=1e-6)
    # Only a running world's non-finite pose counts.
    assert runs["nonfinite"].tolist() == [True, False, False, False, False]
    assert (int(out["nefc"]), int(out["nacon"]), int(out["ncollision"])) == (9, 40, 16)
    # Every world ends parked at rest.
    base_z = scan.arena_top(env._arena) + scan.PARK_LIFT + env._z0
    np.testing.assert_allclose(out["final"]["qpos"], scan.park_pose(np.zeros(3), 0, start, base_z), rtol=1e-6)
    assert not out["final"]["qvel"].any()
    r = scan.reduce_runs({k: runs[k] for k in scan.RUN_KEYS})
    assert (r.passed, r.falls, r.falls_in_settle, r.timeouts, r.measured, r.steps_max) == (1, 2, 1, 2, 4, 10)


def test_scan_overrides_pass_complete_sub_blocks():
    """The measurement merge is one level deep, so each terrain sub-block
    replaces the run's own and must carry every key."""
    suite = ts.ROBOTO_SUITE
    budgets = scan.scan_budgets(suite)
    assert budgets == {"naconmax_per_env": 256, "njmax": 2048, "naccdmax_per_env": None}
    assert scan.scan_budgets(suite, 300, 4096, 128)["naccdmax_per_env"] == 128
    o = scan.scan_overrides(suite, 3072, "warp", budgets)
    defaults = terrain_default_config().terrain
    t = o["terrain"]
    assert set(t["spawn"]) == set(defaults.spawn.to_dict())
    assert set(t["command_bias"]) == set(defaults.command_bias.to_dict())
    assert set(t["base_contact"]) == set(defaults.base_contact.to_dict())
    assert (t["spawn"]["mode"], t["spawn"]["yaw"], t["spawn"]["pad_jitter"], t["spawn"]["level"]) == (
        "pad", False, 0.0, -1,
    )
    assert t["spawn"]["grace_sec"] == 0.0
    assert t["base_contact"] == {"terminate": True, "tol": suite.base_contact_tol}
    assert t["command_bias"]["enable"] is False
    assert params_from_config(t["arena"]) == suite.arena
    assert o["sim"] == {
        "backend": "warp", "num_envs": 3072, "naconmax_per_env": 256, "njmax": 2048, "naccdmax_per_env": None,
    }


def test_command_box_warnings():
    run = {"env_config": {"command": {"vx": [-0.6, 0.5]}}}
    warnings = scan.command_box_warnings(run, (0.3, 0.6))
    assert len(warnings) == 1 and "vx 0.6" in warnings[0] and "[-0.6, 0.5]" in warnings[0]
    assert scan.command_box_warnings({"env_config": {"command": {"vx": [-0.6, 1.0]}}}, (0.3, 0.6)) == []
    assert scan.command_box_warnings({}, (0.3,)) == []


def test_check_arena_refuses_a_wrong_fingerprint_or_params():
    suite = _tiny_suite()
    arena = _cpu_arena()
    scan.check_arena(arena, suite)
    with pytest.raises(ValueError, match="bump Suite.version and re-pin") as e:
        scan.check_arena(arena, replace(suite, fingerprint="0" * 64))
    assert "generator is at version" in str(e.value) and "0" * 64 in str(e.value)
    with pytest.raises(scan.ScanRefused, match="params differ"):
        scan.check_arena(ts.eval_arena(ts.ROBOTO_SUITE), replace(suite, fingerprint=ts.ROBOTO_SUITE.fingerprint))


def test_cell_and_speed_selection():
    suite = ts.ROBOTO_SUITE
    names = [suite.cells[5].name, suite.cells[1].name]
    assert [c.name for c in scan.select_cells(suite, names)] == [suite.cells[1].name, suite.cells[5].name]
    assert scan.select_cells(suite, None) == suite.cells
    with pytest.raises(scan.ScanRefused, match="unknown cells"):
        scan.select_cells(suite, ["pyramid_stairs_99cm"])
    with pytest.raises(scan.ScanRefused, match="names no cell"):
        scan.select_cells(suite, [])
    assert scan.select_speeds(suite, [0.6, 0.3]) == (0.3, 0.6)
    assert scan.select_speeds(suite, None) == suite.speeds
    with pytest.raises(scan.ScanRefused, match="not the suite's"):
        scan.select_speeds(suite, [0.45])
    with pytest.raises(scan.ScanRefused, match="comma-separated numbers"):
        scan._speeds("0.3,fast")


# -- the scan with a fake env and runner ---------------------------------------------


def _write_run(tmp_path: Path, robot="roboto_origin", vx=(-0.6, 1.0), task="terrain") -> Path:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    record = {
        "run_name": "fake_run",
        "task": task,
        "checkpoint_dir": str(run_dir / "checkpoints"),
        "ppo_config": {},
        "env_config": {"command": {"vx": list(vx)}},
        "hydra_config": {
            "robot": {"name": robot, "dir": f"robots/{robot}"},
            "actuators": {"name": "deploy_pd", "overrides": {}},
            "task": {"name": task, "env": {}},
        },
    }
    (run_dir / "run.json").write_text(json.dumps(record))
    return run_dir


def _fake_env(arena, n_worlds, backend="jax"):
    return SimpleNamespace(
        _arena=arena,
        dt=0.02,
        _backend=backend,
        mj_model=SimpleNamespace(actuator_forcerange=np.array([[-20.0, 20.0]] * 4)),
        _naconmax_per_env=256,
        _njmax=2048,
        _config=SimpleNamespace(sim={"num_envs": n_worlds, "naconmax_per_env": 256, "naccdmax_per_env": None}),
        _ccd_slot_bytes=5480,
        _actuator_model=object(),
    )


def _no_build(*_a, **_k):
    raise AssertionError("the scan built an env")


def _fake_build(seen: dict, arena=None, backend="jax"):
    """A build_scan_env that reads run.json and returns a fake env on
    `arena`, the CPU arena by default. It records its arguments in `seen`."""

    def build(run_dir_, suite_, n_worlds, backend_, budgets):
        seen.update(n_worlds=n_worlds, backend=backend_, budgets=dict(budgets))
        run = json.loads((Path(run_dir_) / "run.json").read_text())
        env = _fake_env(_cpu_arena() if arena is None else arena, n_worlds, backend)
        return run, env, Path("ckpt/000000001024"), None

    return build


def _fake_runner(seen: dict):
    """A make_batch_runner whose runner encodes the world index in its
    outputs. World w = c * R + r is run r of batch cell c. Batch cell c
    passes c + 1 runs at 0.3 and none at 0.6. The runner records the keys,
    the budget and each speed's deadlines in `seen`."""

    def runner_for(env, suite_, inference):
        runs = suite_.runs_per_cell

        def run(keys, table, speed, deadline, *, budget):
            n = keys.shape[0]
            seen.setdefault("keys", np.asarray(keys))
            seen.setdefault("budget", budget)
            seen.setdefault("deadline", {})[speed] = np.asarray(deadline)
            w = np.arange(n)
            c, r = w // runs, w % runs
            finished = (r < c + 1) if speed == 0.3 else np.zeros(n, bool)
            return {
                "runs": {
                    "finished": finished,
                    "fell": ~finished & (r % 2 == 0),
                    "fell_in_settle": np.zeros(n, bool),
                    "steps": w + 1,
                    "measured": np.ones(n, np.int32),
                    "progress": np.where(finished, 1.0, 0.25),
                    "saturation": c / 10.0,
                    "track_err": np.full(n, 0.1),
                    "clearance": r / 100.0,
                    "nonfinite": np.zeros(n, bool),
                },
                "iterations": np.int32(n + 7),
                "nefc": np.int32(0),
                "nacon": np.int32(0),
                "ncollision": np.int32(0),
            }

        return run

    return runner_for


def test_batched_layout_slices_back_to_cells(tmp_path, monkeypatch):
    """World w = c * R + r belongs to cell c. The fake runner encodes the
    world index in its outputs, and each cell's entry reduces its own
    slice. Cell c passes c + 1 runs at 0.3 and none at 0.6."""
    base = _tiny_suite()
    # A bar on the first cell, so the gate has a pair to check.
    cells = (replace(base.cells[0], bar=0.5), *base.cells[1:])
    suite = replace(base, cells=cells)
    runs = suite.runs_per_cell
    assert runs == 4
    run_dir = _write_run(tmp_path)
    seen = {}
    monkeypatch.setattr(scan, "build_scan_env", _fake_build(seen))
    monkeypatch.setattr(scan, "make_batch_runner", _fake_runner(seen))
    result = scan.scan(run_dir, suite=suite, backend="jax")

    n = runs * len(cells)
    assert seen["n_worlds"] == n and seen["backend"] == "jax"
    assert seen["budgets"] == {"naconmax_per_env": 256, "njmax": 2048, "naccdmax_per_env": None}
    keys = jp.stack([scan.cell_key(c) for c in cells])
    np.testing.assert_array_equal(seen["keys"], scan.batch_reset_keys(keys, runs))
    assert seen["budget"] == max(int(d.max()) for d in seen["deadline"].values())

    assert list(result["cells"]) == [c.name for c in cells]
    for c, cell in enumerate(cells):
        entry = result["cells"][cell.name]
        assert set(scan.speed_entries(entry)) == {"0.3", "0.6"}
        slow, fast = entry["0.3"], entry["0.6"]
        assert slow["passed"] == c + 1 and fast["passed"] == 0
        assert slow["steps_max"] == fast["steps_max"] == (c + 1) * runs
        assert slow["saturation"] == pytest.approx(c / 10.0)
        assert slow["clearance"] == pytest.approx(np.mean(np.arange(runs)) / 100.0)
        for r in (slow, fast):
            assert r["passed"] + r["falls"] + r["timeouts"] == r["of"] == runs
        assert entry["r_out"] == pytest.approx(ts.r_out(1.6 if "stairs" in cell.terrain_type else 1.9, 4.0, 0.15))
    first = result["cells"][cells[0].name]
    assert (first["bar"], first["threshold"], first["provenance"]) == (0.5, 2, "provisional")
    assert result["cells"][cells[1].name]["threshold"] is None

    gate = result["gate"]["absolute"]
    # Cell 0 passes 1 of 4 at 0.3 and 0 at 0.6 against a threshold of 2.
    assert (gate["verdict"], gate["checked"], gate["expected"]) == ("fail", 2, 2)
    assert result["physics_clean"] is True
    assert result["messages"] is None
    assert result["contacts"]["0.3"]["nacon_pool_max"] is None
    assert result["perf"]["env_steps"] == 2 * (n + 7) * n
    assert result["ccd_scratch"]["slots"] == 256 * n
    assert result["suite"]["fingerprint"] == suite.fingerprint
    assert (result["run"], result["checkpoint"], result["trained_task"]) == ("fake_run", "000000001024", "terrain")
    assert any("jax backend" in w for w in result["warnings"])
    assert not any(w.startswith("partial scan") for w in result["warnings"])
    assert result["provenance"]["started_at"] <= result["timestamp"]
    # The record is JSON.
    json.dumps(result)


def _gated_tiny_suite() -> ts.Suite:
    """The tiny suite with a bar on its first cell: 1 of 4 runs."""
    base = _tiny_suite()
    return replace(base, cells=(replace(base.cells[0], bar=0.25), *base.cells[1:]))


def test_a_partial_scan_gates_against_the_full_suite(tmp_path, monkeypatch):
    """`expected` counts the full suite's gated pairs, whatever the subset.
    Batch cell 0 passes 1 run at 0.3 and none at 0.6."""
    suite = _gated_tiny_suite()
    gated, tracked = suite.cells[0].name, suite.cells[1].name
    run_dir = _write_run(tmp_path)
    monkeypatch.setattr(scan, "build_scan_env", _fake_build({}))
    monkeypatch.setattr(scan, "make_batch_runner", _fake_runner({}))

    result = scan.scan(run_dir, suite=suite, speeds=[0.3], backend="jax")
    assert result["gate"]["absolute"] == {"verdict": "incomplete", "checked": 1, "expected": 2, "failures": []}
    assert "partial scan: speeds [0.3] of [0.3, 0.6]" in result["warnings"]

    result = scan.scan(run_dir, suite=suite, cells=[tracked], backend="jax")
    assert result["gate"]["absolute"] == {"verdict": "incomplete", "checked": 0, "expected": 2, "failures": []}
    assert "partial scan: 1 of 3 cells" in result["warnings"]

    # A failure outranks an incomplete subset.
    result = scan.scan(run_dir, suite=suite, cells=[gated], backend="jax")
    gate = result["gate"]["absolute"]
    assert (gate["verdict"], gate["checked"], gate["expected"]) == ("fail", 2, 2)
    assert [(f["cell"], f["speed"], f["passed"]) for f in gate["failures"]] == [(gated, "0.6", 0)]


CCD_LINE = "CCD overflow - please increase naccdmax to 4097"
EPA_LINE = "Warning: EPA horizon = 24 isn't large enough."


def _warp_runner(seen: dict, lines, nefc: dict, nonfinite_at=None):
    """_fake_runner on warp. Each dispatch writes `lines` to fd 1, as
    MJWarp's device printf does, and reports nacon 7, ncollision 9 and
    `nefc[speed]` rows. World 0 goes non-finite at `nonfinite_at`."""
    fake = _fake_runner(seen)

    def runner_for(env, suite_, inference):
        run = fake(env, suite_, inference)

        def warp_run(keys, table, speed, deadline, *, budget):
            for line in lines:
                os.write(1, (line + "\n").encode())
            out = run(keys, table, speed, deadline, budget=budget)
            out.update(nacon=np.int32(7), ncollision=np.int32(9), nefc=np.int32(nefc[speed]))
            out["runs"]["nonfinite"][0] = speed == nonfinite_at
            return out

        return warp_run

    return runner_for


def test_a_warp_scan_counts_messages_counters_and_nonfinite_runs(tmp_path, monkeypatch):
    """The dispatch runs inside the fd 1 capture, and the messages sum over
    the speeds. The pool counters reach pool_report unswapped."""
    suite = _tiny_suite()
    run_dir = _write_run(tmp_path)
    seen = {}
    monkeypatch.setattr(scan, "build_scan_env", _fake_build(seen, backend="warp"))
    runner = _warp_runner(seen, [CCD_LINE, EPA_LINE], {0.3: 11, 0.6: 11}, nonfinite_at=0.3)
    monkeypatch.setattr(scan, "make_batch_runner", runner)
    result = scan.scan(run_dir, suite=suite, backend="warp")
    assert seen["backend"] == "warp" and result["engine"]["backend"] == "warp"
    assert (result["messages"]["ccd_overflow"], result["messages"]["epa_horizon"]) == (2, 2)
    contacts = result["contacts"]["0.3"]
    assert (contacts["nacon_pool_max"], contacts["ncollision_pool_max"], contacts["nefc_max"]) == (7, 9, 11)
    assert result["nonfinite_runs"] == {"0.3": 1, "0.6": 0}
    assert result["physics_clean"] is False
    assert result["gate"]["absolute"]["verdict"] == "invalid"
    assert not any("jax backend" in w for w in result["warnings"])


def test_a_warp_scan_whose_rows_fill_njmax_is_invalid(tmp_path, monkeypatch):
    suite = _tiny_suite()
    run_dir = _write_run(tmp_path)
    monkeypatch.setattr(scan, "build_scan_env", _fake_build({}, backend="warp"))
    monkeypatch.setattr(scan, "make_batch_runner", _warp_runner({}, [EPA_LINE], {0.3: 11, 0.6: 2048}))
    result = scan.scan(run_dir, suite=suite, backend="warp")
    assert result["messages"]["epa_horizon"] == 2
    assert not any(result["messages"][k] for k in fd_capture.GATING)
    assert result["nonfinite_runs"] == {"0.3": 0, "0.6": 0}
    assert result["contacts"]["0.6"]["fill_rows"] == 1.0
    assert result["physics_clean"] is False
    assert result["gate"]["absolute"]["verdict"] == "invalid"


def test_a_suite_arena_off_its_pin_is_refused_before_any_build(tmp_path, monkeypatch):
    monkeypatch.setattr(scan, "build_scan_env", _no_build)
    monkeypatch.setattr(scan, "make_batch_runner", _no_build)
    run_dir = _write_run(tmp_path)
    with pytest.raises(scan.ScanRefused, match="bump Suite.version and re-pin"):
        scan.scan(run_dir, suite=replace(_tiny_suite(), fingerprint="0" * 64), backend="jax")


def test_a_built_env_on_another_arena_is_refused_before_any_runner(tmp_path, monkeypatch, capsys):
    """The scan checks the arena the env built, not only the suite's."""
    suite = _tiny_suite()
    other = arena_for(params_from_config({**CPU_ARENA, "seed": 1}))
    monkeypatch.setattr(scan, "build_scan_env", _fake_build({}, arena=other))
    monkeypatch.setattr(scan, "make_batch_runner", _no_build)
    run_dir = _write_run(tmp_path)
    with pytest.raises(scan.ScanRefused, match="params differ"):
        scan.scan(run_dir, suite=suite, backend="jax")
    monkeypatch.setitem(ts.SUITES, "roboto_origin", suite)
    assert scan.main(["--run", str(run_dir), "--backend", "jax"]) == 2
    assert "params differ" in capsys.readouterr().err
    assert not (run_dir / scan.OUT_NAME).exists()


# -- CLI -----------------------------------------------------------------------------


def test_list_cells_needs_no_model(monkeypatch, capsys):
    monkeypatch.setattr(scan, "build_scan_env", _no_build)
    monkeypatch.setattr(scan, "make_batch_runner", _no_build)
    assert scan.main(["--list-cells"]) == 0
    out = capsys.readouterr().out
    for cell in ts.ROBOTO_SUITE.cells:
        assert cell.name in out
    assert out.count("provisional") == 12
    assert out.count("tracked") == 36


def test_the_cli_passes_every_flag_to_the_scan(tmp_path, monkeypatch):
    suite = _tiny_suite()
    monkeypatch.setitem(ts.SUITES, "roboto_origin", suite)
    seen = {}
    monkeypatch.setattr(scan, "build_scan_env", _fake_build(seen))
    monkeypatch.setattr(scan, "make_batch_runner", _fake_runner(seen))
    run_dir = _write_run(tmp_path)
    a, b = suite.cells[0].name, suite.cells[2].name
    out = tmp_path / "x.json"
    argv = [
        "--run", str(run_dir), "--backend", "warp", "--cells", f"{b},{a}", "--speeds", "0.3",
        "--eval-seed", "3", "--naconmax-per-env", "128", "--njmax", "1024", "--naccdmax-per-env", "64",
        "--out", str(out),
    ]
    assert scan.main(argv) == 0
    assert seen["budgets"] == {"naconmax_per_env": 128, "njmax": 1024, "naccdmax_per_env": 64}
    assert seen["backend"] == "warp"
    runs = suite.runs_per_cell
    assert seen["n_worlds"] == 2 * runs

    def keys(seed):
        # The cells come back in suite order.
        cells = (suite.cells[0], suite.cells[2])
        return scan.batch_reset_keys(jp.stack([scan.cell_key(c, seed) for c in cells]), runs)

    np.testing.assert_array_equal(seen["keys"], keys(3))
    assert not np.array_equal(seen["keys"], keys(0))
    result = json.loads(out.read_text())
    assert result["eval_seed"] == 3
    assert list(result["cells"]) == [a, b]
    assert all(set(scan.speed_entries(e)) == {"0.3"} for e in result["cells"].values())
    assert not (run_dir / scan.OUT_NAME).exists()


def test_a_run_of_another_task_is_refused_before_any_build(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(scan, "build_scan_env", _no_build)
    monkeypatch.setattr(scan, "make_batch_runner", _no_build)
    run_dir = _write_run(tmp_path, task="sizing")
    with pytest.raises(scan.ScanRefused, match="task 'sizing'"):
        scan.scan(run_dir, suite=_tiny_suite(), backend="jax")
    assert scan.main(["--run", str(run_dir)]) == 2
    assert "task 'sizing'" in capsys.readouterr().err
    assert not (run_dir / scan.OUT_NAME).exists()


@pytest.mark.parametrize("seed", [-1, 2**32])
def test_an_eval_seed_outside_uint32_is_refused_before_any_build(tmp_path, monkeypatch, capsys, seed):
    monkeypatch.setattr(scan, "build_scan_env", _no_build)
    monkeypatch.setattr(scan, "make_batch_runner", _no_build)
    run_dir = _write_run(tmp_path)
    assert scan.main(["--run", str(run_dir), "--eval-seed", str(seed)]) == 2
    assert f"--eval-seed must be in [0, 2**32), got {seed}" in capsys.readouterr().err
    assert not (run_dir / scan.OUT_NAME).exists()


def test_a_robot_without_a_suite_is_refused_before_any_build(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(scan, "build_scan_env", _no_build)
    run_dir = _write_run(tmp_path, robot="asimov_v1")
    assert scan.main(["--run", str(run_dir)]) == 2
    assert "no terrain suite for asimov_v1" in capsys.readouterr().err
    assert not (run_dir / scan.OUT_NAME).exists()


def test_the_full_suite_on_jax_is_refused_before_any_build(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(scan, "build_scan_env", _no_build)
    run_dir = _write_run(tmp_path)
    assert scan.main(["--run", str(run_dir), "--backend", "jax"]) == 2
    err = capsys.readouterr().err
    assert "1139 ground boxes" in err and "warp" in err


def test_unknown_cells_are_refused(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(scan, "build_scan_env", _no_build)
    run_dir = _write_run(tmp_path)
    assert scan.main(["--run", str(run_dir), "--cells", "wave_99cm"]) == 2
    assert "unknown cells" in capsys.readouterr().err


def test_run_sh_has_the_terrain_scan_verb_and_does_not_force_cpu():
    text = (paths.REPO_ROOT / "run.sh").read_text()
    line = next(line for line in text.splitlines() if line.strip().startswith("terrain-scan)"))
    assert "humanoid_lab.eval.terrain_scan" in line
    assert "JAX_PLATFORMS" not in line
    assert "terrain-scan" in next(line for line in text.splitlines() if "usage: run.sh" in line)
