"""The terrain scan end to end on a tiny suite, on jax.

The tiny suite holds the CPU arena's two stair cells, one speed, two
headings, one offset and one draw: 4 worlds. Its settle and budget slack
are short, so a run that never falls still ends within 82 control steps.
roboto_origin under deploy_pd. Each run directory holds a random-init
checkpoint (run_fixtures.write_random_run), so the tests check the
plumbing, the counts, the command schedule, the runner's stops and
parking, not a policy. The full suite's arena needs warp.
"""

from __future__ import annotations

import functools
import json
import math
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jp
import mujoco
import numpy as np
import pytest
from run_fixtures import write_random_run

from humanoid_lab import paths
from humanoid_lab.envs import height_scan
from humanoid_lab.envs import terrain_geometry as tg
from humanoid_lab.envs.joystick import default_config as joystick_default_config
from humanoid_lab.eval import terrain_scan as scan
from humanoid_lab.eval import terrain_suite as ts
from humanoid_lab.registry import make_env
from humanoid_lab.terrain import GENERATOR_VERSION, fingerprint, scene
from humanoid_lab.terrain.config import CPU_ARENA, arena_for, params_from_config

ROBOT = "roboto_origin"
ROBOT_DIR = paths.ROBOTS_DIR / ROBOT
PRESET = "deploy_pd"
TERRAIN_OVERRIDES = {
    "terrain": {"arena": CPU_ARENA},
    "obs": {"privileged": [*joystick_default_config().obs.privileged, height_scan.NAME]},
    # roboto_origin's overlay box.
    "command": {"vx": [-0.6, 1.0]},
}
SCHEMA_KEYS = [
    "schema", "suite", "run", "checkpoint", "trained_task", "robot", "preset", "actuator_overrides",
    "action_window", "engine", "protocol", "eval_seed", "cells", "contacts", "nonfinite_runs", "messages",
    "physics_clean", "gate", "warnings", "perf", "ccd_scratch", "provenance", "timestamp",
]
STAIRS = ("pyramid_stairs", "inverted_pyramid_stairs")


def _tiny_suite() -> ts.Suite:
    params = params_from_config(CPU_ARENA)
    return ts.Suite(
        robot=ROBOT,
        version=1,
        arena=params,
        fingerprint=fingerprint(arena_for(params)),
        cells=tuple(c for c in ts.build_cells(params) if c.terrain_type in STAIRS),
        speeds=(0.6,),
        headings=2,
        offsets=(0.03,),
        draws=1,
        settle_steps=10,
        budget_slack=0.5,
    )


TINY = _tiny_suite()
N_WORLDS = TINY.runs_per_cell * len(TINY.cells)


@pytest.fixture(scope="module")
def terrain_run(tmp_path_factory) -> Path:
    env = make_env("terrain", ROBOT_DIR, PRESET, TERRAIN_OVERRIDES)
    run_dir = tmp_path_factory.mktemp("terrain_run")
    write_random_run(env, run_dir, "terrain", TERRAIN_OVERRIDES, preset=PRESET)
    return run_dir


@pytest.fixture(scope="module")
def joystick_run(tmp_path_factory) -> Path:
    env = make_env("joystick", ROBOT_DIR, PRESET, {})
    run_dir = tmp_path_factory.mktemp("joystick_run")
    write_random_run(env, run_dir, "joystick", {}, preset=PRESET)
    return run_dir


@pytest.fixture(scope="module")
def scanned_and_commands(terrain_run) -> tuple[dict, list[np.ndarray]]:
    """The terrain run through the CLI, with the tiny suite standing in for
    its robot's. The policy records the command block of every observation
    it acts on, one (N_WORLDS, 3) array per control step."""
    build = scan.build_scan_env
    commands = []

    def recording_build(*args, **kwargs):
        run, env, ckpt, inference = build(*args, **kwargs)
        block = env.obs_slices("state")["command"]

        def recorded(obs, key):
            jax.debug.callback(lambda c: commands.append(np.asarray(c)), obs["state"][:, block], ordered=True)
            return inference(obs, key)

        return run, env, ckpt, recorded

    with pytest.MonkeyPatch.context() as mp:
        mp.setitem(ts.SUITES, ROBOT, TINY)
        mp.setattr(scan, "build_scan_env", recording_build)
        assert scan.main(["--run", str(terrain_run), "--backend", "jax"]) == 0
    return json.loads((terrain_run / scan.OUT_NAME).read_text()), commands


@pytest.fixture(scope="module")
def scanned(scanned_and_commands) -> dict:
    return scanned_and_commands[0]


def check_counts(result: dict) -> None:
    """Every (cell, speed) entry's counts lie in their bounds, and the perf
    totals match the dispatches."""
    runs = TINY.runs_per_cell
    assert list(result["cells"]) == [c.name for c in TINY.cells]
    deadline = 0
    for cell in TINY.cells:
        entry = result["cells"][cell.name]
        r_out = ts.r_out(1.6, 4.0, TINY.footprint_reach)
        assert entry["r_out"] == pytest.approx(r_out)
        deadline = max(deadline, *ts.run_deadlines(TINY, r_out, 0.6, result["protocol"]["ctrl_dt"]))
        assert (entry["threshold"], entry["provenance"]) == (None, "tracked")
        assert list(scan.speed_entries(entry)) == ["0.6"]
        r = entry["0.6"]
        assert r["of"] == runs
        # No random-init run crosses: 1.72 m in 72 measured steps is 1.2 m/s.
        assert r["passed"] == 0
        assert 0 <= r["falls_in_settle"] <= r["falls"]
        assert 0 <= r["measured"] <= runs
        assert 0.0 <= r["ci95"][0] <= r["rate"] <= r["ci95"][1] <= 1.0
        assert 1 <= r["steps_max"] <= deadline
        for key in ("progress_mean", "track_err", "saturation", "clearance"):
            assert r[key] is not None and math.isfinite(r[key])
        assert 0.0 <= r["progress_mean"] <= 1.0
        assert 0.0 <= r["saturation"] <= 1.0
    iterations = result["perf"]["iterations"]["0.6"]
    assert max(e["0.6"]["steps_max"] for e in result["cells"].values()) == iterations
    assert result["perf"]["env_steps"] == iterations * N_WORLDS
    assert result["nonfinite_runs"] == {"0.6": 0}


def test_the_record_has_its_schema_and_its_counts_add_up(scanned):
    result = scanned
    assert list(result) == SCHEMA_KEYS
    assert result["suite"] == {
        "robot": ROBOT,
        "version": 1,
        "fingerprint": TINY.fingerprint,
        "generator_version": GENERATOR_VERSION,
    }
    assert (result["trained_task"], result["robot"], result["preset"]) == ("terrain", ROBOT, PRESET)
    assert result["checkpoint"] == "000000001024"
    assert result["engine"]["backend"] == "jax"
    assert set(result["engine"]) == {"backend", *scan.ENGINE_VERSIONS}
    assert result["protocol"]["runs_per_cell_speed"] == 2
    assert result["protocol"]["ctrl_dt"] == pytest.approx(0.02)
    check_counts(result)
    # jax has no live counters and prints nothing.
    contacts = result["contacts"]["0.6"]
    assert (contacts["backend"], contacts["num_envs"], contacts["pool"]) == ("jax", 4, 256 * 4)
    assert contacts["nacon_pool_max"] is None and contacts["fill_pool"] is None
    assert result["messages"] is None
    assert result["physics_clean"] is True
    # Every cell is tracked, so a full scan of the tiny suite gates nothing.
    assert result["gate"]["absolute"] == {"verdict": "pass", "checked": 0, "expected": 0, "failures": []}
    assert any("jax backend" in w for w in result["warnings"])
    assert result["ccd_scratch"]["slots"] == 256 * N_WORLDS
    assert result["ccd_scratch"]["bytes_per_slot"] == 5480
    # deploy_pd is a position servo, so every joint has a target window.
    window = result["action_window"]
    assert window and all(lo <= hi for lo, hi in window.values())
    assert result["provenance"]["git_commit"] is not None
    assert result["provenance"]["started_at"] <= result["timestamp"]


def test_the_command_is_zero_through_the_settle_then_v_along_x(scanned_and_commands):
    """The policy acts on the last step's observation, so it reads the walk
    command one control step after the settle. The command channel carries
    no observation noise. Stopped worlds are still commanded."""
    result, commands = scanned_and_commands
    settle = TINY.settle_steps
    assert len(commands) == result["perf"]["iterations"]["0.6"] > settle + 1
    for block in commands[: settle + 1]:
        assert block.shape == (N_WORLDS, 3) and not block.any()
    walk = np.tile([0.6, 0.0, 0.0], (N_WORLDS, 1))
    for block in commands[settle + 1 :]:
        np.testing.assert_allclose(block, walk, atol=1e-6)


def test_a_joystick_run_can_be_scanned(joystick_run):
    """A joystick run rebuilds on the terrain task with the suite's arena.
    It keeps its flat critic."""
    result = scan.scan(joystick_run, suite=TINY, backend="jax")
    assert result["trained_task"] == "joystick"
    check_counts(result)
    assert result["physics_clean"] is True


# Every runner call below has the budget STEPS and the same shapes, so they
# share one compile.
STEPS = TINY.settle_steps + 2


@pytest.fixture(scope="module")
def batch(terrain_run) -> SimpleNamespace:
    """The terrain run's scan env, spawn table, reset keys, reset states and
    runner on the tiny suite, and `roll(table, deadline)`, one runner call.

    jax has no live counters, so the runner is traced with scripted ones:
    after control step k, nefc STEPS - k, nacon 2k and ncollision 3k.
    `first` is the call that traces it, with every deadline at STEPS."""
    budgets = scan.scan_budgets(TINY)
    run, env, _ckpt, inference = scan.build_scan_env(terrain_run, TINY, N_WORLDS, "jax", budgets)
    scan.check_arena(env._arena, TINY)
    table = scan.spawn_table(env._arena.spec, TINY, TINY.cells, env.dt, TINY.speeds)
    keys = scan.batch_reset_keys(jp.stack([scan.cell_key(c) for c in TINY.cells]), TINY.runs_per_cell)
    reset = jax.jit(jax.vmap(functools.partial(scan.scan_reset, env)))
    state = reset(
        keys, jp.asarray(table["start"]), jp.asarray(table["yaw"]), jp.asarray(table["centre"]),
        jp.asarray(table["row"]), jp.asarray(table["type"]),
    )
    runner = scan.make_batch_runner(env, TINY, inference)

    def roll(t=table, deadline=STEPS):
        d = jp.asarray(np.broadcast_to(np.asarray(deadline, np.int32), (N_WORLDS,)))
        return jax.tree.map(np.asarray, runner(keys, scan._device_table(t), 0.6, d, budget=STEPS))

    def counters(data):
        k = jp.round(jp.max(data.time) / env.dt).astype(jp.int32)
        return STEPS - k, 2 * k, 3 * k

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(scan.sim_budget, "traced_counters", counters)
        first = roll()
    return SimpleNamespace(run=run, env=env, table=table, state=state, roll=roll, first=first)


def test_runs_spawn_on_the_pad_facing_their_heading(batch):
    env, table, state = batch.env, batch.table, batch.state
    b = env._base_qadr
    qpos = np.asarray(state.data.qpos)
    reset_qpos = np.asarray(env._reset_qpos)
    joints = np.asarray(env._qadr)
    command = env.obs_slices("state")["command"]
    for w in range(N_WORLDS):
        cell = TINY.cells[int(table["cell"][w])]
        tile = ts.cell_tile(env._arena.spec, cell)
        # On the pad at the run's start: the pad height plus the reset height.
        np.testing.assert_allclose(qpos[w, b : b + 2], table["start"][w], atol=1e-6)
        assert qpos[w, b + 2] == pytest.approx(env._z0 + tile.pad_height, abs=1e-5)
        # Facing the heading: the yaw composed onto the reset quaternion.
        quat = tg.quat_mul(tg.yaw_quat(jp.asarray(table["yaw"][w])), env._reset_quat)
        np.testing.assert_allclose(qpos[w, b + 3 : b + 7], np.asarray(quat), atol=1e-6)
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, qpos[w, b + 3 : b + 7].astype(float))
        forward = R.reshape(3, 3)[:2, 0]
        assert math.atan2(forward[1], forward[0]) == pytest.approx(
            math.remainder(float(table["yaw"][w]), 2 * math.pi), abs=0.02
        )
        # The reset pose, with no joint noise, at rest.
        np.testing.assert_array_equal(qpos[w, joints], reset_qpos[joints])
    assert not np.asarray(state.data.qvel).any()
    assert not np.asarray(state.info["command"]).any()
    assert not np.asarray(state.obs["state"])[:, command].any()
    # The two headings of a cell face opposite ways along x.
    np.testing.assert_allclose(table["yaw"][:2], [0.0, math.pi], atol=1e-6)
    # Both feet rest on the pad.
    clearance = np.asarray(jax.vmap(env._foot_clearance)(state.data))
    assert np.abs(clearance).max() < 0.005
    np.testing.assert_array_equal(np.asarray(state.info["tile_origin"]), table["centre"])
    assert batch.run["task"] == "terrain"


def test_every_stopped_world_parks_clear_of_the_ground(batch):
    """Every deadline lies inside the budget, so the last step parks every
    world. A parked robot's bounding boxes clear every ground geom's, so it
    adds no broadphase pair and no contact."""
    env, out = batch.env, batch.first
    m = env.mj_model
    b = env._base_qadr
    reset_qpos = np.asarray(env._reset_qpos)
    final = out["final"]["qpos"]
    base_z = scan.arena_top(env._arena) + scan.PARK_LIFT + env._z0
    np.testing.assert_allclose(final, scan.park_pose(reset_qpos, b, batch.table["start"], base_z), atol=1e-6)
    assert not out["final"]["qvel"].any()

    ground = scene.ground_geom_ids(m)
    pairs = (m.geom_contype & np.bitwise_or.reduce(m.geom_conaffinity[ground])) | (
        m.geom_conaffinity & np.bitwise_or.reduce(m.geom_contype[ground])
    )
    # The robot's colliders that can pair with a ground geom.
    robot = np.flatnonzero((m.geom_bodyid != 0) & (pairs != 0))
    assert robot.size
    pad = m.geom_margin + m.geom_gap
    data = mujoco.MjData(m)

    def ground_contacts(q) -> int:
        """Ground contacts at `q`. Leaves `data` at `q` for box_z."""
        data.qpos[:] = q
        mujoco.mj_forward(m, data)
        return sum(1 for c in data.contact[: data.ncon] if {int(c.geom1), int(c.geom2)} & set(ground.tolist()))

    def box_z(g) -> tuple[float, float]:
        """The lowest and highest world z of geom g's bounding box."""
        R = data.geom_xmat[g].reshape(3, 3)
        centre = data.geom_xpos[g, 2] + (R @ m.geom_aabb[g, :3])[2]
        half = np.abs(R[2]) @ m.geom_aabb[g, 3:]
        return centre - half, centre + half

    # The spawn's soles sit a few mm above the pad. 1 cm lower, they touch it.
    sunk = np.asarray(batch.state.data.qpos)[0].copy()
    sunk[b + 2] -= 0.01
    assert ground_contacts(sunk) > 0
    for w in range(N_WORLDS):
        # Parked, no world touches the heightfield, an arena box or an apron.
        assert ground_contacts(final[w]) == 0
        # Each side's box widens by its own margin and gap.
        top = max(box_z(g)[1] + pad[g] for g in ground)
        low = min(box_z(g)[0] - pad[g] for g in robot)
        assert low > top


def test_the_runner_keeps_each_counters_peak(batch):
    """The scripted counters peak at different steps, so a swapped or
    unfolded counter reads another value."""
    out = batch.first
    assert int(out["iterations"]) == STEPS
    assert (int(out["nefc"]), int(out["nacon"]), int(out["ncollision"])) == (STEPS - 1, 2 * STEPS, 3 * STEPS)


def test_the_loop_runs_until_its_last_run_stops(batch):
    """World 0's deadline is a step before the others'. Each run stops at
    its fall, its finish or its own deadline."""
    deadline = np.full(N_WORLDS, STEPS, np.int32)
    deadline[0] = STEPS - 1
    out = batch.roll(deadline=deadline)
    runs = out["runs"]
    early = runs["fell"] | runs["finished"]
    assert np.all(early | (runs["steps"] == deadline))
    assert np.all(runs["steps"] <= deadline)
    # A run standing to the later deadline tells the loop's end from the first stop.
    assert np.any(runs["steps"] == STEPS)
    assert int(out["iterations"]) == runs["steps"].max()


def test_a_run_with_no_measured_step_reports_zero_metrics(batch):
    """Every deadline at the settle's end: no run reaches a measured step,
    so no settle step reaches a metric."""
    out = batch.roll(deadline=TINY.settle_steps)
    runs = out["runs"]
    assert int(out["iterations"]) == runs["steps"].max() <= TINY.settle_steps
    assert not runs["measured"].any() and not runs["finished"].any()
    for key in ("progress", "saturation", "track_err", "clearance"):
        assert not runs[key].any(), key


def test_a_run_finishes_on_its_first_measured_step_past_r_out(batch):
    """r_out 0 and d0 10 for every world: any base distance reaches r_out,
    so every run standing after the settle finishes on its first measured
    step. Its progress is 1 - distance / 10."""
    table = {
        **batch.table,
        "r_out": np.zeros(N_WORLDS, np.float32),
        "d0": np.full(N_WORLDS, 10.0, np.float32),
    }
    runs = batch.roll(table)["runs"]
    standing = ~runs["fell_in_settle"]
    assert standing.any()
    assert runs["finished"][standing].all()
    assert (runs["steps"][standing] == TINY.settle_steps + 1).all()
    assert (runs["measured"][standing] == 1).all()
    assert ((runs["progress"][standing] > 0.99) & (runs["progress"][standing] <= 1.0)).all()


def test_a_non_finite_world_is_flagged_alone(batch):
    """World 0 starts at a NaN xy. The other worlds roll on, finite."""
    table = {**batch.table, "start": batch.table["start"].copy()}
    table["start"][0] = np.nan
    out = batch.roll(table)
    assert out["runs"]["nonfinite"].tolist() == [True, False, False, False]
    final = out["final"]["qpos"]
    assert not np.isfinite(final[0]).all()
    assert np.isfinite(final[1:]).all()
