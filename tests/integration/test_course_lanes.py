"""Course lanes on a real model: eval/courses/lane.py.

roboto_origin under deploy_pd. The run is a random-init checkpoint written
by brax's own checkpoint code next to a run.json (run_fixtures), loaded
through battery.load_checkpoint_policy as a measured run is. A random policy
falls within a second, so the shared env loosens the fall test until it
cannot trip, and every lane runs to its short budget. A course fall comes
from a wrapper whose `done` trips at a fixed step count. With the fall test
off, the random policy's follower progress on straight_push at seed 5 climbs
to 0.62 m in 60 course steps. The push test places one kick at 0 m, where it
lands on the first course step. It places another at a threshold reached
mid-lane, and a third at one never reached.

One env and one compiled lane serve most tests. The record length and the
padded path length are the flat class's (lane_shapes), as for a measured
run. Compiles are what cost time here, about 10 s each.
"""

from __future__ import annotations

import dataclasses
import math
import threading

import jax
import jax.numpy as jp
import numpy as np
import pytest
from run_fixtures import write_random_run

from humanoid_lab import paths
from humanoid_lab.eval import battery
from humanoid_lab.eval.courses import families, follower, lane, scoring, spec
from humanoid_lab.registry import make_env

ROBOT = "roboto_origin"
PRESET = "deploy_pd"
# Neither height nor tilt can trip the fall test.
NO_FALL = {"fall": {"min_height": -1.0, "max_tilt_gz": 2.0}}
# The base starts below this, so the fall test trips on the first step.
FALLS_AT_ONCE = {"fall": {"min_height": 2.0}}
# The stub on_stop hooks mark info["step_count"] by adding this. A state
# that went through on_stop once reads its own step count plus this.
MARK = 1_000_000
# ctrl_dt on both robots.
DT = 0.02


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    env = make_env("joystick", paths.ROBOTS_DIR / ROBOT, PRESET, {})
    out = tmp_path_factory.mktemp("random_run")
    write_random_run(env, out, "joystick", {}, preset=PRESET)
    return out


@pytest.fixture(scope="module")
def params():
    return spec.params_for(ROBOT)


@pytest.fixture(scope="module")
def loaded(run_dir):
    """(env, policy) of the random run, with the fall test off."""
    _, env, _, inf = battery.load_checkpoint_policy(run_dir, NO_FALL)
    return env, inf


@pytest.fixture(scope="module")
def env(loaded):
    return loaded[0]


@pytest.fixture(scope="module")
def shapes(env, params):
    return lane.lane_shapes(params, env.dt)


@pytest.fixture(scope="module")
def catalogue(params):
    return families.catalogue(params)


@pytest.fixture(scope="module")
def make_data(env, params, shapes, catalogue):
    def make(name, budget, **fields):
        data = lane.lane_data(catalogue[name], params, shapes[1], env.dt)
        return data._replace(budget=jp.asarray(budget, jp.int32), **fields)

    return make


def fresh_lane(env, inf, shapes, **hooks):
    return lane.make_lane_fn(
        env, inf, max_steps=shapes[0], settle_steps=battery.settle_steps(env.dt), **hooks
    )


@pytest.fixture(scope="module")
def compiled(loaded, shapes, make_data):
    env, inf = loaded
    return lane.compile_lane(fresh_lane(env, inf, shapes), make_data("straight_10m", 1))


def key(seed):
    return jax.random.PRNGKey(seed)


def host(out):
    return jax.device_get(out)


def assert_same_lane(a, b):
    assert int(a.steps) == int(b.steps)
    for flag in ("nonfinite", "settle_fell", "fell", "reached"):
        assert bool(getattr(a, flag)) == bool(getattr(b, flag)), flag
    n = int(a.steps)
    assert set(a.rec) == set(b.rec)
    for name in a.rec:
        assert np.array_equal(np.asarray(a.rec[name])[:n], np.asarray(b.rec[name])[:n]), name


def slip(out):
    n = int(out.steps)
    return float(np.sum(np.asarray(out.rec["foot_speed"])[:n] * np.asarray(out.rec["contact"])[:n]))


def mark_stop(state, data):
    info = {**state.info, "step_count": state.info["step_count"] + MARK}
    return state.replace(info=info)


class HeightFromTime:
    """`env` with its own `_base_height`: 10 m plus the sim time, a value no
    physical base height reaches. Everything else is the wrapped env's.

    d.time starts at 0 at the reset and advances env.dt per control step,
    which is n_substeps physics steps of sim_dt."""

    def __init__(self, env):
        self._env = env

    def __getattr__(self, name):
        return getattr(self._env, name)

    def _base_height(self, d):
        return 10.0 + d.time


class FallsAt:
    """`env` whose `done` trips once info["step_count"] reaches `at`.
    Everything else is the wrapped env's."""

    def __init__(self, env, at):
        self._env, self._at = env, at

    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, state, action):
        nst = self._env.step(state, action)
        fall = nst.info["step_count"] >= self._at
        return nst.replace(done=jp.where(fall, jp.ones_like(nst.done), nst.done))


# -- the model and its constants ---------------------------------------------------


@pytest.mark.parametrize("robot", ["roboto_origin", "asimov_v1"])
def test_pinned_robot_inputs_match_the_models(robot):
    model_env = make_env("joystick", paths.ROBOTS_DIR / robot, PRESET, {})
    measured = lane.measure_robot_inputs(model_env)
    pinned = spec.ROBOT_INPUTS[robot]
    assert measured["stance_halfwidth_m"] == pytest.approx(pinned.stance_halfwidth_m, abs=5e-4)
    assert measured["nominal_height_m"] == pytest.approx(pinned.nominal_height_m, abs=5e-4)


def test_friction_geoms_are_the_world_geoms_plus_every_foot_geom(env):
    m = env.mj_model
    ids = lane.friction_geoms(env)
    assert list(ids) == sorted(set(ids.tolist()))
    assert {m.geom(int(i)).name for i in ids} == {"ground", *env.robot_spec.foot_geoms}
    world = [i for i in range(m.ngeom)
             if m.geom_bodyid[i] == 0 and (m.geom_contype[i] or m.geom_conaffinity[i])]
    assert set(world) <= set(ids.tolist())
    # The nominal rows' effective friction: the max rule over equal values.
    assert np.all(m.geom_friction[ids, 0] == 0.9)


def test_the_flat_class_fixes_the_lane_shapes(shapes, catalogue, env):
    # straight_slow's budget and figure_eight_r15's 1001 densified points
    # plus the padding. lane_shapes takes no row list. It reads the whole
    # class catalogue, so a row subset cannot change the shapes.
    assert shapes == (6350, 1001 + follower.PAD_PTS)
    assert shapes[0] == max(spec.budget_steps(c, env.dt) for c in catalogue.values())


@pytest.mark.parametrize("robot", ["roboto_origin", "asimov_v1"])
def test_lane_data_carries_each_rows_own_fields(robot):
    # Model-free: no env and no compile. Floats compare as float32, the
    # lane's dtype. float(np.float32(2.5 * 2 * pi)) is not 2.5 * 2 * pi.
    p = spec.params_for(robot)
    n = lane.lane_shapes(p, DT)[1]
    cat = families.catalogue(p)
    for name, c in cat.items():
        d = jax.device_get(lane.lane_data(c, p, n, DT))
        assert int(d.budget) == spec.budget_steps(c, DT), name
        assert int(d.kind) == spec.kind_id(c), name
        assert bool(d.anchor_world) is False, name
        assert np.all(d.origin == 0.0), name
        assert d.yaw_cap == np.float32(p.yaw_cap), name
        for leaf in (d.path.pts, d.path.tan, d.path.speed, d.path.cum):
            assert leaf.shape[0] == n, name
        if isinstance(c, spec.SpinCourse):
            # A right spin's goal is a magnitude: progress counts in the
            # spin's own direction.
            assert c.turn_rad > 0, name
            assert d.spin_wz == np.float32(c.wz), name
            assert d.spin_required == np.float32(c.turn_rad), name
            assert d.push_at_m == np.float32(math.inf) and d.push_vel == 0.0, name
        else:
            assert d.spin_wz == 0.0 and d.spin_required == 0.0, name
            push_at = math.inf if c.push_at_m is None else c.push_at_m
            assert d.push_at_m == np.float32(push_at), name
            assert d.push_vel == np.float32(c.push_vel or 0.0), name

    pushed = {name for name, c in cat.items() if getattr(c, "push_at_m", None) is not None}
    assert pushed == {"straight_push", "straight_push_fast"}
    for name in pushed:
        assert lane.lane_data(cat[name], p, n, DT).push_at_m == np.float32(spec.PUSH_AT_M)


# -- friction ------------------------------------------------------------------------


def test_slippery_lane_differs_and_the_restored_lane_reproduces_nominal(
        loaded, shapes, compiled, make_data):
    env, inf = loaded
    data = make_data("straight_10m", 300)
    ids = lane.friction_geoms(env)
    base = env.mj_model.geom_friction[ids, 0].copy()
    nominal = host(compiled(data, key(0)))
    try:
        lane.set_friction(env, ids, spec.SLIPPERY_MU)
        assert np.all(env.mj_model.geom_friction[ids, 0] == spec.SLIPPERY_MU)
        slippery = host(lane.compile_lane(fresh_lane(env, inf, shapes), data)(data, key(0)))
        # The program compiled before the swap keeps the friction it was
        # traced with.
        stale = host(compiled(data, key(0)))
    finally:
        lane.set_friction(env, ids, base)
    restored = host(lane.compile_lane(fresh_lane(env, inf, shapes), data)(data, key(0)))

    assert np.array_equal(env.mj_model.geom_friction[ids, 0], base)
    assert int(nominal.steps) == int(slippery.steps) == 300
    assert not np.array_equal(nominal.rec["foot_speed"][:300], slippery.rec["foot_speed"][:300])
    # Planted feet slide further on the slippery floor.
    assert slip(slippery) > slip(nominal)
    assert_same_lane(stale, nominal)
    assert_same_lane(restored, nominal)


# -- one program, any pool ---------------------------------------------------------


def test_one_compile_serves_path_and_spin_lanes(compiled, make_data, catalogue, env, shapes):
    def avals(data):
        return jax.tree_util.tree_map(lambda x: (x.shape, x.dtype), data)

    ref = avals(make_data("straight_10m", 1))
    for name in catalogue:
        assert avals(make_data(name, 1)) == ref, name
    # The compiled program takes these avals only. Another shape or dtype
    # raises before anything runs, so a recompile cannot happen silently.
    wider = make_data("straight_10m", 1,
                      path=follower.pack_path(catalogue["straight_10m"], shapes[1] + 1))
    with pytest.raises(TypeError, match="Argument types differ"):
        compiled(wider, key(0))
    with pytest.raises(TypeError, match="Argument types differ"):
        compiled(make_data("straight_10m", 1)._replace(budget=jp.float32(1)), key(0))

    path = host(compiled(make_data("circle_r2", 60), key(1)))
    spin = host(compiled(make_data("spin_right", 60), key(1)))
    for out in (path, spin):
        assert int(out.steps) == 60
        assert scoring.seed_out(out, 1)["outcome"] == "timed_out"
    assert np.all(path.rec["cmd"][:60, 1] == 0.0)
    assert path.rec["cmd"][0, 0] > 0.0
    want = np.asarray([0.0, 0.0, catalogue["spin_right"].wz], dtype=np.float32)
    assert np.all(spin.rec["cmd"][:60] == want)
    assert float(path.yaw_rad) == 0.0

    # A start row is laid out at the post-settle pose, which is the pre-step
    # pose of course step 0. That pose is not the world origin.
    np.testing.assert_array_equal(path.anchor[:2], [path.rec["x"][0], path.rec["y"][0]])
    assert float(path.anchor[2]) == pytest.approx(float(path.rec["yaw"][0]), abs=1e-6)
    assert np.any(path.anchor[:2] != 0.0)

    # The world-yaw change over the 60 steps, from the recorded pre-step yaw
    # and the post-step free-joint quaternion.
    b = env._base_qadr
    post = np.asarray(follower.quat_to_yaw(spin.rec["qpos"][:60, b + 3:b + 7]))
    turned = float(np.sum(np.asarray(follower.wrap_angle(post - spin.rec["yaw"][:60]))))
    assert turned != 0.0
    # Progress counts in the command's direction: a right spin (wz < 0)
    # that turned left reads negative.
    wz_sign = float(np.sign(catalogue["spin_right"].wz))
    assert float(spin.yaw_rad) == pytest.approx(wz_sign * turned, abs=1e-3)


def test_a_lane_alone_equals_the_same_lane_in_a_pool(compiled, make_data):
    jobs = [
        lane.LaneJob(make_data("circle_r2", 80), key(0), "circle_r2"),
        lane.LaneJob(make_data("square_3m", 120), key(1), "square_3m"),
        lane.LaneJob(make_data("spin_left", 60), key(2), "spin_left"),
        lane.LaneJob(make_data("straight_push", 100, push_at_m=jp.float32(0.0)), key(3), "push"),
    ]
    alone = host(compiled(jobs[1].data, jobs[1].key))

    started, lock = [], threading.Lock()

    def logged(data, k):
        with lock:
            started.append(int(data.budget))
        return compiled(data, k)

    def finish(job, out):
        return job.tag, host(out)

    one = lane.run_lanes(logged, jobs, workers=1, finish=finish)
    assert started == [120, 100, 80, 60]
    pooled = lane.run_lanes(compiled, jobs, workers=3, finish=finish)
    assert [tag for tag, _ in pooled] == [job.tag for job in jobs]
    assert_same_lane(pooled[1][1], alone)
    for (_, a), (_, b) in zip(one, pooled):
        assert_same_lane(a, b)


# -- the follower, the kick and the stop ----------------------------------------------


def test_in_lane_follower_matches_an_eager_replay_of_the_recorded_poses(
        compiled, make_data, params):
    # Laid out at a world pose 1.2 rad off the robot's heading, so the
    # follower opens in its spin branch.
    origin = jp.asarray([0.0, 0.0, 1.2], jp.float32)
    data = make_data("slalom_05m", 150, anchor_world=jp.asarray(True), origin=origin)
    out = host(compiled(data, key(4)))
    assert np.array_equal(out.anchor, np.asarray(origin))
    n = int(out.steps)
    assert n == 150
    assert bool(out.rec["spinning"][0])

    path = follower.layout(data.path, *out.anchor)
    state = follower.initial_state()
    spun = 0
    for k in range(n):
        fo = follower.follower_step(path, state, out.rec["x"][k], out.rec["y"][k],
                                    out.rec["yaw"][k], params.yaw_cap)
        np.testing.assert_allclose(np.asarray(fo.cmd), out.rec["cmd"][k], atol=1e-5)
        assert float(fo.xte) == pytest.approx(float(out.rec["xte"][k]), abs=1e-5)
        assert float(fo.s) == float(out.rec["s"][k])
        assert bool(fo.state.spinning) == bool(out.rec["spinning"][k])
        spun += bool(fo.state.spinning)
        state = fo.state
    assert spun > 0


def test_push_lands_once_on_the_first_step_and_to_the_left(compiled, make_data, env, params):
    pushed_data = make_data("straight_push", 60, push_at_m=jp.float32(0.0))
    pushed = host(compiled(pushed_data, key(5)))
    twin = host(compiled(pushed_data._replace(push_at_m=jp.float32(math.inf)), key(5)))
    assert int(pushed.steps) == int(twin.steps) == 60

    kicks = np.flatnonzero(np.any(pushed.rec["kick"][:60] != 0.0, axis=1))
    assert kicks.tolist() == [0]
    assert not np.any(twin.rec["kick"][:60])
    # Both lanes reach the course start in the same state.
    for name in ("x", "y", "yaw"):
        assert pushed.rec[name][0] == twin.rec[name][0]

    yaw = float(pushed.rec["yaw"][0])
    left = np.asarray([-math.sin(yaw), math.cos(yaw)])
    np.testing.assert_allclose(pushed.rec["kick"][0], params.push_vel * left, rtol=1e-6, atol=1e-7)
    # World velocity over the kick step, pushed minus twin, from the
    # post-step base positions.
    b = env._base_qadr
    dv = (pushed.rec["qpos"][0, b:b + 2] - twin.rec["qpos"][0, b:b + 2]) / env.dt
    assert float(dv @ left) == pytest.approx(params.push_vel, rel=0.2)
    assert abs(float(dv @ np.asarray([math.cos(yaw), math.sin(yaw)]))) < 0.1 * params.push_vel

    # A threshold reached mid-lane: the kick lands on the first step whose
    # recorded progress reaches it, and every earlier row is the twin's.
    s = np.asarray(twin.rec["s"][:60])
    k = int(np.argmax(s >= s.max() / 2))
    assert k > 0
    mid = host(compiled(pushed_data._replace(push_at_m=jp.float32(s[k])), key(5)))
    assert np.flatnonzero(np.any(mid.rec["kick"][:60] != 0.0, axis=1)).tolist() == [k]
    for name in mid.rec:
        assert np.array_equal(mid.rec[name][:k], twin.rec[name][:k]), name
    # A threshold the lane never reaches: no kick, and the lane is the twin.
    never = host(compiled(pushed_data._replace(push_at_m=jp.float32(s.max() + 0.02)), key(5)))
    assert not np.any(never.rec["kick"][:60])
    assert_same_lane(never, twin)


def test_settle_fall_is_reported_and_on_stop_runs_once(run_dir, params, shapes, make_data):
    _, env, _, inf = battery.load_checkpoint_policy(run_dir, FALLS_AT_ONCE)
    settle = battery.settle_steps(env.dt)
    assert env.n_substeps * env.sim_dt == pytest.approx(env.dt)
    data = make_data("straight_10m", 100)
    fn = fresh_lane(HeightFromTime(env), inf, shapes, on_stop=mark_stop)
    out = host(lane.compile_lane(fn, data)(data, key(0)))

    assert bool(out.settle_fell) and bool(out.fell)
    assert int(out.steps) == 0
    # The settle height reads the env's own _base_height, averaged over the
    # settle's second half: steps settle // 2 + 1 to settle.
    want = 10.0 + env.dt * np.mean(np.arange(settle // 2 + 1, settle + 1))
    assert float(out.settle_height) == pytest.approx(want, rel=1e-4)
    seed = scoring.seed_out(out, 0)
    assert (seed["outcome"], seed["steps"], seed["fell_at"]) == ("settle_fell", 0, 0)
    # on_stop ran on the settled state, once: the settle's steps plus the mark.
    assert int(out.state.info["step_count"]) == settle + MARK

    course = families.catalogue(params)["straight_10m"]
    res = scoring.seed_result(out.rec, seed, env.dt, course, params)
    assert (res["score"], res["subscores"], res["fell_at"]) == (0.0, None, 0)


def test_course_fall_stops_the_lane_at_the_step_that_tripped_done(
        loaded, shapes, compiled, make_data, params):
    env, inf = loaded
    settle = battery.settle_steps(env.dt)
    data = make_data("straight_10m", 100)
    # `done` trips on course step 24, the 25th after the settle.
    fn = fresh_lane(FallsAt(env, settle + 25), inf, shapes, on_stop=mark_stop)
    out = host(lane.compile_lane(fn, data)(data, key(0)))
    full = host(compiled(data, key(0)))

    assert not bool(out.settle_fell) and bool(out.fell)
    assert int(out.steps) == 25 and int(full.steps) == 100
    # on_stop ran once, on the state that fell.
    assert int(out.state.info["step_count"]) == settle + 25 + MARK
    # The stop cuts the trajectory short without changing it: the falling
    # step is the last recorded row, equal to the same step of a lane that
    # cannot fall.
    assert set(out.rec) == set(full.rec)
    for name in out.rec:
        assert np.array_equal(out.rec[name][:25], full.rec[name][:25]), name

    seed = scoring.seed_out(out, 0)
    assert (seed["outcome"], seed["steps"], seed["fell_at"]) == ("fell", 25, 24)
    course = families.catalogue(params)["straight_10m"]
    res = scoring.seed_result(out.rec, seed, env.dt, course, params)
    assert (res["outcome"], res["fell_at"], res["score"]) == ("fell", 24, 0.0)


def test_nonfinite_lane_stops_and_counts_finite_steps_only(
        compiled, make_data, env, params, tmp_path):
    data = make_data("straight_push", 100, push_at_m=jp.float32(0.0),
                     push_vel=jp.float32(math.nan))
    out = host(compiled(data, key(0)))

    assert bool(out.nonfinite)
    assert int(out.steps) == 0
    # The non-finite step was written, at index `steps`, and not counted.
    # Its yaw change is not counted either.
    assert not np.all(np.isfinite(out.rec["qpos"][0]))
    assert float(out.yaw_rad) == 0.0
    seed = scoring.seed_out(out, 0)
    assert (seed["outcome"], seed["fell_at"]) == ("nonfinite", None)
    assert math.isfinite(seed["yaw_rad"])
    course = families.catalogue(params)["straight_push"]
    res = scoring.seed_result(out.rec, seed, env.dt, course, params)
    assert (res["score"], res["subscores"]) == (0.0, None)
    # The JSON writer refuses NaN, so the seed's entry must hold none.
    scoring.write_json(tmp_path / "courses.json", {"per_seed": [res]})


def test_a_path_lane_already_at_its_end_completes_before_any_step(
        compiled, make_data, catalogue, params, shapes):
    # The path ends 0.1 m ahead of the settled pose, inside GOAL_RADIUS_M,
    # and progress 0 is already within GOAL_MIN_PROGRESS_M of the end. The
    # goal holds on the course's first pose.
    short = dataclasses.replace(
        catalogue["straight_10m"], name="straight_0p1m",
        waypoints=np.array([[0.0, 0.0], [0.1, 0.0]]), speeds=(params.v_nom,),
    )
    assert 0.1 < follower.GOAL_RADIUS_M and 0.1 < follower.GOAL_MIN_PROGRESS_M
    out = host(compiled(make_data("straight_10m", 100, path=follower.pack_path(short, shapes[1])),
                        key(9)))
    assert bool(out.reached)
    assert int(out.steps) == 0
    assert scoring.seed_out(out, 9)["outcome"] == "completed"
    # The catalogue's own path at the same seed and budget runs it out, so
    # the short path's goal is what stopped the lane.
    twin = host(compiled(make_data("straight_10m", 100), key(9)))
    assert int(twin.steps) == 100
    assert scoring.seed_out(twin, 9)["outcome"] == "timed_out"


def test_a_goal_reached_on_the_last_budget_step_completes(compiled, make_data):
    # The goal is tested on the pose after every step, the last budget step
    # included. A spin's goal is yaw progress. The random policy's yaw wanders
    # as it falls, so the target is the progress one of its lanes made,
    # taken from the first seed that ended up ahead.
    spin = make_data("spin_left", 60, spin_required=jp.float32(1e9))
    seed, target = next(
        (s, out.yaw_rad) for s in range(8)
        if float((out := host(compiled(spin, key(s)))).yaw_rad) > 0.1
    )
    spin = spin._replace(spin_required=target)
    first = host(compiled(spin._replace(budget=jp.int32(1000)), key(seed)))
    k = int(first.steps)
    assert 0 < k <= 60
    assert scoring.seed_out(first, seed)["outcome"] == "completed"

    on_time = host(compiled(spin._replace(budget=jp.int32(k)), key(seed)))
    assert int(on_time.steps) == k
    assert scoring.seed_out(on_time, seed)["outcome"] == "completed"
    short = host(compiled(spin._replace(budget=jp.int32(k - 1)), key(seed)))
    assert scoring.seed_out(short, seed)["outcome"] == "timed_out"


# -- under vmap --------------------------------------------------------------------


def test_vmapped_lanes_match_unbatched_outcomes_and_apply_the_hooks(
        loaded, shapes, compiled, make_data):
    env, inf = loaded
    settle = battery.settle_steps(env.dt)

    def reset_fn(k, data):
        # Starts the lane's step count at its budget. step_count only drives
        # the training push schedule, which the measurement overrides turn
        # off. The offset leaves the dynamics unchanged, so steps and
        # outcomes still match the hook-free compiled lane.
        st = env.reset(k)
        return st.replace(info={**st.info, "step_count": st.info["step_count"] + data.budget})

    def observe(state, data):
        n = state.info["step_count"].astype(jp.float32)
        since_reset = n - data.budget.astype(jp.float32)
        return jp.stack([n, -n, jp.where(since_reset <= settle, since_reset, -1.0)])

    # Distinct budgets, so each lane's offset is its own. The NaN lane stops
    # on its first course step whatever its budget is.
    lanes = [
        make_data("circle_r2", 20),
        make_data("spin_left", 50),
        make_data("straight_push", 40, push_at_m=jp.float32(0.0), push_vel=jp.float32(math.nan)),
    ]
    keys = [key(6), key(7), key(8)]
    batch = jax.tree_util.tree_map(lambda *xs: jp.stack(xs), *lanes)
    fn = fresh_lane(env, inf, shapes, reset_fn=reset_fn, on_stop=mark_stop, observe=observe)
    out = host(jax.jit(jax.vmap(fn))(batch, jp.stack(keys)))

    # Trajectories are not compared: batched numerics depend on the batch.
    assert int(out.steps[0]) < int(out.steps[1])
    for i, (data, k) in enumerate(zip(lanes, keys)):
        single = host(compiled(data, k))
        steps = int(single.steps)
        assert int(out.steps[i]) == steps
        got = {f: bool(getattr(out, f)[i]) for f in ("nonfinite", "settle_fell", "fell", "reached")}
        want = {f: bool(getattr(single, f)) for f in got}
        assert got == want
        budget = int(data.budget)
        # The steps the env took: a non-finite step was stepped but not
        # counted.
        stepped = settle + steps + int(single.nonfinite)
        # Each lane started from reset_fn's state, at its own budget. It
        # froze when it stopped, at its budget or on the non-finite step:
        # its state went through on_stop once and was not stepped again
        # while the others ran on.
        assert int(out.state.info["step_count"][i]) == budget + stepped + MARK
        # The step counter's largest value over the settle and the course,
        # and its smallest (the reset state's) negated: a max, not the last
        # value. The smallest is the lane's budget, so observe saw the
        # hook's reset state. The third value counts the settle's steps and
        # is -1 on the course, so its peak shows the settle steps are folded.
        assert out.peaks[i].tolist() == [budget + stepped, -budget, float(settle)]

    nan_lane = jax.tree_util.tree_map(lambda x: x[2], out)
    seed = scoring.seed_out(nan_lane, 8)
    assert (seed["outcome"], seed["steps"]) == ("nonfinite", 0)
