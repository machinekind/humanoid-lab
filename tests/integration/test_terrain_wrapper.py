"""The curriculum auto-reset stack (envs/terrain_wrapper.py) on the CPU
arena.

roboto_origin under deploy_pd, 8 worlds, jax on CPU. The critic lists the
height scan, as configs/task/terrain.yaml does. The wrapper ends episodes
at 200 steps while the env config keeps 1000, the way a smoke run's
`ppo.episode_length` does.

Each test steps a crafted copy of one reset state. An episode ends in one
of two ways here. A knocked-over robot (raised 1 m and pitched 90 degrees)
falls on the tilt check at the next step. A timeout comes from
EpisodeWrapper's step count set one short of the end. `info` sets the
curriculum's inputs: level, terrain type, spawn point, commanded and served
distance, steps lived and the band range. The flat row's crossing reads the
displacement from the spawn point. The fail test reads the served distance.
The arena's tiles are 4 m and its pads 0.4 m. Level 0 is the flat row,
level 1 the top row.
"""

from __future__ import annotations

import contextlib
import functools
import math

import jax
import jax.numpy as jp
import numpy as np
import pytest
from brax.envs.wrappers import training as brax_training
from mujoco_playground import wrapper as playground_wrapper

from humanoid_lab import paths
from humanoid_lab.dr.randomize import make_domain_randomize
from humanoid_lab.envs import curriculum, height_scan, terrain_wrapper, wrappers
from humanoid_lab.envs import terrain_geometry as tg
from humanoid_lab.envs.joystick import default_config as joystick_default_config
from humanoid_lab.envs.terrain_joystick import (
    CURRICULUM_METRICS,
    TELEMETRY_METRICS,
    TELEMETRY_PEAKS,
)
from humanoid_lab.registry import make_env
from humanoid_lab.terrain.config import CPU_ARENA

ROBOT_DIR = paths.ROBOTS_DIR / "roboto_origin"
PRESET = "deploy_pd"
N = 8
EPISODE = 200
CRITIC = [*joystick_default_config().obs.privileged, height_scan.NAME]
PROMOTED, DEMOTED = CURRICULUM_METRICS
# pyramid_stairs: its level-1 pad sits 0.425 m up.
STAIRS = 3


def build(**overrides):
    cfg = {
        "terrain": {"arena": CPU_ARENA},
        "obs": {"privileged": CRITIC},
        "sim": {"num_envs": N},
        **overrides,
    }
    return make_env("terrain", ROBOT_DIR, PRESET, cfg)


@pytest.fixture(scope="module")
def env():
    return build()


def jitted(wrapper):
    return wrapper, jax.jit(wrapper.reset), jax.jit(wrapper.step)


@pytest.fixture(scope="module")
def stack(env):
    return jitted(terrain_wrapper.wrap_for_terrain_brax_training(env, episode_length=EPISODE))


def strong(tree):
    """Every leaf with its weak type dropped. A stepped state carries none,
    so reset-shaped and stepped inputs share one compiled step."""
    return jax.tree.map(lambda x: jp.asarray(x, jp.asarray(x).dtype), tree)


@pytest.fixture(scope="module")
def first(stack):
    _, reset, _ = stack
    return strong(reset(jax.random.split(jax.random.PRNGKey(0), N)))


def zeros(env):
    return jp.zeros((N, env.action_size))


def knock_over(env, state, envs):
    """The robots of `envs` raised 1 m and pitched 90 degrees."""
    b = env._base_qadr
    q = state.data.qpos
    pitch = jp.array([math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0])
    for i in envs:
        q = q.at[i, b + 2].add(1.0).at[i, b + 3 : b + 7].set(tg.quat_mul(pitch, q[i, b + 3 : b + 7]))
    return state.replace(data=state.data.replace(qpos=q))


def time_out(state, envs):
    """EpisodeWrapper's count one step short of the end for `envs`. A done
    flag would zero the count, so theirs is cleared."""
    idx = np.asarray(envs)
    steps = state.info["steps"].at[idx].set(EPISODE - 1)
    return state.replace(info={**state.info, "steps": steps}, done=state.done.at[idx].set(0.0))


def with_info(state, envs, **values):
    """`info` keys set for `envs`. A value is per env or one for all."""
    info = dict(state.info)
    idx = np.asarray(envs)
    for k, v in values.items():
        info[k] = info[k].at[idx].set(jp.asarray(v, info[k].dtype))
    return state.replace(info=info)


def base_xy(env, state):
    b = env._base_qadr
    return np.asarray(state.data.qpos[:, b : b + 2])


def walked_from(env, state, envs, dist):
    """`info` with the spawn point `dist` m behind each env along -x."""
    xy = base_xy(env, state)[np.asarray(envs)]
    return with_info(state, envs, spawn_xy=xy - np.array([dist, 0.0]))


def done_of(state):
    return np.flatnonzero(np.asarray(state.done) > 0).tolist()


def reset_yaw(env, quat):
    """The yaw that turns the reset quaternion into `quat`."""
    q = tg.quat_mul(jp.asarray(quat), jp.asarray(env._reset_quat) * jp.array([1.0, -1.0, -1.0, -1.0]))
    return 2.0 * math.atan2(float(q[3]), float(q[0])), np.asarray(q)


def check_spawn(env, before, after, i):
    """Env i's respawn follows the spawn rule on its new tile."""
    b = env._base_qadr
    t = env._tables
    level, ttype = int(after.info["terrain_level"][i]), int(after.info["terrain_type"][i])
    q = np.asarray(after.data.qpos[i])
    xy = q[b : b + 2]
    origin = np.asarray(t.origin_xy[level, ttype])
    assert np.abs(xy - origin).max() <= env._config.terrain.spawn.pad_jitter + 1e-6
    yaw, rel = reset_yaw(env, q[b + 3 : b + 7])
    np.testing.assert_allclose(rel[1:3], 0.0, atol=1e-6)
    cached = before.info["AutoResetWrapper_first_data"].qpos[i]
    kind = after.info["spawn_kind"][i]
    want = np.asarray(env.spawn_qpos(cached, jp.asarray(xy), jp.float32(yaw), kind))
    np.testing.assert_allclose(q, want, atol=2e-6)
    joints = np.r_[0:b, b + 7 : q.shape[0]]
    np.testing.assert_array_equal(q[joints], np.asarray(cached)[joints])
    np.testing.assert_allclose(q[b + 2], env._z0 + float(t.pad_h[level, ttype]), atol=2.4e-7)
    np.testing.assert_array_equal(np.asarray(after.data.qvel[i]), 0.0)
    info = {k: np.asarray(after.info[k][i]) for k in after.info if not k.startswith(("Auto", "episode"))}
    np.testing.assert_array_equal(info["spawn_xy"], xy)
    np.testing.assert_array_equal(info["last_xy"], xy)
    np.testing.assert_array_equal(info["tile_origin"], origin)
    r0 = np.abs(xy - origin).max()
    np.testing.assert_allclose([info["cheby_min"], info["cheby_max"]], r0, atol=1e-6)
    assert int(info["since_spawn"]) == 0 and float(info["commanded_dist"]) == 0.0
    assert float(info["served_dist"]) == 0.0
    assert info["curriculum_free"] == before.info["curriculum_free"][i]
    assert int(info["spawn_kind"]) == tg.SPAWN_PAD


def test_done_envs_teleport_with_the_spawn_rule(env, stack, first):
    """Env 1 walks 3 m on the flat row and falls: it respawns on its stairs
    tile one level up, 0.425 m higher. Env 4 times out and respawns on the
    flat row."""
    _, _, step = stack
    s = with_info(first, [1], terrain_type=STAIRS)
    s = walked_from(env, s, [1], 3.0)
    s = time_out(knock_over(env, s, [1]), [4])
    out = step(s, zeros(env))
    assert done_of(out) == [1, 4]
    assert np.asarray(out.info["terrain_level"]).tolist() == [0, 1, 0, 0, 0, 0, 0, 0]
    for i in (1, 4):
        check_spawn(env, s, out, i)


def test_live_envs_are_untouched(env, stack, first):
    """Every leaf of a live env equals what playground's stock stack gives
    for the same step. Env 5 has walked 3 m on the flat row, which would
    promote it at a done."""
    _, _, step = stack
    stock = playground_wrapper.wrap_for_brax_training(env, episode_length=EPISODE)
    s = knock_over(env, walked_from(env, first, [5], 3.0), [2])
    ours, theirs = step(s, zeros(env)), jax.jit(stock.step)(s, zeros(env))
    assert done_of(ours) == done_of(theirs) == [2]
    live = np.array([i for i in range(N) if i != 2])
    for part in ("data", "obs", "info", "metrics"):
        a, b = getattr(ours, part), getattr(theirs, part)
        assert jax.tree.structure(a) == jax.tree.structure(b), part
        for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
            x, y = np.asarray(x), np.asarray(y)
            if x.shape[:1] == (N,):
                np.testing.assert_array_equal(x[live], y[live], err_msg=part)


def test_an_early_fall_demotes_through_the_wrapper(env, stack, first):
    """Two falls on level 1 at step 50 of 200, with 2 m commanded. The
    projection holds each to 8 m. Env 3 served 2.5 m. That clears half of
    2 m but not half of 8 m. It strikes, and one strike demotes. Env 2
    served 4.5 m, past half of 8 m, and holds."""
    _, _, step = stack
    s = with_info(first, [2, 3], terrain_level=1, since_spawn=49, commanded_dist=2.0, cheby_min=0.1,
                  cheby_max=0.1, curriculum_strikes=0)
    s = with_info(s, [3], served_dist=2.5)
    s = with_info(s, [2], served_dist=4.5)
    out = step(knock_over(env, s, [2, 3]), zeros(env))
    assert done_of(out) == [2, 3]
    assert np.asarray(out.info["terrain_level"])[[2, 3]].tolist() == [1, 0]
    assert np.asarray(out.info["curriculum_strikes"])[[2, 3]].tolist() == [0, 0]
    assert np.asarray(out.metrics[DEMOTED])[[2, 3]].tolist() == [0, 1]
    assert not np.asarray(out.metrics[PROMOTED]).any()
    check_spawn(env, s, out, 3)


def test_a_band_crossing_promotes_through_the_wrapper(env, stack, first):
    """Env 0 crosses the flat row (3 m walked) and goes up a level. Env 5
    reached 2 m from its level-1 tile centre, past every feature radius,
    and is promoted from the top row to a random row. Its key draws row 0,
    below the top row, which a clipped step up could not give. Env 6
    served the same 3 m within 1 m of the centre and holds. Env 7 walked
    1.5 m on the flat row and holds."""
    _, _, step = stack
    s = walked_from(env, first, [0, 5, 6], 3.0)
    s = walked_from(env, s, [7], 1.5)
    s = with_info(s, [5, 6], terrain_level=1, commanded_dist=2.0, served_dist=3.0, since_spawn=EPISODE - 1,
                  cheby_min=0.1)
    s = with_info(s, [5], cheby_max=2.0, terrain_rng=jax.random.PRNGKey(0))
    s = with_info(s, [6], cheby_max=1.0)
    out = step(time_out(s, [0, 5, 6, 7]), zeros(env))
    assert done_of(out) == [0, 5, 6, 7]
    level = np.asarray(out.info["terrain_level"])
    assert (level[0], level[5], level[6], level[7]) == (1, 0, 1, 0)
    assert np.asarray(out.metrics[PROMOTED]).tolist() == [1, 0, 0, 0, 0, 1, 0, 0]
    assert not np.asarray(out.metrics[DEMOTED]).any()


def test_promote_beats_demote_through_the_wrapper(env, stack, first):
    """A crossing and a failed walk in one episode: 3 m walked and served
    against a commanded 40 m projection. It promotes, and the strike count
    clears."""
    _, _, step = stack
    s = walked_from(env, first, [2], 3.0)
    s = with_info(s, [2], since_spawn=9, commanded_dist=2.0, served_dist=3.0, curriculum_strikes=0)
    out = step(knock_over(env, s, [2]), zeros(env))
    assert done_of(out) == [2]
    assert int(out.info["terrain_level"][2]) == 1
    assert int(out.info["curriculum_strikes"][2]) == 0
    assert (float(out.metrics[PROMOTED][2]), float(out.metrics[DEMOTED][2])) == (1.0, 0.0)


def test_a_timeout_is_not_projected_under_a_short_ppo_episode(env, stack, first):
    """Two 200-step timeouts on level 1 with 2 m commanded. The wrapper
    projects onto its own 200-step episodes. Env 4 served 1.4 m, more than
    half of 2 m, and clears. Projected onto the config's 1000 steps that
    would be under half of 10 m. Env 6 served 0.6 m and fails."""
    _, _, step = stack
    s = with_info(first, [4, 6], terrain_level=1, since_spawn=EPISODE - 1, commanded_dist=2.0,
                  cheby_min=0.1, cheby_max=0.1, curriculum_strikes=0)
    s = with_info(s, [4], served_dist=1.4)
    s = with_info(s, [6], served_dist=0.6)
    assert env._config.episode_length == 1000
    out = step(time_out(s, [4, 6]), zeros(env))
    assert done_of(out) == [4, 6]
    assert np.asarray(out.info["terrain_level"])[[4, 6]].tolist() == [1, 0]
    assert np.asarray(out.metrics[DEMOTED])[[4, 6]].tolist() == [0, 1]


@contextlib.contextmanager
def curriculum_block(env, **values):
    block = env._config.terrain.curriculum
    saved = {k: block[k] for k in values}
    block.update(values)
    try:
        yield
    finally:
        block.update(saved)


def test_pinned_envs_hold_their_rungs(env):
    """Half the batch pinned round robin over the two rows, a quarter on
    the flat row, the rest free. The wrapper's reset marks the six pinned
    envs, and respawns keep the mark. Every env crosses the flat row twice.
    The pinned envs reach their rows at the first respawn and stay there.
    The free ones climb. The env's step emits each episode's level, and
    the free-level metrics carry the free envs only."""
    with curriculum_block(env, pinned_frac=0.5, pinned_flat_frac=0.25):
        _, reset, step = jitted(terrain_wrapper.wrap_for_terrain_brax_training(env, episode_length=EPISODE))
    free = np.array([False] * 6 + [True] * 2)
    want = [0, 1, 0, 1, 0, 0, 1, 1]
    s = strong(reset(jax.random.split(jax.random.PRNGKey(0), N)))
    levels = []
    for _ in range(2):
        np.testing.assert_array_equal(s.info["curriculum_free"], free)
        levels.append(np.asarray(s.info["terrain_level"]))
        s = with_info(walked_from(env, s, range(N), 3.0), range(N), cheby_min=0.1, cheby_max=2.0)
        s = step(time_out(s, range(N)), zeros(env))
        assert done_of(s) == list(range(N))
        assert np.asarray(s.info["terrain_level"]).tolist()[:6] == want[:6]
        np.testing.assert_array_equal(s.metrics["terrain/level_per_step"], levels[-1])
        np.testing.assert_array_equal(s.metrics["terrain/level_free_per_step"], levels[-1] * free)
        np.testing.assert_array_equal(s.metrics["terrain/free_per_step"], free)
    np.testing.assert_array_equal(s.info["curriculum_free"], free)
    assert levels[1].tolist() == want
    assert not np.asarray(s.metrics[PROMOTED])[:6].any()
    assert np.asarray(s.metrics[PROMOTED])[6:].all()


def test_curriculum_knobs_reach_the_wrapper(env, first):
    """grace_sec 0.1, demote_fraction 0.8 and demote_strikes 2, set through
    the config. Envs 0 and 1 fall inside the grace window and are neutral:
    env 0 keeps level 1 and its strike, and env 1 is not promoted for its
    3 m on the flat row. Envs 2 and 3 time out on level 1 having served
    1.4 m of 2 m commanded. That fails only under fraction 0.8. Env 2 takes
    its first strike and holds. Env 3 already had one and is demoted."""
    knobs = build(terrain={
        "arena": CPU_ARENA,
        "spawn": {"grace_sec": 0.1},
        "curriculum": {"demote_strikes": 2, "demote_fraction": 0.8},
    })
    grace = knobs._grace_steps
    assert grace == round(0.1 / knobs.dt) > 0
    _, _, step = jitted(terrain_wrapper.wrap_for_terrain_brax_training(knobs, episode_length=EPISODE))
    s = with_info(first, [0, 1], since_spawn=grace - 1)
    s = with_info(s, [0], terrain_level=1, commanded_dist=2.0, curriculum_strikes=1, cheby_min=0.1,
                  cheby_max=0.1)
    s = walked_from(env, s, [1], 3.0)
    s = with_info(s, [2, 3], terrain_level=1, since_spawn=EPISODE - 1, commanded_dist=2.0,
                  cheby_min=0.1, cheby_max=0.1)
    s = with_info(s, [2], curriculum_strikes=0)
    s = with_info(s, [3], curriculum_strikes=1)
    s = with_info(s, [2, 3], served_dist=1.4)
    out = step(time_out(knock_over(env, s, [0, 1]), [2, 3]), zeros(env))
    assert done_of(out) == [0, 1, 2, 3]
    assert np.asarray(out.info["terrain_level"])[:4].tolist() == [1, 0, 1, 0]
    assert np.asarray(out.info["curriculum_strikes"])[:4].tolist() == [1, 0, 1, 0]
    assert np.asarray(out.metrics[PROMOTED])[:4].tolist() == [0, 0, 0, 0]
    assert np.asarray(out.metrics[DEMOTED])[:4].tolist() == [0, 0, 0, 1]


def test_band_state_and_spawn_counter_reset_on_respawn(env, stack, first):
    """After a respawn the band range is the spawn's own radius and the
    counters are 0. The next step counts from there with the command that
    drove it."""
    _, _, step = stack
    s = with_info(first, [5], since_spawn=120, commanded_dist=7.0, served_dist=6.0, cheby_min=0.05,
                  cheby_max=1.2)
    out = step(time_out(s, [5]), zeros(env))
    check_spawn(env, s, out, 5)
    nxt = step(out, zeros(env))
    assert int(nxt.info["since_spawn"][5]) == 1
    cmd = np.asarray(out.info["command"][5])
    np.testing.assert_allclose(float(nxt.info["commanded_dist"][5]), np.linalg.norm(cmd[:2]) * env.dt, rtol=1e-6)
    linvel = env._local_linvel(jax.tree.map(lambda x: x[5], nxt.data))[:2]
    served = curriculum.served_step(linvel, jp.asarray(cmd), env.dt)
    np.testing.assert_allclose(float(nxt.info["served_dist"][5]), float(served), atol=1e-9)
    xy = base_xy(env, nxt)[5]
    np.testing.assert_array_equal(np.asarray(nxt.info["last_xy"][5]), xy)
    r = np.abs(xy - np.asarray(nxt.info["tile_origin"][5])).max()
    r0 = float(out.info["cheby_min"][5])
    np.testing.assert_allclose(float(nxt.info["cheby_min"][5]), min(r0, r), atol=1e-6)
    np.testing.assert_allclose(float(nxt.info["cheby_max"][5]), max(r0, r), atol=1e-6)


def test_reset_shaped_and_stepped_states_share_one_trace(env, stack, first):
    """A reset-shaped and a stepped state trace the step once. A second
    trace would recompile the whole training step on the first respawn."""
    wrapper, _, _ = stack
    traces = []

    def counted(state, action):
        traces.append(1)
        return wrapper.step(state, action)

    step = jax.jit(counted)
    out = step(time_out(first, [5]), zeros(env))
    step(out, zeros(env))
    assert len(traces) == 1


def test_respawn_yaw_does_not_accumulate(env, stack, first):
    """Two respawns in a row. Each places the base at the draw its key
    gives, turned from the reset quaternion by the drawn yaw alone."""
    _, _, step = stack
    b = env._base_qadr
    s = first
    for _ in range(2):
        rng = s.info["terrain_rng"][7]
        s = step(knock_over(env, s, [7]), zeros(env))
        assert done_of(s) == [7]
        # The curriculum splits the key once, and the spawn draw takes the
        # second half of the next split.
        r_spawn = jax.random.split(jax.random.split(rng)[0])[1]
        xy, yaw, _ = env.draw_spawn(r_spawn, s.info["terrain_type"][7], s.info["terrain_level"][7])
        q = np.asarray(s.data.qpos[7])
        np.testing.assert_array_equal(q[b : b + 2], np.asarray(xy))
        want = tg.quat_mul(tg.yaw_quat(yaw), env._reset_quat)
        np.testing.assert_allclose(q[b + 3 : b + 7], np.asarray(want), atol=1e-6)


def test_episode_steps_restart_on_done(env, stack, first):
    _, _, step = stack
    s = first.replace(info={**first.info, "steps": jp.full_like(first.info["steps"], 50.0)})
    out = step(knock_over(env, s, [6]), zeros(env))
    assert done_of(out) == [6]
    nxt = step(out, zeros(env))
    assert np.asarray(nxt.info["steps"]).tolist() == [52.0] * 6 + [1.0, 52.0]


def test_curriculum_metrics_reach_the_episode_accumulator(env, stack, first):
    """The promotion shows on the respawn step and enters EpisodeWrapper's
    sums on the next one, in the episode the respawn starts."""
    _, _, step = stack
    s = walked_from(env, first, [3], 3.0)
    out = step(knock_over(env, s, [3]), zeros(env))
    assert float(out.metrics[PROMOTED][3]) == 1.0
    assert float(out.info["episode_metrics"][PROMOTED][3]) == 0.0
    nxt = step(out, zeros(env))
    assert float(nxt.metrics[PROMOTED][3]) == 0.0
    assert np.asarray(nxt.info["episode_metrics"][PROMOTED]).tolist() == [0, 0, 0, 1, 0, 0, 0, 0]
    assert np.asarray(nxt.info["episode_metrics"][DEMOTED]).sum() == 0


def test_composes_with_domain_randomization(env, stack):
    """Domain randomization is the inner vmap, under EpisodeWrapper, and its
    models differ per world. Without a randomization_fn the inner vmap is
    brax's plain one. The respawn reads geometry only, so it follows the
    spawn rule under randomized models."""
    assert type(stack[0].env.env) is brax_training.VmapWrapper
    randomize = make_domain_randomize(env.mj_model, env.robot_spec, None)
    fn = functools.partial(randomize, rng=jax.random.split(jax.random.PRNGKey(1), N))
    w, reset, step = jitted(
        terrain_wrapper.wrap_for_terrain_brax_training(env, episode_length=EPISODE, randomization_fn=fn)
    )
    assert isinstance(w.env.env, playground_wrapper.BraxDomainRandomizationVmapWrapper)
    friction = np.asarray(w.env.env._mjx_model_v.geom_friction)[..., 0]
    assert np.ptp(friction, axis=0).max() > 0
    s = reset(jax.random.split(jax.random.PRNGKey(0), N))
    s = with_info(walked_from(env, s, [1], 3.0), [1], terrain_type=STAIRS)
    out = step(knock_over(env, s, [1]), zeros(env))
    assert done_of(out) == [1]
    assert int(out.info["terrain_level"][1]) == 1
    check_spawn(env, s, out, 1)


def test_composes_with_progress_reseed(env):
    """With the no-progress cut on, the reseed wraps the terrain stack: a
    respawned env restarts both its terrain counters and its meter."""
    block = env._config.no_progress
    block.enable = True
    try:
        _, reset, step = jitted(wrappers.make_wrap_env_fn(env._config)(env, episode_length=EPISODE))
        s = reset(jax.random.split(jax.random.PRNGKey(0), N))
        s = s.replace(info={**s.info, "progress_ema": jp.full(N, -3.0),
                            "steps_since_cmd": jp.full(N, 40, s.info["steps_since_cmd"].dtype)})
        out = step(knock_over(env, s, [2]), zeros(env))
    finally:
        block.enable = False
    assert done_of(out) == [2]
    demand = float(env._cmd_speed(out.info["command"][2]))
    assert float(out.info["progress_ema"][2]) == pytest.approx(demand)
    assert int(out.info["steps_since_cmd"][2]) == 0
    assert int(out.info["since_spawn"][2]) == 0
    assert float(out.info["progress_ema"][0]) < -2.0 and int(out.info["steps_since_cmd"][0]) == 41


def test_full_reset_is_refused(env):
    with pytest.raises(ValueError, match="full_reset"):
        terrain_wrapper.wrap_for_terrain_brax_training(env, full_reset=True)


@pytest.mark.parametrize(
    "values, match",
    [
        ({"demote_strikes": 0}, "demote_strikes must be at least 1"),
        ({"pinned_frac": 0.7, "pinned_flat_frac": 0.4}, "pins more than the whole batch"),
        ({"pinned_frac": 1.5}, r"must lie in \[0, 1\]"),
    ],
)
def test_bad_curriculum_knobs_are_refused_at_construction(env, values, match):
    with curriculum_block(env, **values), pytest.raises(ValueError, match=match):
        terrain_wrapper.wrap_for_terrain_brax_training(env, episode_length=EPISODE)


def test_telemetry_is_absent_on_jax(env, stack, first):
    _, _, step = stack
    out = step(first, zeros(env))
    assert not set(TELEMETRY_METRICS) & set(out.metrics)
    assert not set(TELEMETRY_PEAKS) & set(out.info)
    assert set(CURRICULUM_METRICS) <= set(first.metrics)


def test_a_flat_env_keeps_the_stock_wrapper():
    flat = make_env("joystick", ROBOT_DIR, PRESET, {"sim": {"num_envs": N}})
    fn = wrappers.make_wrap_env_fn(flat._config)
    assert fn is playground_wrapper.wrap_for_brax_training
    state = jax.eval_shape(fn(flat, episode_length=EPISODE).reset, jax.random.split(jax.random.PRNGKey(0), N))
    assert not [k for k in state.info if k.startswith("terrain_")]
    assert not [k for k in state.metrics if k.startswith("terrain/")]
    with pytest.raises(TypeError, match="no terrain spawns"):
        terrain_wrapper.TerrainAutoResetWrapper(flat, episode_length=EPISODE)
