"""Warm start across critic layouts on real envs and real checkpoints.

roboto_origin under deploy_pd. The flat critic is joystick's list, 111
wide. The terrain critic appends the 98-column height scan, 209 wide. Each
checkpoint is a random-init one written by brax's own checkpoint code
(run_fixtures.write_random_run).
"""

from __future__ import annotations

import jax
import jax.numpy as jp
import numpy as np
import pytest
from brax.training.acme import running_statistics, specs
from brax.training.agents.ppo import networks as ppo_networks
from run_fixtures import make_networks, write_random_run

from humanoid_lab import paths, restore
from humanoid_lab.envs import height_scan
from humanoid_lab.envs.joystick import default_config as joystick_default_config
from humanoid_lab.policy_io import load_params
from humanoid_lab.registry import make_env
from humanoid_lab.terrain.config import CPU_ARENA

ROBOT_DIR = paths.ROBOTS_DIR / "roboto_origin"
PRESET = "deploy_pd"
FLAT_CRITIC = list(joystick_default_config().obs.privileged)
TERRAIN_OVERRIDES = {
    "terrain": {"arena": CPU_ARENA},
    "obs": {"privileged": [*FLAT_CRITIC, height_scan.NAME]},
    "sim": {"num_envs": 2},
}


@pytest.fixture(scope="module")
def flat():
    return make_env("joystick", ROBOT_DIR, PRESET, {})


@pytest.fixture(scope="module")
def terrain():
    return make_env("terrain", ROBOT_DIR, PRESET, TERRAIN_OVERRIDES)


def critic_width(env):
    return env.observation_size["privileged_state"][0]


def value_fn(env):
    net = make_networks(env.observation_size, env.action_size)
    return jax.jit(lambda params, obs: net.value_network.apply(params[0], params[2], obs))


def brax_init(env):
    """(normalizer, policy, value) as brax's PPO trainer initializes them for
    `env`, before a restore replaces them."""
    net = make_networks(env.observation_size, env.action_size)
    spec = {k: specs.Array(tuple(shape), jp.float32) for k, shape in env.observation_size.items()}
    k1, k2 = jax.random.split(jax.random.PRNGKey(0))
    return running_statistics.init_state(spec), net.policy_network.init(k1), net.value_network.init(k2)


def signature(params):
    """{leaf path: (shape, dtype)} of a (normalizer, policy, value) triple."""
    leaves = jax.tree_util.tree_leaves_with_path(tuple(params))
    return {jax.tree_util.keystr(p): (np.shape(x), np.asarray(x).dtype) for p, x in leaves}


def assert_fits(params, ref):
    """brax's restore swaps `params` into the training state it built from
    `ref`, brax_init's output, without a check. Its jitted training step
    needs the tree, shapes and dtypes of its own init."""
    assert jax.tree.structure(tuple(params)) == jax.tree.structure(ref)
    assert signature(params) == signature(ref)


def test_the_critic_widths(flat, terrain):
    assert (critic_width(flat), critic_width(terrain)) == (111, 209)
    assert flat.observation_size["state"] == terrain.observation_size["state"]


def test_a_flat_checkpoint_restores_into_the_terrain_task_with_an_unchanged_critic(flat, terrain, tmp_path):
    ckpt = write_random_run(flat, tmp_path / "flat_run", "joystick", {}, preset=PRESET)
    plan, kwargs = restore.restore_arguments(ckpt, terrain)
    assert (plan.action, plan.added, plan.removed) == (restore.ADAPT, (height_scan.NAME,), ())
    assert (plan.source_run, plan.source_task, plan.source_arena) == ("flat_run", "joystick", None)
    assert plan.priors == {height_scan.NAME: (height_scan.PRIOR_MEAN, height_scan.PRIOR_STD)}
    assert kwargs["restore_checkpoint_path"] is None
    adapted = kwargs["restore_params"]
    scan = slice(111, 209)
    for field, want in (("mean", height_scan.PRIOR_MEAN), ("std", height_scan.PRIOR_STD)):
        np.testing.assert_allclose(np.asarray(getattr(adapted[0], field)["privileged_state"])[scan], want,
                                   rtol=1e-6)

    # The value on the shared columns is the source's, whatever the scan
    # reads.
    source = load_params(ckpt)
    key = jax.random.PRNGKey(3)
    obs = {
        "state": jax.random.uniform(key, (16, terrain.observation_size["state"][0]), minval=-3, maxval=3),
        "privileged_state": jax.random.uniform(jax.random.fold_in(key, 1), (16, 209), minval=-3, maxval=3),
    }
    flat_obs = {**obs, "privileged_state": obs["privileged_state"][:, :111]}
    np.testing.assert_allclose(
        value_fn(terrain)(adapted, obs), value_fn(flat)(source, flat_obs), rtol=1e-6, atol=1e-6
    )

    # The adapted params fit brax's init for the terrain env, and brax's
    # policy runs on them. The source differs from that init exactly in
    # the critic's normalizer columns and first layer.
    ref = brax_init(terrain)
    assert_fits(adapted, ref)
    want, have = signature(ref), signature(source)
    assert have.keys() == want.keys()
    assert sorted(k for k in want if have[k] != want[k]) == [
        "[0].mean['privileged_state']",
        "[0].std['privileged_state']",
        "[0].summed_variance['privileged_state']",
        "[2]['params']['hidden_0']['kernel']",
    ]
    net = make_networks(terrain.observation_size, terrain.action_size)
    policy = ppo_networks.make_inference_fn(net)(adapted, deterministic=True)
    action, _ = policy(jax.tree.map(lambda x: x[0], obs), key)
    assert action.shape == (terrain.action_size,) and np.isfinite(np.asarray(action)).all()


def test_a_terrain_checkpoint_restores_into_the_flat_task(flat, terrain, tmp_path):
    """The scan's columns and kernel rows are dropped. Every other column
    is the source's, and the source arena is on record."""
    ckpt = write_random_run(terrain, tmp_path / "terrain_run", "terrain", TERRAIN_OVERRIDES, preset=PRESET)
    plan, kwargs = restore.restore_arguments(ckpt, flat)
    assert (plan.action, plan.added, plan.removed) == (restore.ADAPT, (), (height_scan.NAME,))
    assert plan.source_task == "terrain"
    record = terrain.arena_record()
    assert plan.source_arena == {
        "fingerprint": record["fingerprint"], "generator_version": record["generator_version"],
    }
    norm, policy, value = kwargs["restore_params"]
    src_norm, src_policy, src_value = load_params(ckpt)
    for field in ("mean", "std", "summed_variance"):
        src = np.asarray(getattr(src_norm, field)["privileged_state"])
        np.testing.assert_array_equal(getattr(norm, field)["privileged_state"], src[:111])
        np.testing.assert_array_equal(getattr(norm, field)["state"], getattr(src_norm, field)["state"])
    np.testing.assert_array_equal(
        value["params"]["hidden_0"]["kernel"], np.asarray(src_value["params"]["hidden_0"]["kernel"])[:111]
    )
    jax.tree.map(np.testing.assert_array_equal, policy, src_policy)
    assert_fits(kwargs["restore_params"], brax_init(flat))
    out = value_fn(flat)((norm, policy, value), {"state": jp.zeros((1, flat.observation_size["state"][0])),
                                                 "privileged_state": jp.zeros((1, 111))})
    assert np.isfinite(np.asarray(out)).all()


def test_an_equal_layout_restores_as_a_path(flat, tmp_path):
    ckpt = write_random_run(flat, tmp_path / "flat_run", "joystick", {}, preset=PRESET)
    plan, kwargs = restore.restore_arguments(ckpt, flat)
    assert plan.action == restore.AS_IS
    assert kwargs == {"restore_checkpoint_path": str(ckpt)}
