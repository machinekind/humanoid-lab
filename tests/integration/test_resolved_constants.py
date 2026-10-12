"""Joystick.resolved_constants, env.config and reward_term_names.

The Torch env reads the task constants only through these public accessors.
resolved_constants hands it numpy copies, and the JAX constants stay as they
are.
"""

import dataclasses

import jax
import jax.numpy as jp
import numpy as np
import pytest

from humanoid_lab import paths
from humanoid_lab.dr import randomize
from humanoid_lab.envs.joystick import Joystick, default_config, reward_term_names

ROBOT = paths.ROBOTS_DIR / "roboto_origin"


def _build(edit=None):
    cfg = default_config()
    cfg.sim.backend = "jax"
    if edit is not None:
        edit(cfg)
    return Joystick(ROBOT, "deploy_pd", cfg)


def _mirror_heading_income(cfg):
    cfg.symmetry.enable = True
    cfg.obs.state = tuple(cfg.obs.state) + ("heading_err",)
    cfg.reward.scales.gait_symmetry_income = 0.1


def _step_length(cfg):
    cfg.reward.scales.step_length = 0.5


@pytest.fixture(scope="module", params=["default", "mirror_heading_income"])
def env(request):
    return _build(None if request.param == "default" else _mirror_heading_income)


def _is_array(x):
    return isinstance(x, (jax.Array, np.ndarray))


def test_every_array_becomes_numpy_with_the_same_values(env):
    out = env.resolved_constants()
    for f in dataclasses.fields(env._consts):
        a, b = getattr(env._consts, f.name), getattr(out, f.name)
        if _is_array(a):
            assert type(b) is np.ndarray, f.name
            assert b.dtype == np.asarray(a).dtype, f.name
            np.testing.assert_array_equal(b, np.asarray(a), err_msg=f.name)
        elif f.name == "floor_geom_id":
            assert a is None
        else:
            assert b == a, f.name
    assert out.actuator_model is env._consts.actuator_model
    assert out.cmd_vmax.shape == () and out.cmd_vmax.dtype == np.float32


def test_no_jax_array_is_left(env):
    out = env.resolved_constants()

    def leaves(x):
        if isinstance(x, dict):
            for v in x.values():
                yield from leaves(v)
        elif isinstance(x, tuple):
            for v in x:
                yield from leaves(v)
        else:
            yield x

    for f in dataclasses.fields(out):
        for leaf in leaves(getattr(out, f.name)):
            assert not isinstance(leaf, jax.Array), f.name


def test_budgets_floor_and_root_are_resolved(env):
    out = env.resolved_constants()
    assert out.floor_geom_id == randomize._find_floor_geom_id(env.mj_model, None)
    assert out.naconmax_per_env == env._naconmax_per_env
    assert out.njmax == env._njmax
    # Torch uses root_body as the DR root.
    assert out.root_body == randomize._find_base_body_id(env.mj_model)


def test_the_jax_constants_are_untouched(env):
    before = {f.name: getattr(env._consts, f.name) for f in dataclasses.fields(env._consts)}
    jax_fields = [k for k, v in before.items() if isinstance(v, jax.Array)]
    assert {"qadr", "default_pose", "action_scale", "ctrl_lo", "cmd_vmax"} <= set(jax_fields)
    env.resolved_constants()
    for name, value in before.items():
        assert getattr(env._consts, name) is value, name
    for name in jax_fields:
        assert isinstance(getattr(env._consts, name), jax.Array), name
    assert env._consts.floor_geom_id is None


def test_config_is_the_envs_config(env):
    assert env.config is env._config


def _reward_keys(env):
    state = jax.jit(env.reset)(jax.random.PRNGKey(0))
    feet = jp.zeros((env._n_feet,), bool)
    rewards, _ = env._compute_rewards(state.data, state.info, jp.zeros(env.action_size), feet, feet)
    return list(rewards)


def test_term_names_match_the_reward_dict(env):
    assert _reward_keys(env) == reward_term_names(env.config.reward.scales)


def test_a_kept_term_is_named_when_its_scale_is_on():
    env = _build(_step_length)
    names = reward_term_names(env.config.reward.scales)
    assert "step_length" in names
    assert _reward_keys(env) == names
