"""reward_term_names against the reward dict the env computes.

The Torch env names its reward terms with reward_term_names, so the names
must match the keys of the JAX env's reward dict in the same order.
"""

import jax
import jax.numpy as jp
import pytest

from humanoid_lab import paths
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


EDITS = {"default": None, "mirror_heading_income": _mirror_heading_income, "step_length": _step_length}


@pytest.fixture(scope="module", params=list(EDITS))
def env(request):
    env = _build(EDITS[request.param])
    if request.param == "step_length":
        # A kept term is named when its scale is on.
        assert "step_length" in reward_term_names(env.config.reward.scales)
    return env


def _reward_keys(env):
    state = jax.jit(env.reset)(jax.random.PRNGKey(0))
    feet = jp.zeros((env._n_feet,), bool)
    rewards, _ = env._compute_rewards(state.data, state.info, jp.zeros(env.action_size), feet, feet)
    return list(rewards)


def test_term_names_match_the_reward_dict(env):
    assert _reward_keys(env) == reward_term_names(env.config.reward.scales)
