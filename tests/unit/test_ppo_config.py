"""ppo_config: the PPO params both trainers assemble, and the repo-owned keys.

Model-free: Hydra composes only, with no env and no jit.
"""

from __future__ import annotations

import inspect
import json

import pytest
from brax.training.agents.ppo import train as ppo
from hydra import compose, initialize_config_dir

from humanoid_lab import paths
from humanoid_lab.ppo_config import (
    brax_train_kwargs,
    compose_ppo_params,
    resolve_episode_length,
)

FIXTURE = paths.REPO_ROOT / "tests" / "data" / "fixtures" / "ppo_params.json"


def _compose(overrides=()):
    with initialize_config_dir(version_base=None, config_dir=str(paths.CONFIGS_DIR)):
        return compose(config_name="config", overrides=list(overrides))


def _fixture() -> dict:
    recorded = json.loads(FIXTURE.read_text())
    recorded.pop("_meta")
    return recorded


def test_repo_keys_are_popped_before_brax():
    training_params, network_factory_cfg, repo = brax_train_kwargs(compose_ppo_params(_compose()))
    assert repo == {"sym_coef": 0.0}
    assert "sym_coef" not in training_params
    assert "network_factory" not in training_params
    assert network_factory_cfg["policy_obs_key"] == "state"
    params = inspect.signature(ppo.train).parameters
    assert set(training_params) - set(params) == set()


@pytest.mark.parametrize("name", sorted(_fixture()))
def test_ppo_params_equal_the_origin_main_snapshot(name):
    """The params are config only, so this runs on every platform."""
    expected = _fixture()[name]
    overrides = ["smoke=true"] if name == "__smoke__" else [f"experiment={name}"]
    got = json.loads(json.dumps(compose_ppo_params(_compose(overrides)).to_dict()))
    assert got.pop("sym_coef") == 0.0
    assert got == expected


@pytest.mark.parametrize(
    ("overrides", "env_length", "expected"),
    [
        (["smoke=true"], 1000, 200),
        (["smoke=true"], 150, 150),
        ([], 1000, 1000),
        (["++ppo.episode_length=50"], 1000, 50),
    ],
)
def test_resolve_episode_length_follows_the_env_unless_ppo_sets_it(overrides, env_length, expected):
    cfg = _compose(overrides)
    p = compose_ppo_params(cfg)
    resolve_episode_length(p, cfg, env_length)
    assert p.episode_length == expected
