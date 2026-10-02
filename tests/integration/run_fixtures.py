"""A training run's on-disk record without training: a randomly initialized
PPO checkpoint saved with brax's own checkpoint code, next to a run.json in
the layout train.py writes. `write_random_run` writes one into a directory,
for a test that loads a run.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path

import jax
import jax.numpy as jp
from brax.training.acme import running_statistics, specs
from brax.training.agents.ppo import checkpoint as ppo_checkpoint
from brax.training.agents.ppo import networks as ppo_networks

from humanoid_lab import paths

# A small network: these tests check shapes and plumbing, not a policy.
NETWORK = {
    "policy_hidden_layer_sizes": [16, 16],
    "value_hidden_layer_sizes": [16, 16],
    "policy_obs_key": "state",
    "value_obs_key": "privileged_state",
}
STEP = 1024


def network_factory():
    return functools.partial(ppo_networks.make_ppo_networks, **NETWORK)


def make_networks(observation_size, action_size):
    return network_factory()(
        observation_size, action_size, preprocess_observations_fn=running_statistics.normalize
    )


def random_params(env, seed: int = 0):
    """(normalizer, policy, value) for `env`'s observation sizes. The
    normalizer holds real statistics of a uniform batch on [-3, 3], so an
    identity normalizer cannot pass for it."""
    net = make_networks(env.observation_size, env.action_size)
    k_policy, k_value, k_stats = jax.random.split(jax.random.PRNGKey(seed), 3)
    obs_spec = {
        key: specs.Array(tuple(shape), jp.dtype("float32")) for key, shape in env.observation_size.items()
    }
    normalizer = running_statistics.init_state(obs_spec)
    batch = {
        key: jax.random.uniform(jax.random.fold_in(k_stats, i), (64, *spec.shape), minval=-3.0, maxval=3.0)
        for i, (key, spec) in enumerate(obs_spec.items())
    }
    normalizer = running_statistics.update(normalizer, batch)
    return normalizer, net.policy_network.init(k_policy), net.value_network.init(k_value)


def write_random_run(env, run_dir: Path, task: str, env_overrides: dict, *, preset: str,
                     seed: int = 0) -> Path:
    """Write a random-init checkpoint and run.json for `env`, built for
    `task` with `env_overrides` under actuator preset `preset`, into
    `run_dir`. Returns the checkpoint step directory, the path `restore=`
    takes."""
    run_dir = Path(run_dir)
    ckpt_dir = run_dir / "checkpoints"
    params = random_params(env, seed)
    ppo_checkpoint.save(
        ckpt_dir,
        step=STEP,
        params=params,
        config=ppo_checkpoint.network_config(
            observation_size=env.observation_size,
            action_size=env.action_size,
            normalize_observations=True,
            network_factory=network_factory(),
        ),
    )
    robot_dir = Path(env.robot_spec.robot_dir).resolve()
    record = {
        "run_name": run_dir.name,
        "task": task,
        "checkpoint_dir": str(ckpt_dir),
        "env_config": env._config.to_dict(),
        "ppo_config": {"network_factory": NETWORK, "normalize_observations": True},
        "hydra_config": {
            "robot": {"name": env.robot_spec.name, "dir": str(robot_dir.relative_to(paths.REPO_ROOT))},
            "actuators": {"name": preset},
            "task": {"name": task, "env": env_overrides},
        },
        "arena": env.arena_record() if hasattr(env, "arena_record") else None,
    }
    (run_dir / "run.json").write_text(json.dumps(record, indent=2, default=str))
    return ckpt_dir / f"{STEP:012d}"
