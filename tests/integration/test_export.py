"""The validating exporter, end to end (port item 5.2).

The checkpoint here is a real one: a PPO network initialized at random,
saved with brax's own checkpoint machinery and loaded back through
`policy_io.load_policy`, which is the path `run.sh battery` and `run.sh
eval` use. Only the weights are untrained, so every shape, every
serialization quirk and every load-time workaround is the production one.

The destination assertions are the point of the item: w01-tek's exporter
writes its artifacts and validates afterwards, so a failure leaves bad
files on disk. Ours validates against a temp directory, and a failed
export leaves the destination untouched.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np
import pytest
from brax.training.acme import running_statistics, specs
from brax.training.agents.ppo import checkpoint as ppo_checkpoint
from brax.training.agents.ppo import networks as ppo_networks

from humanoid_lab import paths
from humanoid_lab.envs.joystick import Joystick, default_config
from humanoid_lab.export import policy as export
from humanoid_lab.export import runtime

ROBOT = "asimov_v1"
PRESET = "deploy_pd"
ENV_OVERRIDES = {"episode_length": 50}
NETWORK = {
    "policy_hidden_layer_sizes": [16, 16],
    "value_hidden_layer_sizes": [16, 16],
    "policy_obs_key": "state",
    "value_obs_key": "privileged_state",
}


@pytest.fixture(scope="module")
def env():
    cfg = default_config()
    for key, value in ENV_OVERRIDES.items():
        setattr(cfg, key, value)
    return Joystick(paths.ROBOTS_DIR / ROBOT, PRESET, cfg)


@pytest.fixture(scope="module")
def run_dir(env, tmp_path_factory) -> Path:
    """A run directory with a real, untrained checkpoint and a run.json."""
    run_dir = tmp_path_factory.mktemp("export_run")
    ckpt_dir = run_dir / "checkpoints"

    network_factory = functools.partial(ppo_networks.make_ppo_networks, **NETWORK)
    ppo_network = network_factory(
        env.observation_size,
        env.action_size,
        preprocess_observations_fn=running_statistics.normalize,
    )
    k_policy, k_value, k_stats = jax.random.split(jax.random.PRNGKey(0), 3)
    policy_params = ppo_network.policy_network.init(k_policy)
    value_params = ppo_network.value_network.init(k_value)

    # A normalizer with real statistics: an identity one (mean 0, std 1)
    # would let a normalization bug pass every check below.
    # observation_size is {key: (width,)}; a tree map over it would descend
    # into the tuples and hand back bare ints.
    obs_spec = {
        key: specs.Array(tuple(shape), jp.dtype("float32"))
        for key, shape in env.observation_size.items()
    }
    normalizer = running_statistics.init_state(obs_spec)
    batch = {
        key: jax.random.uniform(k_stats, (64, *spec.shape), minval=-3.0, maxval=3.0)
        for key, spec in obs_spec.items()
    }
    normalizer = running_statistics.update(normalizer, batch)

    ppo_checkpoint.save(
        ckpt_dir,
        step=1024,
        params=(normalizer, policy_params, value_params),
        config=ppo_checkpoint.network_config(
            observation_size=env.observation_size,
            action_size=env.action_size,
            normalize_observations=True,
            network_factory=network_factory,
        ),
    )

    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "run_name": "export_fixture",
                "task": "joystick",
                "checkpoint_dir": str(ckpt_dir),
                "env_config": env._config.to_dict(),
                "ppo_config": {"network_factory": NETWORK, "normalize_observations": True},
                "hydra_config": {
                    "robot": {"dir": f"robots/{ROBOT}"},
                    "actuators": {"name": PRESET},
                    "task": {"env": ENV_OVERRIDES},
                },
            },
            indent=2,
        )
    )
    return run_dir


@pytest.fixture(scope="module")
def loaded(run_dir):
    """What the exporter loads: env, contract, weights, inference fn."""
    return export.load_for_export(run_dir)


def test_the_fixture_checkpoint_loads_through_the_real_load_path(loaded):
    obs = {
        "state": jp.zeros(loaded.meta["obs_size"]),
        "privileged_state": jp.zeros(loaded.privileged_size),
    }
    action, _ = loaded.inference(obs, jax.random.PRNGKey(0))
    assert action.shape == (loaded.meta["action_size"],)
    assert np.all(np.isfinite(np.asarray(action)))


def test_the_numpy_forward_matches_the_jitted_brax_inference(loaded):
    worst = export.validate_numpy_vs_brax(
        loaded.weights, loaded.meta, loaded.inference, loaded.privileged_size
    )
    assert worst < export.TOLERANCE
    print(f"numpy vs brax: max |diff| = {worst:.2e}")


def test_a_clean_export_writes_both_artifacts_and_the_meta_round_trips(run_dir, tmp_path):
    out_dir = tmp_path / "deploy"
    written = export.export_run(run_dir, out_dir)

    assert written == out_dir
    assert sorted(p.name for p in out_dir.iterdir()) == ["policy.npz", "policy_meta.json"]

    meta = json.loads((out_dir / "policy_meta.json").read_text())
    assert meta["run_name"] == "export_fixture"
    assert meta["robot"] == ROBOT
    assert meta["preset"] == PRESET
    assert meta["checkpoint"].endswith("000000001024")

    policy = runtime.DeployPolicy.load(out_dir)
    assert policy.meta == meta


def test_a_corrupted_weight_matrix_leaves_nothing_at_the_destination(loaded, tmp_path):
    out_dir = tmp_path / "never_written"
    corrupted = dict(loaded.weights)
    corrupted["hidden_0_kernel"] = corrupted["hidden_0_kernel"] + 0.1

    with pytest.raises(AssertionError, match="numpy forward"):
        export.write_validated(
            loaded.env, loaded.meta, corrupted, out_dir, loaded.inference, loaded.privileged_size
        )

    assert not out_dir.exists()


def test_a_corrupted_normalizer_leaves_nothing_at_the_destination(loaded, tmp_path):
    out_dir = tmp_path / "never_written_either"
    corrupted = dict(loaded.weights)
    corrupted["norm_std"] = corrupted["norm_std"] * 2.0

    with pytest.raises(AssertionError):
        export.write_validated(
            loaded.env, loaded.meta, corrupted, out_dir, loaded.inference, loaded.privileged_size
        )

    assert not out_dir.exists()


def test_a_corrupted_anchor_is_caught_by_the_second_validation(loaded, tmp_path):
    """The two validations are independent: this one passes the first."""
    out_dir = tmp_path / "bad_anchor"
    meta = dict(loaded.meta)
    meta["anchor_ctrl"] = [v + 0.05 for v in meta["anchor_ctrl"]]

    with pytest.raises(AssertionError, match="deploy runtime"):
        export.write_validated(
            loaded.env, meta, loaded.weights, out_dir, loaded.inference, loaded.privileged_size
        )

    assert not out_dir.exists()


def test_a_failed_export_leaves_an_earlier_good_export_in_place(loaded, run_dir, tmp_path):
    out_dir = tmp_path / "keeper"
    export.export_run(run_dir, out_dir)
    before = (out_dir / "policy_meta.json").read_bytes()

    corrupted = dict(loaded.weights)
    corrupted["hidden_1_kernel"] = corrupted["hidden_1_kernel"] * 1.5
    with pytest.raises(AssertionError):
        export.write_validated(
            loaded.env, loaded.meta, corrupted, out_dir, loaded.inference, loaded.privileged_size
        )

    assert (out_dir / "policy_meta.json").read_bytes() == before
    assert sorted(p.name for p in out_dir.iterdir()) == ["policy.npz", "policy_meta.json"]


def test_the_deploy_runtime_matches_a_reference_built_from_the_env(loaded, tmp_path):
    """Validation two, run on its own so the measured error is visible."""
    out_dir = tmp_path / "deploy"
    export.write_validated(
        loaded.env, loaded.meta, loaded.weights, out_dir, loaded.inference, loaded.privileged_size
    )
    worst = export.validate_runtime_vs_env(
        loaded.env, out_dir, loaded.inference, loaded.privileged_size
    )
    assert worst < export.TOLERANCE
    print(f"runtime vs env reference: max |diff| = {worst:.2e}")


def test_the_runtime_reproduces_the_envs_own_ctrl_for_a_known_action(loaded, tmp_path):
    """The mapping the reference check covers, pinned against the env itself."""
    out_dir = tmp_path / "deploy_ctrl"
    export.write_validated(
        loaded.env, loaded.meta, loaded.weights, out_dir, loaded.inference, loaded.privileged_size
    )
    policy = runtime.DeployPolicy.load(out_dir)
    env = loaded.env

    action = np.linspace(-1.0, 1.0, env.action_size)
    want = np.asarray(
        jp.clip(
            env._actuator_model.ctrl_from_action(action, env._default_pose, env._action_scale),
            env._ctrl_lo,
            env._ctrl_hi,
        )
    )
    assert np.max(np.abs(policy.ctrl_from_action(action) - want)) < 1e-6


def test_an_unclassified_env_key_stops_the_export_before_anything_is_written(run_dir, tmp_path):
    run = json.loads((run_dir / "run.json").read_text())
    run["env_config"]["knee_snap_guard"] = True
    broken = tmp_path / "broken_run"
    broken.mkdir()
    (broken / "run.json").write_text(json.dumps(run))

    out_dir = tmp_path / "unreachable"
    with pytest.raises(ValueError, match="knee_snap_guard"):
        export.export_run(broken, out_dir)
    assert not out_dir.exists()


def test_the_default_destination_is_the_runs_own_deploy_dir(run_dir):
    export.export_run(run_dir)
    assert (run_dir / "deploy" / "policy_meta.json").exists()
