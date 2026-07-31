"""The numpy deploy runtime's algebra (port item 5.2).

`export/runtime.py` is the reference implementation a robot side vendors,
so it is numpy and nothing else. These tests build small hand-made
artifacts and check the pieces the exporter's round-trip validation would
otherwise only check in aggregate: the observation assembly order, the MLP
forward, the action -> ctrl mapping, and the gait clock.

Unit-suite safe: no env, no model, no jax (see
tests/unit/test_suite_split.py).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from humanoid_lab.export import runtime

JOINTS = ["j0", "j1"]


def toy_meta(**overrides) -> dict:
    meta = {
        "schema_version": 1,
        "run_name": "toy",
        "task": "joystick",
        "checkpoint": "ckpt",
        "robot": "toy",
        "preset": "toy",
        "actuator_model": "pd",
        "obs_layout": [
            {"name": "gyro", "size": 3},
            {"name": "joint_pos", "size": 2},
            {"name": "last_action", "size": 2},
            {"name": "command", "size": 3},
            {"name": "phase", "size": 4},
        ],
        "obs_size": 14,
        "action_size": 2,
        "joint_names": JOINTS,
        "anchor_ctrl": [0.5, -0.5],
        "default_pose": [0.5, -0.5],
        "action_scale": [0.25, 0.5],
        "ctrl_low": [-1.0, -1.0],
        "ctrl_high": [1.0, 1.0],
        "ctrl_unit": "rad",
        "ctrl_dt": 0.02,
        "torque_low": [-10.0, -10.0],
        "torque_high": [10.0, 10.0],
        "gains": {"preset": "toy", "model": "pd", "joints": JOINTS, "kp": [1.0, 1.0], "kd": [0.1, 0.1]},
        "command_low": [-0.8, -0.6, -0.6],
        "command_high": [0.8, 0.6, 0.6],
        "gait_clock": {
            "offsets": [0.0, -np.pi],
            "freq_low": 1.0,
            "freq_high": 2.0,
            "turn_weight": 0.3,
            "speed_deadband": 0.05,
            "cmd_speed_max": 0.8,
        },
    }
    meta.update(overrides)
    return meta


def toy_weights(obs_size: int = 14, action_size: int = 2, hidden: int = 5) -> dict:
    rng = np.random.default_rng(0)
    return {
        "norm_mean": rng.normal(size=obs_size).astype(np.float32),
        "norm_std": rng.uniform(0.5, 1.5, obs_size).astype(np.float32),
        "hidden_0_kernel": rng.normal(size=(obs_size, hidden)).astype(np.float32),
        "hidden_0_bias": rng.normal(size=hidden).astype(np.float32),
        # tanh_normal doubles the output: loc then scale.
        "hidden_1_kernel": rng.normal(size=(hidden, 2 * action_size)).astype(np.float32),
        "hidden_1_bias": rng.normal(size=2 * action_size).astype(np.float32),
    }


def toy_policy(**meta_overrides) -> runtime.DeployPolicy:
    return runtime.DeployPolicy(toy_weights(), toy_meta(**meta_overrides))


def sensors(**overrides) -> dict:
    reading = {
        "gyro": np.zeros(3),
        "gravity": np.array([0.0, 0.0, -1.0]),
        "joint_pos": np.array([0.5, -0.5]),
        "joint_vel": np.zeros(2),
        "command": np.zeros(3),
    }
    reading.update(overrides)
    return reading


def test_the_forward_pass_is_a_silu_mlp_with_a_tanh_head():
    weights = toy_weights()
    obs = np.linspace(-1.0, 1.0, 14).astype(np.float32)

    x = (obs - weights["norm_mean"]) / weights["norm_std"]
    x = x @ weights["hidden_0_kernel"] + weights["hidden_0_bias"]
    x = x / (1.0 + np.exp(-x))  # SiLU
    x = x @ weights["hidden_1_kernel"] + weights["hidden_1_bias"]
    want = np.tanh(x[:2])  # loc half only

    got = runtime.forward(weights, obs, action_size=2)
    assert np.max(np.abs(want - got)) < 1e-6


def test_a_network_output_that_is_not_a_loc_scale_pair_is_refused():
    """The head width says which distribution the checkpoint carries."""
    weights = toy_weights()
    with pytest.raises(ValueError, match="head"):
        runtime.forward(weights, np.zeros(14), action_size=3)


def test_the_observation_is_assembled_in_the_meta_order():
    policy = toy_policy()
    reading = sensors(gyro=np.array([0.1, 0.2, 0.3]), joint_pos=np.array([0.9, -0.1]))
    obs = policy.observe(**reading)

    assert obs.shape == (14,)
    assert np.allclose(obs[:3], [0.1, 0.2, 0.3])
    # joint_pos is offset by the default pose, as the env's catalog does it.
    assert np.allclose(obs[3:5], [0.4, 0.4])
    assert np.allclose(obs[5:7], 0.0)  # last_action, fresh
    assert np.allclose(obs[7:10], 0.0)  # command
    # cos of each leg phase, then sin. sin(-pi) is 8.7e-8 in float32.
    assert np.allclose(obs[10:], [1.0, -1.0, 0.0, 0.0], atol=1e-6)


def test_a_reordered_layout_reorders_the_observation():
    layout = [
        {"name": "command", "size": 3},
        {"name": "gyro", "size": 3},
        {"name": "joint_pos", "size": 2},
        {"name": "last_action", "size": 2},
        {"name": "phase", "size": 4},
    ]
    policy = toy_policy(obs_layout=layout)
    obs = policy.observe(**sensors(gyro=np.array([0.1, 0.2, 0.3]), command=np.array([0.4, 0.0, 0.0])))
    assert np.allclose(obs[:3], [0.4, 0.0, 0.0])
    assert np.allclose(obs[3:6], [0.1, 0.2, 0.3])


def test_an_observation_the_runtime_cannot_produce_is_refused():
    layout = [{"name": "height", "size": 1}]
    with pytest.raises(ValueError, match="height"):
        toy_policy(obs_layout=layout)


def test_the_action_maps_to_ctrl_through_the_anchor_scale_and_clip():
    policy = toy_policy()
    assert np.allclose(policy.ctrl_from_action(np.zeros(2)), [0.5, -0.5])
    assert np.allclose(policy.ctrl_from_action(np.array([1.0, 1.0])), [0.75, 0.0])
    assert np.allclose(policy.ctrl_from_action(np.array([-1.0, -1.0])), [0.25, -1.0])


def test_ctrl_is_clipped_to_the_bounds_in_the_meta():
    policy = toy_policy(ctrl_low=[0.6, -1.0], ctrl_high=[0.7, 1.0])
    assert np.allclose(policy.ctrl_from_action(np.zeros(2)), [0.6, -0.5])
    assert np.allclose(policy.ctrl_from_action(np.array([1.0, 0.0])), [0.7, -0.5])


def test_the_clock_is_frozen_below_the_command_deadband():
    policy = toy_policy()
    policy.reset()
    for _ in range(5):
        policy.step(**sensors(command=np.array([0.01, 0.0, 0.0])))
    assert policy.phase == 0.0


def test_the_clock_advances_at_the_speed_scaled_frequency():
    policy = toy_policy()
    policy.reset()
    policy.step(**sensors(command=np.array([0.8, 0.0, 0.0])))
    # At the top of the box the fraction is 1, so freq is freq_high.
    assert policy.phase == pytest.approx(2 * np.pi * 0.02 * 2.0)

    policy.reset()
    policy.step(**sensors(command=np.array([0.4, 0.0, 0.0])))
    assert policy.phase == pytest.approx(2 * np.pi * 0.02 * 1.5)


def test_the_clock_wraps_into_the_half_open_pi_interval():
    policy = toy_policy()
    policy.reset()
    for _ in range(200):
        policy.step(**sensors(command=np.array([0.8, 0.0, 0.0])))
    assert -np.pi <= policy.phase < np.pi


def test_the_last_action_fed_back_is_the_previous_step_s_action():
    policy = toy_policy()
    policy.reset()
    first = policy.act(**sensors())
    obs = policy.observe(**sensors())
    assert np.allclose(obs[5:7], first)


def test_a_policy_loads_from_the_written_artifacts(tmp_path):
    weights = toy_weights()
    meta = toy_meta()
    np.savez(tmp_path / "policy.npz", **weights)
    (tmp_path / "policy_meta.json").write_text(json.dumps(meta))

    policy = runtime.DeployPolicy.load(tmp_path)
    reference = runtime.DeployPolicy(weights, meta)
    reading = sensors(gyro=np.array([0.3, -0.2, 0.1]))
    assert np.allclose(policy.step(**reading), reference.step(**reading))


def test_a_meta_from_a_newer_schema_is_refused(tmp_path):
    np.savez(tmp_path / "policy.npz", **toy_weights())
    (tmp_path / "policy_meta.json").write_text(json.dumps(toy_meta(schema_version=99)))
    with pytest.raises(ValueError, match="schema"):
        runtime.DeployPolicy.load(tmp_path)
