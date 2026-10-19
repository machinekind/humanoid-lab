"""The numpy deploy runtime observes what the training env observed.

A policy runs in the env for 60 steps. At every step the runtime assembles
its observation from the env's own sensor readings and the env's served
command, and the result must equal the env's actor observation. The runtime
holds its own last_action and gait clock, so a drift in either shows up
here. Obs noise and pushes are off, so the env's actor vector is the clean
catalog.

The step order matches training. Observation k carries the command the env
holds after step k-1, which may be a fresh resample. That command then
drives step k's phase advance in the env and the runtime's phase advance in
act().
"""

from __future__ import annotations

import jax
import jax.numpy as jp
import numpy as np
import pytest

from humanoid_lab import deploy_contract as dc
from humanoid_lab import paths
from humanoid_lab.envs import progress
from humanoid_lab.envs.joystick import Joystick, default_config
from humanoid_lab.export import runtime

ROBOT_DIR = paths.ROBOTS_DIR / "asimov_v1"
STEPS = 60

# Worst |runtime obs - env obs| measured here over 60 steps: 3.2e-6 for the
# default clock and 2.4e-6 for the deadband case (the float32 env clock
# against the runtime's float64 clock, through cos/sin), 6e-8 for both frozen
# clocks. TOL leaves a 3x margin.
TOL = 1e-5


def build_env(preset: str, actuator_overrides: dict | None = None, **overrides):
    cfg = default_config()
    cfg.episode_length = 1000  # longer than the rollout
    for key, value in overrides.items():
        block, _, leaf = key.partition(".")
        if leaf:
            setattr(getattr(cfg, block), leaf, value)
        else:
            setattr(cfg, block, value)
    return Joystick(ROBOT_DIR, preset, cfg, actuator_overrides=actuator_overrides)


def run_for(env, preset: str = "sizing_ideal") -> dict:
    """A parsed run.json for `env`, as train.py writes one."""
    return {
        "run_name": "obs_parity_test",
        "task": "joystick",
        "env_config": env._config.to_dict(),
        "hydra_config": {
            "robot": {"dir": "robots/asimov_v1"},
            "actuators": {"name": preset},
        },
    }


def toy_weights(obs_size: int, action_size: int, hidden: int = 32, seed: int = 0) -> dict:
    """A random actor MLP with an identity normalizer, in the export layout."""
    rng = np.random.default_rng(seed)
    return {
        "norm_mean": np.zeros(obs_size, np.float32),
        "norm_std": np.ones(obs_size, np.float32),
        "hidden_0_kernel": (rng.normal(size=(obs_size, hidden)) / np.sqrt(obs_size)).astype(np.float32),
        "hidden_0_bias": np.zeros(hidden, np.float32),
        # tanh_normal doubles the output: loc then scale.
        "hidden_1_kernel": (rng.normal(size=(hidden, 2 * action_size)) / np.sqrt(hidden)).astype(np.float32),
        "hidden_1_bias": np.zeros(2 * action_size, np.float32),
    }


COMMON = {
    "obs_noise.gyro": 0.0,
    "obs_noise.joint_pos": 0.0,
    "obs_noise.joint_vel": 0.0,
    "push.enable": False,
    "symmetry.enable": False,
    "command.resample_steps": 7,
    "command.zero_prob": 0.2,
}

CASES = {
    "default_clock": {},
    "frozen_cos_sin": {"gait.freq": (0.0, 0.0)},
    "frozen_zeros": {"gait.freq": (0.0, 0.0), "gait.frozen_phase_obs": "zeros"},
    # A narrow box puts commands in (0.05, 0.15], where the wider deadband
    # freezes both clocks.
    "deadband": {
        "command.deadband": 0.15,
        "command.vx": (-0.3, 0.3),
        "command.vy": (-0.1, 0.1),
        "command.wz": (-0.2, 0.2),
    },
}


@pytest.mark.parametrize("case", list(CASES))
def test_the_runtime_observation_matches_the_env(case):
    env = build_env("sizing_ideal", **COMMON, **CASES[case])
    meta = dc.build_contract(env, run_for(env), "ckpt")
    if case == "frozen_zeros":
        assert meta["schema_version"] == 2
        assert meta["gait_clock"]["phase_obs"] == "zeros"
    else:
        assert meta["schema_version"] == 1
    policy = runtime.DeployPolicy(toy_weights(meta["obs_size"], meta["action_size"]), meta)

    reset, step = jax.jit(env.reset), jax.jit(env.step)
    state = reset(jax.random.PRNGKey(3))
    policy.reset()

    worst = 0.0
    resampled = False
    frozen_in_band = 0
    for _ in range(STEPS):
        data = state.data
        inputs = {
            "gyro": np.asarray(env._gyro(data)),
            "gravity": np.asarray(env._gravity_body(data)),
            "joint_pos": np.asarray(data.qpos[env._qadr]),
            "joint_vel": np.asarray(data.qvel[env._vadr]),
            "command": np.asarray(state.info["command"]),
        }
        obs_env = np.asarray(state.obs["state"])
        obs_rt = policy.observe(**inputs)
        assert np.all(np.isfinite(obs_env))
        worst = max(worst, float(np.max(np.abs(obs_rt - obs_env))))
        assert np.allclose(obs_rt, obs_env, atol=TOL), np.max(np.abs(obs_rt - obs_env))

        action = policy.act(**inputs)
        phase_before = float(state.info["phase"])
        state = step(state, jp.asarray(action))
        if int(state.info["steps_since_cmd"]) == 0:
            resampled = True
        speed = float(progress.cmd_speed(inputs["command"]))
        if case == "deadband" and 0.05 < speed <= 0.15:
            assert float(state.info["phase"]) == phase_before
            frozen_in_band += 1

    print(f"{case}: worst |obs_rt - obs_env| = {worst:.3g}")
    assert resampled, "no command resample happened in the rollout"
    if case == "deadband":
        assert frozen_in_band > 0, "no command landed in (0.05, 0.15]"
