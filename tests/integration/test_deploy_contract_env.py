"""The deploy contract read off a live env.

Every field of `policy_meta.json` is a RESOLVED number taken from the env
instance that defined training, so these tests check the numbers against
that same env rather than against a hand-written table: the action -> ctrl
mapping, the actor observation layout, and the per-actuator gain and torque
tables. Two presets produce the same schema with different numbers, which
is the property the deploy side relies on when it re-derives nothing.
"""

from __future__ import annotations

import jax
import jax.numpy as jp
import numpy as np
import pytest

from humanoid_lab import deploy_contract as dc
from humanoid_lab import paths
from humanoid_lab.envs.joystick import Joystick, default_config
from humanoid_lab.export import runtime

ROBOT_DIR = paths.ROBOTS_DIR / "asimov_v1"


def build_env(preset: str, actuator_overrides: dict | None = None, **overrides):
    cfg = default_config()
    cfg.episode_length = 50  # fast tests, not a training config
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
        "run_name": "contract_test",
        "task": "joystick",
        "env_config": env._config.to_dict(),
        "hydra_config": {
            "robot": {"dir": "robots/asimov_v1"},
            "actuators": {"name": preset},
        },
    }


def zero_policy(contract: dict) -> runtime.DeployPolicy:
    """A runtime policy with all-zero weights for `contract`."""
    obs_size, nu = contract["obs_size"], contract["action_size"]
    weights = {
        "norm_mean": np.zeros(obs_size, np.float32),
        "norm_std": np.ones(obs_size, np.float32),
        "hidden_0_kernel": np.zeros((obs_size, 4), np.float32),
        "hidden_0_bias": np.zeros(4, np.float32),
        "hidden_1_kernel": np.zeros((4, 2 * nu), np.float32),
        "hidden_1_bias": np.zeros(2 * nu, np.float32),
    }
    return runtime.DeployPolicy(weights, contract)


@pytest.fixture(scope="module")
def ideal_env():
    return build_env("sizing_ideal")


@pytest.fixture(scope="module")
def pd_env():
    # knees_bent, not the default home: asimov_v1's home keyframe commands
    # every actuated joint to 0.0, which would let an anchor bug pass the
    # anchor-vs-default_pose checks below on a vector of zeros.
    return build_env("deploy_pd", reset_keyframe="knees_bent")


@pytest.fixture(scope="module")
def ideal_contract(ideal_env):
    return dc.build_contract(ideal_env, run_for(ideal_env, "sizing_ideal"), "ckpt/000000001024")


@pytest.fixture(scope="module")
def pd_contract(pd_env):
    return dc.build_contract(pd_env, run_for(pd_env, "deploy_pd"), "ckpt/000000001024")


def test_two_presets_share_one_schema(ideal_contract, pd_contract):
    assert set(ideal_contract) == set(pd_contract)
    assert ideal_contract["schema_version"] == pd_contract["schema_version"]
    assert ideal_contract["obs_layout"] == pd_contract["obs_layout"]
    assert ideal_contract["joint_names"] == pd_contract["joint_names"]


def test_two_presets_resolve_different_numbers(ideal_contract, pd_contract):
    assert ideal_contract["preset"] == "sizing_ideal"
    assert pd_contract["preset"] == "deploy_pd"
    for field in ("action_scale", "gains"):
        assert ideal_contract[field] != pd_contract[field], (
            f"{field} is identical across two presets with different gains -- "
            "the contract is not reading the live model"
        )
    assert ideal_contract["gains"]["kp"] != pd_contract["gains"]["kp"]
    assert ideal_contract["gains"]["preset"] == "sizing_ideal"


def test_provenance_names_the_run_the_robot_and_the_checkpoint(ideal_contract):
    assert ideal_contract["run_name"] == "contract_test"
    assert ideal_contract["task"] == "joystick"
    assert ideal_contract["robot"] == "asimov_v1"
    assert ideal_contract["checkpoint"] == "ckpt/000000001024"


def test_obs_layout_is_the_envs_own_actor_layout(ideal_env, ideal_contract):
    names = [c["name"] for c in ideal_contract["obs_layout"]]
    assert names == list(ideal_env.actor_obs_names)

    state = ideal_env.reset(jax.random.PRNGKey(0))
    assert ideal_contract["obs_size"] == int(state.obs["state"].shape[0])
    assert sum(c["size"] for c in ideal_contract["obs_layout"]) == ideal_contract["obs_size"]
    assert ideal_contract["action_size"] == ideal_env.action_size


def test_the_anchor_and_scale_reproduce_the_envs_ctrl_mapping(ideal_env, ideal_contract):
    """clip(anchor + action * scale, lo, hi) IS what step() sends to mujoco."""
    anchor = np.asarray(ideal_contract["anchor_ctrl"])
    scale = np.asarray(ideal_contract["action_scale"])
    lo = np.asarray(ideal_contract["ctrl_low"])
    hi = np.asarray(ideal_contract["ctrl_high"])

    rng = np.random.default_rng(0)
    for _ in range(8):
        action = rng.uniform(-1.5, 1.5, ideal_env.action_size)
        want = np.asarray(
            np.clip(
                ideal_env._actuator_model.ctrl_from_action(
                    action, ideal_env._default_pose, ideal_env._action_scale
                ),
                ideal_env._ctrl_lo,
                ideal_env._ctrl_hi,
            )
        )
        got = np.clip(anchor + action * scale, lo, hi)
        assert np.max(np.abs(want - got)) < 1e-6


def test_the_joint_pos_observation_anchor_is_the_keyframe_pose(ideal_env, ideal_contract):
    assert np.allclose(ideal_contract["default_pose"], np.asarray(ideal_env._default_pose))


def test_torque_caps_travel_in_the_metadata(ideal_env, ideal_contract):
    forcerange = np.asarray(ideal_env.mj_model.actuator_forcerange)
    assert np.allclose(ideal_contract["torque_low"], forcerange[:, 0])
    assert np.allclose(ideal_contract["torque_high"], forcerange[:, 1])
    assert len(ideal_contract["torque_high"]) == ideal_env.action_size


def test_the_command_box_is_the_trained_one(ideal_env, ideal_contract):
    c = ideal_env._config.command
    assert ideal_contract["command_low"] == [c.vx[0], c.vy[0], c.wz[0]]
    assert ideal_contract["command_high"] == [c.vx[1], c.vy[1], c.wz[1]]


def test_the_gait_clock_parameters_travel_with_the_policy(ideal_env, ideal_contract):
    """The actor observes `phase`, so the runtime has to advance the clock."""
    clock = ideal_contract["gait_clock"]
    assert clock["freq_low"] == ideal_env._config.gait.freq[0]
    assert clock["freq_high"] == ideal_env._config.gait.freq[1]
    assert np.allclose(clock["cmd_speed_max"], float(ideal_env._cmd_vmax))
    assert len(clock["offsets"]) == ideal_env._n_feet
    assert ideal_contract["ctrl_dt"] == ideal_env._config.ctrl_dt
    # The runtime advances its clock by the env's step for every command.
    # Pure turns and commands near the deadband pin the turn weight and the
    # deadband.
    policy = zero_policy(ideal_contract)
    commands = [
        [0.0, 0.0, 0.0],
        [0.03, 0.0, 0.0],
        [0.0, 0.0, 0.1],
        [0.0, 0.0, 0.5],
        [0.04, 0.0, 0.05],
        [0.3, 0.1, 0.4],
        [1.0, 0.0, 0.0],
    ]
    for command in commands:
        want = float(ideal_env._phase_dt(jp.array(command)))
        assert policy.phase_increment(command) == pytest.approx(want, abs=1e-6), command


def test_a_wider_deadband_freezes_the_clock_in_the_env_and_in_the_runtime():
    env = build_env(
        "sizing_ideal",
        **{"command.deadband": 0.15, "command.resample_steps": 10000, "push.enable": False},
    )
    contract = dc.build_contract(env, run_for(env), "ckpt/000000001024")
    assert contract["gait_clock"]["speed_deadband"] == 0.15

    assert float(env._phase_dt(jp.array([0.1, 0.0, 0.0]))) == 0.0
    assert float(env._phase_dt(jp.array([0.2, 0.0, 0.0]))) > 0.0

    state = jax.jit(env.reset)(jax.random.PRNGKey(0))
    state.info["command"] = jp.array([0.1, 0.0, 0.0])
    step = jax.jit(env.step)
    action = jp.zeros(env.action_size)
    for _ in range(5):
        state = step(state, action)
        assert float(state.info["phase"]) == 0.0

    nu = contract["action_size"]
    policy = zero_policy(contract)
    policy.reset()
    reading = {
        "gyro": np.zeros(3),
        "gravity": np.array([0.0, 0.0, -1.0]),
        "joint_pos": np.asarray(contract["default_pose"]),
        "joint_vel": np.zeros(nu),
        "command": np.array([0.1, 0.0, 0.0]),
    }
    for _ in range(5):
        policy.step(**reading)
    assert policy.phase == 0.0


def test_an_ideal_torque_preset_anchors_on_zero_torque():
    """The anchor is whatever a zero action commands, per actuator model."""
    env = build_env(
        "sizing_ideal",
        actuator_overrides={"model": "ideal_torque"},
        reset_keyframe="knees_bent",
    )
    contract = dc.build_contract(env, run_for(env, "sizing_ideal"), "ckpt")
    assert contract["ctrl_unit"] == "Nm"
    assert np.allclose(contract["anchor_ctrl"], 0.0)
    # The obs anchor does not move with the ctrl anchor.
    assert not np.allclose(contract["default_pose"], 0.0)


def test_a_pd_preset_anchors_on_the_default_pose(pd_env, pd_contract):
    assert pd_contract["ctrl_unit"] == "rad"
    assert np.allclose(pd_contract["anchor_ctrl"], np.asarray(pd_env._default_pose))


def test_an_unclassified_env_key_blocks_the_contract(ideal_env):
    run = run_for(ideal_env)
    run["env_config"]["knee_snap_guard"] = True
    with pytest.raises(ValueError, match="knee_snap_guard"):
        dc.build_contract(ideal_env, run, "ckpt")


def test_a_non_joystick_task_is_refused(ideal_env):
    run = run_for(ideal_env)
    run["task"] = "sizing"
    with pytest.raises(NotImplementedError, match="sizing"):
        dc.build_contract(ideal_env, run, "ckpt")


def test_an_actor_observation_the_robot_cannot_measure_is_refused():
    """A privileged signal on the actor list has no deploy-side source."""
    cfg = default_config()
    cfg.episode_length = 50
    cfg.obs.state = tuple(cfg.obs.state) + ("height",)
    env = Joystick(ROBOT_DIR, "sizing_ideal", cfg)
    with pytest.raises(ValueError, match="height"):
        dc.build_contract(env, run_for(env), "ckpt")


def test_the_default_contract_is_schema_1_with_cos_sin(ideal_contract):
    """A stock export still loads on a robot running an old vendored runtime."""
    assert ideal_contract["schema_version"] == dc.LEGACY_SCHEMA_VERSION == 1
    assert ideal_contract["gait_clock"]["phase_obs"] == "cos_sin"


def test_a_frozen_clock_at_the_default_mode_stays_schema_1_cos_sin():
    env = build_env("sizing_ideal", **{"gait.freq": (0.0, 0.0)})
    meta = dc.build_contract(env, run_for(env), "ckpt")
    assert meta["schema_version"] == 1
    assert meta["gait_clock"]["phase_obs"] == "cos_sin"
    state = env.reset(jax.random.PRNGKey(0))
    phase = np.asarray(env._obs_catalog(state.data, state.info)["phase"])
    assert np.allclose(phase, [1.0, -1.0, 0.0, 0.0], atol=1e-6)


def test_a_frozen_clock_with_zeros_ships_schema_2_and_observes_zeros():
    env = build_env("sizing_ideal", **{"gait.freq": (0.0, 0.0), "gait.frozen_phase_obs": "zeros"})
    meta = dc.build_contract(env, run_for(env), "ckpt")
    assert meta["schema_version"] == dc.SCHEMA_VERSION == 2
    assert meta["gait_clock"]["phase_obs"] == "zeros"
    state = env.reset(jax.random.PRNGKey(0))
    phase = np.asarray(env._obs_catalog(state.data, state.info)["phase"])
    assert phase.shape == (2 * env._n_feet,)
    assert np.array_equal(phase, np.zeros(2 * env._n_feet, np.float32))


def test_zeros_on_a_running_clock_keeps_cos_sin():
    """The mode applies only when gait.freq is (0, 0)."""
    env = build_env("sizing_ideal", **{"gait.frozen_phase_obs": "zeros"})
    meta = dc.build_contract(env, run_for(env), "ckpt")
    assert meta["schema_version"] == 1
    assert meta["gait_clock"]["phase_obs"] == "cos_sin"


def test_a_heading_err_policy_is_refused_at_export():
    """Owner decision 1 (b): the runtime has no yaw input and no yaw_ref."""
    cfg = default_config()
    cfg.episode_length = 50
    cfg.obs.state = tuple(cfg.obs.state) + ("heading_err",)
    env = Joystick(ROBOT_DIR, "sizing_ideal", cfg)
    with pytest.raises(ValueError, match="heading_err"):
        dc.build_contract(env, run_for(env), "ckpt")


def test_heading_err_on_the_critic_list_only_builds_a_contract():
    cfg = default_config()
    cfg.episode_length = 50
    cfg.obs.privileged = tuple(cfg.obs.privileged) + ("heading_err",)
    env = Joystick(ROBOT_DIR, "sizing_ideal", cfg)
    meta = dc.build_contract(env, run_for(env), "ckpt")
    assert "heading_err" not in [c["name"] for c in meta["obs_layout"]]
    assert meta["schema_version"] == 1
