"""The deploy contract read off a live env (port item 5.1).

Every field of `policy_meta.json` is a RESOLVED number taken from the env
instance that defined training, so these tests check the numbers against
that same env rather than against a hand-written table: the action -> ctrl
mapping, the actor observation layout, and the per-actuator gain and torque
tables. Two presets produce the same schema with different numbers, which
is the property the deploy side relies on when it re-derives nothing.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

from humanoid_lab import deploy_contract as dc
from humanoid_lab import paths
from humanoid_lab.envs.joystick import Joystick, default_config

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


@pytest.fixture(scope="module")
def ideal_env():
    return build_env("sizing_ideal")


@pytest.fixture(scope="module")
def pd_env():
    return build_env("deploy_pd")


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


def test_an_ideal_torque_preset_anchors_on_zero_torque():
    """The anchor is whatever a zero action commands, per actuator model."""
    env = build_env("sizing_ideal", actuator_overrides={"model": "ideal_torque"})
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
