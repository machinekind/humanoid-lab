"""Unit tests for the training-time respawn reseeds (envs/wrappers.py).

The wrappers only rewrite `info` leaves where `done` is set, so they are
tested here around a stub env that returns a prepared state, without
building a model. What they do inside a real training rollout is covered by
tests/integration/test_no_progress_env.py for the progress reseed.
"""

import jax.numpy as jp
import numpy as np
import pytest
from mujoco_playground import wrapper as playground_wrapper
from mujoco_playground._src import mjx_env

from humanoid_lab.envs import gait_state, signals, wrappers
from humanoid_lab.envs.joystick import default_config, gait_dur_ema_on


class _StubEnv:
    """Steps by marking the prepared envs done, leaving info as it is."""

    def __init__(self, done):
        self._done = jp.asarray(done, dtype=jp.float32)

    def step(self, state, action):
        return state.replace(done=self._done)


def _gait_info():
    # Two envs, two feet: mid-episode values the respawn must clear.
    return {
        "feet_air_time": jp.array([[0.3, 0.0], [0.3, 0.0]]),
        "feet_contact_time": jp.array([[0.0, 0.4], [0.0, 0.4]]),
        "swing_apex": jp.array([[0.08, 0.0], [0.08, 0.0]]),
        "last_apex": jp.array([[0.06, 0.07], [0.06, 0.07]]),
        "last_contact": jp.array([[False, True], [False, True]]),
        "air_dur_ema": jp.array([[0.35, 0.3], [0.35, 0.3]]),
        "stance_dur_ema": jp.array([[0.4, 0.45], [0.4, 0.45]]),
        "command": jp.array([[0.5, 0.0, 0.0], [0.5, 0.0, 0.0]]),
    }


def _state(info):
    return mjx_env.State(
        data=None,
        obs=jp.zeros((2, 1)),
        reward=jp.zeros(2),
        done=jp.zeros(2),
        metrics={},
        info=info,
    )


def test_gait_reseed_restarts_the_trackers_of_respawned_envs_only():
    before = _gait_info()
    out = wrappers.GaitReseedWrapper(_StubEnv([1.0, 0.0])).step(_state(before), None)

    for k in wrappers.GAIT_TRACKER_KEYS:
        np.testing.assert_array_equal(
            out.info[k][0], np.zeros_like(before[k][0]), err_msg=k
        )
        np.testing.assert_array_equal(out.info[k][1], before[k][1], err_msg=k)
        assert out.info[k].dtype == before[k].dtype, k


def test_gait_reseed_leaves_other_info_alone():
    before = _gait_info()
    out = wrappers.GaitReseedWrapper(_StubEnv([1.0, 1.0])).step(_state(before), None)
    np.testing.assert_array_equal(out.info["command"], before["command"])


def test_gait_dur_ema_on_follows_either_symmetry_scale():
    scales = default_config().reward.scales
    assert not gait_dur_ema_on(scales)
    scales.gait_symmetry = -1.0
    assert gait_dur_ema_on(scales)
    scales.gait_symmetry = 0.0
    scales.gait_symmetry_income = 0.5
    assert gait_dur_ema_on(scales)


def test_wrap_env_fn_is_stock_with_every_reseed_off():
    cfg = default_config()
    assert wrappers.make_wrap_env_fn(cfg) is playground_wrapper.wrap_for_brax_training


def test_wrap_env_fn_adds_the_gait_reseed_for_a_symmetry_term():
    cfg = default_config()
    cfg.reward.scales.gait_symmetry_income = 0.5
    env = wrappers.make_wrap_env_fn(cfg)(_StubEnv([0.0]))
    assert isinstance(env, wrappers.GaitReseedWrapper)
    assert isinstance(env.env, playground_wrapper.BraxAutoResetWrapper)


def test_wrap_env_fn_stacks_both_reseeds():
    cfg = default_config()
    cfg.no_progress.enable = True
    cfg.reward.scales.gait_symmetry = -1.0
    env = wrappers.make_wrap_env_fn(cfg)(_StubEnv([0.0]))
    assert isinstance(env, wrappers.GaitReseedWrapper)
    assert isinstance(env.env, wrappers.ProgressReseedWrapper)
    assert isinstance(env.env.env, playground_wrapper.BraxAutoResetWrapper)


class _YawEnv(_StubEnv):
    """A stub that also answers `unwrapped` and `_quat` for the heading layer."""

    def __init__(self, done, yaw):
        super().__init__(done)
        self._yaw = yaw

    @property
    def unwrapped(self):
        return self

    def _quat(self, data):
        half = jp.asarray(self._yaw) / 2
        return jp.stack([jp.cos(half), jp.zeros_like(half), jp.zeros_like(half), jp.sin(half)], axis=-1)


def test_heading_reseed_rewrites_yaw_ref_of_respawned_envs_only():
    info = {**_gait_info(), "yaw_ref": jp.array([1.5, 1.5])}
    stub = _YawEnv([1.0, 0.0], [0.2, 0.9])
    out = wrappers.HeadingReseedWrapper(stub).step(_state(info), None)

    assert float(out.info["yaw_ref"][0]) == pytest.approx(0.2, abs=1e-6)
    assert float(out.info["yaw_ref"][1]) == 1.5
    assert out.info["yaw_ref"].dtype == info["yaw_ref"].dtype
    np.testing.assert_array_equal(out.info["command"], info["command"])


def test_heading_reseed_value_is_the_shared_base_yaw():
    stub = _YawEnv([1.0, 1.0], [0.4, -2.0])
    info = {**_gait_info(), "yaw_ref": jp.array([0.0, 0.0])}
    out = wrappers.HeadingReseedWrapper(stub).step(_state(info), None)
    np.testing.assert_array_equal(out.info["yaw_ref"], signals.base_yaw(stub._quat(None)))


@pytest.mark.parametrize(
    "wrapper,flag,reset_keys",
    [
        (wrappers.ProgressReseedWrapper, "progress", ["progress_ema"]),
        (wrappers.GaitReseedWrapper, "gait", []),
        (wrappers.HeadingReseedWrapper, "heading", ["yaw_ref"]),
        (wrappers.LandReseedWrapper, "land", []),
    ],
)
def test_each_reseed_clears_through_gait_state(monkeypatch, wrapper, flag, reset_keys):
    # The clearing is written once, in gait_state.clear_done. A wrapper that
    # cleared info itself would pass the behavior tests above but not this one.
    calls = []

    def record(info, done, flags, reset_values):
        calls.append((flags, sorted(reset_values), np.asarray(done).tolist()))
        return info

    monkeypatch.setattr(gait_state, "clear_done", record)
    info = {**_gait_info(), "yaw_ref": jp.array([1.5, 1.5])}
    wrapper(_YawEnv([1.0, 0.0], [0.2, 0.9])).step(_state(info), None)
    assert calls == [((flag,), reset_keys, [True, False])]


def test_wrap_env_fn_adds_the_heading_layer_only_with_heading_err():
    cfg = default_config()
    assert wrappers.make_wrap_env_fn(cfg) is playground_wrapper.wrap_for_brax_training
    for block in ("state", "privileged"):
        cfg = default_config()
        cfg.obs[block] = tuple(cfg.obs[block]) + ("heading_err",)
        env = wrappers.make_wrap_env_fn(cfg)(_StubEnv([0.0]))
        assert isinstance(env, wrappers.HeadingReseedWrapper)
        assert isinstance(env.env, playground_wrapper.BraxAutoResetWrapper)


def test_heading_layer_stacks_outermost():
    cfg = default_config()
    cfg.no_progress.enable = True
    cfg.reward.scales.gait_symmetry = -1.0
    cfg.obs.state = tuple(cfg.obs.state) + ("heading_err",)
    env = wrappers.make_wrap_env_fn(cfg)(_StubEnv([0.0]))
    assert isinstance(env, wrappers.HeadingReseedWrapper)
    assert isinstance(env.env, wrappers.GaitReseedWrapper)
    assert isinstance(env.env.env, wrappers.ProgressReseedWrapper)
    assert isinstance(env.env.env.env, playground_wrapper.BraxAutoResetWrapper)


# -- the land layer --------------------------------------------------------


def _land_info():
    # Two envs, two feet, three slots: the last landing fractions.
    return {
        **_gait_info(),
        "land_frac": jp.array([[[0.5, 0.6, 0.7], [0.2, 0.3, 0.4]], [[0.5, 0.6, 0.7], [0.2, 0.3, 0.4]]]),
    }


def test_land_reseed_zeroes_land_frac_of_respawned_envs_only():
    before = _land_info()
    out = wrappers.LandReseedWrapper(_StubEnv([1.0, 0.0])).step(_state(before), None)
    np.testing.assert_array_equal(out.info["land_frac"][0], np.zeros((2, 3)))
    np.testing.assert_array_equal(out.info["land_frac"][1], before["land_frac"][1])
    assert out.info["land_frac"].dtype == before["land_frac"].dtype
    for k in before:
        if k != "land_frac":
            np.testing.assert_array_equal(out.info[k], before[k], err_msg=k)


def _land_cfg(term="step_length", scale=1.0, pair_min=True):
    cfg = default_config()
    cfg.reward.scales[term] = scale
    cfg.reward[term].pair_min = pair_min
    return cfg


@pytest.mark.parametrize("term", ["step_length", "foot_reach", "knee_extend"])
def test_wrap_env_fn_adds_the_land_layer_only_with_land_frac(term):
    env = wrappers.make_wrap_env_fn(_land_cfg(term))(_StubEnv([0.0]))
    assert isinstance(env, wrappers.LandReseedWrapper)
    assert isinstance(env.env, playground_wrapper.BraxAutoResetWrapper)
    # a scale without pair_min, or pair_min without a scale, keeps no land_frac
    for cfg in (_land_cfg(term, pair_min=False), _land_cfg(term, scale=0.0)):
        assert wrappers.make_wrap_env_fn(cfg) is playground_wrapper.wrap_for_brax_training


def test_wrap_env_fn_is_stock_for_forward_lean_alone():
    cfg = default_config()
    cfg.reward.scales.forward_lean = 1.0
    assert wrappers.make_wrap_env_fn(cfg) is playground_wrapper.wrap_for_brax_training


def test_land_layer_stacks_outermost():
    cfg = _land_cfg()
    cfg.no_progress.enable = True
    cfg.reward.scales.gait_symmetry = -1.0
    cfg.obs.state = tuple(cfg.obs.state) + ("heading_err",)
    env = wrappers.make_wrap_env_fn(cfg)(_StubEnv([0.0]))
    assert isinstance(env, wrappers.LandReseedWrapper)
    assert isinstance(env.env, wrappers.HeadingReseedWrapper)
    assert isinstance(env.env.env, wrappers.GaitReseedWrapper)
    assert isinstance(env.env.env.env, wrappers.ProgressReseedWrapper)
    assert isinstance(env.env.env.env.env, playground_wrapper.BraxAutoResetWrapper)
