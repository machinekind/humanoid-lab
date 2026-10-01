"""Unit tests for the training-time respawn reseeds (envs/wrappers.py).

The wrappers only rewrite `info` leaves where `done` is set, so they are
tested here around a stub env that returns a prepared state, without
building a model. What they do inside a real training rollout is covered by
tests/integration/test_no_progress_env.py for the progress reseed.
"""

import jax.numpy as jp
import numpy as np
from mujoco_playground import wrapper as playground_wrapper
from mujoco_playground._src import mjx_env

from humanoid_lab.envs import wrappers
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


def test_wrap_env_fn_is_stock_with_both_reseeds_off():
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
