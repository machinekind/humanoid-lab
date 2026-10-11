"""A respawn restarts the heading reference (HeadingReseedWrapper).

`BraxAutoResetWrapper(full_reset=False)` restores `data` and `obs` on done
and leaves `info` alone, so `info['yaw_ref']` would carry the dying
episode's heading into the respawn. The script below turns the robot to a
yaw of 0.7, holds a straight command so yaw_ref freezes there, drops the
base below the fall height, and watches the respawn. With the heading layer
the observation reads 0 afterwards. With playground's wrapper alone it reads
a large error, which is what proves the test can fail.
"""

from __future__ import annotations

import jax
import jax.numpy as jp
import numpy as np
import pytest
from mujoco_playground import wrapper as playground_wrapper

from humanoid_lab import paths
from humanoid_lab.envs import progress, signals
from humanoid_lab.envs.joystick import default_config
from humanoid_lab.envs.wrappers import HeadingReseedWrapper, make_wrap_env_fn
from humanoid_lab.registry import make_env

N = 2
YAW = 0.7


@pytest.fixture(scope="module")
def env():
    state = tuple(default_config().obs.state) + ("heading_err",)
    return make_env(
        "joystick",
        paths.ROBOTS_DIR / "asimov_v1",
        "deploy_pd",
        {
            "sim": {"backend": "jax"},
            "obs": {"state": state},
            "push": {"enable": False},
            "command": {"resample_steps": 100000},
            "episode_length": 1000,
        },
    )


def _script(env, wenv):
    """Run the turn, straight, fall script. Returns the trace the tests read."""
    reset, step = jax.jit(wenv.reset), jax.jit(wenv.step)
    a = jp.zeros((N, env.action_size))
    s = reset(jax.random.split(jax.random.PRNGKey(0), N))
    adr = env._base_qadr
    half = YAW / 2
    q = jp.array([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=s.data.qpos.dtype)
    s = s.replace(data=s.data.replace(qpos=s.data.qpos.at[:, adr + 3 : adr + 7].set(q)))
    out = {}

    # (1) turn: yaw_ref follows the yaw
    s.info["command"] = jp.tile(jp.array([0.0, 0.0, 0.6]), (N, 1))
    assert 0.6 > progress.TURN_WZ_THRESHOLD
    s = step(s, a)
    out["turn_yaw_ref"] = np.asarray(s.info["yaw_ref"])

    # (2) straight: yaw_ref freezes
    s.info["command"] = jp.tile(jp.array([0.3, 0.0, 0.0]), (N, 1))
    for _ in range(2):
        s = step(s, a)
    out["frozen_yaw_ref"] = np.asarray(s.info["yaw_ref"])
    assert not np.asarray(s.done).any()

    # (3) fall: base below fall.min_height
    low = s.data.qpos.at[:, adr + 2].set(0.1)
    s = s.replace(data=s.data.replace(qpos=low))
    s = step(s, a)
    out["done"] = np.asarray(s.done)
    out["respawn_yaw_ref"] = np.asarray(s.info["yaw_ref"])
    out["respawn_yaw"] = np.asarray(signals.base_yaw(env._quat(s.data)))

    # (4) the next steps
    errs = [np.asarray(s.obs["state"][:, -1])]
    for _ in range(3):
        s = step(s, a)
        errs.append(np.asarray(s.obs["state"][:, -1]))
    out["errs"] = np.stack(errs)
    return out


def test_the_heading_layer_is_outermost(env):
    wrap = make_wrap_env_fn(env._config)
    wenv = wrap(env, episode_length=1000, action_repeat=1)
    assert isinstance(wenv, HeadingReseedWrapper)


def test_a_respawn_restarts_the_heading_reference(env):
    wenv = make_wrap_env_fn(env._config)(env, episode_length=1000, action_repeat=1)
    out = _script(env, wenv)

    np.testing.assert_allclose(out["turn_yaw_ref"], YAW, atol=0.05)
    np.testing.assert_allclose(out["frozen_yaw_ref"], YAW, atol=0.1)
    assert out["done"].all()
    np.testing.assert_array_equal(out["respawn_yaw_ref"], out["respawn_yaw"])
    assert np.abs(out["errs"]).max() < 0.01, out["errs"]


def test_without_the_heading_layer_the_error_survives_the_respawn(env):
    """The control: playground's wrapper alone leaves yaw_ref at the old heading."""
    wenv = playground_wrapper.wrap_for_brax_training(env, episode_length=1000, action_repeat=1)
    out = _script(env, wenv)

    assert out["done"].all()
    assert np.abs(out["errs"][1:]).min() > 0.5, out["errs"]
