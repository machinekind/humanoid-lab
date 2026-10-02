"""make_wrap_env_fn: which training stack a config gets, and the terrain
wrapper's warp telemetry, model-free.

The stacks are swapped for stubs that record their call, and the telemetry
reads warp-shaped stub data, so nothing here builds an env.
"""

from __future__ import annotations

import types

import jax
import jax.numpy as jp
import numpy as np
from mujoco_playground import wrapper as playground_wrapper

from humanoid_lab.envs import terrain_wrapper, wrappers
from humanoid_lab.envs.joystick import default_config as joystick_default_config
from humanoid_lab.envs.terrain_joystick import TELEMETRY_METRICS, TELEMETRY_PEAKS
from humanoid_lab.envs.terrain_joystick import default_config as terrain_default_config


def config(task, no_progress=False):
    cfg = terrain_default_config() if task == "terrain" else joystick_default_config()
    cfg.no_progress.enable = no_progress
    return cfg


def recording_stack(calls, name):
    def stack(env, **kwargs):
        calls.append((name, env, kwargs))
        return types.SimpleNamespace(stack=name)

    return stack


def test_a_flat_config_returns_the_stock_function():
    assert wrappers.make_wrap_env_fn(config("joystick")) is playground_wrapper.wrap_for_brax_training


def test_a_terrain_config_returns_the_terrain_stack():
    fn = wrappers.make_wrap_env_fn(config("terrain"))
    assert fn is terrain_wrapper.wrap_for_terrain_brax_training


def test_no_progress_wraps_outside_the_terrain_stack(monkeypatch):
    """The reseed is the outermost layer on either stack, and brax's keyword
    arguments reach the stack untouched."""
    calls = []
    monkeypatch.setattr(wrappers, "wrap_for_terrain_brax_training", recording_stack(calls, "terrain"))
    monkeypatch.setattr(
        wrappers.playground_wrapper, "wrap_for_brax_training", recording_stack(calls, "stock")
    )
    env = object()
    for task, stack in (("terrain", "terrain"), ("joystick", "stock")):
        calls.clear()
        wrapped = wrappers.make_wrap_env_fn(config(task, no_progress=True))(
            env, episode_length=200, action_repeat=1, randomization_fn=None
        )
        assert isinstance(wrapped, wrappers.ProgressReseedWrapper)
        assert wrapped.env.stack == stack
        assert calls == [(stack, env, {"episode_length": 200, "action_repeat": 1, "randomization_fn": None})]


def telemetry_stub(n):
    """A wrapper holding only its telemetry state, and its counter tracking
    under jit, as in the training step. The counters are warp-shaped stub
    data: one row count per env and the two pool scalars."""
    wrapper = object.__new__(terrain_wrapper.TerrainAutoResetWrapper)
    wrapper._caps = (100, 1000, 1000)
    wrapper._warned = set()

    @jax.jit
    def track(info, nefc, nacon, ncollision):
        impl = types.SimpleNamespace(nefc=nefc, nacon=nacon, ncollision=ncollision)
        info, metrics = dict(info), {}
        wrapper._track_counters(types.SimpleNamespace(_impl=impl), info, metrics)
        return info, metrics

    zeros = {k: jp.zeros(n, jp.int32) for k in TELEMETRY_PEAKS}
    return track, zeros


def test_warp_telemetry_writes_batch_peaks_and_warns_once(capsys):
    """The batch peaks land in every env's info, the metrics carry the row
    peak and the pool fractions, and each counter warns once, at its first
    90% crossing."""
    n = 4
    track, info = telemetry_stub(n)
    for _ in range(2):
        info, metrics = track(info, np.array([10, 95, 3, 7]), np.array([400]), np.array([950]))
        jax.effects_barrier()
    assert {k: np.asarray(info[k]).tolist() for k in TELEMETRY_PEAKS} == {
        "nefc_peak": [95] * n, "nacon_peak": [400] * n, "ncollision_peak": [950] * n,
    }
    nefc, nacon, ncollision = TELEMETRY_METRICS
    np.testing.assert_allclose(metrics[nefc], 95.0)
    np.testing.assert_allclose(metrics[nacon], 0.4)
    np.testing.assert_allclose(metrics[ncollision], 0.95)
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 2
    assert "nefc peaked at 95 of njmax 100" in lines[0]
    assert "ncollision peaked at 950 of the naconmax pool 1000" in lines[1]


def test_warp_telemetry_warns_once_across_resets(capsys):
    """A reset zeroes the info peaks, as brax's evaluator does before every
    eval, so the traced crossing fires again. The wrapper prints the rows
    once all the same. A counter that first crosses after the reset still
    warns."""
    track, zeros = telemetry_stub(2)
    track(zeros, np.array([95, 0]), np.array([10]), np.array([10]))
    jax.effects_barrier()
    track(zeros, np.array([0, 96]), np.array([950]), np.array([10]))
    jax.effects_barrier()
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 2
    assert "nefc peaked at 95 of njmax 100" in lines[0]
    assert "nacon peaked at 950 of the naconmax pool 1000" in lines[1]
