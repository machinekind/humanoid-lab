"""The no-progress termination wired into the env.

`envs/progress.py`'s math is unit-tested model-free in
tests/unit/test_no_progress.py. What is tested here is the wiring: the info
state and its two reseeds (command resample and respawn), the metrics, the
arming window measured in real control steps, and that the cut lands on
`done` without touching the reward.

`no_progress.enable` is static config read at trace time, so one model serves
the whole file: the flag block is flipped on the live env and the env is
reset again inside the flag context (reset is what seeds `progress_ema` and
the metrics).

The tests below craft `info` directly rather than waiting out a real 2 s
grace window: `progress_ema` and `steps_since_cmd` are exactly the state a
long ignored command would have produced, and driving them straight makes the
arming boundary a single step instead of a hundred. `p_max=1.0` turns the
bernoulli draw into a certainty, so "the hazard fired" is an assertion rather
than a sample.
"""

from __future__ import annotations

import contextlib

import jax
import jax.numpy as jp
import pytest

from mujoco_playground import wrapper as playground_wrapper

from humanoid_lab import paths
from humanoid_lab.envs import wrappers
from humanoid_lab.envs.joystick import Joystick, default_config

ROBOT_DIR = paths.ROBOTS_DIR / "asimov_v1"
PRESET = "sizing_ideal"

NEW_METRICS = ("no_progress_cut", "progress_ratio_per_step")
FORWARD = jp.array([0.8, 0.0, 0.0])


@pytest.fixture(scope="module")
def env():
    cfg = default_config()
    cfg.episode_length = 50  # fast tests, not a training config
    return Joystick(ROBOT_DIR, PRESET, cfg)


@contextlib.contextmanager
def block_values(block, **values):
    """Set keys on a live config block for the body of the with-block and
    restore them after. Every key here is read inside reset()/step(), so the
    env picks the new value up on the next call."""
    saved = {k: block[k] for k in values}
    block.update(values)
    try:
        yield
    finally:
        block.update(saved)


def cut_state(state, command=FORWARD, ema=0.0, steps_since_cmd=99):
    """The reset state re-labelled as an env that has been ignoring its
    command: zero smoothed progress against a real command, one step short of
    the grace window (99 -> 100 after step()'s increment, and 100*0.02 s is
    exactly grace_sec=2.0)."""
    info = dict(state.info)
    info["command"] = command
    info["progress_ema"] = jp.array(ema)
    info["steps_since_cmd"] = jp.array(steps_since_cmd)
    return state.replace(info=info)


def step_once(env, state):
    return env.step(state, jp.zeros(env.action_size))


# -- off: the mechanism does not exist -------------------------------------


def test_off_keeps_progress_state_out_of_info(env):
    state = env.reset(jax.random.PRNGKey(0))
    assert "progress_ema" not in state.info


def test_off_adds_no_metric_keys(env):
    state = env.reset(jax.random.PRNGKey(0))
    stepped = step_once(env, state)

    for key in NEW_METRICS:
        assert key not in state.metrics
        assert key not in stepped.metrics
    assert set(stepped.metrics) == set(state.metrics)


# -- on: state and metrics -------------------------------------------------


def test_reset_seeds_the_meter_at_ratio_one(env):
    """A fresh episode starts at progress ratio 1, so the hazard can only
    come from measured shortfall, never from the seed."""
    with block_values(env._config.no_progress, enable=True):
        state = env.reset(jax.random.PRNGKey(0))

        assert "progress_ema" in state.info
        assert float(state.info["progress_ema"]) == pytest.approx(
            float(env._cmd_speed(state.info["command"]))
        )


def test_metrics_keyset_is_identical_at_reset_and_after_a_step(env):
    """brax's training scan carries the metrics pytree; a key that appears
    only in step() changes its structure and blows up the scan."""
    with block_values(env._config.no_progress, enable=True):
        state = env.reset(jax.random.PRNGKey(0))
        stepped = step_once(env, state)

        for key in NEW_METRICS:
            assert key in state.metrics
            assert key in stepped.metrics
        assert set(stepped.metrics) == set(state.metrics)


def test_the_ratio_metric_is_clipped_to_two(env):
    """Over-delivery on a slow command must not average away shortfall
    elsewhere in the episode."""
    with block_values(env._config.no_progress, enable=True):
        state = env.reset(jax.random.PRNGKey(0))
        stepped = step_once(env, cut_state(state, ema=100.0, steps_since_cmd=0))

        assert float(stepped.metrics["progress_ratio_per_step"]) == pytest.approx(2.0)


# -- on: the arming window -------------------------------------------------


def test_no_cut_is_possible_before_the_grace_window_elapses(env):
    """Zero progress against a real command with a certain hazard: the only
    thing holding the cut back is the grace window."""
    with block_values(env._config.no_progress, enable=True, p_max=1.0, risk_below=1.0):
        state = env.reset(jax.random.PRNGKey(0))
        stepped = step_once(env, cut_state(state, steps_since_cmd=98))

        assert float(stepped.done) == 0.0
        assert float(stepped.metrics["no_progress_cut"]) == 0.0


def test_the_cut_fires_once_the_grace_window_has_elapsed(env):
    with block_values(env._config.no_progress, enable=True, p_max=1.0, risk_below=1.0):
        state = env.reset(jax.random.PRNGKey(0))
        stepped = step_once(env, cut_state(state, steps_since_cmd=99))

        assert float(stepped.done) == 1.0
        assert float(stepped.metrics["no_progress_cut"]) == 1.0


def test_a_zero_command_never_cuts_however_long_it_stands(env):
    """Standing still is what a zero command asks for; demand below 0.05
    never arms the cut."""
    with block_values(env._config.no_progress, enable=True, p_max=1.0, risk_below=1.0):
        state = env.reset(jax.random.PRNGKey(0))
        stepped = step_once(
            env, cut_state(state, command=jp.zeros(3), ema=-5.0, steps_since_cmd=5000)
        )

        assert float(stepped.done) == 0.0
        assert float(stepped.metrics["no_progress_cut"]) == 0.0


def test_progress_that_meets_the_command_never_cuts(env):
    """The hazard is zero at or above risk_below of demand, so a tracked
    command survives the same armed state that kills a stalled one."""
    with block_values(env._config.no_progress, enable=True, p_max=1.0):
        state = env.reset(jax.random.PRNGKey(0))
        stepped = step_once(
            env, cut_state(state, ema=float(env._cmd_speed(FORWARD)), steps_since_cmd=5000)
        )

        assert float(stepped.done) == 0.0


# -- on: the resample reseed -----------------------------------------------


def test_a_command_resample_restarts_the_meter_at_ratio_one(env):
    """A fresh command re-seeds the EMA to the new demand, so the robot is
    not billed for the previous command's shortfall."""
    with block_values(env._config.no_progress, enable=True), block_values(
        env._config.command, resample_steps=1
    ):
        state = env.reset(jax.random.PRNGKey(0))
        stepped = step_once(env, cut_state(state, ema=-5.0, steps_since_cmd=0))

        assert int(stepped.info["steps_since_cmd"]) == 0  # the resample fired
        assert float(stepped.info["progress_ema"]) == pytest.approx(
            float(env._cmd_speed(stepped.info["command"]))
        )


def test_without_a_resample_the_meter_keeps_its_history(env):
    """The counterpart: the reseed is the resample's doing, not something
    every step does."""
    with block_values(env._config.no_progress, enable=True):
        state = env.reset(jax.random.PRNGKey(0))
        stepped = step_once(env, cut_state(state, ema=-5.0, steps_since_cmd=0))

        assert int(stepped.info["steps_since_cmd"]) == 1  # no resample
        # One EMA step from -5 toward a standing robot's ~0 served progress:
        # alpha = dt/ema_sec = 0.02, so the meter is still deep in the red.
        assert float(stepped.info["progress_ema"]) < -4.0


# -- on: the respawn ---------------------------------------------------------
#
# The trainer's env is wrapped by mujoco_playground's
# `wrap_for_brax_training`, which ends in `BraxAutoResetWrapper(full_reset=
# False)`. That wrapper restores `data` and `obs` from the cached first state
# on done and returns `state.info` untouched -- its own docstring says "only
# data and obs are reset, not the environment info". So a cut env respawns
# carrying the dying episode's meter unless something puts it back.


def wrap(env, reseed: bool):
    """The trainer's wrapping, with or without the reseed layer. One env, one
    episode: `episode_length` is high enough that the truncation never fires
    inside these tests."""
    wrap_fn = wrappers.make_wrap_env_fn(env._config) if reseed else (
        playground_wrapper.wrap_for_brax_training
    )
    return wrap_fn(env, episode_length=10_000, action_repeat=1, randomization_fn=None)


def cut_a_wrapped_episode(wrapped, env):
    """Reset the wrapped env, relabel its single env as one that has been
    ignoring its command, and step it once into the cut."""
    state = wrapped.reset(jax.random.split(jax.random.PRNGKey(0), 1))
    info = dict(state.info)
    info["command"] = FORWARD[None]
    info["progress_ema"] = jp.zeros(1)
    info["steps_since_cmd"] = jp.full(1, 99, dtype=info["steps_since_cmd"].dtype)
    stepped = wrapped.step(state.replace(info=info), jp.zeros((1, env.action_size)))

    assert float(stepped.done[0]) == 1.0  # the premise: this episode was cut
    return stepped


def test_the_stock_auto_reset_leaves_the_respawn_with_the_dead_meter(env):
    """What the installed wrapper actually does, pinned. This is the failure
    the reseed exists for: the respawn is armed on its very first step (a cut
    can only fire past the grace window, and steps_since_cmd carries over)
    with a meter deep enough in the red to cut again inside a second."""
    with block_values(env._config.no_progress, enable=True, p_max=1.0, risk_below=1.0):
        stepped = cut_a_wrapped_episode(wrap(env, reseed=False), env)

        assert float(stepped.info["progress_ema"][0]) < 0.1
        assert int(stepped.info["steps_since_cmd"][0]) == 100


def test_the_reseed_wrapper_respawns_the_meter_at_ratio_one(env):
    """The same respawn under the trainer's own wrapping: EMA back at the
    command's demand and the grace window back on, exactly what a command
    resample does."""
    with block_values(env._config.no_progress, enable=True, p_max=1.0, risk_below=1.0):
        stepped = cut_a_wrapped_episode(wrap(env, reseed=True), env)

        assert float(stepped.info["progress_ema"][0]) == pytest.approx(
            float(env._cmd_speed(FORWARD))
        )
        assert int(stepped.info["steps_since_cmd"][0]) == 0


def test_a_live_episode_keeps_its_meter_under_the_reseed_wrapper(env):
    """The reseed is the respawn's doing. A step that does not end an episode
    goes through untouched, so the wrapper cannot launder a shortfall."""
    with block_values(env._config.no_progress, enable=True):
        wrapped = wrap(env, reseed=True)
        state = wrapped.reset(jax.random.split(jax.random.PRNGKey(0), 1))
        info = dict(state.info)
        info["command"] = FORWARD[None]
        info["progress_ema"] = jp.full(1, -5.0)
        info["steps_since_cmd"] = jp.full(1, 40, dtype=info["steps_since_cmd"].dtype)
        stepped = wrapped.step(state.replace(info=info), jp.zeros((1, env.action_size)))

        assert float(stepped.done[0]) == 0.0
        assert float(stepped.info["progress_ema"][0]) < -4.0
        assert int(stepped.info["steps_since_cmd"][0]) == 41


def test_the_reseed_layer_is_not_wrapped_around_a_run_with_the_cut_off(env):
    """Off, the trainer's wrapping is the stock function itself -- the same
    object, so the training path a golden run takes cannot have moved."""
    assert env._config.no_progress.enable is False
    assert (
        wrappers.make_wrap_env_fn(env._config)
        is playground_wrapper.wrap_for_brax_training
    )


# -- on: the cut is a termination, not a reward -----------------------------


def test_the_cut_costs_the_episode_and_nothing_else(env):
    """A true termination: it flips done, the fall-only termination reward
    stays at zero because nothing fell, and no reward key is added."""
    scales = set(default_config().reward.scales)
    with block_values(env._config.no_progress, enable=True, p_max=1.0, risk_below=1.0):
        state = env.reset(jax.random.PRNGKey(0))
        stepped = step_once(env, cut_state(state, steps_since_cmd=99))

        assert float(stepped.done) == 1.0
        assert float(stepped.metrics["reward/termination"]) == 0.0

        assert "no_progress" not in scales
        assert set(env._config.reward.scales) == scales
        reward_keys = {k[len("reward/") :] for k in stepped.metrics if k.startswith("reward/")}
        assert reward_keys == scales
