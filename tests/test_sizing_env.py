"""Build-order step 7 gate (part 1): the sizing task constructs on
robots/asimov_v1 and runs a jitted reset/step without NaN, carries the three
sizing/* metrics identically through reset() and step() (scan-carry
parity), and penalizes torque/energy 5x joystick's starting scales.

CPU-fast on purpose, mirroring tests/test_env_joystick.py's shape.
"""

from __future__ import annotations

import jax
import jax.numpy as jp
import pytest

from humanoid_lab import paths
from humanoid_lab.envs.joystick import default_config as joystick_default_config
from humanoid_lab.envs.sizing import Sizing, default_config

ROBOT_DIR = paths.ROBOTS_DIR / "asimov_v1"
PRESET = "sizing_ideal"

SIZING_METRIC_KEYS = {"sizing/tau_frac_max", "sizing/omega_frac_max", "sizing/mech_power"}


@pytest.fixture(scope="module")
def env():
    cfg = default_config()
    cfg.episode_length = 50  # fast tests, not a training config
    return Sizing(ROBOT_DIR, PRESET, cfg)


def test_reward_scales_are_5x_joystick():
    sizing_cfg = default_config()
    joystick_cfg = joystick_default_config()

    assert sizing_cfg.reward.scales.torques == pytest.approx(
        joystick_cfg.reward.scales.torques * 5.0
    )
    assert sizing_cfg.reward.scales.energy == pytest.approx(
        joystick_cfg.reward.scales.energy * 5.0
    )
    # Every other reward scale is untouched.
    for key in joystick_cfg.reward.scales:
        if key in ("torques", "energy"):
            continue
        assert sizing_cfg.reward.scales[key] == pytest.approx(joystick_cfg.reward.scales[key]), key


def test_reset_gives_finite_obs_and_sizing_metrics(env):
    state = env.reset(jax.random.PRNGKey(0))

    assert jp.all(jp.isfinite(state.obs["state"]))
    assert jp.all(jp.isfinite(state.obs["privileged_state"]))
    assert SIZING_METRIC_KEYS.issubset(state.metrics.keys())
    for key in SIZING_METRIC_KEYS:
        assert bool(jp.isfinite(state.metrics[key]))


def test_step_is_finite_and_metrics_keyset_matches_reset(env):
    state = env.reset(jax.random.PRNGKey(0))
    step_fn = jax.jit(env.step)
    action = jp.zeros(env.action_size)
    next_state = step_fn(state, action)

    # CRITICAL for brax's scan-carry: the metrics pytree structure must not
    # change shape between reset() and step(), sizing/* keys included.
    assert set(next_state.metrics.keys()) == set(state.metrics.keys())
    assert SIZING_METRIC_KEYS.issubset(next_state.metrics.keys())
    assert bool(jp.isfinite(next_state.reward))
    for key in SIZING_METRIC_KEYS:
        assert bool(jp.isfinite(next_state.metrics[key]))


def test_sizing_metrics_are_nonnegative_fractions_and_power(env):
    """tau_frac_max/omega_frac_max are |value|/cap ratios (>=0, though not
    capped at 1.0 -- sizing_ideal's generous caps mean a smoke policy can
    still exceed them); mech_power is a sum of |tau*omega| (>=0)."""
    state = env.reset(jax.random.PRNGKey(0))
    step_fn = jax.jit(env.step)
    action = jp.ones(env.action_size) * 0.5
    next_state = step_fn(state, action)

    assert float(next_state.metrics["sizing/tau_frac_max"]) >= 0.0
    assert float(next_state.metrics["sizing/omega_frac_max"]) >= 0.0
    assert float(next_state.metrics["sizing/mech_power"]) >= 0.0
