"""The velocity-tracking kernel block of `Joystick._compute_rewards`.

Port items 1.1 to 1.4 (see docs/port-details.md): `tracking_product`,
`tracking_relative`, `tracking_far_weight`, and `shaping_tracking_gate`. All
four are static config read at trace time, so a test flips the flag on the
live env and calls `_compute_rewards` again -- no rebuild, one model for the
whole file.

The env resets at rest, so a command is the only thing that sets the tracking
error: at rest `linvel` and `gyro` are ~0 and the error is the command
itself. That makes every kernel value below computable by hand from the
command, which is what these tests assert against.

The kernel math itself is unit-tested in tests/unit/test_rewards.py. What is
tested here is the composition: which kernel feeds which term, in which
order.
"""

from __future__ import annotations

import contextlib

import jax
import jax.numpy as jp
import pytest

from humanoid_lab import paths
from humanoid_lab.envs.joystick import Joystick, default_config

ROBOT_DIR = paths.ROBOTS_DIR / "asimov_v1"
PRESET = "sizing_ideal"


@pytest.fixture(scope="module")
def env():
    cfg = default_config()
    cfg.episode_length = 50  # fast tests, not a training config
    return Joystick(ROBOT_DIR, PRESET, cfg)


@pytest.fixture(scope="module")
def reset_state(env):
    return env.reset(jax.random.PRNGKey(0))


@contextlib.contextmanager
def reward_flags(env, **flags):
    """Set reward config keys for the body of the with-block and restore
    them after. Every flag here is read inside `_compute_rewards`, so a live
    env picks the new value up on the next call."""
    r = env._config.reward
    saved = {k: r[k] for k in flags}
    r.update(flags)
    try:
        yield
    finally:
        r.update(saved)


def rewards_for(env, reset_state, command, **info_overrides):
    """The reward dict at the reset state under `command`."""
    info = dict(reset_state.info)
    info["command"] = jp.asarray(command, dtype=float)
    info.update(info_overrides)
    n_feet = env._n_feet
    rewards, _fall = env._compute_rewards(
        reset_state.data,
        info,
        jp.zeros(env.action_size),
        jp.zeros(n_feet, dtype=bool),
        jp.zeros(n_feet, dtype=bool),
    )
    return {k: float(v) for k, v in rewards.items()}


# -- 1.1 tracking_product --------------------------------------------------

# Both kernels land mid-range at rest under this command (err 0.25 each, so
# exp(-1) ~ 0.368), which keeps the product distinguishable from either
# factor and from a sequential reassignment.
_MIXED_CMD = (0.5, 0.0, 0.5)


def test_tracking_product_is_off_by_default(env):
    assert env._config.reward.tracking_product is False


def test_tracking_product_off_leaves_the_kernels_independent(env, reset_state):
    r = rewards_for(env, reset_state, _MIXED_CMD)
    # Additive tracking: standing still under a half-linear command still
    # scores each kernel on its own error.
    assert r["tracking_lin_vel"] == pytest.approx(r["tracking_ang_vel"], rel=1e-3)
    assert r["tracking_lin_vel"] > 0.3


def test_tracking_product_collapses_the_untracked_half(env, reset_state):
    """A pure-spin command that the robot ignores: standing still tracks the
    zero linear command perfectly and earns the full tracking_lin_vel under
    additive tracking. The product must take that payout away."""
    spin = (0.0, 0.0, 1.5)
    additive = rewards_for(env, reset_state, spin)
    with reward_flags(env, tracking_product=True):
        product = rewards_for(env, reset_state, spin)

    assert additive["tracking_lin_vel"] > 0.99  # perfect zero-linear tracking
    assert product["tracking_lin_vel"] < 1e-3
    assert product["tracking_ang_vel"] < 1e-3


def test_tracking_product_reassignment_is_simultaneous(env, reset_state):
    """`k_lin, k_ang = k_lin * k_ang, k_ang * k_lin` -- both sides read the
    PRE-product values, so both kernels come out equal to the same product. A
    sequential assignment would fold the new k_lin into k_ang and give
    ang * lin * ang."""
    pre = rewards_for(env, reset_state, _MIXED_CMD)
    with reward_flags(env, tracking_product=True):
        post = rewards_for(env, reset_state, _MIXED_CMD)

    expected = pre["tracking_lin_vel"] * pre["tracking_ang_vel"]
    assert post["tracking_lin_vel"] == pytest.approx(expected, rel=1e-6)
    assert post["tracking_ang_vel"] == pytest.approx(expected, rel=1e-6)


def test_tracking_product_leaves_the_other_terms_untouched(env, reset_state):
    off = rewards_for(env, reset_state, _MIXED_CMD)
    with reward_flags(env, tracking_product=True):
        on = rewards_for(env, reset_state, _MIXED_CMD)

    for key in off:
        if key.startswith("tracking_"):
            continue
        assert on[key] == off[key], key


def test_tracking_product_keeps_the_reward_key_order(env, reset_state):
    """The reward sum's float addition order follows dict insertion order, so
    the product must not move or add a key."""
    info = dict(reset_state.info)
    n_feet = env._n_feet
    args = (
        reset_state.data,
        info,
        jp.zeros(env.action_size),
        jp.zeros(n_feet, dtype=bool),
        jp.zeros(n_feet, dtype=bool),
    )
    off, _ = env._compute_rewards(*args)
    with reward_flags(env, tracking_product=True):
        on, _ = env._compute_rewards(*args)
    assert list(on) == list(off)
