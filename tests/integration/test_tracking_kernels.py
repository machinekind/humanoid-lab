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


# -- 1.2 tracking_relative -------------------------------------------------


def test_tracking_relative_is_off_by_default(env):
    r = env._config.reward
    assert r.tracking_relative is False
    assert r.tracking_rel_sigma == 0.25
    assert r.tracking_rel_floor_lin == 0.3
    assert r.tracking_rel_floor_ang == 0.4


def test_relative_kernels_are_command_magnitude_invariant(env, reset_state):
    """At rest the robot tracks 0% of any command, so the relative kernel
    must score both commands identically -- that invariance is the mechanism.
    Both yaw rates sit above tracking_rel_floor_ang (0.4), so neither is
    floored."""
    with reward_flags(env, tracking_relative=True):
        slow = rewards_for(env, reset_state, (0.0, 0.0, 0.8))
        fast = rewards_for(env, reset_state, (0.0, 0.0, 1.5))

    assert slow["tracking_ang_vel"] == pytest.approx(fast["tracking_ang_vel"], rel=1e-3)
    # exp(-cmd^2 / (rel_sigma * cmd^2)) = exp(-1/0.25) at any command.
    assert slow["tracking_ang_vel"] == pytest.approx(float(jp.exp(-4.0)), rel=1e-3)


def test_the_absolute_kernel_collapses_where_the_relative_one_does_not(env, reset_state):
    """The cliff being removed: the same 0%-tracked states score orders of
    magnitude apart under the absolute kernel, so a fast command has no
    reachable gradient."""
    slow = rewards_for(env, reset_state, (0.0, 0.0, 0.8))
    fast = rewards_for(env, reset_state, (0.0, 0.0, 1.5))
    assert fast["tracking_ang_vel"] < 0.01 * slow["tracking_ang_vel"]


def test_relative_lin_kernel_uses_the_planar_command_norm(env, reset_state):
    """denom_lin is norm(cmd_xy), so a command of the same magnitude split
    across x and y scores the same as one along x alone."""
    with reward_flags(env, tracking_relative=True):
        along_x = rewards_for(env, reset_state, (0.8, 0.0, 0.0))
        diagonal = rewards_for(env, reset_state, (0.8 / 2**0.5, 0.8 / 2**0.5, 0.0))
    assert along_x["tracking_lin_vel"] == pytest.approx(
        diagonal["tracking_lin_vel"], rel=1e-3
    )
    # Both are 0%-tracked, so both are exp(-1/rel_sigma). Scoring the
    # diagonal off cmd[0] alone would give exp(-8) instead.
    assert along_x["tracking_lin_vel"] == pytest.approx(float(jp.exp(-4.0)), rel=1e-3)


def test_a_zero_command_divides_by_the_floor_and_stays_finite(env, reset_state):
    with reward_flags(env, tracking_relative=True):
        r = rewards_for(env, reset_state, (0.0, 0.0, 0.0))
    for key in ("tracking_lin_vel", "tracking_ang_vel"):
        assert jp.isfinite(jp.asarray(r[key])), key
        # At rest under a zero command the error is 0, so any finite width
        # gives exactly 1.0. A zero width would give nan.
        assert r[key] == pytest.approx(1.0)


def test_small_commands_share_the_floored_width(env, reset_state):
    """Below tracking_rel_floor_ang (0.4) the width stops shrinking, so a
    tiny command cannot sharpen the kernel toward a cliff of its own."""
    with reward_flags(env, tracking_relative=True):
        floored = rewards_for(env, reset_state, (0.0, 0.0, 0.2))
        at_floor = rewards_for(env, reset_state, (0.0, 0.0, 0.4))
    # width = 0.25 * 0.4^2 = 0.04 for both; only the error differs.
    assert floored["tracking_ang_vel"] == pytest.approx(float(jp.exp(-0.04 / 0.04)), rel=1e-3)
    assert at_floor["tracking_ang_vel"] == pytest.approx(float(jp.exp(-0.16 / 0.04)), rel=1e-3)


def test_the_product_gate_composes_with_the_relative_branch(env, reset_state):
    """tracking_product runs after the branch, so it multiplies whichever
    pair of kernels the branch produced."""
    cmd = (0.5, 0.0, 0.5)
    with reward_flags(env, tracking_relative=True):
        pre = rewards_for(env, reset_state, cmd)
        with reward_flags(env, tracking_product=True):
            post = rewards_for(env, reset_state, cmd)

    expected = pre["tracking_lin_vel"] * pre["tracking_ang_vel"]
    assert post["tracking_lin_vel"] == pytest.approx(expected, rel=1e-6)
    assert post["tracking_ang_vel"] == pytest.approx(expected, rel=1e-6)


def test_tracking_relative_leaves_the_other_terms_untouched(env, reset_state):
    off = rewards_for(env, reset_state, _MIXED_CMD)
    with reward_flags(env, tracking_relative=True):
        on = rewards_for(env, reset_state, _MIXED_CMD)
    for key in off:
        if key.startswith("tracking_"):
            continue
        assert on[key] == off[key], key


# -- 1.3 tracking_far ------------------------------------------------------


def test_tracking_far_is_off_by_default(env):
    r = env._config.reward
    assert r.tracking_far_weight == 0.0
    assert r.tracking_far_sigma == 2.5


def test_tracking_far_is_live_under_absolute_kernels(env, reset_state):
    spin = (0.0, 0.0, 1.5)
    bare = rewards_for(env, reset_state, spin)
    with reward_flags(env, tracking_far_weight=0.25):
        blended = rewards_for(env, reset_state, spin)

    # bare absolute kernel exp(-2.25/0.25) ~ 1.2e-4, gradient-free; the far
    # term exp(-2.25/2.5) ~ 0.41 at weight 0.25 lifts it to ~0.10.
    assert bare["tracking_ang_vel"] < 1e-3
    assert blended["tracking_ang_vel"] > 0.1
    assert blended["tracking_ang_vel"] <= 1.0


def test_tracking_far_is_live_under_relative_kernels(env, reset_state):
    """The regression test, ported from w01-tek
    (training/tests/integration/test_env.py:151, commit 042ada4). The far
    mix-in must blend into the relative kernels too. It used to apply only in
    the absolute branch, so terrain_blind_v3 (tracking_relative with
    tracking_far dropped as inert) lost the far-field gradient that fixed
    stiff_b's dead spin. A robot at rest under a pure-spin command must score
    visibly higher tracking_ang_vel with the blend on."""
    spin = (0.0, 0.0, 1.5)
    with reward_flags(env, tracking_relative=True):
        bare = rewards_for(env, reset_state, spin)
        with reward_flags(env, tracking_far_weight=0.25):
            blended = rewards_for(env, reset_state, spin)

    # at rest: bare relative kernel ~exp(-1/rel_sigma), far term
    # ~exp(-2.25/2.5) -- the blend must lift the reward well clear
    assert blended["tracking_ang_vel"] > 3.0 * bare["tracking_ang_vel"]
    assert blended["tracking_ang_vel"] <= 1.0


def test_tracking_far_blends_the_linear_kernel_too(env, reset_state):
    fast = (1.5, 0.0, 0.0)
    bare = rewards_for(env, reset_state, fast)
    with reward_flags(env, tracking_far_weight=0.25):
        blended = rewards_for(env, reset_state, fast)
    assert blended["tracking_lin_vel"] > 3.0 * bare["tracking_lin_vel"]


def test_the_far_kernel_stays_absolute_across_the_branches(env, reset_state):
    """The far kernel reads the raw squared error, never the relative width,
    so a state far off the command sees the same pull at any commanded speed.
    Subtracting the (1-w)-weighted bare kernels leaves the same far term in
    both branches."""
    spin = (0.0, 0.0, 1.5)
    far_parts = {}
    for relative in (False, True):
        with reward_flags(env, tracking_relative=relative):
            bare = rewards_for(env, reset_state, spin)["tracking_ang_vel"]
            with reward_flags(env, tracking_far_weight=0.25):
                blended = rewards_for(env, reset_state, spin)["tracking_ang_vel"]
        far_parts[relative] = blended - 0.75 * bare

    assert far_parts[False] == pytest.approx(far_parts[True], rel=1e-3)
    assert far_parts[False] == pytest.approx(0.25 * float(jp.exp(-2.25 / 2.5)), rel=1e-3)


def test_tracking_far_leaves_the_other_terms_untouched(env, reset_state):
    off = rewards_for(env, reset_state, _MIXED_CMD)
    with reward_flags(env, tracking_far_weight=0.25):
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
