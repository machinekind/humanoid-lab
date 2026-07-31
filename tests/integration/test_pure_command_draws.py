"""Pure command draws wired into the env (port item 1.6).

The five draws (wz, vy, slow, fast, back) are RNG-coupled sampling logic with
no piece worth extracting, so everything is tested through the env's own
`_sample_command` and through `step`'s resample branch. `_sample_command`
reads its config at trace time and each draw is gated by a plain `if p:`, so
one model serves the whole file: the probabilities are flipped on the live
config block and the sampler is called (or vmapped) inside the flag context.

What the tests pin down:
- With every probability at zero, and with the keys absent entirely, the
  sampler is bit-identical to the pre-1.6 draw. `test_golden_baseline.py` is
  the wider gate; this is the local one.
- Each draw folds its own index into `_sample_command`'s own rng, so enabling
  one draw cannot move another draw's stream.
- Each draw is clean (the axes it does not own are exactly 0) and in range.
- The zero-command overwrite still runs last, after every pure draw.
"""

from __future__ import annotations

import contextlib

import jax
import jax.numpy as jp
import numpy as np
import pytest

from humanoid_lab import paths
from humanoid_lab.envs.joystick import Joystick, default_config

ROBOT_DIR = paths.ROBOTS_DIR / "asimov_v1"
PRESET = "sizing_ideal"

PROB_KEYS = (
    "pure_wz_prob",
    "pure_vy_prob",
    "pure_slow_prob",
    "pure_fast_prob",
    "pure_back_prob",
)

# One fixed bank of keys, vmapped: 128 draws is enough that a stream that
# moved shows up, and a single trace keeps the file fast.
KEYS = jax.random.split(jax.random.PRNGKey(0), 128)


@pytest.fixture(scope="module")
def env():
    cfg = default_config()
    cfg.episode_length = 50  # fast tests, not a training config
    return Joystick(ROBOT_DIR, PRESET, cfg)


@contextlib.contextmanager
def block_values(block, **values):
    """Set keys on a live config block for the body and restore them after."""
    saved = {k: block[k] for k in values}
    block.update(values)
    try:
        yield
    finally:
        block.update(saved)


@contextlib.contextmanager
def without_keys(block, *keys):
    """Delete keys for the body of the with-block: a resolved config recorded
    before 1.6 landed, which is what the sampler's `c.get(key, 0.0)` defaults
    are there to serve. The env locks its config, and a locked ConfigDict
    refuses both the delete and the restore, hence the unlock."""
    saved = {k: block[k] for k in keys}
    with block.unlocked():
        for key in keys:
            del block[key]
        try:
            yield
        finally:
            block.update(saved)


def samples(env, keys=KEYS):
    """(n, 3) commands, one per key, drawn under the config as it stands."""
    return np.asarray(jax.vmap(env._sample_command)(keys))


# -- off: the draws do not exist -------------------------------------------


def test_all_probabilities_zero_matches_a_config_without_the_keys(env):
    """Off is not "draws nothing useful", it is "does not run": no key is
    folded in, so the sampler's output is the pre-1.6 output bit for bit."""
    with block_values(env._config.command, **{k: 0.0 for k in PROB_KEYS}):
        with_keys = samples(env)
    with without_keys(env._config.command, *PROB_KEYS):
        legacy = samples(env)

    np.testing.assert_array_equal(with_keys, legacy)


# -- the draw keys live outside the base split ------------------------------


def test_no_draw_key_aliases_a_key_from_the_base_split(env, monkeypatch):
    """`fold_in(key, i)` IS `split(key, n)[i]` for every i < n -- measured, not
    assumed (see the non-vacuity guard below). So a draw folding in its small
    table index would key off one of `_sample_command`'s own base split keys,
    and a draw that used the folded key directly instead of re-splitting it
    would ship a selector that is a deterministic function of another axis's
    value. The offset domain is what keeps that impossible.

    The indices are captured from the sampler itself rather than copied from
    the source, so this fails if a draw is ever added on a raw index."""
    seen = []
    real_fold_in = jax.random.fold_in

    def spy(key, data):
        seen.append(int(data))
        return real_fold_in(key, data)

    monkeypatch.setattr(jax.random, "fold_in", spy)
    with block_values(env._config.command, **{k: 0.5 for k in PROB_KEYS}):
        env._sample_command(jax.random.PRNGKey(11))

    assert len(seen) == len(PROB_KEYS)  # every draw ran
    assert len(set(seen)) == len(seen)  # and none shares an index

    probe = jax.random.PRNGKey(7)
    # A 5-way split covers the base 4-way split and the no_progress step's
    # own arity, so widening either one cannot silently collide with a draw.
    base = [np.asarray(jax.random.key_data(k)) for k in jax.random.split(probe, 5)]
    for index in seen:
        folded = np.asarray(jax.random.key_data(real_fold_in(probe, index)))
        for j, key in enumerate(base):
            assert not np.array_equal(folded, key), f"draw index {index} aliases split key {j}"

    # Not vacuous: the raw table indices this offset replaces DO alias. 1, 2
    # and 3 are exactly the base draw's vy, wz and zero_prob keys.
    for idx in range(1, 5):
        raw = np.asarray(jax.random.key_data(real_fold_in(probe, idx)))
        assert np.array_equal(raw, base[idx])


# -- stream independence ----------------------------------------------------


@pytest.mark.parametrize(
    "earlier", ["pure_wz_prob", "pure_vy_prob", "pure_slow_prob", "pure_fast_prob"]
)
def test_an_earlier_draw_does_not_perturb_the_back_draw(env, earlier):
    """Every draw folds its own index into `_sample_command`'s argument, so
    an enabled draw consumes nothing from any other draw's stream. `back` is
    the last of the five and overwrites whatever ran before it, so an
    identical result proves the earlier draw's keys came from somewhere else
    entirely -- not from the split that feeds `back`."""
    with block_values(env._config.command, zero_prob=0.0, pure_back_prob=1.0):
        back_only = samples(env)
    with block_values(env._config.command, zero_prob=0.0, pure_back_prob=1.0, **{earlier: 1.0}):
        with_earlier = samples(env)

    np.testing.assert_array_equal(back_only, with_earlier)
    # Not vacuous: these really are back draws, not the base uniform.
    assert (back_only[:, 0] < 0.0).all()


# -- each draw is clean and in range ---------------------------------------


@pytest.mark.parametrize(
    "prob_key, axis, range_key",
    [
        ("pure_wz_prob", 2, "wz"),
        ("pure_vy_prob", 1, "vy"),
        ("pure_slow_prob", 0, "slow_vx"),
        ("pure_fast_prob", 0, "fast_vx"),
        ("pure_back_prob", 0, "back_vx"),
    ],
)
def test_a_draw_at_probability_one_is_clean_and_in_range(env, prob_key, axis, range_key):
    """Clean is the whole point: the axes the draw does not own are exactly
    0.0, so the command is a corner of the box and not a contaminated one."""
    with block_values(env._config.command, zero_prob=0.0, **{prob_key: 1.0}):
        drawn = samples(env)
        lo, hi = env._config.command[range_key]

    other = [i for i in range(3) if i != axis]
    assert (drawn[:, other] == 0.0).all()
    assert (drawn[:, axis] >= lo).all()
    assert (drawn[:, axis] <= hi).all()
    # A redraw that silently returned zeros would satisfy the bounds for
    # pure_wz/pure_vy; a continuous draw never lands exactly on 0.0.
    assert (drawn[:, axis] != 0.0).all()


# -- ordering ---------------------------------------------------------------


@pytest.mark.parametrize("prob_key", PROB_KEYS)
def test_the_zero_overwrite_still_runs_last(env, prob_key):
    """Standing still wins over every pure draw: the zero overwrite is the
    final statement of the sampler."""
    with block_values(env._config.command, zero_prob=1.0, **{prob_key: 1.0}):
        drawn = samples(env)

    np.testing.assert_array_equal(drawn, np.zeros_like(drawn))


# -- the resample path ------------------------------------------------------


def test_a_resampled_command_is_drawn_the_same_way(env):
    """step()'s resample calls the same sampler, so a mid-episode command is
    as clean as a reset one. Nothing about the draws is reset-only."""
    with block_values(
        env._config.command, zero_prob=0.0, pure_back_prob=1.0, resample_steps=1
    ):
        state = env.reset(jax.random.PRNGKey(3))
        stepped = env.step(state, jp.zeros(env.action_size))
        lo, hi = env._config.command.back_vx

    assert int(stepped.info["steps_since_cmd"]) == 0  # the resample fired
    command = np.asarray(stepped.info["command"])
    assert command[1] == 0.0 and command[2] == 0.0
    assert lo <= command[0] <= hi
