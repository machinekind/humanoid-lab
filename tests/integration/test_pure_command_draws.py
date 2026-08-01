"""Pure command draws wired into the env.

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
- Each draw folds an index of its own, offset out of the base split's domain,
  into `_sample_command`'s own rng. So no draw can key off a base split key,
  and enabling one draw moves neither the base sample nor another draw's
  stream. Both of those are tested at a probability BELOW 1: at p=1 the draw
  overwrites every row and a stolen key is invisible.
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
from humanoid_lab.registry import make_env

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


def fired(drawn):
    """Rows a pure draw rewrote. Every draw leaves exactly one nonzero axis
    (`pure_wz` zeroes the linear part, the rest zero everything but vx),
    while the base uniform lands on all three -- a continuous draw never
    returns exactly 0.0. Every test below sets `zero_prob=0.0`, which keeps
    the standing overwrite (all three axes zero) out of the picture."""
    return (drawn != 0.0).sum(axis=1) == 1


@pytest.mark.parametrize("prob_key", PROB_KEYS)
def test_a_draw_leaves_the_rows_it_did_not_redraw_untouched(env, prob_key):
    """A draw at p=0.5 rewrites about half the keys and must leave the other
    half exactly as the all-off sampler drew them. Those surviving rows are
    the only window onto the base split there is: a draw that took a key out
    of the base split unconditionally would move
    every base sample, including the ones it does not overwrite.

    The probability has to be below 1 for this to see anything. At p=1 the
    draw overwrites every row and there is nothing left to compare."""
    with block_values(env._config.command, zero_prob=0.0):
        baseline = samples(env)
    with block_values(env._config.command, zero_prob=0.0, **{prob_key: 0.5}):
        drawn = samples(env)

    survived = ~fired(drawn)
    # Not vacuous in either direction: the draw fired, and it left rows.
    assert 0 < survived.sum() < len(drawn)
    np.testing.assert_array_equal(drawn[survived], baseline[survived])


def test_two_draws_fire_on_different_rows_and_keep_doing_so_together(env):
    """`slow` and `back` at p=0.5 each own a stream, so which keys they fire
    on is decided independently: neither mask is the other's, and enabling
    both changes neither. Two draws sharing a fold_in index would instead
    fire on exactly the same keys, and since `back` runs later and overwrites
    `slow`, the earlier draw would contribute nothing to training at all --
    silently, because every other test here enables one draw at a time.

    `slow_vx` is positive and `back_vx` negative, so with both on the sign of
    vx says which draw wrote each row."""
    with block_values(env._config.command, zero_prob=0.0, pure_slow_prob=0.5):
        slow_only = samples(env)
    with block_values(env._config.command, zero_prob=0.0, pure_back_prob=0.5):
        back_only = samples(env)
    with block_values(
        env._config.command, zero_prob=0.0, pure_slow_prob=0.5, pure_back_prob=0.5
    ):
        both = samples(env)

    slow_mask, back_mask = fired(slow_only), fired(back_only)
    # The catch: identical masks are what a shared index produces.
    assert (slow_mask & ~back_mask).sum() > 0
    assert (back_mask & ~slow_mask).sum() > 0

    np.testing.assert_array_equal(fired(both), slow_mask | back_mask)
    # And the rows keep their values: back's wherever back fired, slow's
    # wherever slow fired and back did not.
    np.testing.assert_array_equal(both[back_mask], back_only[back_mask])
    survivors = slow_mask & ~back_mask
    np.testing.assert_array_equal(both[survivors], slow_only[survivors])


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


# -- the construct-time range check -----------------------------------------


def test_an_armed_draw_outside_the_command_box_refuses_to_construct():
    """The range is a REDRAW of the axis, not a widening of it: the box is
    what the exported contract says the policy trained under. Refused here
    as well as at export, so the run fails before the GPU hours."""
    cfg = default_config()
    cfg.episode_length = 50
    cfg.command.pure_fast_prob = 0.2
    cfg.command.fast_vx = (0.9, 1.4)  # command.vx tops out at 0.8

    with pytest.raises(ValueError) as excinfo:
        Joystick(ROBOT_DIR, PRESET, cfg)

    message = str(excinfo.value)
    assert "pure_fast_prob" in message  # the draw
    assert "0.9" in message and "1.4" in message  # the range
    assert "0.8" in message  # the box


def test_an_armed_draw_inside_the_command_box_constructs():
    cfg = default_config()
    cfg.episode_length = 50
    cfg.command.pure_back_prob = 0.3  # back_vx = (-0.8, -0.2), inside vx

    assert Joystick(ROBOT_DIR, PRESET, cfg) is not None


def test_a_disarmed_out_of_box_range_constructs(env):
    """Every draw is off by default, so the shipped config validates nothing
    -- the off path stays bit-exact (test_golden_baseline.py is the gate).
    Widening a range without arming its draw is not a violation."""
    cfg = default_config()
    cfg.episode_length = 50
    cfg.command.fast_vx = (0.9, 1.4)
    cfg.command.pure_fast_prob = 0.0

    assert Joystick(ROBOT_DIR, PRESET, cfg) is not None


def test_the_box_a_robot_overlay_narrows_is_the_one_checked():
    """configs/robot/roboto_origin.yaml sets command.vx to [-0.6, 1.0],
    narrower on the backward side than the shipped back_vx = (-0.8, -0.2).
    The check reads the COMPOSED config, so arming pure_back under that
    overlay is refused even though the same defaults pass on asimov_v1,
    whose vx is [-0.8, 0.8]. Composed the way training composes it."""
    overrides = {
        "episode_length": 50,
        "command": {"vx": [-0.6, 1.0], "pure_back_prob": 0.3},
    }

    with pytest.raises(ValueError, match="back_vx"):
        make_env("joystick", ROBOT_DIR, PRESET, overrides)
