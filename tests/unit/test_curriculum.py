"""The terrain curriculum rules (envs/curriculum.py), model-free.

The arena here has five levels and 4 m tiles. Episodes run 1000 steps of
0.02 s.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jp
import numpy as np
import pytest

from humanoid_lab.envs import curriculum

N_ROWS = 5
TILE = 4.0
PAD_R = 0.4
FEATURE_R = 1.6
EPISODE = 1000
DT = 0.02


def step(level, served, commanded, steps_lived=EPISODE, *, strikes=0, demote_strikes=1,
         crossed=False, grace_steps=0, pinned=False, key=0, demote_fraction=0.5):
    """(level, strikes, promoted, demoted) as plain Python values."""
    out = curriculum.curriculum_step(
        jp.int32(level),
        jp.float32(served),
        jp.float32(commanded),
        jp.int32(steps_lived),
        jax.random.PRNGKey(key),
        episode_length=EPISODE,
        n_rows=N_ROWS,
        demote_fraction=demote_fraction,
        strikes=jp.int32(strikes),
        demote_strikes=demote_strikes,
        crossed=jp.asarray(crossed),
        grace_steps=grace_steps,
        pinned=jp.asarray(pinned),
    )
    lvl, s, _, promoted, demoted = out
    return int(lvl), int(s), bool(promoted), bool(demoted)


def crossed(on_flat, walked=0.0, cmin=0.0, cmax=0.0, feature_r=FEATURE_R):
    return bool(
        curriculum.crossed_rule(
            jp.asarray(on_flat), jp.float32(feature_r), jp.float32(walked), jp.float32(cmin),
            jp.float32(cmax), PAD_R, TILE,
        )
    )


def test_promote_on_the_band():
    """Off the flat row an episode crosses when its Chebyshev range reached
    the pad and the features' outer radius. Walked distance plays no part."""
    assert crossed(False, walked=0.1, cmin=0.2, cmax=FEATURE_R)
    assert not crossed(False, walked=3.0, cmin=0.2, cmax=FEATURE_R - 0.01)
    assert not crossed(False, walked=3.0, cmin=PAD_R + 0.01, cmax=2.0)
    assert step(2, served=0.1, commanded=3.0, crossed=True) == (3, 0, True, False)


def test_the_flat_row_promotes_on_half_a_tile():
    assert crossed(True, walked=TILE / 2 + 0.01, feature_r=0.0)
    assert not crossed(True, walked=TILE / 2, cmin=0.0, cmax=3.0, feature_r=0.0)


def test_an_early_fall_projects_to_a_failure():
    """A fall at step 50 having served 0.4 m of the 0.5 m commanded so far
    is held to a 10 m full episode, and fails."""
    assert step(3, served=0.4, commanded=0.5, steps_lived=50) == (2, 0, False, True)


def test_a_timeout_is_not_projected():
    """A timeout reads steps_lived == episode_length: the same 0.4 of 0.5 m
    passes."""
    assert step(3, served=0.4, commanded=0.5, steps_lived=EPISODE) == (3, 0, False, False)
    assert step(3, served=0.2, commanded=0.5, steps_lived=EPISODE) == (2, 0, False, True)


def test_stand_and_spin_are_strike_transparent():
    """An episode that commanded no distance neither strikes nor clears:
    fail, spin, fail still demotes at two strikes."""
    fail = {"served": 0.1, "commanded": 3.0, "demote_strikes": 2}
    spin = {"served": 0.02, "commanded": 0.0, "demote_strikes": 2}
    lvl, s, _, _ = step(3, **fail)
    assert (lvl, s) == (3, 1)
    lvl, s, _, _ = step(3, strikes=s, **spin)
    assert (lvl, s) == (3, 1)
    assert step(3, strikes=s, **fail) == (2, 0, False, True)


def test_a_near_stand_can_strike():
    """A timeout that commanded 0.04 m, under MOVED_DIST, and served none of
    it fails. The same timeout that commanded 0 m is neutral."""
    assert step(1, served=0.0, commanded=0.04) == (0, 0, False, True)
    assert step(1, served=0.0, commanded=0.0) == (1, 0, False, False)


def served(velocities, commands):
    """The sum of served_step over body-frame planar velocities (T, 2)
    under commands (T, 3)."""
    each = jax.vmap(curriculum.served_step, (0, 0, None))(
        jp.asarray(velocities, jp.float32), jp.asarray(commands, jp.float32), DT
    )
    return float(jp.sum(each))


def displacement(commands, substeps=4):
    """How far a base that holds each command's body-frame velocity and yaw
    rate for one step ends from its start."""
    h = DT / substeps
    yaw, x, y = 0.0, 0.0, 0.0
    for vx, vy, wz in np.asarray(commands, float):
        for _ in range(substeps):
            yaw += wz * h
            c, s = math.cos(yaw), math.sin(yaw)
            x += h * (c * vx - s * vy)
            y += h * (s * vx + c * vy)
    return math.hypot(x, y)


def test_served_step_projects_onto_the_command():
    """Velocity along the command serves its speed. Across it serves
    nothing, against it a negative distance. A zero linear command serves
    0, and its yaw rate plays no part."""
    cmd = jp.array([0.3, 0.4, 0.8])

    def one(v, c=cmd):
        return float(curriculum.served_step(jp.asarray(v, jp.float32), c, DT))

    assert one([0.3, 0.4]) == pytest.approx(0.5 * DT)
    assert one([0.6, 0.8]) == pytest.approx(1.0 * DT)
    assert one([-0.4, 0.3]) == pytest.approx(0.0, abs=1e-9)
    assert one([-0.3, -0.4]) == pytest.approx(-0.5 * DT)
    assert one([0.5, 0.0], jp.array([0.0, 0.0, 1.2])) == 0.0


@pytest.mark.parametrize(
    "segments",
    [
        # 0.69 m circles, walked 3.8 times.
        [(0.8, 0.2, 1.2)],
        # Four 250-step straight segments: out, back, left, right.
        [(0.8, 0.0, 0.0), (-0.5, 0.0, 0.0), (0.0, 0.5, 0.0), (0.0, -0.5, 0.0)],
    ],
    ids=["circle", "out_and_back"],
)
def test_a_tracked_timeout_does_not_strike(segments):
    """A robot holds each command for its share of a 1000-step timeout.
    It ends near where it started. It served the whole commanded distance,
    so it does not strike and its count clears. Its displacement would
    strike, and so does standing still under the same commands."""
    commands = np.repeat(np.array(segments), EPISODE // len(segments), axis=0)
    commanded = float(np.sum(np.linalg.norm(commands[:, :2], axis=1)) * DT)
    got = served(commands[:, :2], commands)
    assert got == pytest.approx(commanded, rel=1e-5)
    walked = displacement(commands)
    assert walked < 0.15 * commanded
    common = {"commanded": commanded, "strikes": 1, "demote_strikes": 2}
    assert step(3, served=got, **common) == (3, 0, False, False)
    assert step(3, served=walked, **common) == (2, 0, False, True)
    assert step(3, served=served(np.zeros((EPISODE, 2)), commands), **common) == (2, 0, False, True)


def test_motion_across_or_against_the_command_strikes():
    """A timeout under a forward 0.5 m/s command, 10 m in all. Moving
    sideways at the commanded speed serves nothing. Moving backward serves
    -10 m. Both strike, though each moved 10 m. Moving forward at 60% of
    the command clears."""
    commands = np.tile([0.5, 0.0, 0.0], (EPISODE, 1))
    for velocity, fails in (([0.0, 0.5], True), ([-0.5, 0.0], True), ([0.3, 0.0], False)):
        got = served(np.tile(velocity, (EPISODE, 1)), commands)
        assert step(3, served=got, commanded=10.0)[3] is fails, velocity


def test_sway_in_place_strikes():
    """A base that steps in place under a forward 0.5 m/s command. It sways
    3 cm sideways at 1.5 Hz and 1 cm fore and aft at 3 Hz. The sway serves
    under 1 mm, and the timeout strikes."""
    t = np.arange(1, EPISODE + 1) * DT
    velocity = np.stack(
        [0.01 * 6 * np.pi * np.cos(6 * np.pi * t + 0.3), 0.03 * 3 * np.pi * np.cos(3 * np.pi * t + 1.1)],
        axis=-1,
    )
    got = served(velocity, np.tile([0.5, 0.0, 0.0], (EPISODE, 1)))
    assert abs(got) < 1e-3
    assert step(3, served=got, commanded=10.0) == (2, 0, False, True)


def test_grace_is_neutral_and_keeps_strikes():
    """An episode that ends within grace_steps of its spawn neither strikes,
    promotes nor clears. One step later the same episode strikes."""
    common = {"served": 0.0, "commanded": 0.5, "strikes": 2, "demote_strikes": 3, "grace_steps": 50}
    assert step(3, steps_lived=50, crossed=True, **common) == (3, 2, False, False)
    assert step(3, steps_lived=51, **common) == (2, 0, False, True)


def test_promote_beats_demote():
    """A crossing promotes even when the walk also fails, and clears the
    count."""
    assert step(2, served=0.1, commanded=3.0, strikes=0, crossed=True) == (3, 0, True, False)
    assert step(2, served=0.1, commanded=3.0, strikes=2, demote_strikes=3, crossed=True) == (
        3, 0, True, False,
    )


def test_three_strikes_demote_and_a_clean_walk_clears():
    fail = {"served": 0.2, "commanded": 3.0, "demote_strikes": 3}
    assert step(3, strikes=0, **fail) == (3, 1, False, False)
    assert step(3, strikes=1, **fail) == (3, 2, False, False)
    assert step(3, strikes=2, **fail) == (2, 0, False, True)
    # Served most of what it was asked to, crossed nothing: the count clears
    # and the level holds.
    assert step(3, served=1.0, commanded=1.2, strikes=2, demote_strikes=3) == (3, 0, False, False)


def test_demote_at_the_floor_stays_at_zero():
    """The strike count clears, and the level cannot go lower, so nothing
    reads as demoted."""
    assert step(0, served=0.1, commanded=3.0) == (0, 0, False, False)


def test_a_non_promoting_top_level_holds():
    assert step(N_ROWS - 1, served=1.0, commanded=1.2) == (N_ROWS - 1, 0, False, False)


def test_the_top_level_promotes_to_a_random_row():
    levels = {step(N_ROWS - 1, served=0.0, commanded=0.0, crossed=True, key=k)[0] for k in range(64)}
    assert levels == set(range(N_ROWS))
    for k in range(8):
        assert step(N_ROWS - 1, served=0.0, commanded=0.0, crossed=True, key=k)[2]


def test_pinned_never_moves():
    assert step(2, served=3.0, commanded=3.0, crossed=True, pinned=True) == (2, 0, False, False)
    assert step(2, served=0.1, commanded=3.0, strikes=1, demote_strikes=2, pinned=True) == (
        2, 1, False, False,
    )


def test_zero_steps_lived_holds():
    """A done before any step ran: the division stays finite and the episode
    is neutral."""
    assert step(2, served=0.0, commanded=5.0, steps_lived=0, strikes=1, demote_strikes=2,
                crossed=True) == (2, 1, False, False)


def test_the_key_always_splits_once():
    rng = jax.random.PRNGKey(7)
    want = jax.random.split(rng)[0]
    for crossed_, pinned in ((False, False), (True, False), (True, True)):
        out = curriculum.curriculum_step(
            jp.int32(N_ROWS - 1), jp.float32(0.0), jp.float32(0.0), jp.int32(10), rng,
            episode_length=EPISODE, n_rows=N_ROWS, demote_fraction=0.5, strikes=jp.int32(0),
            demote_strikes=1, crossed=jp.asarray(crossed_), grace_steps=0, pinned=jp.asarray(pinned),
        )
        np.testing.assert_array_equal(np.asarray(out[2]), np.asarray(want))


def test_pinned_layout():
    """Ten envs, 40% on the rungs round robin, 20% on level 0, the rest
    free."""
    pinned, level = curriculum.pinned_layout(10, 0.4, 0.2, n_rows=3)
    assert pinned.dtype == np.bool_ and level.dtype == np.int32
    np.testing.assert_array_equal(pinned, [True] * 6 + [False] * 4)
    np.testing.assert_array_equal(level, [0, 1, 2, 0, 0, 0, 0, 0, 0, 0])
    pinned, level = curriculum.pinned_layout(8, 0.0, 0.0, n_rows=3)
    assert not pinned.any() and not level.any()
    # Rounding never pins more than the batch.
    pinned, _ = curriculum.pinned_layout(3, 0.5, 0.5, n_rows=2)
    assert pinned.all()


@pytest.mark.parametrize("fracs", [(-0.1, 0.0), (0.0, 1.1), (0.7, 0.4)])
def test_pinned_layout_refuses_impossible_fractions(fracs):
    with pytest.raises(ValueError, match="terrain.curriculum.pinned"):
        curriculum.pinned_layout(8, *fracs, n_rows=3)
