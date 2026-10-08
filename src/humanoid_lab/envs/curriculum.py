"""The terrain curriculum: how an episode moves its env's level.

Model-free. `served_step`, `crossed_rule` and `curriculum_step` are
jax.numpy over one env's values. The env accumulates `served_step` every
step. The auto-reset wrapper (envs/terrain_wrapper.py) vmaps the other two
over the batch on done. `pinned_layout` is numpy, static in the batch size.

A level is an arena row. The flat row, when the arena has one, is level 0.
"""

from __future__ import annotations

import jax
import jax.numpy as jp
import numpy as np

# Full-episode commanded distance (m) at or below which an episode cannot
# clear the strike count. An episode that commanded no distance, a pure
# stand or spin, cannot fail either.
MOVED_DIST = 0.05
# Planar command speed (m/s) at or below which a step serves no distance.
SERVED_MIN_SPEED = 1e-6


def served_step(linvel_xy, command, dt):
    """Distance (m) one step served along the planar command.

    The base's planar velocity is projected onto the commanded direction
    and multiplied by `dt`. `linvel_xy` is the env's `_local_linvel`, the
    body-frame velocity the tracking reward compares with `command[:2]`.
    It is the linear term of the no-progress cut's `progress.served`.

    A robot that tracks its command serves the commanded distance on any
    path, an arc or a circle included. Motion across the command serves
    nothing. Motion against it serves a negative distance. Sway cancels
    over a gait cycle. A command at or below SERVED_MIN_SPEED serves 0."""
    cmd = command[:2]
    speed = jp.linalg.norm(cmd)
    along = jp.dot(linvel_xy, cmd) / jp.maximum(speed, SERVED_MIN_SPEED)
    return jp.where(speed > SERVED_MIN_SPEED, along, 0.0) * dt


def crossed_rule(on_flat, feature_r, walked, cheby_min, cheby_max, pad_radius, tile_size):
    """Whether an episode crossed its tile.

    On the flat row an episode crosses when it walked more than half a
    tile from its spawn. On every other row it crosses when its Chebyshev
    range around the tile centre reached the pad (`cheby_min <=
    pad_radius`) and the features' outer radius (`cheby_max >=
    feature_r`). From a pad spawn the first condition holds at the spawn.
    The band rule needs the features reached. A walk of half a tile from
    the pad, on a diagonal, can end short of them."""
    band = (cheby_min <= pad_radius) & (cheby_max >= feature_r)
    return jp.where(on_flat, walked > 0.5 * tile_size, band)


def curriculum_step(
    level,
    served,
    commanded_dist,
    steps_lived,
    rng,
    *,
    episode_length: int,
    n_rows: int,
    demote_fraction: float,
    strikes,
    demote_strikes: int,
    crossed,
    grace_steps: int,
    pinned,
):
    """(level, strikes, rng, promoted, demoted) after one episode.

    `crossed` promotes. Promotion from the top row lands on a uniform
    random row, so the easier rows stay in training.

    `served` is the episode's sum of `served_step`: its progress along the
    commanded direction. `commanded_dist` is the sum of the commanded
    planar speed times dt. Both use the command that drove each step. A
    robot that tracks its commands serves the full commanded distance,
    however much the yaw rate curves its path.

    An episode fails when it served less than `demote_fraction` of the
    distance its commands asked for, projected onto a full episode:
    `commanded_dist * episode_length / steps_lived`. A fall at step 50 of
    1000 is held to twenty times the distance commanded so far, so almost
    every early fall fails. A timeout reads `steps_lived ==
    episode_length` and is not projected. `episode_length` must be the
    length the episodes actually run to, brax's, not the env config's.

    A failure adds a strike. The failure that brings the count to
    `demote_strikes` drops a level and clears the count. A promotion
    clears the count. So does a clean episode whose full-episode commanded
    distance exceeds MOVED_DIST. An episode that commanded no distance
    cannot fail, so a stand or spin episode neither strikes nor clears.
    One that commanded a little, at most MOVED_DIST, can still strike.
    Promotion wins over a failure in the same episode.

    An episode that ended within `grace_steps` of its spawn is neutral: no
    strike, no promotion, the count kept. `pinned` holds the level and the
    count.

    `promoted` is True when the episode earned a promotion. `demoted` is
    True when a demotion moved the level down, so a demotion at level 0
    reads False. Both read False when pinned. The key always splits once."""
    full = commanded_dist * episode_length / jp.maximum(steps_lived, 1)
    graced = steps_lived <= grace_steps
    promote = crossed & ~graced
    fail = (served < demote_fraction * full) & ~graced & ~promote
    moved = full > MOVED_DIST
    clear = promote | (~fail & ~graced & moved)
    new_strikes = jp.where(clear, 0, jp.where(fail, strikes + 1, strikes))
    # Only a failure demotes, so a neutral episode never acts on a count
    # it did not add to.
    demote = fail & (new_strikes >= demote_strikes)
    new_strikes = jp.where(demote, 0, new_strikes)
    stepped = jp.clip(level + jp.where(promote, 1, jp.where(demote, -1, 0)), 0, n_rows - 1)
    rng, sub = jax.random.split(rng)
    random_row = jax.random.randint(sub, (), 0, n_rows)
    new_level = jp.where(promote & (level >= n_rows - 1), random_row, stepped)
    promoted = promote & ~pinned
    demoted = demote & (new_level < level) & ~pinned
    new_level = jp.where(pinned, level, new_level).astype(jp.asarray(level).dtype)
    new_strikes = jp.where(pinned, strikes, new_strikes).astype(jp.asarray(strikes).dtype)
    return new_level, new_strikes, rng, promoted, demoted


def pinned_layout(n: int, pinned_frac: float, pinned_flat_frac: float, n_rows: int):
    """(pinned, level) for a batch of `n` envs, numpy.

    The first round(pinned_frac * n) envs hold one row each, round robin:
    env i holds row i % n_rows. The next round(pinned_flat_frac * n) hold
    level 0, the flat row when the arena has one. The rest ride the
    curriculum. A pinned env reaches its row at its first respawn."""
    for name, frac in (("pinned_frac", pinned_frac), ("pinned_flat_frac", pinned_flat_frac)):
        if not 0.0 <= frac <= 1.0:
            raise ValueError(f"terrain.curriculum.{name} must lie in [0, 1], got {frac}")
    if pinned_frac + pinned_flat_frac > 1.0:
        raise ValueError(
            f"terrain.curriculum.pinned_frac {pinned_frac} plus pinned_flat_frac "
            f"{pinned_flat_frac} pins more than the whole batch"
        )
    n_pin = min(round(pinned_frac * n), n)
    n_flat = min(round(pinned_flat_frac * n), n - n_pin)
    idx = np.arange(n)
    pinned = idx < n_pin + n_flat
    level = np.where(idx < n_pin, idx % n_rows, 0).astype(np.int32)
    return pinned, level
