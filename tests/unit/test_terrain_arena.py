"""The terrain arena generator: layout, difficulty, per-type geometry, the
lookup grid, serialization and the fingerprint.

`terrain.generate` is numpy only -- no model, no scene, no device -- so the
whole generator is checked here. Checks against compiled physics geometry
(ray casts, contacts) need a scene and belong in tests/integration.
"""

from __future__ import annotations

import functools
import json
import math
import time
from dataclasses import fields, replace

import numpy as np
import pytest

from humanoid_lab import terrain
from humanoid_lab.terrain.arena import _ROW, _TILE, _stream
from humanoid_lab.terrain.params import (
    _BOOLS,
    _FLOATS,
    _INTS,
    _RAMPS,
    _RANGES,
    rubble_cells,
)

# sha256 of two arenas. If a change moves either, that arena changed: bump
# GENERATOR_VERSION, then re-pin both. The tread-range arena covers the
# per-tile tread draw, which the default arena never takes.
PINNED_FINGERPRINTS = {
    "default": "837fbf8ab9fca3292bca2b1b87976c0deb242e859504b28650e5d9bef155765f",
    "tread_range": "70684a28401e597cf72dda4dc0eed7298ad129c2f5a315472fdc2ae6dd60b3cb",
}

SLOPES = ("pyramid_slope", "inverted_pyramid_slope")
SCATTERED = ("rough_uniform", "discrete_obstacles", "random_grid", "wave")
BOX_SCATTER = ("discrete_obstacles", "random_grid")

# Arenas that more than one test reads, built once each.
VARIANTS = {
    "default": {},
    "rows3": {"n_rows": 3},
    "flat_row": {"flat_row": True},
    "3m": {"tile_size": 3.0, "n_rows": 3},
    # The summit floor is wider than this pad, so it sets the summit. It is
    # off its 0.3 m default and a whole number of cells.
    "small_pad": {"pad_radius": 0.2, "summit_floor": 0.36, "n_rows": 3},
    "tread_range": {"n_rows": 6, "stair_tread": (0.25, 0.45)},
    # The narrowest tread ArenaParams admits: two cells.
    "narrowest_tread": {"n_rows": 3, "stair_tread": 0.08},
    # The flat row beside a non-default pad, and beside a non-default tile
    # and border.
    "flat_row_pad": {"flat_row": True, "pad_radius": 0.52, "n_rows": 2},
    "flat_row_tile": {"flat_row": True, "tile_size": 3.6, "border": 1.2, "n_rows": 4},
    # Every geometry field off its default, each in the direction where code
    # that read the default instead breaks a property checked below. The
    # pad is a whole number of cells, so the pit rim is a node line. Every
    # wave count is at least 2, so each wave row crosses zero. The tread
    # leaves room for 5 risers, so the cap of 4 binds.
    "tuned": {
        "seed": 3,
        "n_rows": 4,
        "row_jitter": 0.1,
        "tile_size": 3.6,
        "border": 1.2,
        "cell_size": 0.04,
        "pad_radius": 0.52,
        "pad_taper": 0.4,
        "edge_taper": 0.4,
        "rough_amplitude": (0.01, 0.02),
        "slope_angle": (0.0, 0.25),
        "stair_riser": (0.03, 0.10),
        "obstacle_height": (0.02, 0.06),
        "wave_amplitude": (0.02, 0.03),
        "coarse_step": 0.2,
        "overlay_fraction": 0.5,
        "slope_platform_half": 0.7,
        "stair_tread": 0.22,
        "summit_floor": 0.45,
        "rim_margin": 0.35,
        "stair_max_steps": 4,
        "discrete_count": 6,
        "discrete_half_range": (0.08, 0.15),
        "discrete_height_fraction": (0.4, 0.9),
        "edge_margin": 0.2,
        "pad_clearance": 0.1,
        "grid_pitch": 0.2,
        "grid_fill_prob": 0.5,
        "grid_half_range": (0.03, 0.08),
        "grid_height_fraction": (0.4, 0.9),
        "wave_half_periods": (2, 4),
    },
    # Every obstacle and rubble box at the smallest half-size that still
    # covers a node.
    "smallest_boxes": {
        "n_rows": 3,
        "discrete_half_range": (0.04 / math.sqrt(2), 0.25),
        "grid_half_range": (0.04 / math.sqrt(2), 0.04 / math.sqrt(2)),
    },
}

# One value off the default for every ArenaParams field. Each changes what
# the 3-row arena builds, so code that reads a default constant in place of
# its field leaves that arena unchanged and fails the sweep.
ALTERNATIVES = {
    "seed": 1,
    "n_rows": 4,
    "ordered": True,
    "difficulties": (0.0, 0.3, 1.0),
    "flat_row": True,
    "row_jitter": 0.1,
    "type_caps": {"wave": 0.5},
    "tile_size": 3.6,
    "border": 1.2,
    "cell_size": 0.05,
    "pad_radius": 0.52,
    "pad_taper": 0.4,
    "edge_taper": 0.4,
    "rough_amplitude": (0.01, 0.02),
    "slope_angle": (0.0, 0.25),
    "stair_riser": (0.03, 0.10),
    "obstacle_height": (0.02, 0.06),
    "wave_amplitude": (0.02, 0.03),
    "coarse_step": 0.2,
    "overlay_fraction": 0.5,
    "slope_platform_half": 0.7,
    "stair_tread": 0.25,
    "summit_floor": 0.45,
    "rim_margin": 0.5,
    "stair_min_steps": 6,
    "stair_max_steps": 4,
    "discrete_count": 6,
    "discrete_half_range": (0.08, 0.15),
    "discrete_height_fraction": (0.4, 0.9),
    "edge_margin": 0.2,
    "pad_clearance": 0.1,
    "grid_pitch": 0.2,
    "grid_fill_prob": 0.5,
    "grid_half_range": (0.03, 0.08),
    "grid_height_fraction": (0.4, 0.9),
    "wave_half_periods": (2, 4),
}


@functools.cache
def _variant(name):
    return terrain.generate(**VARIANTS[name])


@pytest.fixture(scope="module")
def arena():
    return _variant("default")


def _tile(arena, ttype, row):
    return next(t for t in arena.spec.tiles if t.terrain_type == ttype and t.row == row)


def _layout(arena):
    return [(t.row, t.col, t.terrain_type) for t in arena.spec.tiles]


def _row_difficulties(arena):
    s = arena.spec
    return [next(t.difficulty for t in s.tiles if t.row == r) for r in range(s.n_rows)]


def _relief(arena, tile, inset=0.2):
    """Peak-to-peak surface height over a tile, inset so the samples never
    reach a neighbour across the shared edge."""
    cx, cy, _ = tile.origin
    reach = arena.spec.params.tile_size / 2 - inset
    g = np.linspace(-reach, reach, 60)
    gx, gy = np.meshgrid(cx + g, cy + g)
    return float(np.ptp(terrain.lookup_height(arena, gx.ravel(), gy.ravel())))


def _perimeter_max_abs(arena, tile, inset=0.02, n=41):
    """Max |surface height| along the four tile edges, a small inset in."""
    cx, cy, _ = tile.origin
    h = arena.spec.params.tile_size / 2 - inset
    lin = np.linspace(-h, h, n)
    ones = np.ones(n)
    xs = np.concatenate([cx + lin, cx + lin, cx - h * ones, cx + h * ones])
    ys = np.concatenate([cy - h * ones, cy + h * ones, cy + lin, cy + lin])
    return float(np.max(np.abs(terrain.lookup_height(arena, xs, ys))))


def _tile_cheby(arena, tile):
    """Chebyshev distance of every lookup node from the tile centre."""
    xs, ys = terrain.grid_axes(arena.spec)
    gx, gy = np.meshgrid(xs - tile.origin[0], ys - tile.origin[1])
    return np.maximum(np.abs(gx), np.abs(gy))


def _tile_window(arena, tile):
    """The tile's own lookup nodes, perimeter included."""
    p = arena.spec.params
    n = round(p.tile_size / p.cell_size)
    bc = round(p.border / p.cell_size)
    return (
        slice(bc + tile.row * n, bc + tile.row * n + n + 1),
        slice(bc + tile.col * n, bc + tile.col * n + n + 1),
    )


def _box_tile(arena, box):
    """The tile whose square holds `box`'s centre."""
    s = arena.spec
    size = s.params.tile_size
    col = math.floor((box.pos[0] + s.n_cols * size / 2) / size)
    row = math.floor((box.pos[1] + s.n_rows * size / 2) / size)
    return next(t for t in s.tiles if t.row == row and t.col == col)


def _box_reach(box, tile):
    """Chebyshev distance from the tile centre to the far side of the box's
    rotated extent."""
    hx, hy, _ = box.half
    c, s = abs(math.cos(box.yaw)), abs(math.sin(box.yaw))
    ax, ay = hx * c + hy * s, hx * s + hy * c
    return max(
        abs(box.pos[0] - tile.origin[0]) + ax, abs(box.pos[1] - tile.origin[1]) + ay
    )


def _scattered_boxes(arena):
    """{(type, row): [(u, v, box)]} for every obstacle and rubble tile, with
    each box's centre (u, v) in tile-local coordinates. A tile that kept no
    box maps to an empty list."""
    out = {
        (t.terrain_type, t.row): []
        for t in arena.spec.tiles
        if t.terrain_type in BOX_SCATTER and t.feature_radius > 0
    }
    for b in arena.boxes:
        t = _box_tile(arena, b)
        if t.terrain_type in BOX_SCATTER:
            u, v = b.pos[0] - t.origin[0], b.pos[1] - t.origin[1]
            out[(t.terrain_type, t.row)].append((u, v, b))
    return out


def _footprint_margin(xs, ys, box):
    """Window under `box` and each node's signed distance outside its yawed
    footprint (negative inside)."""
    (px, py, _), (hx, hy, _) = box.pos, box.half
    reach = hx + hy + 1e-3
    c0, c1 = np.searchsorted(xs, [px - reach, px + reach])
    r0, r1 = np.searchsorted(ys, [py - reach, py + reach])
    gx, gy = np.meshgrid(xs[c0:c1] - px, ys[r0:r1] - py)
    c, s = math.cos(box.yaw), math.sin(box.yaw)
    margin = np.maximum(np.abs(gx * c + gy * s) - hx, np.abs(gy * c - gx * s) - hy)
    return (slice(r0, r1), slice(c0, c1)), margin


def _touched_floor(surface, xs, ys, box):
    """Lowest node of every heightfield cell that `box`'s yawed footprint
    touches, edges included. MuJoCo's triangles and bilinear both stay
    within a cell's node range, so neither surface dips below this anywhere
    under the box. A cell touches the footprint unless the world axes or the
    box's own axes separate them."""
    (px, py, _), (hx, hy, _) = box.pos, box.half
    cell = xs[1] - xs[0]
    reach = hx + hy + 2 * cell
    c0, c1 = np.searchsorted(xs, [px - reach, px + reach])
    r0, r1 = np.searchsorted(ys, [py - reach, py + reach])
    s = surface[r0:r1, c0:c1]
    low = np.minimum(
        np.minimum(s[:-1, :-1], s[:-1, 1:]), np.minimum(s[1:, :-1], s[1:, 1:])
    )
    gx, gy = np.meshgrid(
        (xs[c0 : c1 - 1] + xs[c0 + 1 : c1]) / 2 - px,
        (ys[r0 : r1 - 1] + ys[r0 + 1 : r1]) / 2 - py,
    )
    c, sn = math.cos(box.yaw), math.sin(box.yaw)
    ac, as_ = abs(c), abs(sn)
    h = cell / 2
    touch = (
        (np.abs(gx) <= h + hx * ac + hy * as_ + 1e-9)
        & (np.abs(gy) <= h + hx * as_ + hy * ac + 1e-9)
        & (np.abs(gx * c + gy * sn) <= hx + h * (ac + as_) + 1e-9)
        & (np.abs(gy * c - gx * sn) <= hy + h * (ac + as_) + 1e-9)
    )
    return float(low[touch].min())


def _surface(arena):
    """The heightfield surface MuJoCo compiles: pos_z + data * elevation_z."""
    hf = arena.spec.hfield
    return hf.pos_z + arena.hfield_data.astype(np.float64) * hf.elevation_z


def _assert_slope_profile(arena, tile):
    """Each of the four arms is a straight ramp of grade
    tan(slope_angle(d)) from the plateau to 0 at the tile edge, within the
    overlay noise amplitude.

    The overlay is a bilinear blend of uniform [-1, 1] draws, scaled by
    overlay_fraction of the rough amplitude and by tapers no larger than 1.
    Its magnitude never exceeds that amplitude, at any seed."""
    p = arena.spec.params
    half = p.tile_size / 2
    cx, cy, _ = tile.origin
    sign = 1.0 if tile.terrain_type == "pyramid_slope" else -1.0
    r = np.linspace(p.slope_platform_half, half, 80)
    want = sign * math.tan(p.slope_angle(tile.difficulty)) * (half - r)
    # 1e-6 covers the float32 lookup.
    bound = p.overlay_fraction * p.rough_amplitude(tile.difficulty) + 1e-6
    for ux, uy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        h = terrain.lookup_height(arena, cx + ux * r, cy + uy * r)
        assert np.abs(h - want).max() <= bound, (tile.terrain_type, tile.row, ux, uy)


def _assert_noise_lattice(arena, tile):
    """The rough noise surface is linear between lattice lines spaced
    tile_size / round(tile_size / coarse_step) apart, starting at the tile
    edge.

    Along one node row inside both tapers, a nonzero second difference
    marks a kink within one cell of the node. Every kink sits within a cell
    of a lattice line. Most lattice lines show a kink, so a coarser lattice
    with the same origin fails too."""
    s = arena.spec
    p = s.params
    half = p.tile_size / 2
    cell = p.cell_size
    cx, cy, _ = tile.origin
    xs, ys = terrain.grid_axes(s)
    row = int(np.abs(ys - (cy + half - p.edge_taper - 0.02)).argmin())
    flat = half - p.edge_taper
    cols = np.abs(xs - cx) <= flat + 1e-9
    x = xs[cols] - cx
    h = arena.lookup[row, cols].astype(np.float64)
    kinks = x[1:-1][np.abs(np.diff(h, 2)) > 1e-5]
    n_lines = round(p.tile_size / p.coarse_step)
    lines = -half + p.tile_size / n_lines * np.arange(n_lines + 1)

    def gap(points, targets):
        return np.abs(points[:, None] - targets[None, :]).min(axis=1)

    assert kinks.size > 0
    assert gap(kinks, lines).max() <= cell + 1e-9, tile.row
    inner = lines[np.abs(lines) < flat - cell]
    found = gap(inner, kinks) <= cell + 1e-9
    assert found.mean() >= 0.8, (tile.row, found.mean())


# -- layout ---------------------------------------------------------------


def test_a_seed_builds_one_arena_and_another_seed_a_different_one(arena):
    again = terrain.generate()
    assert np.array_equal(arena.lookup, again.lookup)
    assert np.array_equal(arena.hfield_data, again.hfield_data)
    assert arena.boxes == again.boxes
    assert arena.spec == again.spec
    other = terrain.generate(seed=1)
    assert not np.array_equal(arena.lookup, other.lookup)
    assert _layout(arena) != _layout(other)


def test_every_row_holds_each_type_once(arena):
    s = arena.spec
    assert s.n_cols == len(terrain.TYPES)
    assert len(s.tiles) == s.n_rows * s.n_cols
    for r in range(s.n_rows):
        row = sorted(t.terrain_type for t in s.tiles if t.row == r)
        assert row == sorted(terrain.TYPES)
    # Every row draws its own shuffle, and no type keeps one column up the
    # whole curriculum.
    orders = [
        tuple(
            t.terrain_type
            for t in sorted((t for t in s.tiles if t.row == r), key=lambda t: t.col)
        )
        for r in range(s.n_rows)
    ]
    assert len(set(orders)) == s.n_rows
    assert all(len({o[c] for o in orders}) > 1 for c in range(s.n_cols))


def test_ordered_keeps_the_type_order_and_the_exact_ramp(arena):
    a = terrain.generate(ordered=True)
    s = a.spec
    for r in range(s.n_rows):
        cols = sorted((t for t in s.tiles if t.row == r), key=lambda t: t.col)
        assert [t.terrain_type for t in cols] == list(terrain.TYPES)
        assert all(t.difficulty == r / (s.n_rows - 1) for t in cols)
    assert _layout(a) != _layout(arena)


@pytest.mark.parametrize("variant", ["default", "tuned"])
def test_row_difficulty_is_exact_at_the_ends_jittered_and_strictly_increasing(
    variant,
):
    arena = _variant(variant)
    ds = _row_difficulties(arena)
    n = arena.spec.n_rows
    assert ds[0] == 0.0 and ds[-1] == 1.0
    assert all(np.diff(ds) > 0)
    nominal = [r / (n - 1) for r in range(n)]
    assert all(d != d0 for d, d0 in zip(ds[1:-1], nominal[1:-1]))
    # Under half a row gap off nominal, which is what keeps the order.
    jitter = arena.spec.params.row_jitter / (n - 1)
    assert all(abs(d - d0) <= jitter + 1e-12 for d, d0 in zip(ds, nominal))


def test_explicit_difficulties_are_taken_as_given():
    rows = (0.1, 0.25, 0.25, 0.6, 1.3)
    a = terrain.generate(difficulties=rows)
    assert a.spec.n_rows == len(rows)
    assert _row_difficulties(a) == list(rows)
    # The shuffle draws first from the row's stream, so the seed gives the
    # same column order with or without an explicit list.
    default = terrain.generate(n_rows=len(rows))
    assert _layout(a) == _layout(default)


def test_every_tile_has_its_own_rng_stream():
    """Two tiles of one type and difficulty still differ."""
    a = terrain.generate(ordered=True, difficulties=(0.5, 0.5))
    rough = [t for t in a.spec.tiles if t.terrain_type == "rough_uniform"]
    xs, ys = terrain.grid_axes(a.spec)
    patches = []
    for t in rough:
        cols = np.abs(xs - t.origin[0]) < 1.0
        rows = np.abs(ys - t.origin[1]) < 1.0
        patches.append(a.lookup[np.ix_(rows, cols)])
    assert not np.array_equal(*patches)


def test_row_and_tile_streams_never_coincide():
    """SeedSequence pads a key with zeros, so the keys [seed, i] and
    [seed, i, 0] seed one stream. A tag after the seed keeps every row and
    tile key apart."""
    assert np.array_equal(
        np.random.SeedSequence([0, 3]).generate_state(4),
        np.random.SeedSequence([0, 3, 0]).generate_state(4),
    )
    p = terrain.ArenaParams()
    n_cols = len(terrain.TYPES)
    streams = [_stream(p, _ROW, i) for i in range(p.n_rows)]
    streams += [_stream(p, _TILE, i, j) for i in range(p.n_rows) for j in range(n_cols)]
    states = {tuple(g.bit_generator.seed_seq.generate_state(4)) for g in streams}
    assert len(states) == p.n_rows + p.n_rows * n_cols
    row = _stream(p, _ROW, 3).bit_generator.random_raw(2)
    tile = _stream(p, _TILE, 3, 0).bit_generator.random_raw(2)
    assert not np.array_equal(row, tile)


def test_a_tile_draws_nothing_from_its_neighbours_streams():
    """A tread range draws on stair tiles only. Every other tile is
    unchanged."""
    a = terrain.generate(n_rows=3, stair_tread=0.30)
    b = terrain.generate(n_rows=3, stair_tread=(0.25, 0.45))
    p = a.spec.params
    xs, ys = terrain.grid_axes(a.spec)
    reach = p.tile_size / 2 - p.cell_size
    for ta, tb in zip(a.spec.tiles, b.spec.tiles):
        assert (ta.row, ta.col, ta.terrain_type) == (tb.row, tb.col, tb.terrain_type)
        if ta.terrain_type in terrain.STAIR_TYPES:
            continue
        cols = np.abs(xs - ta.origin[0]) <= reach
        rows = np.abs(ys - ta.origin[1]) <= reach
        window = np.ix_(rows, cols)
        assert np.array_equal(a.lookup[window], b.lookup[window]), ta.terrain_type


# -- difficulty ramps -----------------------------------------------------


def test_ramp_inverses_round_trip():
    p = terrain.ArenaParams()
    targets = {
        "rough_amplitude": (0.01, 0.025, 0.04),
        "slope_angle": (0.1, 0.2, 0.35),
        "stair_riser": (0.05, 0.10, 0.15),
        "obstacle_height": (0.03, 0.07, 0.11),
        "wave_amplitude": (0.02, 0.04, 0.06),
    }
    for name, values in targets.items():
        ramp = getattr(p, name)
        for value in values:
            assert ramp(ramp.difficulty(value)) == pytest.approx(value, abs=1e-12)
        for d in (0.0, 0.5, 1.0, 1.4):
            assert ramp.difficulty(ramp(d)) == pytest.approx(d, abs=1e-12)


def test_ramps_rise_from_near_flat():
    p = terrain.ArenaParams()
    ds = np.linspace(0.0, 1.0, 10)
    for ramp in (
        p.rough_amplitude,
        p.slope_angle,
        p.stair_riser,
        p.obstacle_height,
        p.wave_amplitude,
    ):
        assert all(np.diff([ramp(d) for d in ds]) > 0)
    assert p.slope_angle(0) == 0.0
    assert p.rough_amplitude(0) <= 0.01
    assert p.stair_riser(0) <= 0.02
    assert p.obstacle_height(0) <= 0.01
    assert p.wave_amplitude(0) <= 0.01


def test_relief_rises_with_difficulty_and_row_0_is_near_flat(arena):
    """Realized relief, read from the lookup grid. Slope and stair relief is
    the ramp itself, so it rises row by row. The scattered types draw more
    randomness, so they only have to end well above where they start. Row 0
    stays under 3 cm except on stairs, whose 2 cm riser still stacks one
    per step."""
    last = arena.spec.n_rows - 1
    for ttype in SLOPES + terrain.STAIR_TYPES:
        relief = [_relief(arena, _tile(arena, ttype, r)) for r in range(last + 1)]
        assert all(np.diff(relief) > 0), (ttype, relief)
    for ttype in SCATTERED:
        r0 = _relief(arena, _tile(arena, ttype, 0))
        assert r0 <= 0.03, (ttype, r0)
        assert r0 < _relief(arena, _tile(arena, ttype, last))
    p = arena.spec.params
    for ttype in terrain.STAIR_TYPES:
        t = _tile(arena, ttype, 0)
        assert _relief(arena, t) == pytest.approx(
            t.n_steps * p.stair_riser(0), abs=1e-6
        )
    for ttype in SLOPES:
        assert _relief(arena, _tile(arena, ttype, 0)) <= 0.01


@pytest.mark.parametrize("variant", ["default", "3m", "tuned"])
def test_scattered_types_follow_their_ramps(variant):
    """An obstacle or rubble box's top stands its type's fraction of the
    obstacle ramp above the highest ground under it. Nodes within 1 um of
    its edge could count either way, so the lower bound reads the nodes
    strictly inside and the upper bound those on the edge too. Rough noise
    reaches at least 0.8 of the rough amplitude both above and below zero,
    and never exceeds it. The wave peaks at the wave amplitude."""
    a = _variant(variant)
    p = a.spec.params
    surface = _surface(a)
    xs, ys = terrain.grid_axes(a.spec)
    fraction = {
        "discrete_obstacles": p.discrete_height_fraction,
        "random_grid": p.grid_height_fraction,
    }
    checked = dict.fromkeys(fraction, 0)
    for b in a.boxes:
        t = _box_tile(a, b)
        if t.terrain_type not in fraction:
            continue
        lo, hi = fraction[t.terrain_type]
        ramp = p.obstacle_height(t.difficulty)
        window, margin = _footprint_margin(xs, ys, b)
        under = surface[window]
        top = b.pos[2] + b.half[2]
        assert top - under[margin < -1e-6].max() >= lo * ramp - 1e-6, (t.row, b)
        assert top - under[margin <= 1e-6].max() <= hi * ramp + 1e-6, (t.row, b)
        checked[t.terrain_type] += 1
    assert all(n > 0 for n in checked.values()), checked
    # A flat row 0 has no relief to measure.
    for t in (t for t in a.spec.tiles if t.row >= int(p.flat_row)):
        if t.terrain_type not in ("rough_uniform", "wave"):
            continue
        h = a.lookup[_tile_window(a, t)].astype(np.float64)
        peak = float(np.abs(h).max())
        if t.terrain_type == "rough_uniform":
            amp = p.rough_amplitude(t.difficulty)
            assert peak <= amp + 1e-6, (t.row, peak, amp)
            assert h.max() >= 0.8 * amp, (t.row, h.max(), amp)
            assert h.min() <= -0.8 * amp, (t.row, h.min(), amp)
        else:
            amp = p.wave_amplitude(t.difficulty)
            assert peak == pytest.approx(amp, rel=0.02), (t.row, peak, amp)


@pytest.mark.parametrize("variant", ["default", "tuned"])
def test_rough_noise_sits_on_its_lattice(variant):
    a = _variant(variant)
    for t in a.spec.tiles:
        if t.terrain_type == "rough_uniform":
            _assert_noise_lattice(a, t)


@pytest.mark.parametrize("variant", ["default", "tuned"])
def test_each_wave_axis_draws_its_count_from_the_params(variant):
    """Along a node line off the pad a wave changes sign k - 1 times, where
    k is that axis's half-period count. The line with the largest swing is
    read, so the other axis's sine is far from its zeros there. Both counts
    lie in wave_half_periods."""
    a = _variant(variant)
    p = a.spec.params
    n = round(p.tile_size / p.cell_size)
    axis = (np.arange(n + 1) - n / 2) * p.cell_size
    # Every node of such a line lies past the pad taper.
    off_pad = np.abs(axis) >= p.pad_radius + p.pad_taper
    counts = set()
    for t in (t for t in a.spec.tiles if t.terrain_type == "wave"):
        h = a.lookup[_tile_window(a, t)].astype(np.float64)
        tol = 1e-3 * p.wave_amplitude(t.difficulty)
        for grid in (h, h.T):  # lines along x give kx, along y give ky
            lines = grid[off_pad]
            line = lines[np.abs(lines).max(axis=1).argmax()]
            signs = np.sign(line[np.abs(line) > tol])
            k = int(np.count_nonzero(np.diff(signs))) + 1
            assert k in p.wave_half_periods, (t.row, k)
            counts.add(k)
    assert counts == set(p.wave_half_periods)


@pytest.mark.parametrize("variant", ["default", "tuned"])
def test_slopes_ramp_at_their_angle(variant):
    arena = _variant(variant)
    for t in arena.spec.tiles:
        if t.terrain_type in SLOPES:
            _assert_slope_profile(arena, t)


# -- stairs ---------------------------------------------------------------


def test_the_summit_is_the_pad_never_below_its_floor():
    p = terrain.ArenaParams()
    assert terrain.summit_platform_half(p) == p.pad_radius == 0.4
    assert terrain.summit_platform_half(replace(p, pad_radius=0.2)) == p.summit_floor
    assert terrain.summit_platform_half(replace(p, pad_radius=0.55)) == 0.55
    assert terrain.generate(n_rows=2).spec.stair_platform_half == 0.4


def test_stair_steps_fill_the_room_between_summit_and_rim():
    p = terrain.ArenaParams()
    # A 0.4 m summit and a 0.25 m rim leave 1.35 m on a 4 m tile.
    assert terrain.stair_steps(0.30, p) == 5  # 4 treads, 1.20 m
    assert terrain.stair_steps(0.45, p) == 4  # 3 treads, 1.35 m (exact in floats)
    assert terrain.stair_steps(0.25, p) == 6  # 5 treads, 1.25 m
    # A 0.3 m rim leaves 1.3 m, which four 0.325 m treads fill exactly on
    # paper. In floats the quotient comes out one ulp under 4 and the flight
    # ends one ulp past the rim. A result of 4 means the floor lost its
    # slack. A ValueError means the rim check did.
    tight = replace(p, rim_margin=0.3)
    rim = tight.tile_size / 2 - tight.rim_margin
    assert (rim - 0.4) / 0.325 < 4
    assert terrain.stair_flight_half(0.325, 5, tight) > rim
    assert terrain.stair_steps(0.325, tight) == 5
    # Short treads hit the riser ceiling. The flight stops short of the rim.
    assert terrain.stair_steps(0.13, p) == p.stair_max_steps
    # A wider summit leaves room for fewer treads.
    assert terrain.stair_steps(0.30, replace(p, pad_radius=0.55)) == 5
    assert terrain.stair_steps(0.30, replace(p, pad_radius=0.55, tile_size=3.6)) == 4


def test_a_flight_past_the_rim_is_a_build_error():
    p = terrain.ArenaParams()
    # The two-tread floor at 0.7 m reaches 1.80 m of the 1.75 m available.
    with pytest.raises(ValueError, match="rim"):
        terrain.stair_steps(0.7, p)
    with pytest.raises(ValueError, match="rim"):
        terrain.generate(stair_tread=(0.3, 0.7), n_rows=2)
    # The same tread fits a wider tile.
    assert terrain.stair_steps(0.7, replace(p, tile_size=5.0)) == 3


@pytest.mark.parametrize("variant", ["default", "small_pad", "tuned"])
def test_stair_flights_match_their_risers_and_treads(variant):
    """Every lookup node on a flight reads its tread's height: one riser
    down per tread on a pyramid, one riser up in a pit. Nodes within a cell
    of a tread edge are skipped. Probes on all four arms then pin the
    summit edge, each tread's inner edge and the flat ground past the
    flight."""
    a = _variant(variant)
    s = a.spec
    p = s.params
    platform = s.stair_platform_half
    assert platform == terrain.summit_platform_half(p)
    if variant == "small_pad":
        assert platform == p.summit_floor != p.pad_radius
    cell = p.cell_size
    for t in (t for t in s.tiles if t.terrain_type in terrain.STAIR_TYPES):
        cx, cy, _ = t.origin
        riser = p.stair_riser(t.difficulty)
        pyramid = t.terrain_type == "pyramid_stairs"

        def tread_height(k, t=t, riser=riser, pyramid=pyramid):
            return (t.n_steps - k) * riser if pyramid else (k - t.n_steps) * riser

        cheby = _tile_cheby(a, t)
        k = np.floor((cheby - platform) / t.stair_tread) + 1
        inner = platform + np.clip(k - 1, 0, None) * t.stair_tread
        edge = np.minimum(np.abs(cheby - inner), np.abs(cheby - inner - t.stair_tread))
        on_flight = (k >= 1) & (k <= t.n_steps - 1) & (edge > cell)
        assert on_flight.any()
        np.testing.assert_allclose(
            a.lookup[on_flight],
            tread_height(k[on_flight]),
            rtol=0,
            atol=1e-5,
            err_msg=f"{t.terrain_type} row {t.row}",
        )
        for ux, uy in ((1, 0), (-1, 0), (0, 1), (0, -1)):

            def at(r, ux=ux, uy=uy, cx=cx, cy=cy):
                return float(terrain.lookup_height(a, cx + ux * r, cy + uy * r))

            assert at(0.0) == pytest.approx(t.pad_height, abs=1e-6)
            assert at(platform - cell) == pytest.approx(t.pad_height, abs=1e-5)
            for kk in range(1, t.n_steps):
                r_in = platform + (kk - 1) * t.stair_tread
                assert at(r_in + 1.5 * cell) == pytest.approx(
                    tread_height(kk), abs=1e-5
                ), (t.terrain_type, t.row, kk)
            assert at(t.feature_radius + 2 * cell) == 0.0


@pytest.mark.parametrize(
    "variant", ["default", "small_pad", "tread_range", "narrowest_tread"]
)
def test_stair_rings_tile_the_flight_without_overlap(variant):
    """A stair tile's boxes are unyawed and cover its flight square once: a
    pyramid's summit block and four boxes per ring, or a pit's rings around
    its bare floor. Overlapping or duplicate boxes keep the surface but add
    geoms and corner contacts to the contact budget."""
    a = _variant(variant)
    s = a.spec
    by_tile: dict[tuple[int, int], list] = {}
    for b in a.boxes:
        t = _box_tile(a, b)
        if t.terrain_type in terrain.STAIR_TYPES:
            by_tile.setdefault((t.row, t.col), []).append(b)
    stairs = [t for t in s.tiles if t.terrain_type in terrain.STAIR_TYPES]
    assert len(by_tile) == len(stairs)
    for t in stairs:
        boxes = by_tile[(t.row, t.col)]
        pyramid = t.terrain_type == "pyramid_stairs"
        assert all(b.yaw == 0.0 for b in boxes)
        assert len(boxes) == int(pyramid) + 4 * (t.n_steps - 1)
        lo = np.array([(b.pos[0] - b.half[0], b.pos[1] - b.half[1]) for b in boxes])
        hi = np.array([(b.pos[0] + b.half[0], b.pos[1] + b.half[1]) for b in boxes])
        # Two boxes overlap inside when they overlap along both axes.
        span = np.minimum(hi[:, None], hi[None]) - np.maximum(lo[:, None], lo[None])
        inside = (span > 1e-9).all(axis=2)
        np.fill_diagonal(inside, False)
        assert not inside.any(), (t.terrain_type, t.row)
        area = sum(4 * b.half[0] * b.half[1] for b in boxes)
        want = (2 * t.feature_radius) ** 2
        if not pyramid:
            want -= (2 * s.stair_platform_half) ** 2
        assert area == pytest.approx(want, abs=1e-9), (t.terrain_type, t.row)


@pytest.mark.parametrize("stair_tread", [0.30, (0.25, 0.45), 0.08])
def test_the_pit_wall_hides_under_the_outermost_ring(stair_tread):
    """The heightfield past a pit's last carved node is ground level, and
    that node sits under the outermost ring. No groove opens beyond the
    flight wherever its edge falls on the node grid. A drawn tread almost
    never puts the edge on a node. 0.08 m is the narrowest tread
    ArenaParams admits, two cells."""
    a = terrain.generate(n_rows=4, stair_tread=stair_tread)
    s = a.spec
    cell = s.params.cell_size
    surface = _surface(a)
    for t in (t for t in s.tiles if t.terrain_type == "inverted_pyramid_stairs"):
        cheby = _tile_cheby(a, t)
        tile = cheby <= s.params.tile_size / 2
        carved = tile & (surface < -1e-6)
        assert carved.any()
        last = cheby[carved].max()
        assert last <= t.feature_radius - cell + 1e-6
        assert last >= t.feature_radius - t.stair_tread - 1e-6
        assert np.abs(surface[tile & ~carved]).max() < 1e-6


def test_a_tread_range_draws_per_tile_within_the_range():
    a = _variant("tread_range")
    p = a.spec.params
    tiles = [t for t in a.spec.tiles if t.terrain_type in terrain.STAIR_TYPES]
    treads = np.array([t.stair_tread for t in tiles])
    assert ((treads >= 0.25) & (treads <= 0.45)).all()
    assert len(set(treads.tolist())) == len(tiles)  # a draw per tile
    for t in tiles:
        # The tread is the first draw from the tile's own stream. With no
        # flat row, (row, col) is the stream's key.
        draw = _stream(p, _TILE, t.row, t.col).uniform(*p.stair_tread)
        assert t.stair_tread == float(draw), (t.terrain_type, t.row)
        assert t.n_steps == terrain.stair_steps(t.stair_tread, p)
        assert t.feature_radius == terrain.stair_flight_half(
            t.stair_tread, t.n_steps, p
        )
        assert t.feature_radius <= p.tile_size / 2 - p.rim_margin + 1e-9
    # A fixed tread draws nothing and lands on every stair tile.
    fixed = [
        t
        for t in terrain.generate(n_rows=3).spec.tiles
        if t.terrain_type in terrain.STAIR_TYPES
    ]
    assert {t.stair_tread for t in fixed} == {0.30}
    assert {t.n_steps for t in fixed} == {5}


@pytest.mark.parametrize("variant", ["default", "tuned"])
def test_no_flight_passes_the_riser_cap(variant):
    """Every flight keeps stair_min_steps to stair_max_steps risers. The
    tuned tread leaves room for more risers than its cap, so there the cap
    sets every count."""
    a = _variant(variant)
    p = a.spec.params
    stairs = [t for t in a.spec.tiles if t.terrain_type in terrain.STAIR_TYPES]
    assert all(p.stair_min_steps <= t.n_steps <= p.stair_max_steps for t in stairs)
    if variant == "tuned":
        room = p.tile_size / 2 - p.rim_margin - a.spec.stair_platform_half
        assert math.floor(room / p.stair_tread) + 1 > p.stair_max_steps
        assert {t.n_steps for t in stairs} == {p.stair_max_steps}


@pytest.mark.parametrize("variant", ["default", "tuned"])
def test_feature_radius_per_type_bounds_every_box(variant):
    arena = _variant(variant)
    p = arena.spec.params
    for t in arena.spec.tiles:
        if t.terrain_type in terrain.STAIR_TYPES:
            assert t.feature_radius == terrain.stair_flight_half(
                t.stair_tread, t.n_steps, p
            )
        else:
            assert t.feature_radius == p.tile_size / 2 - p.edge_margin
            assert t.n_steps == 0 and t.stair_tread == 0.0
    for b in arena.boxes:
        t = _box_tile(arena, b)
        assert _box_reach(b, t) <= t.feature_radius + 1e-9, (t.terrain_type, b)


@pytest.mark.parametrize("variant", ["default", "tread_range", "tuned"])
def test_past_the_feature_radius_a_stair_tile_is_flat_and_a_slope_is_not(variant):
    """A stair tile is flat ground from its feature radius out to its edge.
    The top row's hill keeps climbing there, down to 0 only at the edge."""
    a = _variant(variant)
    s = a.spec
    half = s.params.tile_size / 2
    top = s.n_rows - 1
    for t in s.tiles:
        stairs = t.terrain_type in terrain.STAIR_TYPES
        if not stairs and (t.terrain_type, t.row) != ("pyramid_slope", top):
            continue
        cheby = _tile_cheby(a, t)
        band = a.lookup[(cheby > t.feature_radius + 1e-6) & (cheby < half - 1e-6)]
        assert band.size > 0
        if stairs:
            assert np.all(band == 0.0), (t.terrain_type, t.row)
        else:
            assert np.all(band > 0.0)


# -- caps -------------------------------------------------------------------


def test_type_caps_compress_a_column():
    caps = {"pyramid_stairs": 0.7, "random_grid": 0.55}
    a = terrain.generate(n_rows=5, type_caps=caps)
    b = terrain.generate(n_rows=5)
    for ta, tb in zip(a.spec.tiles, b.spec.tiles):
        assert ta.terrain_type == tb.terrain_type
        assert ta.difficulty == pytest.approx(
            tb.difficulty * caps.get(ta.terrain_type, 1.0)
        )
    top = max(t.difficulty for t in a.spec.tiles if t.terrain_type == "pyramid_stairs")
    assert top == pytest.approx(0.7)


@pytest.mark.parametrize("ttype, cap", [("pyramid_stairs", 0.7), ("random_grid", 0.55)])
def test_a_capped_tile_is_built_at_its_capped_difficulty(ttype, cap):
    """A capped tile matches the same tile of an uncapped arena whose rows
    are given the capped difficulty. Explicit difficulties keep the column
    order and every tile's stream."""
    rows = [0.0, 0.3, 0.6, 1.0]
    a = terrain.generate(difficulties=rows, type_caps={ttype: cap})
    b = terrain.generate(difficulties=[cap * d for d in rows])
    checked = 0
    for ta, tb in zip(a.spec.tiles, b.spec.tiles):
        if ta.terrain_type != ttype:
            continue
        assert ta == tb
        window = _tile_window(a, ta)
        np.testing.assert_array_equal(a.lookup[window], b.lookup[window])
        checked += 1
    assert checked == len(rows)


# -- params -----------------------------------------------------------------


def test_every_params_field_is_normalized():
    """A field added later must join a normalization group."""
    groups = _FLOATS + _INTS + _BOOLS + _RAMPS + _RANGES
    special = ("difficulties", "type_caps", "stair_tread", "wave_half_periods")
    assert len(set(groups + special)) == len(groups + special)
    assert set(groups + special) == {f.name for f in fields(terrain.ArenaParams)}


def _geometry(arena):
    """Everything an arena built, without the params that name it."""
    return (
        arena.lookup.shape,
        arena.lookup.tobytes(),
        arena.hfield_data.tobytes(),
        arena.boxes,
    )


def test_the_sweep_moves_every_params_field():
    """A field added later needs an alternative value in the sweep."""
    assert set(ALTERNATIVES) == {f.name for f in fields(terrain.ArenaParams)}


@pytest.mark.parametrize("name", sorted(ALTERNATIVES))
def test_every_params_field_reaches_the_arena(name):
    """Moving any one field off its default changes what the 3-row arena
    builds. The pinned fingerprint covers the default arena only, so this
    catches code that reads a default constant in place of its field."""
    overrides = {"n_rows": 3, name: ALTERNATIVES[name]}
    if name == "stair_min_steps":
        # stair_steps lays as many treads as fit, so a riser floor that
        # binds always asks for one tread more than the rim allows.
        with pytest.raises(ValueError, match="rim"):
            terrain.generate(**overrides)
        return
    a = terrain.generate(**overrides)
    assert _geometry(a) != _geometry(_variant("rows3"))
    # Spawn code reads the pad and summit sizes from the spec.
    p = a.spec.params
    assert all(t.pad_radius == p.pad_radius for t in a.spec.tiles)
    assert a.spec.stair_platform_half == terrain.summit_platform_half(p)


def test_int_and_numpy_scalars_name_the_same_arena(arena):
    """A yaml `4` and a numpy integer seed build and hash like the Python
    float and int."""
    as_ints = terrain.generate(tile_size=4, border=2, row_jitter=0.4)
    assert terrain.fingerprint(as_ints) == terrain.fingerprint(arena)
    a = terrain.generate(seed=np.int64(1), n_rows=2)
    assert terrain.fingerprint(a) == terrain.fingerprint(
        terrain.generate(seed=1, n_rows=2)
    )
    assert type(a.spec.params.seed) is int
    json.dumps(terrain.spec_to_dict(a.spec))


@pytest.mark.parametrize(
    "bad",
    [{"seed": 1.5}, {"n_rows": 2.0}, {"ordered": 1}, {"flat_row": "false"}],
)
def test_params_refuse_a_scalar_of_the_wrong_kind(bad):
    with pytest.raises(TypeError):
        terrain.ArenaParams(**bad)


def test_params_take_a_config_block_as_it_is():
    block = {
        "tile_size": 3.0,
        "stair_tread": [0.25, 0.45],
        "type_caps": {"pyramid_stairs": 0.7},
        "stair_riser": {"base": 0.02, "gain": 0.1},
        "slope_angle": [0.0, 0.3],
        "difficulties": [0.2, 0.4],
    }
    p = terrain.params_from_dict(block)
    assert p.stair_tread == (0.25, 0.45)
    assert p.type_caps == (("pyramid_stairs", 0.7),)
    assert p.stair_riser == terrain.Ramp(0.02, 0.1)
    assert p.slope_angle == terrain.Ramp(0.0, 0.3)
    assert p.n_rows == 2
    assert terrain.params_from_dict(terrain.params_to_dict(p)) == p
    with pytest.raises(TypeError):
        terrain.params_from_dict({"tile_sise": 3.0})


@pytest.mark.parametrize(
    "bad, match",
    [
        ({"type_caps": {"stairs": 0.5}}, "unknown terrain types"),
        ({"cell_size": 0.03}, "whole number"),
        ({"cell_size": 0.0}, "cell_size"),
        ({"pad_radius": 0.7}, "slope plateau"),
        ({"n_rows": 1}, "at least 2"),
        ({"difficulties": (-0.5,)}, "difficulties"),
        ({"row_jitter": 0.5}, "strictly increasing"),
        ({"stair_tread": (0.4, 0.3)}, "lo <= hi"),
        ({"pad_taper": 0}, "pad_taper"),
        ({"edge_taper": 0}, "edge_taper"),
        ({"coarse_step": -0.1}, "coarse_step"),
        ({"grid_pitch": 0}, "grid_pitch"),
        ({"discrete_half_range": (-0.1, 0.1)}, "discrete_half_range"),
        ({"discrete_count": -1}, "discrete_count"),
        ({"wave_half_periods": ()}, "wave_half_periods"),
        ({"wave_half_periods": (0, 2)}, "wave_half_periods"),
        ({"stair_riser": (0, 0.13)}, "stair_riser"),
        ({"rim_margin": 0.0}, "rim_margin"),
        ({"rim_margin": -0.1}, "rim_margin"),
        ({"edge_margin": -0.2}, "edge_margin"),
        ({"slope_platform_half": 2.0}, "slope_platform_half"),
        ({"slope_platform_half": 1.97}, "slope_platform_half"),
        ({"slope_angle": {"base": -0.2, "gain": 0.35}}, "slope_angle"),
        ({"slope_angle": {"base": 0.0, "gain": 20.0}}, "slope_angle"),
        ({"difficulties": (0.0, 4.6)}, "slope_angle"),
        ({"type_caps": {"inverted_pyramid_slope": 5.0}}, "slope_angle"),
        ({"pad_clearance": -0.01}, "pad_clearance"),
        ({"discrete_half_range": (0.1, 1.5)}, "discrete_half_range"),
        # Boxes that could fall between nodes, where the lookup misses them.
        ({"cell_size": 0.1}, "grid_half_range .* no node"),
        ({"grid_half_range": (0.008, 0.012)}, "grid_half_range .* no node"),
        # Each half-size alone covers a node, but the shrink to a 0.1 m
        # cell takes the shortest below cell_size / sqrt(2).
        ({"grid_pitch": 0.1}, "grid_half_range .* no node"),
        ({"discrete_half_range": (0.02, 0.25)}, "discrete_half_range .* no node"),
        # Treads under two cells: the pit wall leaves the outermost ring, and
        # a ring band can miss every node line.
        ({"stair_tread": 0.045}, "stair_tread .* two cell_size"),
        ({"stair_tread": 0.06}, "stair_tread .* two cell_size"),
        ({"stair_tread": (0.045, 0.30)}, "stair_tread .* two cell_size"),
        ({"stair_tread": (0.06, 0.3)}, "stair_tread .* two cell_size"),
        # The bound follows the cell: a 0.2 m cell refuses the 0.30 m default
        # tread. The box sizes rise to pass their own node checks.
        (
            {
                "cell_size": 0.2,
                "edge_margin": 0.2,
                "discrete_half_range": (0.15, 0.25),
                "grid_pitch": 0.5,
                "grid_half_range": (0.15, 0.15),
            },
            "stair_tread .* two cell_size",
        ),
        ({"stair_tread": 0.0}, "stair_tread .* must be positive"),
        ({"stair_min_steps": 5, "stair_max_steps": 4}, "stair steps"),
        ({"stair_min_steps": 1}, "stair steps"),
        ({"type_caps": {"wave": 0.0}}, "type_caps must be positive"),
        ({"obstacle_height": (0.0, 0.10)}, "obstacle_height base"),
        ({"rough_amplitude": (0.005, 0.0)}, "ramp gain"),
        ({"pad_radius": 0.0}, "pad_radius"),
        # Redundant while pad_radius is positive: this pins config hygiene,
        # not a build failure.
        ({"summit_floor": 0.0}, "summit_floor"),
        ({"border": -0.04}, "border must not be negative"),
        ({"border": 0.02}, "whole number"),
        ({"border": 2.02}, "2.02 m is not a whole number"),
        ({"discrete_half_range": (0.2, 0.1)}, "lo <= hi"),
        ({"difficulties": ()}, "at least 1 terrain rows"),
    ],
)
def test_params_refuse_an_arena_that_cannot_build(bad, match):
    with pytest.raises(ValueError, match=match):
        terrain.ArenaParams(**bad)


def test_a_zero_border_puts_the_tiles_on_the_heightfield_edge():
    """The tiles are the bordered arena's, node for node, without the flat
    frame around them."""
    a = terrain.generate(n_rows=3, border=0.0)
    framed = _variant("rows3")
    s = a.spec
    p = s.params
    n = round(p.tile_size / p.cell_size)
    assert s.hfield.nrow == p.n_rows * n + 1
    assert s.hfield.ncol == len(terrain.TYPES) * n + 1
    assert s.hfield.radius_x == len(terrain.TYPES) * p.tile_size / 2
    assert s.hfield.radius_y == p.n_rows * p.tile_size / 2
    bc = round(framed.spec.params.border / p.cell_size)
    assert np.array_equal(a.lookup, framed.lookup[bc:-bc, bc:-bc])
    assert a.boxes == framed.boxes


def test_the_smallest_box_bound_is_exact():
    """A box covers a node when its shorter half-size, after the rubble
    shrink, reaches cell_size / sqrt(2). The defaults pass, and so do
    sizes at the bound and grids that only just meet it."""
    terrain.ArenaParams()
    need = 0.04 / math.sqrt(2)
    terrain.ArenaParams(discrete_half_range=(need, 0.25), grid_half_range=(need, need))
    # 0.04 m rubble against a 0.0354 m bound.
    terrain.ArenaParams(cell_size=0.05)
    # Shrunk to 0.0335 m against the 0.0283 m bound.
    terrain.ArenaParams(grid_pitch=0.2)


def test_explicit_difficulties_past_1_still_build():
    """Explicit rows are not clamped at 1. The slope check refuses only an
    angle at or past pi/2."""
    a = terrain.generate(difficulties=(0.0, 1.5))
    top = _tile(a, "pyramid_slope", 1)
    assert top.difficulty == 1.5
    assert top.pad_height == pytest.approx(
        terrain.slope_plateau_height(1.5, a.spec.params)
    )


# -- serialization ----------------------------------------------------------


def test_the_spec_round_trips_through_a_dict_and_json():
    a = terrain.generate(
        n_rows=3, flat_row=True, stair_tread=(0.25, 0.45), type_caps={"wave": 0.5}
    )
    d = terrain.spec_to_dict(a.spec)
    assert terrain.spec_from_dict(d) == a.spec
    via_json = json.loads(json.dumps(d))
    assert via_json == d
    assert terrain.spec_from_dict(via_json) == a.spec
    assert d["generator_version"] == terrain.GENERATOR_VERSION
    assert d["params"]["stair_tread"] == [0.25, 0.45]
    assert d["params"]["type_caps"] == {"wave": 0.5}
    # Each stair tile carries its own tread and riser count. The flat row's
    # stair columns have neither.
    stairs = [t for t in d["tiles"] if t["terrain_type"] in terrain.STAIR_TYPES]
    assert all((t["n_steps"] >= 3) == (t["row"] > 0) for t in stairs)
    assert all((0.25 <= t["stair_tread"] <= 0.45) == (t["row"] > 0) for t in stairs)


# -- lookup grid ------------------------------------------------------------


@pytest.mark.parametrize("variant", ["default", "tuned"])
def test_the_lookup_is_the_heightfield_surface_with_box_tops_written_in(variant):
    """Rasterized independently here. A node no box covers reads the
    heightfield surface MuJoCo compiles (pos_z + data * elevation_z). A
    node inside boxes reads the highest of their tops. Nodes within 1 um of
    a box edge could go either way and are skipped."""
    arena = _variant(variant)
    s = arena.spec
    surface = _surface(arena)
    xs, ys = terrain.grid_axes(s)
    expected = surface.copy()
    covered = np.zeros(surface.shape, dtype=bool)
    edge = np.zeros(surface.shape, dtype=bool)
    for b in arena.boxes:
        window, margin = _footprint_margin(xs, ys, b)
        inside = margin < -1e-6
        covered[window] |= inside
        edge[window] |= np.abs(margin) <= 1e-6
        top = b.pos[2] + b.half[2]
        expected[window] = np.where(
            inside, np.maximum(expected[window], top), expected[window]
        )
    bare = ~covered & ~edge
    on_box = covered & ~edge
    assert bare.sum() > 0.5 * bare.size
    assert on_box.sum() > 1_000 * s.n_rows
    np.testing.assert_allclose(arena.lookup[bare], surface[bare], rtol=0, atol=1e-5)
    np.testing.assert_allclose(
        arena.lookup[on_box], expected[on_box], rtol=0, atol=1e-5
    )


def test_the_lookup_blurs_a_box_edge_between_nodes_both_ways(arena):
    """The top row's pyramid_stairs ring 1 ends at 0.70 m, mid-cell on the
    0.04 m grid. Just inside that edge the lookup reads the ring late,
    below its top but within one riser. Just outside it reads the next ring
    early, above its top. A node more than a cell inside reads the top
    exactly."""
    p = arena.spec.params
    t = _tile(arena, "pyramid_stairs", arena.spec.n_rows - 1)
    edge = arena.spec.stair_platform_half + t.stair_tread
    assert edge == pytest.approx(0.70)
    assert edge / p.cell_size == pytest.approx(17.5)
    riser = p.stair_riser(t.difficulty)
    ring1, ring2 = (t.n_steps - 1) * riser, (t.n_steps - 2) * riser
    cx, cy, _ = t.origin
    for ux, uy in ((1, 0), (-1, 0), (0, 1), (0, -1)):

        def at(r, ux=ux, uy=uy):
            return float(terrain.lookup_height(arena, cx + ux * r, cy + uy * r))

        assert ring2 <= at(0.69) < ring1 - 1e-3, (ux, uy)
        assert ring2 + 1e-3 < at(0.71) <= ring1, (ux, uy)
        assert at(0.66) == pytest.approx(ring1, abs=1e-5), (ux, uy)


@pytest.mark.parametrize(
    "ttype, ratio", [("pyramid_stairs", 0.75), ("inverted_pyramid_stairs", 0.5)]
)
def test_a_stair_top_reads_at_most_three_quarters_of_a_riser_low(arena, ttype, ratio):
    """On the top row's stairs, points within one cell inside a tread edge
    read below the tread. A pyramid tread's outer corner, with both edges
    mid-cell, has one of its cell's four nodes on the tread. Approaching
    it, the lookup reads 3/4 of a riser low. Treads span at least two
    cells, so the other three nodes lie one riser lower and nothing reads
    lower. A pit tread's outer corners border the higher ring. Its low side
    is the inner edge, which reads at most half a riser low."""
    p = arena.spec.params
    cell = p.cell_size
    t = _tile(arena, ttype, arena.spec.n_rows - 1)
    riser = p.stair_riser(t.difficulty)
    cx, cy, _ = t.origin
    boxes = [
        b
        for b in arena.boxes
        if abs(b.pos[0] - cx) < p.tile_size / 2 and abs(b.pos[1] - cy) < p.tile_size / 2
    ]

    def top_at(x, y):
        z = np.full_like(x, -np.inf)
        for b in boxes:
            inside = (np.abs(x - b.pos[0]) <= b.half[0]) & (np.abs(y - b.pos[1]) <= b.half[1])
            z = np.where(inside, np.maximum(z, b.pos[2] + b.half[2]), z)
        return z

    worst = 0.0
    for b in boxes:
        hx, hy, _ = b.half
        # The end points sit 1 um inside the corners.
        u, v = np.meshgrid(
            np.linspace(-hx + 1e-6, hx - 1e-6, 201), np.linspace(-hy + 1e-6, hy - 1e-6, 201)
        )
        band = (hx - np.abs(u) <= cell) | (hy - np.abs(v) <= cell)
        x, y = b.pos[0] + u[band], b.pos[1] + v[band]
        top = b.pos[2] + b.half[2]
        mine = np.abs(top_at(x, y) - top) < 1e-9
        low = top - terrain.lookup_height(arena, x[mine], y[mine])
        worst = max(worst, float(low.max()))
    assert ratio * riser - 1e-3 < worst <= ratio * riser + 1e-6, worst


def test_bilinear_and_the_triangle_split_differ_by_under_4_5_mm_on_bare_cells(arena):
    """Within a cell, bilinear and either of MuJoCo's two triangles differ
    by at most a quarter of the cell's twist |h00 - h01 - h10 + h11|,
    reached at its centre. A cell is bare when no box top is written onto
    any of its four nodes. The worst bare cells lie on the diagonal creases
    of the d = 1 slopes. The crease alone twists by
    tan(slope_angle(1)) * cell_size there, and the overlay adds the rest."""
    p = arena.spec.params
    surface = _surface(arena)
    node = np.abs(arena.lookup - surface) < 1e-6
    bare = node[:-1, :-1] & node[:-1, 1:] & node[1:, :-1] & node[1:, 1:]
    twist = np.abs(
        surface[:-1, :-1] - surface[:-1, 1:] - surface[1:, :-1] + surface[1:, 1:]
    )
    gap = float(twist[bare].max()) / 4
    crease = math.tan(p.slope_angle(1.0)) * p.cell_size / 4
    assert crease <= gap < 4.5e-3, (crease, gap)


@pytest.mark.parametrize("variant", ["default", "tread_range", "tuned"])
def test_no_gap_opens_under_any_box(variant):
    """Every box's base lies at or below the lowest node of every
    heightfield cell its footprint touches, so neither MuJoCo's triangles
    nor bilinear dip below it anywhere under the box. A stair box stands on
    flat ground or the pit floor. Outermost pit rings embed into the pit
    wall. A scattered box on rolling ground reaches past its highest node
    down to the low side, and no deeper than the ground within two cells of
    its bounding square."""
    a = _variant(variant)
    surface = _surface(a)
    xs, ys = terrain.grid_axes(a.spec)
    cell = a.spec.params.cell_size
    reach_down = []
    for b in a.boxes:
        ttype = _box_tile(a, b).terrain_type
        base = b.pos[2] - b.half[2]
        assert base <= _touched_floor(surface, xs, ys, b) + 1e-5, (ttype, b)
        if ttype in terrain.STAIR_TYPES:
            continue
        hx, hy, _ = b.half
        near = (np.abs(xs - b.pos[0]) <= hx + hy + 2 * cell)[None, :] & (
            np.abs(ys - b.pos[1]) <= hx + hy + 2 * cell
        )[:, None]
        assert base >= surface[near].min() - 1e-5, (ttype, b)
        window, margin = _footprint_margin(xs, ys, b)
        reach_down.append(surface[window][margin <= 1e-6].max() - base)
    assert len(reach_down) > 100 * a.spec.n_rows
    # The overlay rolls, so some box reaches well below its highest node.
    assert max(reach_down) > 1e-3


@pytest.mark.parametrize(
    "variant", ["default", "3m", "tuned", "smallest_boxes", "narrowest_tread"]
)
def test_every_box_shows_in_the_lookup(variant):
    """Every box's footprint holds a node, and the lookup reads at least
    the box's top there. The smallest boxes and the narrowest stair rings
    ArenaParams admits pass too."""
    a = _variant(variant)
    xs, ys = terrain.grid_axes(a.spec)
    for b in a.boxes:
        window, margin = _footprint_margin(xs, ys, b)
        on = margin <= 1e-9
        assert on.any(), b
        top = b.pos[2] + b.half[2]
        assert a.lookup[window][on].max() >= top - 1e-5, b


@pytest.mark.parametrize("variant", ["default", "flat_row"])
def test_every_box_and_the_heightfield_have_positive_size(variant):
    a = _variant(variant)
    assert all(min(b.half) > 0 for b in a.boxes)
    assert a.spec.hfield.elevation_z > 0


def test_heightfield_data_is_normalized_over_its_own_range(arena):
    hf = arena.spec.hfield
    assert arena.hfield_data.dtype == np.float32
    assert arena.hfield_data.shape == (hf.nrow, hf.ncol)
    assert arena.hfield_data.min() == 0.0 and arena.hfield_data.max() == 1.0
    # The floor of the deepest pit, the top row's inverted stairs.
    deepest = _tile(arena, "inverted_pyramid_stairs", arena.spec.n_rows - 1)
    assert hf.pos_z == deepest.pad_height
    assert hf.elevation_z > 0 and hf.base_z > 0


@pytest.mark.parametrize("variant", ["default", "flat_row", "3m", "tuned"])
def test_the_heightfield_spans_the_extent_on_the_cell_grid(variant):
    a = _variant(variant)
    s = a.spec
    p = s.params
    hf = s.hfield
    assert (s.x_min, s.x_max) == (-hf.radius_x, hf.radius_x)
    assert (s.y_min, s.y_max) == (-hf.radius_y, hf.radius_y)
    assert hf.radius_x == pytest.approx(s.n_cols * p.tile_size / 2 + p.border)
    assert hf.radius_y == pytest.approx(s.n_rows * p.tile_size / 2 + p.border)
    assert hf.ncol == round(2 * hf.radius_x / p.cell_size) + 1
    assert hf.nrow == round(2 * hf.radius_y / p.cell_size) + 1
    assert a.lookup.shape == a.hfield_data.shape == (hf.nrow, hf.ncol)
    # The physics extent and the sampler's node grid agree.
    _, dx, _, dy = terrain.sample_frame(s)
    assert dx == pytest.approx(2 * hf.radius_x / (hf.ncol - 1))
    assert dx == pytest.approx(p.cell_size)
    assert dy == pytest.approx(2 * hf.radius_y / (hf.nrow - 1))
    assert dy == pytest.approx(p.cell_size)


@pytest.mark.parametrize("variant", ["default", "flat_row", "tuned"])
def test_spawn_pads_are_flat(variant):
    """Every pad is flat to 1 mm at its declared height, sampled out to one
    cell inside its rim. That radius is the spawn reach budget: footprint
    plus jitter.

    A point at radius r has Chebyshev radius at most r. Out to
    pad_radius - cell, its bilinear nodes stay short of the pit's ring 1,
    which starts at the summit edge, a node line here. Taper nodes just
    past the rim add under 0.1 mm. In the last cell the pit's lookup rises
    toward ring 1's tread top, which the summit-edge node holds."""
    arena = _variant(variant)
    s = arena.spec
    p = s.params
    cell = p.cell_size
    assert s.stair_platform_half / cell == pytest.approx(
        round(s.stair_platform_half / cell)
    )
    for t in s.tiles:
        cx, cy, _ = t.origin
        r = np.linspace(0.0, p.pad_radius - cell, 8)
        ang = np.linspace(0.0, 2 * np.pi, 32)
        h = terrain.lookup_height(
            arena,
            cx + np.outer(r, np.cos(ang)).ravel(),
            cy + np.outer(r, np.sin(ang)).ravel(),
        )
        assert np.abs(h - t.pad_height).max() < 1e-3, (t.terrain_type, t.row)
        # A flat row's stair columns hold no flight (n_steps 0).
        if t.terrain_type == "inverted_pyramid_stairs" and t.n_steps:
            rim = float(terrain.lookup_height(arena, cx + s.stair_platform_half, cy))
            riser = p.stair_riser(t.difficulty)
            assert rim == pytest.approx(t.pad_height + riser, abs=1e-5)


@pytest.mark.parametrize("variant", ["default", "flat_row_pad", "flat_row_tile"])
def test_every_tile_origin_is_its_centre_node(variant):
    """Spawn code reads each tile's origin and pad radius from the spec. The
    origin is the centre node of the tile's own node window, which holds
    its surface. The flat row's tiles, built apart from the terrain rows,
    are held to the same rule."""
    a = _variant(variant)
    s = a.spec
    p = s.params
    xs, ys = terrain.grid_axes(s)
    n = round(p.tile_size / p.cell_size)
    bc = round(p.border / p.cell_size)
    assert len(s.tiles) == s.n_rows * s.n_cols
    for t in s.tiles:
        assert t.origin == pytest.approx(
            (xs[bc + t.col * n + n // 2], ys[bc + t.row * n + n // 2], 0.0), abs=1e-9
        ), (t.terrain_type, t.row)
        assert t.pad_radius == p.pad_radius


@pytest.mark.parametrize("variant", ["default", "tuned"])
def test_scattered_boxes_clear_the_pad(variant):
    """Every obstacle and rubble box stays pad_clearance outside the spawn
    pad, measured to the nearest point of its yawed footprint."""
    arena = _variant(variant)
    p = arena.spec.params
    closest = math.inf
    for b in arena.boxes:
        t = _box_tile(arena, b)
        if t.terrain_type not in BOX_SCATTER:
            continue
        u, v = b.pos[0] - t.origin[0], b.pos[1] - t.origin[1]
        c, s = math.cos(b.yaw), math.sin(b.yaw)
        lu, lv = abs(u * c + v * s), abs(-u * s + v * c)
        hx, hy, _ = b.half
        closest = min(closest, math.hypot(max(lu - hx, 0.0), max(lv - hy, 0.0)))
    assert closest >= p.pad_radius + p.pad_clearance - 1e-9


def _rubble_cells(p):
    """Tile-local centre of every rubble cell along one axis."""
    n = rubble_cells(p)
    return (np.arange(n) - (n - 1) / 2) * p.grid_pitch


def _rubble_cell(p, u, v):
    """(row, column) of the rubble cell whose centre is nearest (u, v)."""
    start = _rubble_cells(p)[0]
    return round((v - start) / p.grid_pitch), round((u - start) / p.grid_pitch)


@pytest.mark.parametrize(
    "tile_size, grid_pitch, cells, outer",
    [(4.0, 0.2, 19, 1.8), (3.0, 0.2, 14, 1.3), (3.0, 0.4, 7, 1.2)],
)
def test_rubble_fills_a_room_its_pitch_divides(tile_size, grid_pitch, cells, outer):
    """A grid_pitch that divides the room inside edge_margin lays cells out
    to that margin on every side. In each case room / grid_pitch comes out
    one ulp short of the whole count. Boxes as wide as the pitch shrink to
    fill their cell and barely jitter, so each box centre names its cell."""
    a = terrain.generate(
        difficulties=(1.0,),
        tile_size=tile_size,
        grid_pitch=grid_pitch,
        grid_fill_prob=1.0,
        grid_half_range=(grid_pitch, grid_pitch),
    )
    p = a.spec.params
    assert outer + grid_pitch / 2 == pytest.approx(tile_size / 2 - p.edge_margin)
    boxes = next(b for (tt, _), b in _scattered_boxes(a).items() if tt == "random_grid")
    half = grid_pitch / 2
    # Cell centres sit on whole multiples of half a pitch, odd or even count.
    centres = np.array([(u, v) for u, v, _ in boxes])
    for along in centres.T:
        idx = np.unique(np.round(along / half))
        assert len(idx) == cells, idx
        assert idx.min() * half == pytest.approx(-outer)
        assert idx.max() * half == pytest.approx(outer)


@pytest.mark.parametrize("variant", ["default", "tuned"])
def test_scattered_boxes_keep_their_size_count_and_cell(variant):
    """Obstacles keep their drawn half-sizes, at most discrete_count to a
    tile. Each rubble box's bounding circle stays inside its own
    grid_pitch cell, so neighbours never fuse. Both types turn through more
    than a quarter turn of yaw."""
    a = _variant(variant)
    p = a.spec.params
    lo, hi = p.discrete_half_range
    centres = _rubble_cells(p)
    yaws = {ttype: [] for ttype in BOX_SCATTER}
    for (ttype, row), boxes in _scattered_boxes(a).items():
        for u, v, b in boxes:
            hx, hy, _ = b.half
            yaws[ttype].append(b.yaw)
            if ttype == "discrete_obstacles":
                assert lo - 1e-12 <= min(hx, hy), (row, b)
                assert max(hx, hy) <= hi + 1e-12, (row, b)
                continue
            iv, iu = _rubble_cell(p, u, v)
            assert 0 <= iu < len(centres) and 0 <= iv < len(centres), (row, b)
            off = math.hypot(u - centres[iu], v - centres[iv])
            assert off + math.hypot(hx, hy) <= p.grid_pitch / 2 + 1e-9, (row, b)
            assert max(hx, hy) <= p.grid_half_range[1] + 1e-12, (row, b)
        if ttype == "discrete_obstacles":
            assert len(boxes) <= p.discrete_count, row
    for ttype, turns in yaws.items():
        assert max(turns) - min(turns) > math.pi / 2, ttype


def test_scatter_density_follows_its_params(arena):
    """Obstacles per tile and the rubble fill track discrete_count and
    grid_fill_prob.

    An obstacle draw drops only when its centre falls within
    pad_radius + pad_clearance + hypot(hi, hi) = 0.80 m of the tile centre.
    The centre is uniform over a square of half-width at least
    reach - hi * sqrt(2) = 1.55 m, so at most 21% of draws drop. The mean
    keeps at least 0.79 of discrete_count, and 0.75 leaves room for the
    draws. A rubble cell whose centre lies at least pad_radius +
    pad_clearance + grid_pitch / 2 from the tile centre never drops its
    box, so the fill over those cells estimates grid_fill_prob."""
    p = arena.spec.params
    hi = p.discrete_half_range[1]
    drop = p.pad_radius + p.pad_clearance + math.hypot(hi, hi)
    half_width = p.tile_size / 2 - p.edge_margin - math.sqrt(2) * hi
    assert math.pi * drop**2 / (2 * half_width) ** 2 < 0.22
    scattered = _scattered_boxes(arena)
    obstacles = [
        len(b) for (tt, _), b in scattered.items() if tt == "discrete_obstacles"
    ]
    assert len(obstacles) == arena.spec.n_rows
    assert np.mean(obstacles) >= 0.75 * p.discrete_count, np.mean(obstacles)

    centres = _rubble_cells(p)
    cu, cv = np.meshgrid(centres, centres)
    far = np.hypot(cu, cv) >= p.pad_radius + p.pad_clearance + p.grid_pitch / 2
    filled = sum(
        bool(far[_rubble_cell(p, u, v)])
        for (tt, _), boxes in scattered.items()
        if tt == "random_grid"
        for u, v, _ in boxes
    )
    fill = filled / (arena.spec.n_rows * int(far.sum()))
    assert fill == pytest.approx(p.grid_fill_prob, abs=0.05)

    bare = terrain.generate(n_rows=2, discrete_count=0)
    assert all(
        _box_tile(bare, b).terrain_type != "discrete_obstacles" for b in bare.boxes
    )


@pytest.mark.parametrize("variant", ["default", "tuned"])
def test_tile_borders_are_flat(variant):
    """Every tile is within 1 cm of z = 0 two centimetres inside its edge,
    so a robot crossing a border meets no wall or cliff. The default
    arena's steepest ramp rises 0.02 m x tan(0.35 rad) = 7 mm over that
    inset."""
    arena = _variant(variant)
    for t in arena.spec.tiles:
        assert _perimeter_max_abs(arena, t) <= 0.01, (t.terrain_type, t.row)


@pytest.mark.parametrize("variant", ["default", "tuned"])
def test_arena_seams_have_no_step(variant):
    """No one-cell step above 2 cm across any tile seam, the border's
    included, within two cells of the seam line. The default arena's
    steepest ramp steps 0.04 m x tan(0.35 rad) = 1.5 cm per cell. The
    overlay noise adds the rest. Boxes and the pit rim are held further in
    than two cells."""
    arena = _variant(variant)
    s = arena.spec
    size = s.params.tile_size
    lookup = arena.lookup.astype(np.float64)
    xs, ys = terrain.grid_axes(s)
    band = 2 * s.params.cell_size + 1e-9
    worst = 0.0
    for j in range(s.n_cols + 1):
        cols = np.nonzero(np.abs(xs - (-s.n_cols * size / 2 + j * size)) <= band)[0]
        block = lookup[:, cols.min() : cols.max() + 1]
        worst = max(worst, float(np.abs(np.diff(block, axis=1)).max()))
    for i in range(s.n_rows + 1):
        rows = np.nonzero(np.abs(ys - (-s.n_rows * size / 2 + i * size)) <= band)[0]
        block = lookup[rows.min() : rows.max() + 1, :]
        worst = max(worst, float(np.abs(np.diff(block, axis=0)).max()))
    assert worst <= 0.02, worst


# -- tile size --------------------------------------------------------------


@pytest.mark.parametrize("tile_size", [3.0, 4.0])
def test_each_tile_size_shapes_its_own_tiles(tile_size):
    """Every feature follows the arena's own tile size: borders flat at the
    tile's edge, slopes from its half-width, the noise lattice from its
    edge, features and boxes inside its ring, and the scatter spread across
    all of it."""
    a = terrain.generate(tile_size=tile_size, n_rows=3, ordered=True)
    s = a.spec
    p = s.params
    half = tile_size / 2
    assert s.x_max == pytest.approx(s.n_cols * half + p.border)
    assert s.hfield.ncol == round(2 * s.x_max / p.cell_size) + 1
    for t in s.tiles:
        assert _perimeter_max_abs(a, t) <= 0.01, (t.terrain_type, t.row)
        if t.terrain_type in SLOPES:
            plateau = math.tan(p.slope_angle(t.difficulty)) * (
                half - p.slope_platform_half
            )
            assert abs(t.pad_height) == pytest.approx(plateau)
            _assert_slope_profile(a, t)
        if t.terrain_type not in terrain.STAIR_TYPES:
            assert t.feature_radius == pytest.approx(half - p.edge_margin)
        cx, cy, _ = t.origin
        assert float(terrain.lookup_height(a, cx, cy)) == pytest.approx(
            t.pad_height, abs=1e-6
        )
        if t.terrain_type == "rough_uniform" and t.row > 0:
            _assert_noise_lattice(a, t)
    spread = dict.fromkeys(BOX_SCATTER, 0.0)
    for b in a.boxes:
        t = _box_tile(a, b)
        reach = _box_reach(b, t)
        assert reach <= t.feature_radius + 1e-9
        if t.terrain_type in spread:
            spread[t.terrain_type] = max(spread[t.terrain_type], reach)
    for ttype, reach in spread.items():
        assert reach > half - p.edge_margin - p.grid_pitch, (ttype, reach)


# -- fingerprint and cost ---------------------------------------------------


@pytest.mark.parametrize("variant", sorted(PINNED_FINGERPRINTS))
def test_the_fingerprint_is_pinned(variant):
    assert terrain.fingerprint(_variant(variant)) == PINNED_FINGERPRINTS[variant]


def test_the_fingerprint_names_version_params_and_arrays(arena):
    base = terrain.fingerprint(arena)
    other = terrain.fingerprint(terrain.generate(seed=1))
    assert other != base
    # Params and keyword overrides name the same arena.
    assert terrain.fingerprint(terrain.generate(terrain.ArenaParams(seed=1))) == other
    assert terrain.fingerprint(terrain.generate(terrain.ArenaParams(), seed=1)) == other
    # One input changed per row, each by more than float32 rounding.
    s = arena.spec
    lookup = arena.lookup.copy()
    lookup[0, 0] += 1e-3
    data = arena.hfield_data.copy()
    data[0, 0] += 1e-3
    first = arena.boxes[0]
    yawed = (replace(first, yaw=first.yaw + 0.1),) + arena.boxes[1:]
    perturbed = {
        "version": replace(
            arena, spec=replace(s, generator_version=s.generator_version + 1)
        ),
        "lookup": replace(arena, lookup=lookup),
        "hfield": replace(arena, hfield_data=data),
        "boxes": replace(arena, boxes=yawed),
        "params": replace(
            arena, spec=replace(s, params=replace(s.params, pad_clearance=0.06))
        ),
    }
    hashes = {name: terrain.fingerprint(a) for name, a in perturbed.items()}
    assert base not in hashes.values(), hashes
    assert len(set(hashes.values())) == len(hashes)


def _cpu_seconds(fn):
    start = time.process_time()
    fn()
    return time.process_time() - start


def test_the_default_arena_builds_in_under_a_second():
    # The budget measures the generator's CPU cost, so MJX jobs running
    # beside the unit loop do not trip it.
    assert min(_cpu_seconds(terrain.generate) for _ in range(3)) < 1.0


# -- sampler ----------------------------------------------------------------


@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_bilinear_interpolates_and_clamps(backend):
    """Hand values on a 3 x 4 grid whose axes differ in origin and spacing:
    node (r, c) sits at (1 + 0.5 c, -3 + 2 r). The points are a cell centre
    and an edge midpoint, each with unequal fractions on the two axes, the
    far corner node, and one point past each of the four edges. A sampler
    that swaps the axes or their spacings misreads them."""
    xp = np if backend == "numpy" else pytest.importorskip("jax").numpy
    grid = np.arange(12.0).reshape(3, 4)
    x = xp.asarray([1.25, 1.75, 2.5, 0.0, 9.0, 1.0, 2.5])
    y = xp.asarray([-2.0, -3.0, 1.0, -1.0, -1.0, -99.0, 99.0])
    got = terrain.bilinear(grid, 1.0, 0.5, -3.0, 2.0, x, y, xp=xp)
    np.testing.assert_allclose(
        np.asarray(got), [2.5, 1.5, 11.0, 4.0, 7.0, 0.0, 11.0], rtol=0, atol=1e-6
    )


def test_bilinear_reads_the_same_on_jax_numpy(arena):
    jax = pytest.importorskip("jax")
    jnp = jax.numpy
    rng = np.random.default_rng(0)
    s = arena.spec
    x = rng.uniform(s.x_min - 1, s.x_max + 1, 500)
    y = rng.uniform(s.y_min - 1, s.y_max + 1, 500)
    host = terrain.lookup_height(arena, x, y)
    frame = terrain.sample_frame(s)
    device = terrain.bilinear(
        jnp.asarray(arena.lookup), *frame, jnp.asarray(x), jnp.asarray(y), xp=jnp
    )
    # jax holds the coordinates at float32, about 2 um at 20 m. Across a
    # 0.75 m step over one 4 cm cell that moves the height by under 0.1 mm.
    np.testing.assert_allclose(np.asarray(device), host, rtol=0, atol=1e-4)
    # Under jit the raw numpy lookup becomes a constant.
    jitted = jax.jit(lambda x, y: terrain.bilinear(arena.lookup, *frame, x, y, xp=jnp))
    got = jitted(jnp.asarray(x), jnp.asarray(y))
    np.testing.assert_allclose(np.asarray(got), host, rtol=0, atol=1e-4)
