"""Terrain geometry for the terrain task: height samplers, lowest points,
spawn height and spawn draws.

Model-free. `tables_from_arena`, `box_tables` and `dilate_max` run on the
host with numpy. Everything else is jax.numpy and traces inside the env.

Sampler. `height` is `bilinear` on the arena's lookup grid, the same body
the generator uses. On bare heightfield it stays within a
quarter of each cell's twist of MuJoCo's triangles: a median of 2
micrometres, and 4.3 mm at worst on the default arena. On an all-zero
patch it returns exactly zero, so `z - H` is `z` bit for bit there. Within
one 4 cm cell of a box edge it blends the box top with the ground beside
it:
- Outside a box it never reads below the physical surface. Beside a face
  on a node line it reads up to the full step high. Beside a face mid-cell
  it reads up to half the step high. On the default arena the tallest step
  is a 15 cm stair riser at d = 1.
- Inside a box, within one cell of an edge that falls between node lines,
  it reads low. At the edge it reads low by the step times the fraction of
  the cell the box covers, half the step for an edge mid-cell. At a corner
  mid-cell in both axes one node of four is on the box, and it reads 3/4 of
  the step low. On the default arena that is 11.25 cm under the 15 cm
  riser at d = 1. No point inside a stair reads lower.
- Yawed boxes (rubble and obstacles) put most of their edges between node
  lines. Their tops read up to 11.7 cm low within one cell of an edge.

Spawn grid. `lookup_spawn` is `dilate_max(lookup, 1)`, a 3 x 3-node max.
It is never below the lookup. On the default arena it reads no stair top
low. Within one cell of a rubble or obstacle edge, 0.02% of top points
still read more than 1 cm low, up to 2.8 cm, where a yawed corner reaches
into a cell past the nodes that see it. It reads high beside every box, up
to the box's step within two cells.

Exact ground. `ground` reads the heightfield alone on MuJoCo's own
triangles, through `triangle`. Where a box contains the point and its top
is higher, it reads the top. Each lookup cell lists the boxes whose
footprint meets it, at most 3 on the default arena. It has no box-edge
band. A box top reads exact up to its edge, and the ground beside a box
reads the ground. Over 300,000 uniform points of the default arena it
matches mj_ray within 0.08 mm, with p99.99 at 0.01 mm. The larger gaps
are float32 rounding of the point's coordinates. They grow with the drop
across the cell, up to 0.12 mm on the pit walls, where a cell drops
0.75 m. On an all-zero patch it reads exactly +0.0. The flat row reads
exactly zero.

Flat band. The flat row's tiles form an axis-aligned rectangle. Its ground
reads exactly zero, and the measurements there are the flat floor's own
expressions. `EMPTY_BAND` contains no point.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import jax
import jax.numpy as jp
import numpy as np

from humanoid_lab.terrain import (
    TYPES,
    Arena,
    bilinear,
    grid_axes,
    sample_frame,
    triangle,
)

# A band no point lies in: an arena without a flat row.
EMPTY_BAND = (math.inf, -math.inf, math.inf, -math.inf)

# Spawn kinds, recorded per episode.
SPAWN_PAD = 0
SPAWN_FEATURE = 1
SPAWN_FALLBACK = 2

# Capsule sample offsets along the axis, in half-lengths.
SPAWN_CAPSULE_POINTS = (-1.0, -0.5, 0.0, 0.5, 1.0)

# Sole samples of a foot capsule: (place on the axis in half-lengths,
# horizontal offset along the axis in radii, horizontal offset across it in
# radii). Five axis points, each also 0.8 r to either side. One point 0.8 r
# beyond each end of the axis, under its end cap. The radius-scaled offsets
# cover the capsule's footprint, so a box edge under its side or end cap
# meets a sample.
SOLE_SAMPLES = (
    *((a, 0.0, 0.0) for a in SPAWN_CAPSULE_POINTS),
    *((a, 0.0, s) for a in SPAWN_CAPSULE_POINTS for s in (-0.8, 0.8)),
    (-1.0, -0.8, 0.0),
    (1.0, 0.8, 0.0),
)
# Depth of the capsule's underside below its axis at each sole sample, in
# radii: sqrt(1 - d^2) at horizontal offset d. Exactly 1 on the axis.
_SOLE_DEPTH = tuple(math.sqrt(1.0 - e * e - s * s) for _, e, s in SOLE_SAMPLES)

# Most boxes one lookup cell may list. `tables_from_arena` refuses an arena
# that crowds more into a cell.
MAX_CELL_BOXES = 8
# Slack (m) of the cell-box overlap test. A float32 point near a node line
# may resolve to the neighbouring cell. That cell then still lists the box.
_CELL_SLACK = 1e-4


class TerrainTables(NamedTuple):
    """The arena as the env reads it. Grids and per-tile tables are device
    arrays indexed by (arena row, `TYPES` index); each terrain row shuffles
    its columns, so the type, not the column, names a tile."""

    lookup: jax.Array  # (nrow, ncol) f32 surface height, box tops written in
    lookup_spawn: jax.Array  # dilate_max(lookup, 1)
    heights: jax.Array  # (nrow, ncol) f32 heightfield alone, no box tops
    # (nrow - 1, ncol - 1, K) the boxes whose footprint meets each cell,
    # padded with the sentinel row's index
    cell_boxes: jax.Array
    # (B + 1, 7) f32 per box: centre x, y, half x, y, cos and sin of its
    # yaw, top z. The last row is a sentinel that contains no point.
    boxes: jax.Array
    frame: tuple[float, float, float, float]  # (x0, dx, y0, dy)
    origin_xy: jax.Array  # (R, T, 2) tile centres
    pad_h: jax.Array  # (R, T) spawn pad heights
    feature_r: jax.Array  # (R, T) Chebyshev radius bounding each tile's features
    flat_band: tuple[float, float, float, float]  # (x_lo, x_hi, y_lo, y_hi)
    n_rows: int  # curriculum levels, the flat row included
    flat_row: bool
    tile_size: float
    pad_radius: float
    max_step: float  # largest height step between neighbouring lookup nodes


def dilate_max(grid: np.ndarray, k: int) -> np.ndarray:
    """(2k + 1)^2-node max filter of `grid`, edge-padded. Host only."""
    g = np.asarray(grid)
    if k < 0:
        raise ValueError(f"k must not be negative, got {k}")
    nr, nc = g.shape
    padded = np.pad(g, k, mode="edge")
    out = g.copy()
    for dr in range(2 * k + 1):
        for dc in range(2 * k + 1):
            np.maximum(out, padded[dr : dr + nr, dc : dc + nc], out=out)
    return out


def _box_rows(boxes) -> np.ndarray:
    """(B + 1, 7) float32 rows of `boxes` (see `TerrainTables.boxes`), then
    the sentinel."""
    rows = np.zeros((len(boxes) + 1, 7), np.float32)
    for i, b in enumerate(boxes):
        rows[i] = (
            b.pos[0],
            b.pos[1],
            b.half[0],
            b.half[1],
            math.cos(b.yaw),
            math.sin(b.yaw),
            b.pos[2] + b.half[2],
        )
    rows[-1] = (0.0, 0.0, -1.0, -1.0, 1.0, 0.0, -math.inf)
    return rows


def box_tables(boxes, frame, shape, cap: int = MAX_CELL_BOXES) -> tuple[np.ndarray, np.ndarray]:
    """(cell_boxes, rows) for `boxes` (each with pos, half and yaw, like
    terrain.Box) on a node grid of `shape` (nrow, ncol) at `frame`. Host
    only.

    A cell lists every box whose yawed footprint meets the closed cell, by
    a separating-axis test with _CELL_SLACK of slack. K is the most boxes
    any cell lists, at least 1. Raises when K exceeds `cap`. The message
    names the world centre of the most crowded cell."""
    x0, dx, y0, dy = frame
    nr, nc = shape[0] - 1, shape[1] - 1
    rows = _box_rows(boxes)
    sentinel = len(boxes)
    count = np.zeros((nr, nc), np.int64)
    hits = []
    for i, (cx, cy, hx, hy, c, s, _) in enumerate(rows[:-1].astype(float)):
        ax, ay = hx * abs(c) + hy * abs(s), hx * abs(s) + hy * abs(c)
        j0 = max(math.floor((cx - ax - _CELL_SLACK - x0) / dx), 0)
        j1 = min(math.floor((cx + ax + _CELL_SLACK - x0) / dx), nc - 1)
        i0 = max(math.floor((cy - ay - _CELL_SLACK - y0) / dy), 0)
        i1 = min(math.floor((cy + ay + _CELL_SLACK - y0) / dy), nr - 1)
        if j1 < j0 or i1 < i0:
            continue
        rx = x0 + (np.arange(j0, j1 + 1) + 0.5) * dx - cx
        ry = (y0 + (np.arange(i0, i1 + 1) + 0.5) * dy - cy)[:, None]
        hdx, hdy = dx / 2, dy / 2
        meets = (
            (np.abs(rx) <= ax + hdx + _CELL_SLACK)
            & (np.abs(ry) <= ay + hdy + _CELL_SLACK)
            & (np.abs(rx * c + ry * s) <= hx + hdx * abs(c) + hdy * abs(s) + _CELL_SLACK)
            & (np.abs(ry * c - rx * s) <= hy + hdx * abs(s) + hdy * abs(c) + _CELL_SLACK)
        )
        window = (slice(i0, i1 + 1), slice(j0, j1 + 1))
        count[window] += meets
        hits.append((i, window, meets))
    k = max(int(count.max()) if count.size else 0, 1)
    if k > cap:
        i, j = np.unravel_index(count.argmax(), count.shape)
        raise ValueError(
            f"the lookup cell centred at ({x0 + (j + 0.5) * dx:.2f}, {y0 + (i + 0.5) * dy:.2f}) "
            f"meets {k} boxes, over the limit of {cap}. The exact foot ground reads every "
            "listed box per sole sample."
        )
    dtype = np.int16 if sentinel < np.iinfo(np.int16).max else np.int32
    cell = np.full((nr, nc, k), sentinel, dtype)
    fill = np.zeros((nr, nc), np.int64)
    for i, window, meets in hits:
        rr, cc = np.nonzero(meets)
        sub, f = cell[window], fill[window]
        sub[rr, cc, f[rr, cc]] = i
        f[rr, cc] += 1
    return cell, rows


def _heightfield_grid(arena: Arena, lookup: np.ndarray) -> np.ndarray:
    """The lookup with every node under a box footprint, or within
    _CELL_SLACK of one, set back to the heightfield there. Every other node
    keeps the lookup's value bit for bit."""
    hf = arena.spec.hfield
    xs, ys = grid_axes(arena.spec)
    out = lookup.copy()
    for b in arena.boxes:
        hx, hy = b.half[0], b.half[1]
        c, s = math.cos(b.yaw), math.sin(b.yaw)
        ax, ay = hx * abs(c) + hy * abs(s), hx * abs(s) + hy * abs(c)
        c0 = int(np.searchsorted(xs, b.pos[0] - ax - _CELL_SLACK))
        c1 = int(np.searchsorted(xs, b.pos[0] + ax + _CELL_SLACK, side="right"))
        r0 = int(np.searchsorted(ys, b.pos[1] - ay - _CELL_SLACK))
        r1 = int(np.searchsorted(ys, b.pos[1] + ay + _CELL_SLACK, side="right"))
        gx = xs[c0:c1][None, :] - b.pos[0]
        gy = ys[r0:r1][:, None] - b.pos[1]
        u, v = np.abs(gx * c + gy * s), np.abs(gy * c - gx * s)
        under = (u <= hx + _CELL_SLACK) & (v <= hy + _CELL_SLACK)
        field = hf.pos_z + hf.elevation_z * arena.hfield_data[r0:r1, c0:c1].astype(float)
        patch = out[r0:r1, c0:c1]
        patch[under] = field[under].astype(np.float32)
    return out


def tables_from_arena(arena: Arena) -> TerrainTables:
    """The env's tables for `arena`. Raises when a lookup cell meets more
    than MAX_CELL_BOXES boxes. A cell meets at most 3 stair treads or 4
    rubble boxes, so only discrete obstacles reach the limit. The message
    names the arena keys that set their density."""
    spec = arena.spec
    n_types = len(TYPES)
    index = {t: i for i, t in enumerate(TYPES)}
    origin_xy = np.zeros((spec.n_rows, n_types, 2), np.float32)
    pad_h = np.zeros((spec.n_rows, n_types), np.float32)
    feature_r = np.zeros((spec.n_rows, n_types), np.float32)
    for t in spec.tiles:
        j = index[t.terrain_type]
        origin_xy[t.row, j] = t.origin[:2]
        pad_h[t.row, j] = t.pad_height
        feature_r[t.row, j] = t.feature_radius

    p = spec.params
    if p.flat_row:
        half = p.tile_size / 2
        flat = [t for t in spec.tiles if t.row == 0]
        xs = [t.origin[0] for t in flat]
        ys = [t.origin[1] for t in flat]
        band = (min(xs) - half, max(xs) + half, min(ys) - half, max(ys) + half)
    else:
        band = EMPTY_BAND

    lookup = np.asarray(arena.lookup, np.float32)
    max_step = max(
        float(np.abs(np.diff(lookup, axis=0)).max()),
        float(np.abs(np.diff(lookup, axis=1)).max()),
    )
    frame = tuple(float(v) for v in sample_frame(spec))
    try:
        cell_boxes, box_rows = box_tables(arena.boxes, frame, lookup.shape)
    except ValueError as e:
        raise ValueError(
            f"{e} Only discrete obstacles crowd a cell this far. Lower "
            "terrain.arena.discrete_count or discrete_half_range, or pick another "
            "terrain.arena.seed."
        ) from e
    return TerrainTables(
        lookup=jp.asarray(lookup),
        lookup_spawn=jp.asarray(dilate_max(lookup, 1)),
        heights=jp.asarray(_heightfield_grid(arena, lookup)),
        cell_boxes=jp.asarray(cell_boxes),
        boxes=jp.asarray(box_rows),
        frame=frame,
        origin_xy=jp.asarray(origin_xy),
        pad_h=jp.asarray(pad_h),
        feature_r=jp.asarray(feature_r),
        flat_band=tuple(float(v) for v in band),
        n_rows=int(spec.n_rows),
        flat_row=bool(p.flat_row),
        tile_size=float(p.tile_size),
        pad_radius=float(p.pad_radius),
        max_step=max_step,
    )


# -- samplers ------------------------------------------------------------------


def height(t: TerrainTables, xy, grid: str = "lookup"):
    """Ground height under world `xy` (..., 2) -> (...). `grid` is
    "lookup" (the plain lookup) or "spawn" (the dilated spawn grid). The
    module docstring states where each reads low or high near a box edge."""
    if grid == "lookup":
        g = t.lookup
    elif grid == "spawn":
        g = t.lookup_spawn
    else:
        raise ValueError(f"grid must be 'lookup' or 'spawn', got {grid!r}")
    x0, dx, y0, dy = t.frame
    return bilinear(g, x0, dx, y0, dy, xy[..., 0], xy[..., 1], xp=jp)


def ground(t: TerrainTables, xy):
    """Exact ground under world `xy` (..., 2) -> (...): the heightfield on
    MuJoCo's triangles, or the top of a box that contains the point,
    whichever is higher. On an all-zero patch it reads exactly +0.0."""
    x0, dx, y0, dy = t.frame
    x, y = xy[..., 0], xy[..., 1]
    field = triangle(t.heights, x0, dx, y0, dy, x, y, xp=jp)
    nr, nc = t.cell_boxes.shape[:2]
    col = jp.clip(jp.floor((x - x0) / dx), 0, nc - 1).astype(jp.int32)
    row = jp.clip(jp.floor((y - y0) / dy), 0, nr - 1).astype(jp.int32)
    b = t.boxes[t.cell_boxes[row, col]]  # (..., K, 7)
    ox = x[..., None] - b[..., 0]
    oy = y[..., None] - b[..., 1]
    inside = (jp.abs(ox * b[..., 4] + oy * b[..., 5]) <= b[..., 2]) & (
        jp.abs(oy * b[..., 4] - ox * b[..., 5]) <= b[..., 3]
    )
    top = jp.max(jp.where(inside, b[..., 6], -jp.inf), axis=-1)
    return jp.maximum(field, top)


def in_band(t: TerrainTables, xy):
    """Whether `xy` (..., 2) lies in the flat row's rectangle, edges
    included."""
    x_lo, x_hi, y_lo, y_hi = t.flat_band
    x, y = xy[..., 0], xy[..., 1]
    return (x >= x_lo) & (x <= x_hi) & (y >= y_lo) & (y <= y_hi)


def chebyshev(xy, origin):
    """Chebyshev distance from `origin`, over the last axis."""
    return jp.max(jp.abs(xy - origin), axis=-1)


# -- rotations -----------------------------------------------------------------


def yaw_quat(yaw):
    """(w, x, y, z) quaternion of a rotation by `yaw` about +z."""
    half = 0.5 * yaw
    zero = jp.zeros_like(half)
    return jp.stack([jp.cos(half), zero, zero, jp.sin(half)], axis=-1)


def quat_mul(a, b):
    """Hamilton product a * b of (w, x, y, z) quaternions: b first, then a."""
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return jp.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=-1,
    )


def rotate_xy(u, yaw):
    """Horizontal offsets `u` (..., 2) turned by `yaw` about +z."""
    c, s = jp.cos(yaw), jp.sin(yaw)
    return jp.stack([c * u[..., 0] - s * u[..., 1], s * u[..., 0] + c * u[..., 1]], axis=-1)


# -- collider points -------------------------------------------------------------


def capsule_points(centre, axis, half, ts):
    """Points `centre + t * half * axis` for each t in `ts`.

    centre and axis (..., 3), half (...) -> (..., len(ts), 3). A sphere is
    a capsule of half-length 0."""
    ts = jp.asarray(ts, dtype=centre.dtype)
    return centre[..., None, :] + (ts[:, None] * jp.asarray(half)[..., None, None]) * axis[..., None, :]


def sole_gaps(t: TerrainTables, centre, axis, half, r):
    """Height of a foot capsule's underside above the exact ground at each
    of its SOLE_SAMPLES. centre and axis (..., 3), half and r (...) ->
    (..., S).

    A sample offset by d horizontally from an axis point reads the
    underside of that point's sphere, sqrt(r^2 - d^2) below it. That point
    lies on the capsule, so no sample reads the capsule lower than it is.
    For a level capsule it is the capsule's own underside. On the axis the
    gap is the axis point's height above the ground less r. A capsule
    standing upright, or a sphere, takes +x as its along-axis
    direction."""
    samples = np.asarray(SOLE_SAMPLES)
    pts = capsule_points(centre, axis, half, samples[:, 0])
    flat = axis[..., :2]
    norm = jp.linalg.norm(flat, axis=-1, keepdims=True)
    fwd = jp.where(norm > 1e-6, flat / jp.maximum(norm, 1e-6), jp.array([1.0, 0.0], flat.dtype))
    side = jp.stack([-fwd[..., 1], fwd[..., 0]], axis=-1)
    along = jp.asarray(samples[:, 1], flat.dtype)[:, None]
    across = jp.asarray(samples[:, 2], flat.dtype)[:, None]
    r = jp.asarray(r)
    off = (along * fwd[..., None, :] + across * side[..., None, :]) * r[..., None, None]
    depth = jp.asarray(_SOLE_DEPTH, pts.dtype) * r[..., None]
    return pts[..., 2] - depth - ground(t, pts[..., :2] + off)


def lowest_point_box(c, R, half):
    """(xy, z) of a box's lowest point. c (..., 3), R (..., 3, 3) its
    rotation, half (..., 3).

    The lowest corner is c - R @ (sign(R[2, :]) * half). Its height is
    c_z - sum_j |R[2, j]| * half_j. An axis lying level contributes no
    offset, which picks the middle of that edge or face."""
    s = jp.sign(R[..., 2, :])
    p = c - jp.einsum("...ij,...j->...i", R, s * half)
    z = c[..., 2] - jp.sum(jp.abs(R[..., 2, :]) * half, axis=-1)
    return p[..., :2], z


def lowest_point_capsule(c, R, half, r):
    """(xy, z) of a capsule's lowest point: the lower end-cap centre, less
    the radius. c (..., 3), R (..., 3, 3), half and r (...)."""
    axis = R[..., :, 2]
    s = jp.sign(R[..., 2, 2])
    p = c - (s * half)[..., None] * axis
    return p[..., :2], p[..., 2] - r


def lowest_point_sphere(c, r):
    """(xy, z) of a sphere's lowest point."""
    return c[..., :2], c[..., 2] - r


# -- spawn height ------------------------------------------------------------------


def spawn_z(t: TerrainTables, u, lift, z0, xy, yaw, grid: str):
    """Base height that sets every sample point's bottom on or above the
    ground: z0 + max_k (G(xy + R(yaw) u_k) - lift_k).

    `u` (P, 2) are the points' horizontal offsets from the base and `lift`
    (P,) how far each bottom sits above the lowest one, both at the reset
    pose. On ground of constant height c the lowest point binds, and the
    result is exactly z0 + c."""
    g = height(t, xy + rotate_xy(u, yaw), grid)
    return z0 + jp.max(g - lift)


def support_spread(t: TerrainTables, u_sole, xy, yaw):
    """The larger of two spreads (max minus min) under the sole points
    `u_sole`: on the dilated spawn grid and on the lookup.

    The spawn grid sets a feature spawn's height. Beside a box it reads the
    top for up to two cells, so a foot there reads level while it hangs
    over the lower ground. The lookup reads that drop."""
    p = xy + rotate_xy(u_sole, yaw)
    spawn = height(t, p, "spawn")
    plain = height(t, p)
    return jp.maximum(jp.max(spawn) - jp.min(spawn), jp.max(plain) - jp.min(plain))


# -- draws ---------------------------------------------------------------------


def first_level(rng, n_rows: int, init_level_frac: float, spawn_level: int):
    """An env's first level: U[0, max(1, round(n_rows * init_level_frac))),
    or `spawn_level` when it is 0 or more. The draw runs either way."""
    hi = max(1, round(n_rows * init_level_frac))
    drawn = jax.random.randint(rng, (), 0, hi)
    if spawn_level >= 0:
        return jp.full((), spawn_level, dtype=drawn.dtype)
    return drawn


def pad_draw(rng, origin, jitter: float):
    """`origin` plus U(-jitter, jitter) on each axis: a square."""
    return origin + jax.random.uniform(rng, (2,), minval=-jitter, maxval=jitter)


def feature_draw(rng, t: TerrainTables, ttype, level, yaw, half_extent: float, k: int,
                 max_spread: float, u_sole):
    """(xy, ok): the first of `k` candidates within `half_extent` of the
    tile centre, per axis, whose sole spread is at most `max_spread`. Every
    candidate shares `yaw`. ok is False when none qualifies, and xy is then
    the first candidate."""
    origin = t.origin_xy[level, ttype]
    cands = origin + jax.random.uniform(rng, (k, 2), minval=-half_extent, maxval=half_extent)
    spread = jax.vmap(lambda p: support_spread(t, u_sole, p, yaw))(cands)
    good = spread <= max_spread
    return cands[jp.argmax(good)], jp.any(good)


def spawn_draw(rng, t: TerrainTables, ttype, level, *, feature: bool, yaw_enable: bool,
               pad_jitter: float, k: int, half_extent: float, max_spread: float, u_sole):
    """(xy, yaw, kind) of one spawn on tile (level, ttype).

    The yaw is drawn first, then the pad point, then the feature
    candidates. The key splits the same way in every mode, so the pad
    point and the yaw a key gives never depend on the mode, the level or
    the outcome. The flat row always takes the pad. A feature draw with no
    level-footed candidate falls back to the pad (kind SPAWN_FALLBACK)."""
    r_yaw, r_pad, r_feat = jax.random.split(rng, 3)
    yaw = jax.random.uniform(r_yaw, minval=-jp.pi, maxval=jp.pi) if yaw_enable else jp.zeros(())
    pad = pad_draw(r_pad, t.origin_xy[level, ttype], pad_jitter)
    if not feature:
        return pad, yaw, jp.full((), SPAWN_PAD, jp.int32)
    xy, ok = feature_draw(r_feat, t, ttype, level, yaw, half_extent, k, max_spread, u_sole)
    on_flat = (level == 0) if t.flat_row else jp.zeros((), bool)
    kind = jp.where(on_flat, SPAWN_PAD, jp.where(ok, SPAWN_FEATURE, SPAWN_FALLBACK))
    xy = jp.where(kind == SPAWN_FEATURE, xy, pad)
    return xy, yaw, kind.astype(jp.int32)
