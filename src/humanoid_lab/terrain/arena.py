"""Procedural terrain arena: tiles of eight terrain types on one heightfield.

An arena is a grid of square tiles inside a flat border. Each terrain type
has one column. Each difficulty level has one row, and difficulty rises
from 0 to 1 up the rows. `ArenaParams` sets every size and the layout:
shuffled or ordered columns, jittered or explicit row difficulties, and an
optional flat row 0. Its docstring has the details.

Rough ground, slopes, waves and the inverted-stairs pit are carved into one
heightfield that covers the whole arena. Stairs, obstacles and rubble are
static boxes, which give crisp edges and cheap contacts. A low overlay of
rough noise rides the slope and box-tile grounds, so those are not
perfectly smooth. Every tile is flat (z = 0) on its perimeter, so
neighbours join without a step. Every tile has a flat circular spawn pad
at its centre. Every tile is built on the same tile-local node grid. Each
tile draws from its own rng stream, so no tile's content depends on what
another tile drew.

The heightfield and the boxes belong in the world body. MuJoCo skips
contacts between geoms of one body, so the arena adds no terrain-terrain
contacts.

`Arena.lookup` is the ground-truth surface height on the heightfield's own
node grid. Each box's top is written onto the nodes inside its footprint,
edges included. `ArenaParams` refuses box sizes that could cover no node:
obstacle and rubble half-sizes under cell_size / sqrt(2), and stair treads
under two cell_size, so every ring band spans a node line. Every box
therefore shows in it. Height queries read it through `bilinear` instead of
casting rays, so it has to match the physics geometry node for node.
Between nodes it can only blend.

Where a box edge lies on a node line, the lookup reaches the top at that
edge node and ramps up over the cell outside it. It reads the box up to
one cell early and never late. On the default arena the summit and pad
rim lie on node lines, so the pit's flat pad reads flat out to
pad_radius - cell_size. Where an edge falls between node lines, the ramp
straddles it. Within one cell of the edge the lookup reads the higher side
low and the lower side high, by up to the full step. On the default arena
the 0.7 m and 1.3 m tread edges fall mid-cell, and the lookup reads a
d = 1 tread lip up to half its 15 cm riser low. Nearly every yawed
obstacle and rubble edge falls between node lines too. A consumer that
must never read the ground below a box, such as fall termination or foot
clearance, can take the max of the four nodes around a point. That reads
no axis-aligned box a cell or more wide low. A yawed box's corner can
still reach into a cell past all four of its nodes.

MuJoCo splits each heightfield cell into two triangles, where `bilinear`
blends four nodes. At the cell centre the two differ by a quarter of the
cell's twist |h00 - h01 - h10 + h11|, whichever diagonal MuJoCo splits on,
and nowhere in the cell by more. On the default arena's bare heightfield
the gap peaks at 4.3 mm, on the diagonal creases of the d = 1 slope tiles.
The crease alone gives tan(slope_angle(1)) * cell_size / 4 = 3.7 mm, and
the overlay adds the rest. A pit's corner cell, one carved node and three
at 0, twists by the whole pit depth. At two of a pit's four corners
MuJoCo's far triangle there stays at ground level, above the outermost
ring, and the lookup reads up to a quarter riser below it: 37.5 mm at
d = 1. mj_ray reproduces these figures at those cells.

MuJoCo rescales heightfield data to [0, 1] on compile and puts the surface
at `pos_z + data * elevation_z`. The arena stores data already normalized
over its own range. `pos_z` is the minimum height and `elevation_z` the
range, so the compiled surface reproduces the heights.

Everything here is in memory and numpy only.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace

import numpy as np

from humanoid_lab.terrain.params import (
    STAIR_TYPES,
    TYPES,
    ArenaParams,
    node_count,
    params_from_dict,
    params_to_dict,
    rubble_cells,
    rubble_shrink,
    slope_plateau_height,
    stair_flight_half,
    stair_steps,
    stair_tread_bounds,
    summit_platform_half,
)
from humanoid_lab.terrain.sampling import bilinear

# Bump whenever a change to this package alters the arena some ArenaParams
# builds. The version and the params name an arena. `fingerprint` also
# covers the arrays they produce. Pinned fingerprints of the default and
# tread-range arenas catch a change that forgot to bump. The tests pin the
# other layout options by equivalence.
GENERATOR_VERSION = 1

# Thickness of the heightfield's solid below its geom frame. The frame sits
# at the arena's minimum height (pos_z), so any positive thickness clears
# the deepest pit. MuJoCo requires it positive.
HFIELD_BASE_Z = 1.0

# Slack so a node exactly on a box edge or the pit rim counts as inside,
# whatever the last bit of its coordinate.
_EPS = 1e-9

# Every rng key starts with the seed and one of these tags. A row key then
# never equals a tile key, even after SeedSequence pads keys with zeros.
_ROW, _TILE = 0, 1


@dataclass(frozen=True)
class Box:
    """Static box: centre, half-sizes (m) and yaw about +z (rad)."""

    pos: tuple[float, float, float]
    half: tuple[float, float, float]
    yaw: float = 0.0


@dataclass(frozen=True)
class TileSpec:
    """One tile: its place, type, realized difficulty, spawn pad and stair counts."""

    row: int
    col: int
    terrain_type: str
    difficulty: float  # realized: the row's difficulty times the type's cap
    origin: tuple[float, float, float]  # tile-centre ground point (z = 0)
    pad_radius: float
    pad_height: float  # world z of the flat spawn pad at the tile centre
    # Chebyshev radius that bounds the tile's boxes and stair flight. It is 0
    # on the flat row, which has no feature. Past it a stair tile is flat
    # ground. Slope, rough and wave relief and the box tiles' noise overlay
    # run on to the perimeter, where they reach 0.
    feature_radius: float = 0.0
    # Stair tiles only, 0 elsewhere: this tile's tread and riser count.
    stair_tread: float = 0.0
    n_steps: int = 0


@dataclass(frozen=True)
class HFieldSpec:
    """The heightfield's node counts, MuJoCo size and the geom's z position."""

    nrow: int  # rows index y
    ncol: int  # columns index x
    radius_x: float
    radius_y: float
    elevation_z: float
    base_z: float
    pos_z: float


@dataclass(frozen=True)
class TerrainSpec:
    """An arena's layout and sizes without its boxes or arrays.

    `spec_to_dict` serializes it."""

    generator_version: int
    params: ArenaParams
    n_rows: int  # arena rows, the flat row included when there is one
    n_cols: int
    types: tuple[str, ...]
    stair_platform_half: float
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    hfield: HFieldSpec
    tiles: tuple[TileSpec, ...]


@dataclass(frozen=True, eq=False)
class Arena:
    """A generated arena: its spec, static boxes, normalized heightfield data
    and lookup grid."""

    spec: TerrainSpec
    boxes: tuple[Box, ...]
    hfield_data: np.ndarray  # (nrow, ncol) float32 in [0, 1], rows index y
    lookup: np.ndarray  # (nrow, ncol) float32 surface height (m), box tops written in


@dataclass(frozen=True, eq=False)
class _TileGrid:
    """A tile's nodes in tile-local coordinates, perimeter included."""

    axis: np.ndarray  # (n,) node offsets from the tile centre, either axis
    lx: np.ndarray  # (n, n), rows index y
    ly: np.ndarray
    cheby: np.ndarray  # Chebyshev distance from the tile centre
    radius: np.ndarray  # Euclidean distance from the tile centre


@dataclass(frozen=True, eq=False)
class _Tile:
    heights: np.ndarray  # heightfield patch on the tile grid
    boxes: list[Box]  # tile-local
    pad_height: float
    feature_radius: float
    stair_tread: float = 0.0
    n_steps: int = 0


def _tile_grid(p: ArenaParams) -> _TileGrid:
    cells = node_count(p.tile_size, p.cell_size)
    # Integer offsets scaled last, so the perimeter nodes sit at exactly
    # +-tile_size / 2 and every tapered feature reaches exactly 0 there.
    axis = (np.arange(cells + 1) - cells / 2) * p.tile_size / cells
    lx, ly = np.meshgrid(axis, axis)
    return _TileGrid(
        axis=axis,
        lx=lx,
        ly=ly,
        cheby=np.maximum(np.abs(lx), np.abs(ly)),
        radius=np.sqrt(lx * lx + ly * ly),
    )


def _pad_taper(g: _TileGrid, p: ArenaParams) -> np.ndarray:
    """0 on the spawn pad, rising to 1 over pad_taper past its rim."""
    return np.clip((g.radius - p.pad_radius) / p.pad_taper, 0.0, 1.0)


def _edge_taper(g: _TileGrid, p: ArenaParams) -> np.ndarray:
    """1 inside the tile, falling to exactly 0 on its perimeter."""
    return np.clip((p.tile_size / 2 - g.cheby) / p.edge_taper, 0.0, 1.0)


def _noise(g: _TileGrid, amplitude: float, rng, p: ArenaParams) -> np.ndarray:
    """Uniform noise on the coarse_step lattice, bilinearly upsampled. It is
    flat on the pad and 0 on the tile perimeter."""
    n = max(round(p.tile_size / p.coarse_step), 1) + 1
    half = p.tile_size / 2
    step = p.tile_size / (n - 1)
    coarse = rng.uniform(-1.0, 1.0, size=(n, n))
    noise = bilinear(coarse, -half, step, -half, step, g.lx, g.ly)
    return amplitude * noise * _pad_taper(g, p) * _edge_taper(g, p)


def _slope(g: _TileGrid, d: float, sign: float, p: ArenaParams) -> np.ndarray:
    """A frustum: the flat plateau, then a straight ramp to 0 at the tile
    edge. sign +1 is a hill, -1 a pit."""
    run = p.tile_size / 2 - np.maximum(g.cheby, p.slope_platform_half)
    return sign * math.tan(p.slope_angle(d)) * run


def _wave(g: _TileGrid, d: float, rng, p: ArenaParams) -> np.ndarray:
    """Product of sines with whole half-periods across the tile, so the
    surface is 0 on all four edges. The pad is flattened like rough."""
    kx = int(rng.choice(p.wave_half_periods))
    ky = int(rng.choice(p.wave_half_periods))
    half = p.tile_size / 2
    wave = np.sin(kx * np.pi * (g.lx + half) / p.tile_size) * np.sin(
        ky * np.pi * (g.ly + half) / p.tile_size
    )
    return p.wave_amplitude(d) * wave * _pad_taper(g, p)


def _ring(r_in: float, r_out: float, top: float, base: float) -> list[Box]:
    """Four boxes tiling the square annulus between Chebyshev radii r_in and
    r_out, from z = base to z = top."""
    mid = (r_in + r_out) / 2
    band = (r_out - r_in) / 2
    cz = (base + top) / 2
    hz = (top - base) / 2
    return [
        Box((0.0, mid, cz), (r_out, band, hz)),
        Box((0.0, -mid, cz), (r_out, band, hz)),
        Box((-mid, 0.0, cz), (band, r_in, hz)),
        Box((mid, 0.0, cz), (band, r_in, hz)),
    ]


def _pyramid_stair_boxes(
    platform: float, tread: float, n_steps: int, riser: float
) -> list[Box]:
    """The summit block n_steps risers high, then rings one riser lower each,
    the last a riser above the flat tile ground."""
    top = n_steps * riser
    boxes = [Box((0.0, 0.0, top / 2), (platform, platform, top / 2))]
    for k in range(1, n_steps):
        r_in = platform + (k - 1) * tread
        boxes += _ring(r_in, r_in + tread, (n_steps - k) * riser, 0.0)
    return boxes


def _pit_stair_boxes(
    platform: float, tread: float, n_steps: int, riser: float
) -> list[Box]:
    """Rings on the floor of a pit n_steps risers deep, each one riser higher
    outward. The summit is the bare pit floor. The last riser, from -riser
    up to the ground, is the pit wall in the heightfield (see _pit)."""
    depth = n_steps * riser
    boxes = []
    for k in range(1, n_steps):
        r_in = platform + (k - 1) * tread
        boxes += _ring(r_in, r_in + tread, -depth + k * riser, -depth)
    return boxes


def _aabb_half(hx: float, hy: float, yaw: float) -> tuple[float, float]:
    c, s = abs(math.cos(yaw)), abs(math.sin(yaw))
    return hx * c + hy * s, hx * s + hy * c


def _clears_pad(u: float, v: float, hx: float, hy: float, p: ArenaParams) -> bool:
    """The box's corner radius stays pad_clearance outside the spawn pad."""
    corner = math.sqrt(hx * hx + hy * hy)
    return math.sqrt(u * u + v * v) >= p.pad_radius + corner + p.pad_clearance


def _discrete_boxes(d: float, rng, p: ArenaParams) -> list[Box]:
    """Up to discrete_count yawed boxes scattered over the tile. They stay
    off the pad. Their rotated extent stays edge_margin inside the tile."""
    max_h = p.obstacle_height(d)
    reach = p.tile_size / 2 - p.edge_margin
    boxes = []
    for _ in range(p.discrete_count):
        hx = float(rng.uniform(*p.discrete_half_range))
        hy = float(rng.uniform(*p.discrete_half_range))
        yaw = float(rng.uniform(0.0, math.pi))
        ax, ay = _aabb_half(hx, hy, yaw)
        u = float(rng.uniform(-reach + ax, reach - ax))
        v = float(rng.uniform(-reach + ay, reach - ay))
        if not _clears_pad(u, v, hx, hy, p):
            continue
        h = float(rng.uniform(*p.discrete_height_fraction)) * max_h
        boxes.append(Box((u, v, h / 2), (hx, hy, h / 2), yaw))
    return boxes


def _rubble_boxes(d: float, rng, p: ArenaParams) -> list[Box]:
    """A grid_pitch cell grid, each cell raising one small yawed box with
    probability grid_fill_prob. Every box is a bump, never a hole. Each
    box, at any yaw, stays inside its own cell, so neighbours never fuse
    into slabs."""
    max_h = p.obstacle_height(d)
    n = rubble_cells(p)
    start = -(n - 1) * p.grid_pitch / 2
    half_cell = p.grid_pitch / 2
    boxes = []
    for iy in range(n):
        for ix in range(n):
            if rng.uniform() > p.grid_fill_prob:
                continue
            hx = float(rng.uniform(*p.grid_half_range))
            hy = float(rng.uniform(*p.grid_half_range))
            yaw = float(rng.uniform(0.0, math.pi))
            # The bounding circle plus the jitter stays inside the cell:
            # shrink an oversized box, then jitter within the slack.
            shrink = rubble_shrink(hx, hy, p)
            hx, hy = hx * shrink, hy * shrink
            jr = (half_cell - math.hypot(hx, hy)) * math.sqrt(rng.uniform())
            jth = rng.uniform(0.0, 2 * math.pi)
            u = start + ix * p.grid_pitch + jr * math.cos(jth)
            v = start + iy * p.grid_pitch + jr * math.sin(jth)
            if not _clears_pad(u, v, hx, hy, p):
                continue
            h = float(rng.uniform(*p.grid_height_fraction)) * max_h
            boxes.append(Box((u, v, h / 2), (hx, hy, h / 2), yaw))
    return boxes


_BOX_SCATTER = {"discrete_obstacles": _discrete_boxes, "random_grid": _rubble_boxes}


def _footprint(axis: np.ndarray, box: Box):
    """The grid window under `box`'s bounding square and the mask of nodes
    in it that lie inside its yawed footprint, edges included."""
    u, v, _ = box.pos
    hx, hy, _ = box.half
    ax, ay = _aabb_half(hx, hy, box.yaw)
    c0 = int(np.searchsorted(axis, u - ax - _EPS))
    c1 = int(np.searchsorted(axis, u + ax + _EPS, side="right"))
    r0 = int(np.searchsorted(axis, v - ay - _EPS))
    r1 = int(np.searchsorted(axis, v + ay + _EPS, side="right"))
    gx = axis[c0:c1][None, :] - u
    gy = axis[r0:r1][:, None] - v
    c, s = math.cos(box.yaw), math.sin(box.yaw)
    inside = (np.abs(gx * c + gy * s) <= hx + _EPS) & (
        np.abs(gy * c - gx * s) <= hy + _EPS
    )
    return (slice(r0, r1), slice(c0, c1)), inside


def _seat(boxes: list[Box], heights: np.ndarray, axis: np.ndarray) -> list[Box]:
    """Seat each drawn box on rolling ground.

    The top rests at the drawn height (2 * half z) above the highest node
    under the footprint. The box reaches down to the lowest node one index
    past its bounding square on every side. That window holds all four
    nodes of every heightfield cell the footprint touches. MuJoCo's
    triangles and `bilinear` both stay within a cell's node range, so the
    base is at or below either surface everywhere under the box, and no gap
    opens under its low side. The overlay's floor would also do, but it
    sinks the box deeper than the ground around it dips."""
    seated = []
    for b in boxes:
        (rows, cols), inside = _footprint(axis, b)
        # ArenaParams admits no box that could fall between nodes.
        assert inside.any(), b
        top = float(heights[rows, cols][inside].max()) + 2 * b.half[2]
        r0, c0 = rows.start - 1, cols.start - 1
        # edge_margin >= cell_size keeps the widened window inside the tile.
        assert r0 >= 0 and c0 >= 0, b
        base = float(heights[r0 : rows.stop + 1, c0 : cols.stop + 1].min())
        seated.append(
            Box(
                (b.pos[0], b.pos[1], (top + base) / 2),
                (b.half[0], b.half[1], (top - base) / 2),
                b.yaw,
            )
        )
    return seated


def _raster(heights: np.ndarray, boxes: list[Box], axis: np.ndarray) -> np.ndarray:
    """The tile's ground-truth surface: its heights, with each box's top
    written onto the nodes the box covers."""
    surface = heights.copy()
    for b in boxes:
        window, inside = _footprint(axis, b)
        patch = surface[window]
        patch[inside] = np.maximum(patch[inside], b.pos[2] + b.half[2])
    return surface


def _pit(g: _TileGrid, flight: float, depth: float, cell: float) -> np.ndarray:
    """The inverted-stairs pit: -depth on every node at least one cell inside
    the flight's outer edge, 0 elsewhere.

    Between its last carved node and the next, the heightfield climbs from
    -depth to 0. Carving one cell short puts that climb under the outermost
    ring: the last carved node lies under two cells inside the flight edge,
    and ArenaParams holds every tread to at least two cells. The ring's top
    hides all of it below -riser. The wall is then the last riser, and
    nothing past the flight dips below the ground. Carved out to the edge,
    the climb falls outside the ring. mj_ray found a V-groove there 0.73 m
    deep beside the default arena's 0.75 m pit."""
    return np.where(g.cheby <= flight - cell + _EPS, -depth, 0.0)


def _stairs_tile(inverted: bool, d: float, rng, g: _TileGrid, p: ArenaParams) -> _Tile:
    lo, hi = stair_tread_bounds(p)
    tread = float(rng.uniform(lo, hi)) if isinstance(p.stair_tread, tuple) else lo
    n_steps = stair_steps(tread, p)
    riser = p.stair_riser(d)
    platform = summit_platform_half(p)
    flight = stair_flight_half(tread, n_steps, p)
    if inverted:
        depth = n_steps * riser
        heights = _pit(g, flight, depth, g.axis[1] - g.axis[0])
        boxes = _pit_stair_boxes(platform, tread, n_steps, riser)
        return _Tile(heights, boxes, -depth, flight, tread, n_steps)
    boxes = _pyramid_stair_boxes(platform, tread, n_steps, riser)
    return _Tile(np.zeros_like(g.lx), boxes, n_steps * riser, flight, tread, n_steps)


def _build_tile(ttype: str, d: float, rng, g: _TileGrid, p: ArenaParams) -> _Tile:
    if ttype in STAIR_TYPES:
        return _stairs_tile(ttype == "inverted_pyramid_stairs", d, rng, g, p)
    # Boxes stay edge_margin inside the tile edge. The continuous surfaces
    # use the same radius even though their relief reaches 0 only at the edge.
    feature = p.tile_size / 2 - p.edge_margin
    if ttype == "rough_uniform":
        return _Tile(_noise(g, p.rough_amplitude(d), rng, p), [], 0.0, feature)
    if ttype == "wave":
        return _Tile(_wave(g, d, rng, p), [], 0.0, feature)
    overlay = _noise(g, p.overlay_fraction * p.rough_amplitude(d), rng, p)
    if ttype in ("pyramid_slope", "inverted_pyramid_slope"):
        sign = 1.0 if ttype == "pyramid_slope" else -1.0
        heights = _slope(g, d, sign, p) + overlay
        return _Tile(heights, [], sign * slope_plateau_height(d, p), feature)
    boxes = _seat(_BOX_SCATTER[ttype](d, rng, p), overlay, g.axis)
    return _Tile(overlay, boxes, 0.0, feature)


def _stream(p: ArenaParams, tag: int, *idx: int) -> np.random.Generator:
    """The rng stream keyed by the seed, a tag and the row or tile index."""
    return np.random.default_rng(np.random.SeedSequence([p.seed, tag, *idx]))


def _row_layout(i: int, p: ArenaParams) -> tuple[float, tuple[str, ...]]:
    """Terrain row i's difficulty and column order.

    One rng stream per row draws the shuffle and then the jitter, so an
    explicit difficulty list keeps the column order the seed gives. Jitter
    moves interior rows only, so rows 0 and n - 1 keep their exact
    difficulty: 0 and 1 by default, the given values when `difficulties` is
    set."""
    n = p.n_rows
    d = p.difficulties[i] if p.difficulties is not None else i / (n - 1)
    if p.ordered:
        return float(d), TYPES
    rng = _stream(p, _ROW, i)
    types = tuple(TYPES[k] for k in rng.permutation(len(TYPES)))
    if p.difficulties is None and 0 < i < n - 1:
        d += rng.uniform(-p.row_jitter, p.row_jitter) / (n - 1)
    return float(d), types


def generate(params: ArenaParams | None = None, **overrides) -> Arena:
    """Build one arena from `params` (the defaults when None), with any
    field overridden by keyword: `generate(seed=1)`,
    `generate(params, flat_row=True)`.

    The flat row is additive. Every terrain row keeps its difficulty, its
    column order and its rng streams. It moves up one row and one tile."""
    p = ArenaParams(**overrides) if params is None else replace(params, **overrides)
    # Fail before any work if the widest tread cannot fit the tile.
    stair_steps(stair_tread_bounds(p)[1], p)

    n_cols = len(TYPES)
    n_rows = p.n_rows + int(p.flat_row)
    tile_cells = node_count(p.tile_size, p.cell_size)
    border_cells = node_count(p.border, p.cell_size, minimum=0)
    nrow = n_rows * tile_cells + 2 * border_cells + 1
    ncol = n_cols * tile_cells + 2 * border_cells + 1
    radius_x = n_cols * p.tile_size / 2 + p.border
    radius_y = n_rows * p.tile_size / 2 + p.border
    grid_x0 = -n_cols * p.tile_size / 2
    grid_y0 = -n_rows * p.tile_size / 2
    g = _tile_grid(p)

    heights = np.zeros((nrow, ncol))
    lookup = np.zeros((nrow, ncol))
    boxes: list[Box] = []
    tiles: list[TileSpec] = []
    if p.flat_row:
        # Flat to the bit and box-free. It still holds one tile of every
        # type, so every (row, type) pair names a tile. Here the type only
        # names the column.
        cy = grid_y0 + 0.5 * p.tile_size
        for j, ttype in enumerate(TYPES):
            cx = grid_x0 + (j + 0.5) * p.tile_size
            tiles.append(
                TileSpec(
                    row=0,
                    col=j,
                    terrain_type=ttype,
                    difficulty=0.0,
                    origin=(cx, cy, 0.0),
                    pad_radius=p.pad_radius,
                    pad_height=0.0,
                )
            )
    for i in range(p.n_rows):
        row = i + int(p.flat_row)
        d, row_types = _row_layout(i, p)
        cy = grid_y0 + (row + 0.5) * p.tile_size
        r0 = border_cells + row * tile_cells
        for j, ttype in enumerate(row_types):
            cx = grid_x0 + (j + 0.5) * p.tile_size
            c0 = border_cells + j * tile_cells
            dt = d * p.cap(ttype)
            tile = _build_tile(ttype, dt, _stream(p, _TILE, i, j), g, p)
            # Neighbours share their perimeter nodes, where both are 0.
            window = (slice(r0, r0 + tile_cells + 1), slice(c0, c0 + tile_cells + 1))
            heights[window] = tile.heights
            lookup[window] = _raster(tile.heights, tile.boxes, g.axis)
            boxes += [
                Box((cx + b.pos[0], cy + b.pos[1], b.pos[2]), b.half, b.yaw)
                for b in tile.boxes
            ]
            tiles.append(
                TileSpec(
                    row=row,
                    col=j,
                    terrain_type=ttype,
                    difficulty=dt,
                    origin=(cx, cy, 0.0),
                    pad_radius=p.pad_radius,
                    # + 0.0 turns the flat inverted slope's -0.0 into 0.0.
                    pad_height=float(tile.pad_height) + 0.0,
                    feature_radius=float(tile.feature_radius),
                    stair_tread=tile.stair_tread,
                    n_steps=tile.n_steps,
                )
            )

    if not np.isfinite(heights).all():
        raise ValueError("the arena's heights are not all finite")
    hmin, hmax = float(heights.min()), float(heights.max())
    # Every row holds an inverted-stairs pit, at least stair_min_steps
    # positive risers deep, so the range is positive as MuJoCo requires.
    assert hmax > hmin
    elevation_z, pos_z = hmax - hmin, hmin
    hfield_data = ((heights - hmin) / (hmax - hmin)).astype(np.float32)

    spec = TerrainSpec(
        generator_version=GENERATOR_VERSION,
        params=p,
        n_rows=n_rows,
        n_cols=n_cols,
        types=TYPES,
        stair_platform_half=summit_platform_half(p),
        x_min=-radius_x,
        x_max=radius_x,
        y_min=-radius_y,
        y_max=radius_y,
        hfield=HFieldSpec(
            nrow=nrow,
            ncol=ncol,
            radius_x=radius_x,
            radius_y=radius_y,
            elevation_z=elevation_z,
            base_z=HFIELD_BASE_Z,
            pos_z=pos_z,
        ),
        tiles=tuple(tiles),
    )
    return Arena(
        spec=spec,
        boxes=tuple(boxes),
        hfield_data=hfield_data,
        lookup=lookup.astype(np.float32),
    )


def sample_frame(spec: TerrainSpec) -> tuple[float, float, float, float]:
    """(x0, dx, y0, dy) of the arena's node grid, as `bilinear` takes it."""
    dx = (spec.x_max - spec.x_min) / (spec.hfield.ncol - 1)
    dy = (spec.y_max - spec.y_min) / (spec.hfield.nrow - 1)
    return spec.x_min, dx, spec.y_min, dy


def grid_axes(spec: TerrainSpec) -> tuple[np.ndarray, np.ndarray]:
    """World x of every node column and world y of every node row."""
    xs = np.linspace(spec.x_min, spec.x_max, spec.hfield.ncol)
    ys = np.linspace(spec.y_min, spec.y_max, spec.hfield.nrow)
    return xs, ys


def lookup_height(arena: Arena, x, y) -> np.ndarray:
    """Ground-truth surface height at world (x, y), bilinear in the lookup."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return bilinear(arena.lookup, *sample_frame(arena.spec), x, y)


def spec_to_dict(spec: TerrainSpec) -> dict:
    """JSON-ready view of the spec. `spec_from_dict` inverts it exactly."""
    return {
        "generator_version": spec.generator_version,
        "params": params_to_dict(spec.params),
        "n_rows": spec.n_rows,
        "n_cols": spec.n_cols,
        "types": list(spec.types),
        "stair_platform_half": spec.stair_platform_half,
        "extent": {
            "x_min": spec.x_min,
            "x_max": spec.x_max,
            "y_min": spec.y_min,
            "y_max": spec.y_max,
        },
        "hfield": asdict(spec.hfield),
        "tiles": [{**asdict(t), "origin": list(t.origin)} for t in spec.tiles],
    }


def spec_from_dict(data: dict) -> TerrainSpec:
    """The `TerrainSpec` that `spec_to_dict` wrote."""
    return TerrainSpec(
        generator_version=data["generator_version"],
        params=params_from_dict(data["params"]),
        n_rows=data["n_rows"],
        n_cols=data["n_cols"],
        types=tuple(data["types"]),
        stair_platform_half=data["stair_platform_half"],
        **data["extent"],
        hfield=HFieldSpec(**data["hfield"]),
        tiles=tuple(
            TileSpec(**{**t, "origin": tuple(t["origin"])}) for t in data["tiles"]
        ),
    )


def fingerprint(arena: Arena) -> str:
    """sha256 naming an arena: the generator version, the params, and the
    lookup, heightfield and box arrays they built.

    The arrays enter at float32, the precision the grids are stored at. Box
    parameters are rounded to float32 too, so a last-bit difference in sin
    or tan between machines only rarely moves the hash."""
    h = hashlib.sha256()
    head = {
        "generator_version": arena.spec.generator_version,
        "params": params_to_dict(arena.spec.params),
    }
    h.update(json.dumps(head, sort_keys=True).encode())
    box_array = np.array(
        [b.pos + b.half + (b.yaw,) for b in arena.boxes], dtype=np.float64
    ).reshape(-1, 7)
    for a in (arena.lookup, arena.hfield_data, box_array):
        a = np.ascontiguousarray(a, dtype="<f4")
        h.update(np.asarray(a.shape, dtype="<i8").tobytes())
        h.update(a.tobytes())
    return h.hexdigest()
