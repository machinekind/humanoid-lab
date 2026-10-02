"""envs/terrain_geometry.py on synthetic grids: samplers, lowest points,
spawn height and spawn draws. Model-free, jax on CPU."""

from __future__ import annotations

import math

import jax
import jax.numpy as jp
import numpy as np
import pytest

from humanoid_lab import terrain
from humanoid_lab.envs import terrain_geometry as tg
from humanoid_lab.terrain.config import CPU_ARENA, arena_for, params_from_config

CELL = 0.04
N = 101  # nodes per axis: a 4 m square centred on the origin
X0 = -2.0
# The CPU arena with rubble and obstacles. It keeps the CPU arena's 33
# stair treads and adds yawed boxes.
RUBBLE_ARENA = {**CPU_ARENA, "grid_fill_prob": 0.7, "discrete_count": 12}


def _tables(lookup, *, spawn=None, heights=None, boxes=(), flat_row=False, band=tg.EMPTY_BAND, n_rows=2):
    """Tables over a synthetic N x N grid. Every tile of every row is
    centred on the origin. The heightfield grid is the lookup unless
    given."""
    lookup = np.asarray(lookup, np.float32)
    spawn = tg.dilate_max(lookup, 1) if spawn is None else np.asarray(spawn, np.float32)
    heights = lookup if heights is None else np.asarray(heights, np.float32)
    frame = (X0, CELL, X0, CELL)
    cell_boxes, rows = tg.box_tables(boxes, frame, lookup.shape)
    n_types = len(terrain.TYPES)
    return tg.TerrainTables(
        lookup=jp.asarray(lookup),
        lookup_spawn=jp.asarray(spawn),
        heights=jp.asarray(heights),
        cell_boxes=jp.asarray(cell_boxes),
        boxes=jp.asarray(rows),
        frame=frame,
        origin_xy=jp.zeros((n_rows, n_types, 2)),
        pad_h=jp.zeros((n_rows, n_types)),
        feature_r=jp.full((n_rows, n_types), 1.6),
        flat_band=band,
        n_rows=n_rows,
        flat_row=flat_row,
        tile_size=4.0,
        pad_radius=0.4,
        max_step=0.0,
    )


def _axes():
    return X0 + CELL * np.arange(N)


def _slope(deg):
    xs = _axes()
    return np.tile(np.tan(np.radians(deg)) * xs, (N, 1))


def _sole():
    """A level sole 0.16 m long and 0.04 m wide, three rows of five points."""
    xs = np.linspace(-0.08, 0.08, 5)
    ys = (-0.02, 0.0, 0.02)
    return jp.asarray([(x, y) for y in ys for x in xs], dtype=jp.float32)


SPAWN_KW = {"yaw_enable": True, "pad_jitter": 0.15, "k": 8, "half_extent": 1.5, "max_spread": 0.02}


# -- samplers ------------------------------------------------------------------


def test_jnp_sampler_equals_the_numpy_sampler():
    rng = np.random.default_rng(0)
    grid = rng.uniform(-0.2, 0.3, (N, N)).astype(np.float32)
    t = _tables(grid)
    xy = rng.uniform(-2.1, 2.1, (5000, 2)).astype(np.float32)
    want = terrain.bilinear(grid.astype(float), X0, CELL, X0, CELL, xy[:, 0], xy[:, 1])
    got = np.asarray(jax.jit(lambda p: tg.height(t, p))(jp.asarray(xy)))
    # float32 coordinates resolve a node offset to about 6e-6 of a cell.
    np.testing.assert_allclose(got, want, atol=1e-5)


def test_an_all_zero_grid_reads_positive_zero():
    t = _tables(np.zeros((N, N)))
    xy = jp.asarray(np.random.default_rng(1).uniform(-2.2, 2.2, (2000, 2)), jp.float32)
    for grid in ("lookup", "spawn"):
        h = np.asarray(tg.height(t, xy, grid))
        assert np.all(h == 0.0) and not np.signbit(h).any()
    z = jp.linspace(-1.0, 1.0, 2000)
    np.testing.assert_array_equal(np.asarray(z - tg.height(t, xy)), np.asarray(z))
    # The exact ground reads +0.0 on any mix of signed zeros.
    signed = np.where(np.random.default_rng(2).random((N, N)) < 0.5, -0.0, 0.0)
    g = np.asarray(tg.ground(_tables(np.zeros((N, N)), heights=signed), xy))
    assert np.all(g == 0.0) and not np.signbit(g).any()


def test_dilate_max_is_a_neighbourhood_max_and_identity_on_flat():
    rng = np.random.default_rng(2)
    grid = rng.normal(size=(7, 9))
    for k in (1, 2):
        out = tg.dilate_max(grid, k)
        for r in range(7):
            for c in range(9):
                window = grid[max(r - k, 0) : r + k + 1, max(c - k, 0) : c + k + 1]
                assert out[r, c] == window.max()
    assert np.array_equal(tg.dilate_max(np.full((5, 5), 0.3), 1), np.full((5, 5), 0.3))
    assert np.array_equal(tg.dilate_max(grid, 0), grid)


def test_in_band_is_the_flat_row_rectangle():
    t = tg.tables_from_arena(arena_for(params_from_config(CPU_ARENA)))
    assert t.flat_band == (-16.0, 16.0, -4.0, 0.0)
    inside = jp.array([[-16.0, -4.0], [16.0, 0.0], [0.0, -2.0], [-3.3, -0.01]])
    outside = jp.array([[0.0, 0.01], [16.01, -2.0], [-16.01, -2.0], [0.0, -4.01]])
    assert np.all(np.asarray(tg.in_band(t, inside)))
    assert not np.any(np.asarray(tg.in_band(t, outside)))
    empty = _tables(np.zeros((N, N)))
    assert not np.any(np.asarray(tg.in_band(empty, inside)))
    # The same arena without a flat row has an empty band.
    no_flat = tg.tables_from_arena(arena_for(params_from_config({**CPU_ARENA, "flat_row": False})))
    assert no_flat.flat_row is False
    assert no_flat.flat_band == tg.EMPTY_BAND
    assert not np.any(np.asarray(tg.in_band(no_flat, inside)))


def test_the_flat_band_reads_zero_on_the_cpu_arena():
    t = tg.tables_from_arena(arena_for(params_from_config(CPU_ARENA)))
    xy = jax.random.uniform(
        jax.random.PRNGKey(0), (20000, 2), minval=jp.array([-16.0, -4.0]), maxval=jp.array([16.0, 0.0])
    )
    assert np.all(np.asarray(tg.height(t, xy)) == 0.0)
    assert np.all(np.asarray(tg.ground(t, xy)) == 0.0)


# -- exact ground ------------------------------------------------------------------


def _random_boxes(n, seed):
    rng = np.random.default_rng(seed)
    return [
        terrain.Box(
            (float(rng.uniform(-1.5, 1.5)), float(rng.uniform(-1.5, 1.5)), 0.0),
            (float(rng.uniform(0.03, 0.2)), float(rng.uniform(0.03, 0.2)), float(rng.uniform(0.02, 0.15))),
            float(rng.uniform(0.0, math.pi)),
        )
        for _ in range(n)
    ]


def _box_frame(b, xy):
    """|u|, |v| of points `xy` (n, 2) in box `b`'s own frame, float64."""
    c, s = math.cos(b.yaw), math.sin(b.yaw)
    ox, oy = xy[:, 0] - b.pos[0], xy[:, 1] - b.pos[1]
    return np.abs(ox * c + oy * s), np.abs(oy * c - ox * s)


def _inside(boxes, xy):
    """(n_points, n_boxes) whether each point lies in each box's footprint,
    float64 on the host."""
    out = np.zeros((len(xy), len(boxes)), bool)
    for j, b in enumerate(boxes):
        u, v = _box_frame(b, xy)
        out[:, j] = (u <= b.half[0]) & (v <= b.half[1])
    return out


def test_box_tables_list_every_box_a_cell_meets():
    """A point inside a box finds that box in its cell's list, and K is the
    most boxes any cell lists."""
    boxes = _random_boxes(40, 0)
    cell, rows = tg.box_tables(boxes, (X0, CELL, X0, CELL), (N, N))
    assert cell.shape[:2] == (N - 1, N - 1) and rows.shape == (41, 7)
    assert np.isinf(rows[-1, 6]) and rows[-1, 6] < 0
    listed = (cell != 40).sum(axis=-1)
    assert listed.max() == cell.shape[2]
    xy = np.random.default_rng(1).uniform(-1.8, 1.8, (20000, 2))
    inside = _inside(boxes, xy)
    col = np.floor((xy[:, 0] - X0) / CELL).astype(int)
    row = np.floor((xy[:, 1] - X0) / CELL).astype(int)
    for i, j in zip(*np.nonzero(inside)):
        assert j in cell[row[i], col[i]], (i, j)


def test_box_tables_refuse_a_crowded_cell():
    """The refusal names the first crowded cell in row order, by its world
    centre."""
    stack = [terrain.Box((0.0, 0.0, 0.1), (0.1, 0.1, 0.1), yaw) for yaw in (0.0, 0.3, 0.6)]
    cell, _ = tg.box_tables(stack, (X0, CELL, X0, CELL), (N, N))
    assert cell.shape[2] == 3
    with pytest.raises(
        ValueError, match=r"the lookup cell centred at \(-0\.06, -0\.10\) meets 3 boxes, over the limit of 2"
    ):
        tg.box_tables(stack, (X0, CELL, X0, CELL), (N, N), cap=2)


def test_tables_refuse_a_crowded_arena_and_name_its_knobs():
    """200 discrete obstacles per tile crowd 10 boxes into one cell at seed
    0. At seed 1 they crowd 8, the limit, and the tables build."""
    crowded = {**CPU_ARENA, "discrete_count": 200}
    with pytest.raises(
        ValueError,
        match=r"meets 10 boxes, over the limit of 8\..*Lower terrain\.arena\.discrete_count or "
        r"discrete_half_range, or pick another terrain\.arena\.seed",
    ):
        tg.tables_from_arena(arena_for(params_from_config({**crowded, "seed": 0})))
    t = tg.tables_from_arena(arena_for(params_from_config({**crowded, "seed": 1})))
    assert t.cell_boxes.shape[2] == tg.MAX_CELL_BOXES == 8


def test_ground_is_the_heightfield_or_the_top_of_a_containing_box():
    """On rolling ground with yawed boxes, the ground is the heightfield on
    MuJoCo's triangles, or the highest top of the boxes that contain the
    point."""
    rng = np.random.default_rng(2)
    heights = rng.uniform(-0.05, 0.05, (N, N))
    boxes = _random_boxes(40, 3)
    t = _tables(heights, boxes=boxes)
    xy = rng.uniform(-1.8, 1.8, (20000, 2))
    got = np.asarray(jax.jit(lambda p: tg.ground(t, p))(jp.asarray(xy, jp.float32)))
    field = terrain.triangle(heights.astype(np.float32).astype(float), X0, CELL, X0, CELL, xy[:, 0], xy[:, 1])
    tops = np.array([b.pos[2] + b.half[2] for b in boxes])
    inside = _inside(boxes, xy)
    want = np.maximum(field, np.where(inside, tops[None, :], -np.inf).max(axis=1))
    # float32 coordinates move a point by about 1e-7 m. Points that close to
    # an edge may land on either side of it.
    margin = np.ones(len(xy), bool)
    for b in boxes:
        u, v = _box_frame(b, xy)
        margin &= (np.abs(u - b.half[0]) >= 1e-5) & (np.abs(v - b.half[1]) >= 1e-5)
    assert inside.any(axis=1).mean() > 0.1
    np.testing.assert_allclose(got[margin], want[margin], atol=1e-5)


def test_ground_equals_the_lookup_off_the_boxes_on_a_rubble_arena():
    """The heightfield grid keeps the lookup's value bit for bit at every
    node away from the boxes, the flat row included. Under a box it holds
    the heightfield, not the top. Most boxes are yawed."""
    arena = arena_for(params_from_config(RUBBLE_ARENA))
    assert sum(b.yaw != 0.0 for b in arena.boxes) > 100
    t = tg.tables_from_arena(arena)
    lookup, heights = np.asarray(t.lookup), np.asarray(t.heights)
    xs, ys = terrain.grid_axes(arena.spec)
    gx, gy = np.meshgrid(xs, ys)
    nodes = np.stack([gx.ravel(), gy.ravel()], axis=1)
    near = np.zeros(len(nodes), bool)
    for b in arena.boxes:
        grown = terrain.Box(b.pos, (b.half[0] + 1e-3, b.half[1] + 1e-3, b.half[2]), b.yaw)
        near |= _inside([grown], nodes)[:, 0]
    near = near.reshape(lookup.shape)
    assert near.any()
    np.testing.assert_array_equal(heights[~near], lookup[~near])
    np.testing.assert_array_equal(np.signbit(heights[~near]), np.signbit(lookup[~near]))
    hf = arena.spec.hfield
    field = (hf.pos_z + hf.elevation_z * arena.hfield_data.astype(float)).astype(np.float32)
    np.testing.assert_allclose(heights[near], field[near], atol=1e-6)
    assert (lookup[near] > heights[near] + 0.01).mean() > 0.9


def test_ground_reads_the_cell_on_the_diagonal_mujoco_splits():
    """One cell with node (r, c + 1) raised 0.1 m. MuJoCo splits the cell
    on the diagonal from node (r, c) to node (r + 1, c + 1). At (tx, ty) =
    (0.75, 0.25) the point lies in the raised node's triangle and reads
    0.05 m. At (0.25, 0.75) it lies in the other triangle and reads 0.
    Bilinear would read 0.05625 and 0.00625. The other diagonal would read
    0.075 and 0.025."""
    heights = np.zeros((N, N))
    heights[50, 51] = 0.1  # the cell from (0, 0) to (0.04, 0.04)
    t = _tables(np.zeros((N, N)), heights=heights)
    xy = jp.array([[0.75 * CELL, 0.25 * CELL], [0.25 * CELL, 0.75 * CELL]])
    np.testing.assert_allclose(np.asarray(tg.ground(t, xy)), [0.05, 0.0], atol=1e-6)
    np.testing.assert_allclose(
        terrain.triangle(np.array([[0.0, 0.1], [0.0, 0.0]]), 0.0, 1.0, 0.0, 1.0,
                         np.array([0.75, 0.25]), np.array([0.25, 0.75])),
        [0.05, 0.0],
        atol=1e-12,
    )


def test_ground_reads_the_box_tops_on_a_rubble_arena():
    """Tables built from an arena of stairs, rubble and obstacles: at every
    lookup node at least one cell inside a box, the exact ground reads the
    top the lookup holds. Most boxes are yawed."""
    arena = arena_for(params_from_config(RUBBLE_ARENA))
    assert sum(b.yaw != 0.0 for b in arena.boxes) > 100
    t = tg.tables_from_arena(arena)
    lookup = np.asarray(t.lookup)
    xs, ys = terrain.grid_axes(arena.spec)
    gx, gy = np.meshgrid(xs, ys)
    nodes = np.stack([gx.ravel(), gy.ravel()], axis=1)
    shrunk = [terrain.Box(b.pos, (b.half[0] - CELL, b.half[1] - CELL, b.half[2]), b.yaw) for b in arena.boxes]
    deep = _inside(shrunk, nodes).any(axis=1)
    assert deep.sum() > 1000
    got = np.asarray(tg.ground(t, jp.asarray(nodes[deep], jp.float32)))
    np.testing.assert_allclose(got, lookup.ravel()[deep], atol=1e-6)


# -- foot soles --------------------------------------------------------------------


def test_sole_samples_cover_the_capsule_footprint():
    """Five axis points, each 0.8 r to either side, and one point 0.8 r
    beyond each end of the axis, under its end cap. 17 samples, every one
    within the radius of the axis segment."""
    s = np.asarray(tg.SOLE_SAMPLES)
    assert s.shape == (17, 3)
    assert sorted(set(s[:, 0].tolist())) == [-1.0, -0.5, 0.0, 0.5, 1.0]
    assert np.all(np.hypot(s[:, 1], s[:, 2]) <= 0.8)
    assert np.all((s[:, 1] == 0.0) | (np.abs(s[:, 0]) == 1.0))
    assert np.all(np.sign(s[:, 1]) * np.sign(s[:, 0]) >= 0)


def test_sole_gaps_read_the_underside_on_level_ground_and_a_slope():
    """A level capsule 3 cm above flat ground reads 3 cm at the axis
    samples and 3 cm + 0.4 r beside them. Lying along a 20 degree slope at
    rest it reads r / cos(20 deg) - r at its lowest sample."""
    r, half = 0.012, 0.067
    flat = _tables(np.zeros((N, N)))
    centre = jp.array([[0.3, -0.2, 0.03 + r]])
    axis = jp.array([[math.cos(0.4), math.sin(0.4), 0.0]])
    gaps = np.asarray(tg.sole_gaps(flat, centre, axis, jp.array([half]), jp.array([r])))[0]
    on_axis = (np.asarray(tg.SOLE_SAMPLES)[:, 1:] == 0.0).all(axis=1)
    np.testing.assert_allclose(gaps[on_axis], 0.03, atol=1e-6)
    np.testing.assert_allclose(gaps[~on_axis], 0.03 + 0.4 * r, atol=1e-6)

    a = math.radians(20.0)
    slope = _tables(_slope(20.0))
    centre = jp.array([[0.1, 0.0, math.tan(a) * 0.1 + r / math.cos(a)]])
    axis = jp.array([[math.cos(a), 0.0, math.sin(a)]])
    gaps = np.asarray(tg.sole_gaps(slope, centre, axis, jp.array([half]), jp.array([r])))[0]
    assert gaps.min() == pytest.approx(r / math.cos(a) - r, abs=1e-5)
    # A sphere is a capsule of half-length 0.
    sphere = np.asarray(tg.sole_gaps(flat, jp.array([[0.0, 0.0, 0.05]]), jp.array([[0.0, 0.0, 1.0]]),
                                     jp.array([0.0]), jp.array([0.02])))[0]
    assert sphere.min() == pytest.approx(0.03, abs=1e-6)


@pytest.mark.parametrize("yaw", [0.0, 1.0])
def test_a_box_edge_under_a_capsule_side_meets_the_side_samples(yaw):
    """A box whose edge runs along a level capsule, 0.5 r out from its
    axis, holds the capsule up by its side. No axis sample is over the
    box. The samples 0.8 r out on that side are, and read a gap of
    0.27 r, inside the 5 mm contact band. The capsule and the box turn by
    the same yaw."""
    r, half, top = 0.012, 0.067, 0.08
    fwd = np.array([math.cos(yaw), math.sin(yaw)])
    side = np.array([-fwd[1], fwd[0]])
    cx, cy = (0.5 * r + 0.5) * side
    box = terrain.Box((float(cx), float(cy), top / 2), (0.5, 0.5, top / 2), yaw)
    t = _tables(np.zeros((N, N)), boxes=[box])
    z = top + math.sqrt(r * r - (0.5 * r) ** 2)
    centre, axis = jp.array([[0.0, 0.0, z]]), jp.array([[fwd[0], fwd[1], 0.0]])
    gaps = np.asarray(tg.sole_gaps(t, centre, axis, jp.array([half]), jp.array([r])))[0]
    s = np.asarray(tg.SOLE_SAMPLES)
    over = s[:, 2] > 0
    np.testing.assert_allclose(gaps[over], z - 0.6 * r - top, atol=1e-6)
    assert gaps[over].max() < 0.005
    # Every other sample reads the ground beside the box, 8 cm down.
    np.testing.assert_allclose(gaps[~over].min(), z - r, atol=1e-6)


@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("yaw", [0.0, 1.2])
def test_a_box_edge_under_a_capsule_end_cap_meets_its_end_sample(yaw, sign):
    """A box edge crosses a level capsule's axis 0.5 r beyond one end,
    under the end cap, and holds the capsule up there. Only that end's
    sample, 0.8 r beyond the axis end, is over the box. It reads a gap of
    (sqrt(0.75) - 0.6) r, inside the 5 mm contact band. Every other sample
    reads the ground beside the box."""
    r, half, top = 0.012, 0.067, 0.08
    fwd = np.array([math.cos(yaw), math.sin(yaw)])
    cx, cy = sign * (half + 0.5 * r + 0.5) * fwd
    box = terrain.Box((float(cx), float(cy), top / 2), (0.5, 0.5, top / 2), yaw)
    t = _tables(np.zeros((N, N)), boxes=[box])
    z = top + math.sqrt(0.75) * r
    centre, axis = jp.array([[0.0, 0.0, z]]), jp.array([[fwd[0], fwd[1], 0.0]])
    gaps = np.asarray(tg.sole_gaps(t, centre, axis, jp.array([half]), jp.array([r])))[0]
    assert gaps.min() == pytest.approx((math.sqrt(0.75) - 0.6) * r, abs=1e-6)
    assert (gaps < 0.005).sum() == 1


# -- lowest points ----------------------------------------------------------------


def _rotations(n, seed):
    q = np.random.default_rng(seed).normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
            np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
            np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
        ],
        axis=1,
    )


def test_box_lowest_point_matches_brute_force_corners():
    rng = np.random.default_rng(3)
    R = _rotations(500, 4)
    c = rng.normal(size=(500, 3))
    half = rng.uniform(0.02, 0.2, (500, 3))
    xy, z = tg.lowest_point_box(jp.asarray(c), jp.asarray(R), jp.asarray(half))
    signs = np.array([(sx, sy, sz) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    corners = c[:, None, :] + np.einsum("nij,nkj->nki", R, signs[None, :, :] * half[:, None, :])
    low = corners[np.arange(500), corners[..., 2].argmin(1)]
    np.testing.assert_allclose(np.asarray(z), low[:, 2], atol=1e-5)
    np.testing.assert_allclose(np.asarray(xy), low[:, :2], atol=1e-5)


def test_capsule_lowest_point_matches_brute_force():
    rng = np.random.default_rng(5)
    R = _rotations(500, 6)
    c = rng.normal(size=(500, 3))
    half = rng.uniform(0.0, 0.2, 500)
    r = rng.uniform(0.01, 0.07, 500)
    xy, z = tg.lowest_point_capsule(jp.asarray(c), jp.asarray(R), jp.asarray(half), jp.asarray(r))
    ts = np.linspace(-1, 1, 2001)
    axis_pts = c[:, None, :] + ts[None, :, None] * half[:, None, None] * R[:, None, :, 2]
    k = axis_pts[..., 2].argmin(1)
    low = axis_pts[np.arange(500), k]
    np.testing.assert_allclose(np.asarray(z), low[:, 2] - r, atol=1e-5)
    np.testing.assert_allclose(np.asarray(xy), low[:, :2], atol=1e-5)
    sxy, sz = tg.lowest_point_sphere(jp.asarray(c), jp.asarray(r))
    np.testing.assert_allclose(np.asarray(sz), c[:, 2] - r, atol=1e-6)
    np.testing.assert_allclose(np.asarray(sxy), c[:, :2])


def test_capsule_points_are_the_centre_and_both_end_cap_centres():
    c = jp.array([[0.1, 0.2, 0.3]])
    axis = jp.array([[0.0, 0.6, 0.8]])
    pts = np.asarray(tg.capsule_points(c, axis, jp.array([0.05]), (-1.0, 0.0, 1.0)))
    np.testing.assert_allclose(pts[0], [[0.1, 0.17, 0.26], [0.1, 0.2, 0.3], [0.1, 0.23, 0.34]], atol=1e-7)


# -- spawn height ------------------------------------------------------------------


def _points(seed=7, n=40):
    rng = np.random.default_rng(seed)
    u = rng.uniform(-0.3, 0.3, (n, 2))
    lift = np.concatenate([[0.0], rng.uniform(0.0, 1.0, n - 1)])
    return jp.asarray(u, jp.float32), jp.asarray(lift, jp.float32)


@pytest.mark.parametrize("grid", ["lookup", "spawn"])
def test_spawn_z_on_flat_ground_is_z0_plus_ground(grid):
    u, lift = _points()
    z0 = 0.75
    flat = _tables(np.zeros((N, N)))
    for xy, yaw in (((0.0, 0.0), 0.0), ((0.31, -0.57), 1.1), ((-1.13, 0.4), -2.9)):
        z = tg.spawn_z(flat, u, lift, z0, jp.asarray(xy), yaw, grid)
        assert float(z) == np.float32(z0)
    # On a raised plane the bilinear weights round: within one float32 ulp.
    raised = _tables(np.full((N, N), 0.2))
    for xy, yaw in (((0.0, 0.0), 0.0), ((0.31, -0.57), 1.1)):
        z = tg.spawn_z(raised, u, lift, z0, jp.asarray(xy), yaw, grid)
        assert abs(float(z) - (z0 + 0.2)) <= 1.2e-7


def test_a_raised_point_under_one_collider_lifts_the_spawn():
    """A 0.3 m block under a point that sits 0.1 m above the sole lifts the
    base by 0.2 m. Under a point 0.5 m up it lifts nothing."""
    xs = _axes()
    grid = np.zeros((N, N))
    block = (np.abs(xs - 0.4) <= 0.1)[None, :] & (np.abs(xs) <= 0.1)[:, None]
    grid[block] = 0.3
    t = _tables(grid)
    u = jp.array([[0.0, 0.0], [0.4, 0.0]])
    z = tg.spawn_z(t, u, jp.array([0.0, 0.1]), 0.75, jp.zeros(2), 0.0, "lookup")
    np.testing.assert_allclose(float(z), 0.75 + 0.2, atol=1e-6)
    z = tg.spawn_z(t, u, jp.array([0.0, 0.5]), 0.75, jp.zeros(2), 0.0, "lookup")
    np.testing.assert_allclose(float(z), 0.75, atol=1e-7)


def test_spawn_z_rotates_offsets_with_yaw():
    """A point at +y of the base reaches a block at +x when the base turns
    by -pi/2."""
    xs = _axes()
    grid = np.zeros((N, N))
    grid[(np.abs(xs) <= 0.08)[:, None] & (np.abs(xs - 0.4) <= 0.08)[None, :]] = 0.3
    t = _tables(grid)
    u = jp.array([[0.0, 0.0], [0.0, 0.4]])
    lift = jp.zeros(2)
    assert float(tg.spawn_z(t, u, lift, 0.75, jp.zeros(2), 0.0, "lookup")) == pytest.approx(0.75)
    z = tg.spawn_z(t, u, lift, 0.75, jp.zeros(2), -math.pi / 2, "lookup")
    assert float(z) == pytest.approx(1.05, abs=1e-6)


def test_support_spread_is_zero_on_flat_and_rejects_a_straddled_riser_and_a_steep_slope():
    sole = _sole()
    flat = _tables(np.zeros((N, N)))
    assert float(tg.support_spread(flat, sole, jp.zeros(2), 0.7)) == 0.0

    xs = _axes()
    riser = _tables(np.tile(np.where(xs > 0.0, 0.1, 0.0), (N, 1)))
    assert float(tg.support_spread(riser, sole, jp.zeros(2), 0.0)) >= 0.1

    steep = _tables(_slope(10.0))
    assert float(tg.support_spread(steep, sole, jp.zeros(2), 0.0)) > 0.02
    gentle = _tables(_slope(3.0))
    assert float(tg.support_spread(gentle, sole, jp.zeros(2), 0.0)) <= 0.02


def test_support_spread_rejects_a_foot_beside_a_box():
    """A 10 cm box from x = 0.039, just short of the node at x = 0.04. The
    spawn grid reads its top from the node at x = 0 on. One foot stands on
    the top. The other lies across x in [0.005, 0.035], over the lower
    ground. Every sole point reads the top on the spawn grid, so its
    spread there is 0. The lookup reads the drop."""
    xs = _axes()
    t = _tables(np.tile(np.where(xs >= 0.039, 0.1, 0.0), (N, 1)))
    ys = np.linspace(-0.08, 0.08, 5)
    sole = jp.asarray(
        [(x, y) for x in (0.005, 0.02, 0.035) for y in ys] + [(x, y) for x in (0.2, 0.215, 0.23) for y in ys],
        jp.float32,
    )
    on_spawn_grid = np.asarray(tg.height(t, sole, "spawn"))
    np.testing.assert_allclose(on_spawn_grid, 0.1, atol=1e-6)
    assert float(tg.support_spread(t, sole, jp.zeros(2), 0.0)) > 0.06


def test_support_spread_rejects_a_foot_the_spawn_grid_lifts():
    """A 10 cm step from x = 0.079, just short of the node at x = 0.08. The
    lookup reads 0 up to x = 0.04. The spawn grid reads the top from the
    node at x = 0.04 on. One foot lies across x in [0, 0.04] on the lower
    ground. Its lookup spread is 0, and its spawn-grid spread is the full
    step. spawn_z sets the base from the spawn grid, so that foot would
    hang."""
    xs = _axes()
    t = _tables(np.tile(np.where(xs >= 0.079, 0.1, 0.0), (N, 1)))
    ys = np.linspace(-0.08, 0.08, 5)
    sole = jp.asarray(
        [(x, y) for x in (0.0, 0.02, 0.04) for y in ys] + [(x, y) for x in (-0.3, -0.285, -0.27) for y in ys],
        jp.float32,
    )
    on_lookup = np.asarray(tg.height(t, sole))
    assert on_lookup.max() - on_lookup.min() == 0.0
    on_spawn_grid = np.asarray(tg.height(t, sole, "spawn"))
    assert on_spawn_grid.max() - on_spawn_grid.min() >= 0.1 - 1e-6
    assert float(tg.support_spread(t, sole, jp.zeros(2), 0.0)) >= 0.1 - 1e-6


# -- draws ---------------------------------------------------------------------


def test_first_level_lies_in_the_lower_fraction_unless_pinned():
    keys = jax.random.split(jax.random.PRNGKey(0), 4000)
    levels = np.asarray(jax.vmap(lambda k: tg.first_level(k, 10, 0.5, -1))(keys))
    assert set(levels.tolist()) == {0, 1, 2, 3, 4}
    levels = np.asarray(jax.vmap(lambda k: tg.first_level(k, 2, 0.5, -1))(keys))
    assert set(levels.tolist()) == {0}
    pinned = np.asarray(jax.vmap(lambda k: tg.first_level(k, 10, 0.5, 7))(keys))
    assert set(pinned.tolist()) == {7}


def test_pad_jitter_is_bounded_per_axis():
    keys = jax.random.split(jax.random.PRNGKey(1), 20000)
    origin = jp.array([1.0, -2.0])
    xy = np.asarray(jax.vmap(lambda k: tg.pad_draw(k, origin, 0.15))(keys)) - np.asarray(origin)
    assert np.abs(xy).max() <= 0.15
    # A square: both axes reach near the jitter in the same draw.
    assert (np.abs(xy).min(axis=1) > 0.14).any()


def _mixed_tables():
    """x < 0 flat, x >= 0 a 10 degree slope."""
    xs = _axes()
    return _tables(np.tile(np.where(xs < 0.0, 0.0, np.tan(np.radians(10.0)) * xs), (N, 1)))


def test_feature_draw_takes_the_first_level_candidate():
    t = _mixed_tables()
    sole = _sole()

    @jax.jit
    def draw(rng):
        xy, ok = tg.feature_draw(rng, t, 3, 1, 0.3, 1.5, 8, 0.02, sole)
        cands = jax.random.uniform(rng, (8, 2), minval=-1.5, maxval=1.5)
        spread = jax.vmap(lambda c: tg.support_spread(t, sole, c, 0.3))(cands)
        return xy, ok, cands, spread

    xy, ok, cands, spread = (np.asarray(a) for a in jax.vmap(draw)(jax.random.split(jax.random.PRNGKey(0), 50)))
    good = spread <= 0.02
    assert good.any(axis=1).any() and not good.any(axis=1).all()
    np.testing.assert_array_equal(ok, good.any(axis=1))
    first = cands[np.arange(50), good.argmax(axis=1)]
    np.testing.assert_array_equal(xy[ok], first[ok])


def test_feature_draw_falls_back_to_the_pad():
    t = _tables(_slope(10.0))
    rng = jax.random.PRNGKey(3)
    _, ok = jax.jit(lambda k: tg.feature_draw(k, t, 0, 1, 0.0, 1.5, 8, 0.02, _sole()))(rng)
    assert not bool(ok)

    def draw(feature):
        return jax.jit(lambda k: tg.spawn_draw(k, t, 0, 1, feature=feature, u_sole=_sole(), **SPAWN_KW))(rng)

    xy, yaw, kind = draw(True)
    assert int(kind) == tg.SPAWN_FALLBACK
    pad, pad_yaw, pad_kind = draw(False)
    assert int(pad_kind) == tg.SPAWN_PAD
    np.testing.assert_array_equal(np.asarray(xy), np.asarray(pad))
    assert float(yaw) == float(pad_yaw)


def test_feature_candidates_stay_inside_the_tile():
    t = _tables(np.zeros((N, N)))
    keys = jax.random.split(jax.random.PRNGKey(4), 2000)
    xy, _, kind = jax.jit(
        jax.vmap(lambda k: tg.spawn_draw(k, t, 2, 1, feature=True, u_sole=_sole(), **SPAWN_KW))
    )(keys)
    assert np.all(np.asarray(kind) == tg.SPAWN_FEATURE)
    assert np.abs(np.asarray(xy)).max() <= SPAWN_KW["half_extent"]
    assert np.abs(np.asarray(xy)).max() > 0.9 * SPAWN_KW["half_extent"]


@pytest.mark.parametrize("flat_row", [True, False])
def test_only_the_flat_row_uses_the_pad(flat_row):
    """Level 0 takes the pad only when it is the flat row. Without a flat
    row it is a terrain row like any other."""
    t = _tables(np.zeros((N, N)), flat_row=flat_row)
    keys = jax.random.split(jax.random.PRNGKey(5), 200)
    draw = jax.jit(
        jax.vmap(
            lambda k, level: tg.spawn_draw(k, t, 4, level, feature=True, u_sole=_sole(), **SPAWN_KW),
            in_axes=(0, None),
        )
    )
    xy, _, kind = draw(keys, 0)
    if flat_row:
        assert np.all(np.asarray(kind) == tg.SPAWN_PAD)
        assert np.abs(np.asarray(xy)).max() <= SPAWN_KW["pad_jitter"]
    else:
        assert np.all(np.asarray(kind) == tg.SPAWN_FEATURE)
    _, _, kind = draw(keys, 1)
    assert np.all(np.asarray(kind) == tg.SPAWN_FEATURE)


def test_spawn_rng_cost_is_fixed():
    """A key gives the same yaw and pad point in every mode, on every tile
    and whatever the outcome."""
    flat = _tables(np.zeros((N, N)), flat_row=True)
    steep = _tables(_slope(10.0), flat_row=True)
    rng = jax.random.PRNGKey(6)
    ref_xy, ref_yaw, _ = tg.spawn_draw(rng, flat, 0, 1, feature=False, u_sole=_sole(), **SPAWN_KW)
    for t, level, feature in ((flat, 0, True), (steep, 1, True), (steep, 1, False), (flat, 1, False)):
        xy, yaw, kind = tg.spawn_draw(rng, t, 5, level, feature=feature, u_sole=_sole(), **SPAWN_KW)
        assert float(yaw) == float(ref_yaw)
        if int(kind) != tg.SPAWN_FEATURE:
            np.testing.assert_array_equal(np.asarray(xy), np.asarray(ref_xy))
    no_yaw = {**SPAWN_KW, "yaw_enable": False}
    xy, yaw, _ = tg.spawn_draw(rng, flat, 0, 1, feature=False, u_sole=_sole(), **no_yaw)
    assert float(yaw) == 0.0
    np.testing.assert_array_equal(np.asarray(xy), np.asarray(ref_xy))


def test_yaw_composes_on_the_left_of_the_reset_quat():
    """yaw_quat(psi) * q turns the pose q about world z by psi: a vector
    rotated by the product is the q-rotated vector turned by psi."""
    q = jp.asarray(np.array([0.9, 0.1, -0.3, 0.2]) / np.linalg.norm([0.9, 0.1, -0.3, 0.2]))
    assert np.array_equal(np.asarray(tg.quat_mul(tg.yaw_quat(jp.float32(0.0)), q)), np.asarray(q))

    def rot(quat, v):
        w, x, y, z = (float(c) for c in quat)
        m = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
            ]
        )
        return m @ v

    psi = 0.83
    v = np.array([0.3, -0.5, 0.8])
    turned = rot(tg.quat_mul(tg.yaw_quat(jp.float32(psi)), q), v)
    c, s = math.cos(psi), math.sin(psi)
    want = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ rot(q, v)
    np.testing.assert_allclose(turned, want, atol=1e-6)
