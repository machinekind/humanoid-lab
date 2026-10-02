"""envs/height_scan.py: the critic scan's grid, placement, values and
normalizer prior. Model-free."""

from __future__ import annotations

import math

import jax
import jax.numpy as jp
import numpy as np
import pytest

from humanoid_lab import terrain
from humanoid_lab.envs import height_scan as hs
from humanoid_lab.envs import terrain_geometry as tg


def _quat(axis, angle):
    """(w, x, y, z) rotation by `angle` about unit `axis`."""
    half = angle / 2
    return np.array([math.cos(half), *(math.sin(half) * np.asarray(axis, float))])


def _mul(a, b):
    """Hamilton product a * b: b first, then a."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ]
    )


def test_grid_shape_ordering_and_endpoints():
    grid = hs.body_grid()
    assert grid.shape == (hs.SIZE, 2)
    xs = np.linspace(-0.4, 0.9, 14)
    ys = np.linspace(-0.3, 0.3, 7)
    np.testing.assert_allclose(np.diff(xs), 0.1)
    np.testing.assert_allclose(np.diff(ys), 0.1)
    # Index ix * NY + iy: x is constant over each run of NY.
    for ix in range(hs.NX):
        for iy in range(hs.NY):
            np.testing.assert_allclose(grid[ix * hs.NY + iy], (xs[ix], ys[iy]), atol=1e-12)
    np.testing.assert_allclose(grid[0], (-0.4, -0.3))
    np.testing.assert_allclose(grid[-1], (0.9, 0.3))


def test_size_matches_the_grid():
    assert (hs.NX, hs.NY, hs.SIZE) == (14, 7, 98)
    assert hs.SIZE == hs.NX * hs.NY == len(hs.body_grid())
    assert hs.NAME == "height_scan_clean"
    np.testing.assert_allclose(np.asarray(hs.body_grid(jp)), hs.body_grid(), atol=1e-6)


def test_world_placement_rotates_by_yaw_then_translates():
    grid = np.array([[1.0, 0.0], [0.0, 1.0]])
    out = hs.world_xy(grid, np.array([2.0, -1.0]), np.pi / 2)
    np.testing.assert_allclose(out, [[2.0, 0.0], [1.0, -1.0]], atol=1e-12)

    # The farthest point ahead on the centre line lands 0.9 m along the
    # heading, and the left end of the nearest row 0.3 m to its left.
    yaw = 0.7
    fwd = np.array([math.cos(yaw), math.sin(yaw)])
    left = np.array([-math.sin(yaw), math.cos(yaw)])
    base = np.array([-3.0, 4.0])
    placed = hs.world_xy(hs.body_grid(), base, yaw)
    np.testing.assert_allclose(placed[(hs.NX - 1) * hs.NY + hs.NY // 2], base + 0.9 * fwd, atol=1e-12)
    np.testing.assert_allclose(placed[hs.NY - 1], base - 0.4 * fwd + 0.3 * left, atol=1e-12)

    on_device = hs.world_xy(jp.asarray(hs.body_grid(jp)), jp.asarray(base), jp.float32(yaw), jp)
    np.testing.assert_allclose(np.asarray(on_device), placed, atol=1e-5)


@pytest.mark.parametrize("yaw", [0.0, 0.5, -2.0, 3.0])
def test_yaw_is_the_heading_whatever_the_pitch_and_roll(yaw):
    """The heading of the body x axis projected on the ground."""
    tilt = _mul(_quat((0, 1, 0), 0.4), _quat((1, 0, 0), -0.3))
    quat = _mul(_quat((0, 0, 1), yaw), tilt)
    assert hs.yaw_from_quat(quat) == pytest.approx(yaw, abs=1e-12)
    assert float(hs.yaw_from_quat(jp.asarray(quat, jp.float32), jp)) == pytest.approx(yaw, abs=1e-6)
    batch = np.stack([quat, _quat((0, 0, 1), 1.0)])
    np.testing.assert_allclose(hs.yaw_from_quat(batch), [yaw, 1.0], atol=1e-12)


def test_scan_values_are_relative_and_clipped():
    heights = np.array([0.7, -0.7, 0.02, 0.51])
    np.testing.assert_allclose(hs.scan_values(heights, 0.01), [0.5, -0.5, 0.01, 0.5], atol=1e-12)
    np.testing.assert_allclose(hs.scan_values(heights, 0.01, clip=0.3), [0.3, -0.3, 0.01, 0.3], atol=1e-12)
    assert hs.CLIP == 0.5


def test_the_prior_is_the_scan_spread_on_the_default_arena():
    """PRIOR_MEAN and PRIOR_STD re-measured on the default arena. The base
    sits at 100 points per tile, uniform over the disc of the tile's
    feature radius about its centre, each with a uniform yaw. The reference
    is the lowest ground under a fixed 3 x 3 footprint at +-0.1 m, not
    roboto_origin's sole points. That raises the mean by about 0.004 m."""
    t = tg.tables_from_arena(terrain.generate(terrain.ArenaParams()))
    rng = np.random.default_rng(0)
    radius = np.asarray(t.feature_r)
    shape = (*radius.shape, 100)
    r = radius[..., None] * np.sqrt(rng.uniform(size=shape))
    theta = rng.uniform(-np.pi, np.pi, shape)
    offset = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=-1)
    base = jp.asarray(np.asarray(t.origin_xy)[..., None, :] + offset)[..., None, :]
    yaw = jp.asarray(rng.uniform(-np.pi, np.pi, shape))[..., None]
    side = np.array([-0.1, 0.0, 0.1])
    footprint = jp.asarray(np.stack(np.meshgrid(side, side, indexing="ij"), axis=-1).reshape(-1, 2))

    # Jitted, it runs in 0.2 s. Eager, it takes 3 s.
    @jax.jit
    def scan_at(base, yaw):
        ref = tg.ground(t, hs.world_xy(footprint, base, yaw, jp)).min(axis=-1)
        heights = tg.ground(t, hs.world_xy(hs.body_grid(jp), base, yaw, jp))
        return hs.scan_values(heights, ref[..., None], xp=jp)

    scan = np.asarray(scan_at(base, yaw))
    assert abs(scan.std() - hs.PRIOR_STD) < 0.15 * scan.std()
    assert abs(scan.mean() - hs.PRIOR_MEAN) < 0.01
