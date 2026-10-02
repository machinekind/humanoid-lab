"""Critic height scan: ground heights on a grid around the base, relative
to the lowest sole.

Pure. Every helper takes `xp`, numpy or jax.numpy. The caller supplies the
ground reader and the reference height, so nothing here needs a model.

The grid lies in the base's yaw frame: x along the heading, y to its left,
level whatever the base's pitch and roll. It has NX x NY points at a 0.1 m
pitch, flattened with index ix * NY + iy, so x is constant over each run of
NY. The extents are constants, not config. The critic's width is then a
property of the code, and an env without a terrain block serves the same
width. None of the values has been tuned by training.

- Ahead, 0.9 m. That is 0.9 s at roboto_origin's 1.0 m/s top forward
  command, or three of the arena's default 0.30 m treads.
- Behind, 0.4 m. roboto_origin's soles reach 0.117 m behind the base at
  its reset pose, and its backward command reaches 0.6 m/s.
- Across, 0.3 m each side. roboto_origin's soles reach 0.10 m each side,
  and its lateral command reaches 0.5 m/s.
- Pitch, 0.1 m. A straight approach puts three rows on each 0.30 m tread.
"""

from __future__ import annotations

import numpy as np

X_RANGE = (-0.4, 0.9)  # m, along the heading
NX = 14
Y_RANGE = (-0.3, 0.3)  # m, across it, +y to the left
NY = 7
SIZE = NX * NY
CLIP = 0.5  # m, each value is clipped to +-CLIP

# The obs catalog name. Critic only: the robot has no source for it.
NAME = "height_scan_clean"


def body_grid(xp=np):
    """(SIZE, 2) yaw-frame points, index ix * NY + iy."""
    xs = xp.linspace(X_RANGE[0], X_RANGE[1], NX)
    ys = xp.linspace(Y_RANGE[0], Y_RANGE[1], NY)
    return xp.stack([xp.repeat(xs, NY), xp.tile(ys, NX)], axis=-1)


def yaw_from_quat(quat, xp=np):
    """Heading of a (w, x, y, z) quaternion: the angle of the body x axis
    projected on the ground, from world +x toward +y."""
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    return xp.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def world_xy(grid, base_xy, yaw, xp=np):
    """Yaw-frame `grid` (..., 2) turned by `yaw` about +z, then moved to
    `base_xy`. World xy (..., 2)."""
    c, s = xp.cos(yaw), xp.sin(yaw)
    gx, gy = grid[..., 0], grid[..., 1]
    return xp.stack([c * gx - s * gy, s * gx + c * gy], axis=-1) + base_xy


def scan_values(heights, ref_z, clip=CLIP, xp=np):
    """Ground `heights` less `ref_z`, clipped to +-clip."""
    return xp.clip(heights - ref_z, -clip, clip)
