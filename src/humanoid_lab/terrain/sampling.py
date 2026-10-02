"""Height reads of a regular node grid, on the host or on device.

`bilinear` reads the lookup. `triangle` reads the heightfield as MuJoCo
collides with it.
"""

from __future__ import annotations

import numpy as np


def bilinear(grid, x0: float, dx: float, y0: float, dy: float, x, y, xp=np):
    """Sample `grid` (rows index y, columns index x) at world (x, y).

    Node (r, c) sits at (x0 + c * dx, y0 + r * dy). Points off the grid
    clamp to its edge. `xp` is numpy or jax.numpy. Host and device code
    read the lookup through this one body. Every lookup read therefore
    agrees with the arena it came from. `grid` may be a numpy array. With
    jax.numpy it becomes a device constant."""
    grid = xp.asarray(grid)
    nr, nc = grid.shape
    fx = xp.clip((x - x0) / dx, 0.0, nc - 1)
    fy = xp.clip((y - y0) / dy, 0.0, nr - 1)
    c0 = xp.floor(fx).astype(xp.int32)
    r0 = xp.floor(fy).astype(xp.int32)
    c1 = xp.minimum(c0 + 1, nc - 1)
    r1 = xp.minimum(r0 + 1, nr - 1)
    tx = fx - c0
    ty = fy - r0
    return (
        grid[r0, c0] * (1 - tx) * (1 - ty)
        + grid[r0, c1] * tx * (1 - ty)
        + grid[r1, c0] * (1 - tx) * ty
        + grid[r1, c1] * tx * ty
    )


def triangle(grid, x0: float, dx: float, y0: float, dy: float, x, y, xp=np):
    """Sample the triangulated surface over `grid` at world (x, y).

    MuJoCo splits each heightfield cell on the diagonal from node (r, c) to
    node (r + 1, c + 1). This reads that surface: each point reads the
    plane through the three nodes of its triangle. The node layout, the
    clamp and `xp` are as in `bilinear`. The grid needs at least two nodes
    per axis."""
    grid = xp.asarray(grid)
    nr, nc = grid.shape
    fx = xp.clip((x - x0) / dx, 0.0, nc - 1)
    fy = xp.clip((y - y0) / dy, 0.0, nr - 1)
    c0 = xp.minimum(xp.floor(fx), nc - 2).astype(xp.int32)
    r0 = xp.minimum(xp.floor(fy), nr - 2).astype(xp.int32)
    tx = fx - c0
    ty = fy - r0
    h00 = grid[r0, c0]
    h01 = grid[r0, c0 + 1]
    h10 = grid[r0 + 1, c0]
    h11 = grid[r0 + 1, c0 + 1]
    return xp.where(
        tx >= ty,
        h00 + tx * (h01 - h00) + ty * (h11 - h01),
        h00 + ty * (h10 - h00) + tx * (h11 - h10),
    )
