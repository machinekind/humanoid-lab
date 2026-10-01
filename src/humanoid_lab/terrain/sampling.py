"""Bilinear sampling of a regular height grid, on the host or on device."""

from __future__ import annotations

import numpy as np


def bilinear(grid, x0: float, dx: float, y0: float, dy: float, x, y, xp=np):
    """Sample `grid` (rows index y, columns index x) at world (x, y).

    Node (r, c) sits at (x0 + c * dx, y0 + r * dy). Points off the grid
    clamp to its edge. `xp` is numpy or jax.numpy. Host and device code
    sample through this one body. Every height read therefore agrees with
    the arena it came from. `grid` may be a numpy array. With jax.numpy it
    becomes a device constant."""
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
