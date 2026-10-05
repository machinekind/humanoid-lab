"""Polyline builders for course waypoints.

numpy only. Every builder returns a (K, 2) float array in whatever frame its
arguments are given in, so the same shapes serve a course laid out from the
robot's post-settle pose and one placed in a world frame. The point counts
are part of a row's definition: the follower resamples between waypoints in
straight chords, so a coarser arc is a different course.
"""

from __future__ import annotations

import math

import numpy as np

# Straight run-up that lead_in() puts before a shape, so a curve never starts
# on the robot's first step: 2.0-2.5 s of walking at the nominal speeds.
LEAD_IN_M = 1.0


def line(length, start=(0.0, 0.0), heading=0.0) -> np.ndarray:
    """Two-point segment of `length` from `start` along `heading` (rad)."""
    x0, y0 = start
    return np.array(
        [[x0, y0], [x0 + length * math.cos(heading), y0 + length * math.sin(heading)]],
        dtype=float,
    )


def arc(radius, sweep, start=(0.0, 0.0), heading=0.0, n=64) -> np.ndarray:
    """`n` points of a constant-radius arc leaving `start` tangent to `heading`.

    `sweep` is signed in radians: positive turns left (CCW), negative right.
    The first point is `start`; join() drops it where it repeats the end of
    the previous part.
    """
    phi = np.linspace(0.0, abs(sweep), n)
    s = math.copysign(1.0, sweep)
    # Local frame: heading along +x, turn centre at (0, s * radius).
    lx = radius * np.sin(phi)
    ly = s * radius * (1.0 - np.cos(phi))
    c, sn = math.cos(heading), math.sin(heading)
    return np.stack([start[0] + c * lx - sn * ly, start[1] + sn * lx + c * ly], -1)


def join(*parts) -> np.ndarray:
    """Concatenate polylines, dropping each shared endpoint once."""
    out = [np.asarray(parts[0], dtype=float)]
    for p in parts[1:]:
        p = np.asarray(p, dtype=float)
        if np.allclose(out[-1][-1], p[0], atol=1e-9):
            p = p[1:]
        out.append(p)
    return np.concatenate(out, axis=0)


def sine_slalom(length, amplitude, wavelength, n=128) -> np.ndarray:
    """y = amplitude * sin(2 pi x / wavelength), `n` points over [0, length]."""
    x = np.linspace(0.0, length, n)
    return np.stack([x, amplitude * np.sin(2 * np.pi * x / wavelength)], -1)


def lead_in(shape) -> np.ndarray:
    """`shape` (starting at the origin, heading +x) behind LEAD_IN_M of straight."""
    return join(line(LEAD_IN_M), np.asarray(shape, dtype=float) + np.array([LEAD_IN_M, 0.0]))


def circle(radius, n=128) -> np.ndarray:
    """One full left circle of `radius` behind the lead-in.

    The last waypoint is the circle's start, so completion also needs
    progress along the course (follower.GOAL_MIN_PROGRESS_M).
    """
    return lead_in(arc(radius, 2 * math.pi, n=n))


def rounded_square(side, r, n=24) -> np.ndarray:
    """A left-turning square of `side` with each corner an arc of radius `r`.

    Starts at the origin heading +x and ends at (r, 0) heading +x after the
    fourth corner, tangent-continuous throughout. Its length is
    4 side - 7 r + 2 pi r. A sharp corner asks for an instant 90 deg turn,
    which no follower executes, so the follower's own error there would be
    the size of a policy's.

    `side` must exceed 2 r. At 2 r the edges vanish and the shape is a
    circle; below it they run backwards.
    """
    if side - 2 * r <= 0:
        raise ValueError(
            f"rounded_square: side {side} m leaves no straight edge between "
            f"corners of radius {r} m; it needs side > 2 r"
        )
    parts = [line(side - r)]
    x, y, h = side - r, 0.0, 0.0
    for k in range(4):
        a = arc(r, math.pi / 2, start=(x, y), heading=h, n=n)
        parts.append(a)
        x, y = a[-1]
        h += math.pi / 2
        if k < 3:
            edge = line(side - 2 * r, start=(x, y), heading=h)
            parts.append(edge)
            x, y = edge[-1]
    return join(*parts)
