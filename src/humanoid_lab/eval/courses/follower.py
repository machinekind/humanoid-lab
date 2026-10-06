"""The frozen pure-pursuit follower that turns a robot pose into a course command.

The step is jax.numpy on fixed-shape arrays, so one function runs eagerly in
the unit tests, inside a jitted course lane and under vmap. Nothing branches
in Python on a traced value. The host helpers (densify, pack_path,
padded_points) build the path arrays in float64 and cast them to float32
once.

The constants are the benchmark's. Changing any of them changes what every
recorded score means: families.catalogue_fingerprint hashes them and
tests/unit/test_courses.py asserts them. The yaw cap is the one per-robot
input (spec.CourseParams.yaw_cap).

The follower is non-holonomic: vy is always exactly 0. One that strafed
would let a robot crab through the turning rows without turning.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jp
import numpy as np

from humanoid_lab.eval.courses.spec import PathCourse

# 0.8 s of preview at 0.5 m/s, about one stride. A 0.24 m lookahead raises the
# follower's accuracy on curves but turns harder after a kick; a trained
# Roboto policy fell on straight_push with it.
LOOKAHEAD_M = 0.40
# Spin-in-place hysteresis: enter above 60 deg of heading error, leave below
# 20 deg. With a single threshold, an error sitting on it flips the command
# between walking and spinning every step, which a policy averages to a
# stand.
SPIN_ENTER_RAD = 1.05
SPIN_EXIT_RAD = 0.35
K_YAW_SPIN = 1.5  # 1/s, proportional yaw gain in the spin branch
# Wider than either robot's foot span.
GOAL_RADIUS_M = 0.25
# A closed course ends where it starts, so the goal radius alone would
# complete it on the lead-in. Completion also needs this close to the end of
# the course's arclength.
GOAL_MIN_PROGRESS_M = 2 * LOOKAHEAD_M
RESAMPLE_DS_M = 0.02
# The closest-point search looks only this far ahead of the last match, so
# progress is monotone and a self-crossing (the figure-eight's) cannot snap
# it back. Longer than one step at the fastest row (0.018 m at 0.9 m/s),
# shorter than the figure-eight's 9.42 m between its two passes of the
# crossing.
PROGRESS_WINDOW_M = 1.0

WINDOW_PTS = round(PROGRESS_WINDOW_M / RESAMPLE_DS_M)
LOOK_PTS = round(LOOKAHEAD_M / RESAMPLE_DS_M)
# Copies of the last point appended to every path, so the search window and
# the lookahead index never run past the array (dynamic_slice would clamp
# the window instead), and argmin still picks the real last point first.
PAD_PTS = WINDOW_PTS + LOOK_PTS + 1


class PathArrays(NamedTuple):
    """A densified, padded path: (P, 2) points and unit tangents, (P,) speed
    and arclength, and the () total length."""

    pts: jax.Array
    tan: jax.Array
    speed: jax.Array
    cum: jax.Array
    total: jax.Array


class FollowerState(NamedTuple):
    i: jax.Array  # int32, index of the last closest point; never decreases
    spinning: jax.Array  # bool, the hysteresis state


class FollowerStep(NamedTuple):
    cmd: jax.Array  # (3,) [vx, 0, wz]
    state: FollowerState
    xte: jax.Array  # signed cross-track error, m, positive left of the path
    s: jax.Array  # arclength progress, m
    reached: jax.Array  # bool: inside the goal radius and near the end


def wrap_angle(a):
    """`a` wrapped into [-pi, pi)."""
    return (a + jp.pi) % (2 * jp.pi) - jp.pi


def quat_to_yaw(q_wxyz):
    """Yaw of a (w, x, y, z) quaternion, rad."""
    w, x, y, z = q_wxyz[..., 0], q_wxyz[..., 1], q_wxyz[..., 2], q_wxyz[..., 3]
    return jp.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


# -- host-side path construction ---------------------------------------------


def densify(waypoints, segment_speeds) -> tuple[np.ndarray, np.ndarray]:
    """Waypoints resampled about RESAMPLE_DS_M apart, float64.

    Each segment contributes its start and evenly spaced interior points; the
    final waypoint closes the list. Every point carries the commanded speed
    of the segment it starts.
    """
    wp = np.asarray(waypoints, dtype=float)
    sp = np.asarray(segment_speeds, dtype=float)
    pts, spd = [], []
    for k in range(len(wp) - 1):
        a, b = wp[k], wp[k + 1]
        n = max(1, round(float(np.hypot(*(b - a))) / RESAMPLE_DS_M))
        t = np.arange(n) / n
        pts.append(a + np.outer(t, b - a))
        spd.append(np.full(n, sp[k]))
    pts.append(wp[-1:])
    spd.append(sp[-1:])
    return np.concatenate(pts, axis=0), np.concatenate(spd, axis=0)


def dense_points(course: PathCourse) -> int:
    return len(densify(course.waypoints, course.segment_speeds)[0])


def padded_points(courses) -> int:
    """The padded length every path of `courses` fits in: the longest
    densified path plus PAD_PTS. Spin rows have no path and count as one
    point, so a spin-only set still gets a valid shape."""
    longest = max(
        (dense_points(c) for c in courses if isinstance(c, PathCourse)), default=1
    )
    return longest + PAD_PTS


def pack_path(course: PathCourse, n_points: int) -> PathArrays:
    """`course`'s path in its course frame, padded to `n_points`, float32.

    Tangents, arclengths and the total are computed in float64 on the host
    before the cast, so float32 rounding does not accumulate along the path.
    """
    pts, speed = densify(course.waypoints, course.segment_speeds)
    if n_points < len(pts) + PAD_PTS:
        raise ValueError(
            f"{course.name}: {len(pts)} points need n_points >= {len(pts) + PAD_PTS}, "
            f"got {n_points}"
        )
    d = np.diff(pts, axis=0, append=pts[-1:] + (pts[-1:] - pts[-2:-1]))
    tan = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
    cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    pad = n_points - len(pts)

    def f32(a):
        return jp.asarray(np.concatenate([a, np.repeat(a[-1:], pad, axis=0)]), dtype=jp.float32)

    return PathArrays(
        pts=f32(pts), tan=f32(tan), speed=f32(speed), cum=f32(cum),
        total=jp.asarray(cum[-1], dtype=jp.float32),
    )


def dummy_path(n_points: int) -> PathArrays:
    """The path a spin lane carries so it shares the path lanes' shapes.
    Finite everywhere; the lane never reads the follower on a spin lane."""
    z = jp.zeros(n_points, dtype=jp.float32)
    return PathArrays(
        pts=jp.zeros((n_points, 2), jp.float32),
        tan=jp.stack([jp.ones(n_points, jp.float32), z], axis=-1),
        speed=z, cum=z, total=jp.zeros((), jp.float32),
    )


# -- the follower --------------------------------------------------------------


def layout(path: PathArrays, x0, y0, yaw0) -> PathArrays:
    """`path` moved from its course frame to the frame whose origin is
    (x0, y0) with +x along yaw0. Speeds and arclengths do not change."""
    c, s = jp.cos(yaw0), jp.sin(yaw0)

    def rot(v):
        return jp.stack([c * v[:, 0] - s * v[:, 1], s * v[:, 0] + c * v[:, 1]], axis=-1)

    return path._replace(pts=rot(path.pts) + jp.stack([x0, y0]), tan=rot(path.tan))


def initial_state() -> FollowerState:
    return FollowerState(i=jp.int32(0), spinning=jp.array(False))


def follower_step(path: PathArrays, state: FollowerState, x, y, yaw, yaw_cap) -> FollowerStep:
    """One follower decision at pose (x, y, yaw), in `path`'s frame.

    1. The closest point within PROGRESS_WINDOW_M ahead of the last match.
    2. Cross-track error against that point's tangent, positive to the left.
    3. The lookahead point LOOKAHEAD_M further on; alpha is the heading error
       toward it.
    4. Hysteresis: spinning continues while |alpha| >= SPIN_EXIT_RAD; walking
       turns into spinning once |alpha| > SPIN_ENTER_RAD.
    5. Spinning: vx = 0, wz = K_YAW_SPIN alpha. Walking: vx = v cos(alpha) at
       the closest point's segment speed v, and the pure-pursuit
       wz = 2 vx sin(alpha) / LOOKAHEAD_M. wz is clipped to +-yaw_cap.
    6. Reached: inside GOAL_RADIUS_M of the last point, with the progress
       index within GOAL_MIN_PROGRESS_M of the end.
    """
    p = jp.stack([x, y])
    seg = jax.lax.dynamic_slice(path.pts, (state.i, 0), (WINDOW_PTS + 1, 2))
    i = state.i + jp.argmin(jp.sum((seg - p) ** 2, axis=1)).astype(jp.int32)

    off = p - path.pts[i]
    tan = path.tan[i]
    xte = tan[0] * off[1] - tan[1] * off[0]

    look = path.pts[jp.minimum(path.pts.shape[0] - 1, i + LOOK_PTS)]
    alpha = wrap_angle(jp.arctan2(look[1] - p[1], look[0] - p[0]) - yaw)

    spinning = jp.where(
        state.spinning, jp.abs(alpha) >= SPIN_EXIT_RAD, jp.abs(alpha) > SPIN_ENTER_RAD
    )
    vx_walk = path.speed[i] * jp.cos(alpha)
    vx = jp.where(spinning, 0.0, vx_walk)
    wz = jp.clip(
        jp.where(spinning, K_YAW_SPIN * alpha, 2.0 * vx_walk * jp.sin(alpha) / LOOKAHEAD_M),
        -yaw_cap,
        yaw_cap,
    )

    s = path.cum[i]
    to_end = path.pts[-1] - p
    reached = (s >= path.total - GOAL_MIN_PROGRESS_M) & (
        jp.hypot(to_end[0], to_end[1]) < GOAL_RADIUS_M
    )
    return FollowerStep(
        cmd=jp.stack([vx, jp.zeros_like(vx), wz]),
        state=FollowerState(i=i, spinning=spinning),
        xte=xte,
        s=s,
        reached=reached,
    )


@jax.jit
def perfect_unicycle(path: PathArrays, yaw_cap, budget_steps, dt) -> dict:
    """The follower driving a unicycle that executes (vx, wz) exactly.

    Starts at the course origin, heading +x, and integrates one explicit
    Euler step of `dt` per command for at most `budget_steps` steps. The goal
    is tested on every pose, the one after the last step included, as in a
    course lane. Returns `completed`, `steps`, `xte_rms_m` (over the poses
    that chose a command), `peak_wz` and `spin_steps`.

    Not a ceiling on what a robot can score: pure pursuit cuts curves to the
    inside, and a robot whose yaw lags is pushed back toward the path.
    """
    zero = jp.zeros((), jp.float32)
    fo0 = follower_step(path, initial_state(), zero, zero, zero, yaw_cap)

    def cond(c):
        t, _, _, _, fo, _, _, _ = c
        return ~fo.reached & (t < budget_steps)

    def body(c):
        t, x, y, yaw, fo, sq, peak, spins = c
        vx, wz = fo.cmd[0], fo.cmd[2]
        x = x + vx * jp.cos(yaw) * dt
        y = y + vx * jp.sin(yaw) * dt
        yaw = yaw + wz * dt
        nxt = follower_step(path, fo.state, x, y, yaw, yaw_cap)
        return (t + 1, x, y, yaw, nxt, sq + fo.xte * fo.xte,
                jp.maximum(peak, jp.abs(wz)), spins + fo.state.spinning.astype(jp.int32))

    init = (jp.int32(0), zero, zero, zero, fo0, zero, zero, jp.int32(0))
    t, _, _, _, fo, sq, peak, spins = jax.lax.while_loop(cond, body, init)
    return {
        "completed": fo.reached,
        "steps": t,
        "xte_rms_m": jp.sqrt(sq / jp.maximum(t, 1)),
        "peak_wz": peak,
        "spin_steps": spins,
    }
