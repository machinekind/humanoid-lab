"""Course lanes: one (row, seed) rollout as one jitted program on CPU jax.

A lane resets the env from its seed key, settles at zero command for
battery.SETTLE_SEC, lays the course out from the settled pose, and then steps
the policy under the follower's command until the goal, a fall, non-finite
physics or the row's budget. Everything runs inside one jax program: the
settle is a `lax.scan`, the course a `lax.while_loop`, and the follower is
jax.numpy (follower.py). Path and spin rows share the program: the row kind
is data, and a spin lane carries a dummy path of the path lanes' shape.

Execution. `make_lane_fn` builds the lane for one env, `compile_lane`
compiles it once on the calling thread, and `run_lanes` calls the compiled
program from a thread pool, one lane per call. 16 lanes of a trained Roboto
policy (straight_10m and circle_r2, 8 seeds each, 6350 record steps, scored
in the workers) ran at 0.23-0.30 ms per env-step on 8 threads of a 10-core
Apple M2 Pro, against 1.22 ms on one thread. The records were bit-identical
at 1, 4, 6, 8 and 10 threads. On the same CPU, vmapped lanes were slower per
env-step than one unbatched lane at every batch size tried (8, 32, 160): the
MJX solver's while-loop runs to the slowest lane, and every `lax.cond` in
the env becomes a select.

The stop test sits in the while-loop condition and the body always steps.
On XLA:CPU a `lax.cond` in the loop body copies the carried record on every
step. On the same CPU, a while-loop with trivial compute carrying a 0.5-8 MB
buffer took about 35-55 us per MB per iteration with a `cond`. It took
under 0.5 us without one. One lane of a trained Roboto policy on
straight_10m, with a 2.3 MB record and run on one thread, was 12-16% slower
per step with a `cond` in the body. Without the `cond` the record is
written in place.

Shapes. `max_steps` (the record length) and the padded path length come
from the whole catalogue of one ground class (`lane_shapes`), never from
the rows being run. A row subset, a friction group and the full catalogue
therefore compile the same program, and a lane's numbers do not depend on
which other rows ran.

Hooks. `reset_fn(key, data)`, `on_stop(state, data)` and
`observe(state, data)` let a batched executor reuse this lane under
`jax.vmap`; each defaults to None and costs nothing when absent. Under vmap
a finished lane's carry is frozen while the batch keeps stepping it, so
`on_stop` runs on the state at which the lane stops, once. `observe` returns
per-step scalars folded into `LaneOut.peaks` by max, over the settle and the
course. Nothing in the lane branches in Python on a traced value.

Friction. Floor rows rebuild the device model (`set_friction`) and need a
fresh `make_lane_fn` closure: env.step reads `env._mjx_model` when it is
traced, so a program compiled before the swap keeps the old friction.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any, NamedTuple

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from mujoco import mjx

from humanoid_lab.eval.courses import families, follower, model_ids
from humanoid_lab.eval.courses.follower import FollowerStep, PathArrays
from humanoid_lab.eval.courses.spec import (
    SPIN,
    Course,
    CourseParams,
    SpinCourse,
    budget_steps,
    kind_id,
)


class LaneData(NamedTuple):
    """One row, as the arrays a compiled lane takes. Every row of a ground
    class gives the same shapes and dtypes."""

    kind: jax.Array  # int32, spec.PATH or spec.SPIN
    path: PathArrays  # course frame; a spin lane carries follower.dummy_path
    budget: jax.Array  # int32 course steps; the lane cuts it to its max_steps
    push_at_m: jax.Array  # float32 follower progress at which the kick lands; inf for none
    push_vel: jax.Array  # float32 m/s, the kick to the left of the heading
    spin_wz: jax.Array  # float32 rad/s, the held spin command; 0 on a path row
    spin_required: jax.Array  # float32 rad of yaw progress that completes a spin
    anchor_world: jax.Array  # bool: lay the course out at `origin`, not the settled pose
    origin: jax.Array  # (3,) float32 world (x, y, yaw), read when anchor_world
    yaw_cap: jax.Array  # float32 rad/s, the follower's wz clip


class LaneOut(NamedTuple):
    """What a lane returns. Only the first `steps` rows of `rec` are
    meaningful. A lane that went non-finite on a course step also wrote that
    step's row, at index `steps`. A lane that went non-finite in the settle
    wrote no row, so `rec` stays all zeros. scoring.seed_out turns the stop
    flags into the seed's outcome."""

    steps: jax.Array  # int32 recorded course steps, finite ones only
    nonfinite: jax.Array  # bool: qpos went non-finite, in the settle or on the course
    settle_fell: jax.Array  # bool: done tripped during the settle
    fell: jax.Array  # bool: done tripped, in the settle or on the course
    reached: jax.Array  # bool: the goal test held on the last pose
    settle_height: jax.Array  # mean base height over the settle's second half, m
    yaw_rad: jax.Array  # yaw progress in the spin's direction over counted steps; 0 on a path row
    anchor: jax.Array  # (3,) world (x, y, yaw) of the course frame
    peaks: jax.Array  # observe's per-step values folded by max; (0,) without observe
    rec: dict  # name -> (max_steps, ...) per-step record
    state: Any  # the env state the lane stopped at, after on_stop


class LaneJob(NamedTuple):
    """One call of a compiled lane. `tag` is the caller's, for `finish`."""

    data: LaneData
    key: jax.Array
    tag: Any = None


def lane_shapes(params: CourseParams, dt: float, ground_class: str = "flat") -> tuple[int, int]:
    """(max_steps, n_points) for every lane of one robot and ground class:
    the class catalogue's largest budget and its longest padded path. Roboto
    Origin's flat class gives (6350, 1072)."""
    rows = list(families.catalogue(params, ground_class).values())
    return max(budget_steps(c, dt) for c in rows), follower.padded_points(rows)


def lane_data(course: Course, params: CourseParams, n_points: int, dt: float) -> LaneData:
    """`course` as lane inputs, with its path padded to `n_points`."""

    def f32(value):
        return jp.asarray(value, dtype=jp.float32)

    if isinstance(course, SpinCourse):
        path = follower.dummy_path(n_points)
        push_at, push_vel = math.inf, 0.0
        spin_wz, spin_required = float(course.wz), course.turn_rad
    else:
        path = follower.pack_path(course, n_points)
        push_at = math.inf if course.push_at_m is None else float(course.push_at_m)
        push_vel = 0.0 if course.push_vel is None else float(course.push_vel)
        spin_wz, spin_required = 0.0, 0.0
    return LaneData(
        kind=jp.asarray(kind_id(course), dtype=jp.int32),
        path=path,
        budget=jp.asarray(budget_steps(course, dt), dtype=jp.int32),
        push_at_m=f32(push_at),
        push_vel=f32(push_vel),
        spin_wz=f32(spin_wz),
        spin_required=f32(spin_required),
        anchor_world=jp.asarray(course.anchor == "world"),
        origin=f32((0.0, 0.0, 0.0) if course.origin is None else course.origin),
        yaw_cap=f32(params.yaw_cap),
    )


# -- env signals -----------------------------------------------------------------


def base_height(env, d):
    """Base height above the ground under the robot. An env with its own
    `_base_height` (a ground that is not flat) supplies it; otherwise it is
    the free joint's z, which is the height above the flat floor."""
    fn = getattr(env, "_base_height", None)
    return fn(d) if fn is not None else d.qpos[env._base_qadr + 2]


def pose(env, d):
    """Base (x, y, yaw) in the world. Yaw comes from the free joint's
    quaternion, not env._quat, which may read an IMU site."""
    b = env._base_qadr
    return d.qpos[b], d.qpos[b + 1], follower.quat_to_yaw(d.qpos[b + 3 : b + 7])


def signals(env, d) -> dict:
    """The physics half of a lane's per-step record, read through the env's
    own helpers, so a ground env's overrides apply."""
    foot_vel = env._foot_linvel(d)
    base_vel = d.qvel[env._base_vadr : env._base_vadr + 2]
    return {
        "v_fwd": env._local_linvel(d)[0],
        "v_planar": jp.hypot(base_vel[0], base_vel[1]),
        "h": base_height(env, d),
        "gyro_z": env._gyro(d)[2],
        "qvel_act": d.qvel[env._vadr],
        "foot_speed": jp.hypot(foot_vel[:, 0], foot_vel[:, 1]),
        "foot_vz": foot_vel[:, 2],
        "contact": env._foot_contact(d),
        "foot_clear": env._foot_clearance(d),
        "qpos": d.qpos,
        "tau": d.actuator_force,
    }


def with_cmd(state, cmd):
    """`state` holding `cmd` as its command for the next step.

    Zeroing `steps_since_cmd` keeps the env's own resample from replacing
    the command. The measurement overrides of battery.load_checkpoint_policy
    already set `command.resample_steps` past any course. The zeroing still
    holds under a `--set` that lowers it to 2 or more. At 1 or 0 the env
    resamples on every step, and the observation the policy acts on carries
    the random command."""
    info = {
        **state.info,
        "command": cmd,
        "steps_since_cmd": jp.zeros_like(state.info["steps_since_cmd"]),
    }
    return state.replace(info=info)


def tree_where(cond, a, b):
    """Leaf-wise `jp.where(cond, a, b)` over two pytrees of one structure."""
    return jax.tree_util.tree_map(lambda x, y: jp.where(cond, x, y), a, b)


# -- the lane --------------------------------------------------------------------


class _Carry(NamedTuple):
    st: Any
    t: jax.Array
    fo: FollowerStep
    fell: jax.Array
    nonfinite: jax.Array
    yaw_prog: jax.Array
    pushed: jax.Array
    peaks: jax.Array
    buf: dict


def make_lane_fn(env, inf, *, max_steps: int, settle_steps: int,
                 reset_fn: Callable | None = None, on_stop: Callable | None = None,
                 observe: Callable | None = None) -> Callable:
    """The lane `(data: LaneData, key) -> LaneOut` for `env` under policy
    `inf` (obs, key) -> (action, extras).

    `key` drives env.reset (the reset pose noise and the random reset
    command) and the per-step observation noise. The policy is deterministic,
    so its action key is a fixed PRNGKey(0).

    Per step, the record holds the pre-step pose (`x`, `y`, `yaw`), the
    follower values that chose the step's command (`s`, `xte`, `spinning`),
    the held command (`cmd`), the planar kick added to the base velocity
    before the step (`kick`, world frame), and `signals` of the post-step
    physics. The policy acts on the observation built before the kick.

    `max_steps` is the record length; a row's budget above it is cut to it.
    Each call of this function returns a new closure, which is what a model
    swap needs (`set_friction`).

    Hooks, all traced into the lane:
      reset_fn(key, data) -> state    replaces env.reset(key)
      on_stop(state, data) -> state   applied once to the state the lane
                                      stops at (after the settle on a settle
                                      fall); it must keep every leaf's shape
                                      and dtype
      observe(state, data) -> (k,)    after the reset and every step; folded
                                      into LaneOut.peaks by max
    """
    max_steps, settle_steps = int(max_steps), int(settle_steps)

    def lane(data: LaneData, key) -> LaneOut:
        act_key = jax.random.PRNGKey(0)
        is_spin = data.kind == SPIN
        budget = jp.minimum(data.budget, max_steps)

        def step(st, cmd):
            action, _ = inf(st.obs, act_key)
            return env.step(with_cmd(st, cmd), action)

        def fold(peaks, st):
            return peaks if observe is None else jp.maximum(peaks, observe(st, data))

        def goal(fo, yaw_prog):
            return jp.where(is_spin, yaw_prog >= data.spin_required, fo.reached)

        def stopped(fell, nonfinite, fo, yaw_prog, t):
            return fell | nonfinite | goal(fo, yaw_prog) | (t >= budget)

        st = reset_fn(key, data) if reset_fn is not None else env.reset(key)
        peaks = observe(st, data) if observe is not None else jp.zeros((0,), jp.float32)

        def settle_body(carry, _):
            st, fell, peaks = carry
            st = step(st, jp.zeros(3))
            return (st, fell | (st.done > 0), fold(peaks, st)), base_height(env, st.data)

        (st, settle_fell, peaks), heights = jax.lax.scan(
            settle_body, (st, jp.array(False), peaks), None, length=settle_steps
        )
        settle_height = jp.mean(heights[settle_steps // 2 :])

        x, y, yaw = pose(env, st.data)
        anchor = jp.where(data.anchor_world, data.origin, jp.stack([x, y, yaw]))
        path = follower.layout(data.path, anchor[0], anchor[1], anchor[2])
        fo = follower.follower_step(path, follower.initial_state(), x, y, yaw, data.yaw_cap)
        nonfinite = ~jp.all(jp.isfinite(st.data.qpos))
        yaw_prog = jp.zeros((), jp.float32)
        t = jp.zeros((), jp.int32)
        if on_stop is not None:
            st = tree_where(stopped(settle_fell, nonfinite, fo, yaw_prog, t), on_stop(st, data), st)

        proto = jax.eval_shape(lambda d: signals(env, d), st.data)
        buf = {k: jp.zeros((max_steps, *v.shape), v.dtype) for k, v in proto.items()}
        for k in ("x", "y", "yaw", "s", "xte"):
            buf[k] = jp.zeros(max_steps, jp.float32)
        buf["spinning"] = jp.zeros(max_steps, bool)
        buf["cmd"] = jp.zeros((max_steps, 3), jp.float32)
        buf["kick"] = jp.zeros((max_steps, 2), jp.float32)

        def cond(c: _Carry):
            return ~stopped(c.fell, c.nonfinite, c.fo, c.yaw_prog, c.t)

        def body(c: _Carry) -> _Carry:
            st = c.st
            x, y, yaw = pose(env, st.data)
            cmd = jp.where(is_spin, jp.stack([0.0, 0.0, data.spin_wz]), c.fo.cmd)
            # One kick, the first time the follower's progress passes
            # push_at_m. Its sign is fixed (left of the heading), so the two
            # push rows differ only in speed.
            kick = ~is_spin & ~c.pushed & (c.fo.s >= data.push_at_m)
            dv = jp.where(kick, data.push_vel * jp.stack([-jp.sin(yaw), jp.cos(yaw)]), 0.0)
            qvel = st.data.qvel
            qvel = jp.where(kick, qvel.at[env._base_vadr : env._base_vadr + 2].add(dv), qvel)
            st = st.replace(data=st.data.replace(qvel=qvel))

            nst = step(st, cmd)
            nonfinite = ~jp.all(jp.isfinite(nst.data.qpos))
            row = {
                **signals(env, nst.data),
                "x": x, "y": y, "yaw": yaw,
                "s": c.fo.s, "xte": c.fo.xte, "spinning": c.fo.state.spinning,
                "cmd": cmd, "kick": dv,
            }
            buf = {k: c.buf[k].at[c.t].set(v) for k, v in row.items()}
            nx, ny, nyaw = pose(env, nst.data)
            # A non-finite step is written past the end. It is counted
            # neither in `t` nor in the yaw progress.
            t = c.t + jp.where(nonfinite, 0, 1).astype(c.t.dtype)
            yaw_prog = jp.where(
                nonfinite,
                c.yaw_prog,
                c.yaw_prog + jp.sign(data.spin_wz) * follower.wrap_angle(nyaw - yaw),
            )
            fell = nst.done > 0
            # The next step's command, and the goal test on this post-step
            # pose, the one after the last budget step included.
            fo = follower.follower_step(path, c.fo.state, nx, ny, nyaw, data.yaw_cap)
            peaks = fold(c.peaks, nst)
            if on_stop is not None:
                nst = tree_where(stopped(fell, nonfinite, fo, yaw_prog, t), on_stop(nst, data), nst)
            return _Carry(nst, t, fo, fell, nonfinite, yaw_prog, c.pushed | kick, peaks, buf)

        carry = _Carry(
            st=st, t=t, fo=fo, fell=settle_fell, nonfinite=nonfinite,
            yaw_prog=yaw_prog, pushed=jp.array(False), peaks=peaks, buf=buf,
        )
        c = jax.lax.while_loop(cond, body, carry)
        return LaneOut(
            steps=c.t,
            nonfinite=c.nonfinite,
            settle_fell=settle_fell,
            fell=c.fell,
            reached=goal(c.fo, c.yaw_prog),
            settle_height=settle_height,
            yaw_rad=c.yaw_prog,
            anchor=anchor,
            peaks=c.peaks,
            rec=c.buf,
            state=c.st,
        )

    return lane


def compile_lane(lane_fn: Callable, data: LaneData):
    """`lane_fn` compiled ahead of time for `data`'s shapes, on the calling
    thread. The result is called as `compiled(data, key)` from any thread. A
    call with other shapes or dtypes raises instead of compiling again."""
    return jax.jit(lane_fn).lower(data, jax.random.PRNGKey(0)).compile()


def run_lanes(compiled, jobs: Sequence[LaneJob], *, workers: int,
              finish: Callable[[LaneJob, LaneOut], Any]) -> list:
    """Run every job through `compiled` on `workers` threads.

    Jobs start longest budget first, so a long lane does not start last and
    set the wall time alone. `finish(job, out)` runs in the worker that ran
    the lane, so scoring overlaps the other lanes and only its result is
    kept. Returns finish's results in the order of `jobs`. The pool size and
    the start order never change a number."""
    order = sorted(range(len(jobs)), key=lambda i: -int(jobs[i].data.budget))
    results: list = [None] * len(jobs)

    def work(i):
        job = jobs[i]
        return i, finish(job, compiled(job.data, job.key))

    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        for i, result in pool.map(work, order):
            results[i] = result
    return results


# -- model edits and measurements ------------------------------------------------


def friction_geoms(env) -> np.ndarray:
    """The geoms a floor row sets: every colliding world geom and every foot
    geom (model_ids.friction_geom_ids)."""
    m = env.mj_model
    return model_ids.friction_geom_ids(
        m.geom_bodyid, m.geom_contype, m.geom_conaffinity, env._foot_geom_ids
    )


def set_friction(env, ids, mu) -> None:
    """Set the sliding friction of geoms `ids` to `mu` (a scalar, or one
    value per id) and put the model on the device again.

    MuJoCo combines two equal-priority geoms' friction by element-wise max,
    so a floor row sets the floor and the feet together. Lanes built before
    the call keep the old friction; build a fresh one with make_lane_fn.
    Never jit a bound env method around a swap: a `jax.jit(env.step)`
    created after it reproduced the old friction bit for bit while an
    earlier `jax.jit(env.step)` was still alive, because jax reused the trace
    of the equal bound method. Restore by passing the values read before
    the first call."""
    ids = np.asarray(ids, dtype=np.int64)
    env.mj_model.geom_friction[ids, 0] = mu
    env._mjx_model = mjx.put_model(env.mj_model, impl=env._backend)


def measure_robot_inputs(env) -> dict:
    """The two keyframe inputs of spec.RobotInputs, measured on `env`'s
    model at its reset keyframe with MuJoCo's C forward pass.

    `stance_halfwidth_m` is half the lateral spread of the foot sites, in
    the base's heading frame. `nominal_height_m` is the base z. Never traced.
    """
    m = env.mj_model
    d = mujoco.MjData(m)
    d.qpos[:] = np.asarray(env._home_qpos)
    mujoco.mj_forward(m, d)
    b = env._base_qadr
    w, qx, qy, qz = d.qpos[b + 3 : b + 7]
    yaw = math.atan2(2.0 * (w * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    sites = d.site_xpos[np.asarray(env._foot_site_ids)]
    lateral = -math.sin(yaw) * sites[:, 0] + math.cos(yaw) * sites[:, 1]
    return {
        "stance_halfwidth_m": float(lateral.max() - lateral.min()) / 2.0,
        "nominal_height_m": float(d.qpos[b + 2]),
    }
