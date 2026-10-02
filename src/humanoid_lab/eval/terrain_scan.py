"""CLI: score a checkpoint on its robot's terrain scan suite.

    ./run.sh terrain-scan --run runs/<name>
    ./run.sh terrain-scan --run runs/<name> --cells pyramid_stairs_9.8cm --speeds 0.3
    ./run.sh terrain-scan --list-cells

eval/terrain_suite.py defines the suite: the arena, its cells, the course
and the protocol constants. The run's robot picks the suite, and a robot
without one is refused before anything is built.

Env. `load_checkpoint_policy(run_dir, flat=False, ...)` rebuilds the run on
the terrain task with the suite's arena, so a joystick run scans too. A run
of any other task is refused. The measurement overrides turn off pushes,
command resampling and the no-progress cut. The scan adds pad spawns
without jitter or yaw, base contact on at the suite's tolerance, no command
bias and the suite's warp budgets. `done` then means a fall: base height,
tilt, or a termination collider within the tolerance of the ground. The scan
refuses an arena whose params or fingerprint differ from the suite's.

Batch. One dispatch per speed holds every selected cell. Run r of cell c is
world `c * runs_per_cell + r`, and its reset key is
`split(cell_key(c), runs_per_cell)[r]`. A cell's runs therefore get the
same keys in any batch. The key drives only the env's observation noise.
The policy acts deterministically.

Reset. Each world resets, then takes the reset pose with no joint noise, on
its tile's pad at the run's start, facing its heading. `spawn_qpos` places
it on the lookup (kind 0). Pads are flat, so the base sits at the pad height
plus the reset height. qvel is zero, ctrl neutral and the command zero.

Step. The command is zero for `settle_steps`, then `[v, 0, 0]`. Each step
writes the command, acts on the last observation and steps the env. A
step's observation carries that step's command, so the policy first sees
the walk command one control step after the settle ends. A run finishes
when its base reaches Chebyshev `r_out` from the tile centre after the
settle. It falls when the env reports done while it runs. It stops at a
fall, a finish or its deadline, and a fall after its finish is not
recorded. A fall in the settle counts as a fall and is also counted apart.

Metrics, over each run's steps after the settle while it runs:
- saturation: the fraction of actuators whose |force| exceeds
  `saturation_frac` of their own force cap, the battery's rule;
- track_err: |v - body vx|;
- clearance: the mean over the feet of the terrain-relative clearance;
- progress: the largest `(d - d0) / (r_out - d0)`, clipped to [0, 1], with
  d the base's Chebyshev distance and d0 the start's.
A run's metrics are means over its measured steps. A cell averages them
over the runs that have one, and reports 0 when none has.

Parking. Every step, each world that has stopped is rewritten at the reset
pose with its soles PARK_LIFT above the arena's highest point, at zero
velocity. Its bounding boxes then clear every ground geom's, so it adds no
ground pair and no contact. A fallen robot cannot fill the contact pool,
and the counters follow the runs still going.

Counters. After each control step the runner reads warp's pool-wide
`nacon` and `ncollision` and the largest per-world `nefc`, and keeps their
peaks. They are sampled at a control step's last physics step. On warp each
dispatch runs under `fd_capture.capture_fd1`, which counts MJWarp's
overflow messages from every physics step. The scan is physics-clean when
no gating message printed, no pool or row fill reached 1 and no running
world's qpos went non-finite. A scan that is not physics-clean keeps its
numbers, and its gate verdict is `invalid`.

jax. The jax backend runs narrowphase on every robot-box pair and refuses
an arena of more than JAX_BOX_LIMIT ground boxes. The eval arena has 1139:
its 1135 boxes and the 4 aprons. The full suite therefore runs on warp. jax
prints no overflow message and has no live counters, so physics-clean there
covers non-finite states only.

Gate. A gated cell passes at a speed when at least `threshold(cell, runs)`
of its runs pass. Tracked cells are reported and never gated. The verdict
is `invalid` when the scan is not physics-clean, `fail` when a gated pair
falls short, `incomplete` when fewer pairs were checked than the full suite
gates, and `pass` otherwise. Every bar is provisional.

The scan writes `<run>/terrain_scan.json` unless `--out` names a path, and
prints a cells x speeds table. A refused request exits 2 before any model
is built. A finished scan exits 0, whatever its verdict.

XLA_PYTHON_CLIENT_PREALLOCATE is set to false unless the environment sets
it. MJWarp allocates its CCD scratch outside the XLA pool. Importing
check_terrain applies the training process's other XLA defaults.
"""

from __future__ import annotations

import os

from humanoid_lab import tasks

# Before jax creates its client: the terrain training process's allocator.
os.environ.update({k: os.environ.get(k, v) for k, v in tasks.TERRAIN_XLA_DEFAULTS.items()})

import argparse
import math
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np

from humanoid_lab import fd_capture, paths, run_record, sim_budget
from humanoid_lab.check_terrain import action_window
from humanoid_lab.envs import terrain_geometry as tg
from humanoid_lab.envs.terrain_joystick import JAX_BOX_LIMIT
from humanoid_lab.envs.terrain_joystick import default_config as terrain_default_config
from humanoid_lab.eval import terrain_suite as ts
from humanoid_lab.terrain import GENERATOR_VERSION, TYPES, Arena, fingerprint, scene
from humanoid_lab.terrain.config import ccd_scratch, config_from_params

SCHEMA = 1
OUT_NAME = "terrain_scan.json"
# m between a parked robot's soles and the arena's highest point.
PARK_LIFT = 2.0
# Two-sided 95% normal quantile, for the Wilson interval.
Z95 = 1.959963984540054
# Per-world runner outputs, sliced per cell.
RUN_KEYS = (
    "finished",
    "fell",
    "fell_in_settle",
    "steps",
    "measured",
    "progress",
    "saturation",
    "track_err",
    "clearance",
    "nonfinite",
)
# Package versions that identify the engine a scan ran on.
ENGINE_VERSIONS = ("jax", "mujoco", "mujoco_mjx", "warp_lang")


class ScanRefused(ValueError):
    """A request the scan refuses before it rolls anything. Exit 2."""


# -- pure helpers: xp is numpy or jax.numpy ------------------------------------------


def tile_distance(xy, centre, xp=np):
    """Chebyshev distance of `xy` from the tile centre, over the last axis.
    The suite's features are concentric squares about the centre."""
    return xp.max(xp.abs(xy - centre), axis=-1)


def still_running(i, finished, fell, deadline):
    """Whether each run is still on its course at step `i`: not fallen, not
    finished and before its own deadline."""
    return ~(fell | finished) & (i < deadline)


def fall_progress(fell, done, running):
    """The sticky fall flag. `done` counts only while the run is running,
    so a fall after a finish or past the deadline is not recorded."""
    return fell | (running & done)


class Outcome(NamedTuple):
    """Each run's state on the course. `steps` counts the steps it ran,
    `measured` the ones after the settle."""

    finished: object
    fell: object
    fell_in_settle: object
    steps: object
    measured: object
    progress: object


def start_outcome(n: int, xp=np) -> Outcome:
    """`n` runs before their first step."""
    no = xp.zeros(n, bool)
    return Outcome(no, no, no, xp.zeros(n, xp.int32), xp.zeros(n, xp.int32), xp.zeros(n, xp.float32))


def track(i, o: Outcome, distance, done, r_out, d0, deadline, settle_steps: int, xp=np):
    """(outcome, running, live) after step `i`.

    `distance` and `done` are read after the step. `running` is whether
    each run was on its course when the step began. `live` is running after
    the settle: the steps a run is measured on."""
    running = still_running(i, o.finished, o.fell, deadline)
    walking = xp.asarray(i) >= settle_steps
    live = running & walking
    fell = fall_progress(o.fell, done, running)
    fell_in_settle = o.fell_in_settle | (running & done & ~walking)
    finished = o.finished | (live & (distance >= r_out))
    gain = xp.clip((distance - d0) / (r_out - d0), 0.0, 1.0)
    progress = xp.where(live, xp.maximum(o.progress, gain), o.progress)
    steps = o.steps + running.astype(o.steps.dtype)
    measured = o.measured + live.astype(o.measured.dtype)
    out = Outcome(finished, fell, fell_in_settle, steps, measured, progress.astype(o.progress.dtype))
    return out, running, live


def arena_top(arena: Arena) -> float:
    """The arena's highest point: box tops, the heightfield's top and the
    aprons' tops at z = 0."""
    hf = arena.spec.hfield
    return max(float(np.max(arena.lookup)), float(hf.pos_z + hf.elevation_z), 0.0)


def park_pose(reset_qpos, base_qadr: int, xy, base_z: float, xp=np):
    """(N, nq) parking poses: `reset_qpos` with each base at `xy` (N, 2) and
    height `base_z`."""
    n = xy.shape[0]
    q = xp.tile(xp.asarray(reset_qpos)[None], (n, 1))
    z = xp.full((n, 1), base_z, dtype=q.dtype)
    b = base_qadr
    return xp.concatenate([q[:, :b], xp.asarray(xy, dtype=q.dtype), z, q[:, b + 3 :]], axis=1)


def park(qpos, qvel, parked, pose, xp=np):
    """(qpos, qvel) with each `parked` world at its `pose` and at rest.
    Other worlds are untouched."""
    p = parked[:, None]
    return xp.where(p, pose, qpos), xp.where(p, 0.0, qvel)


def wilson(passed: int, of: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval of `passed` successes in `of` runs."""
    if of <= 0:
        return 0.0, 1.0
    p = passed / of
    denom = 1.0 + z * z / of
    centre = (p + z * z / (2 * of)) / denom
    half = z * math.sqrt(p * (1 - p) / of + z * z / (4 * of * of)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


@dataclass(frozen=True)
class CellResult:
    """One cell at one speed. Pass, fall and timeout counts are over every
    run. The metrics average over the `measured` runs, those with a step
    after the settle. `falls` includes `falls_in_settle`. `steps_max` is
    the longest run's step count."""

    passed: int
    of: int
    rate: float
    ci95: tuple[float, float]
    falls: int
    falls_in_settle: int
    timeouts: int
    progress_mean: float
    track_err: float
    saturation: float | None
    clearance: float
    measured: int
    steps_max: int


def reduce_runs(out: Mapping) -> CellResult:
    """One cell's runs at one speed into its numbers. A run passes when it
    finished and did not fall. A cell where no run reached a measured step
    reports 0, never NaN."""
    finished = np.asarray(out["finished"], bool)
    fell = np.asarray(out["fell"], bool)
    measured = np.asarray(out["measured"]) > 0
    passed = int((finished & ~fell).sum())
    of = int(finished.size)

    def mean_of(key):
        values = np.asarray(out[key], float)[measured]
        return float(values.mean()) if values.size else 0.0

    return CellResult(
        passed=passed,
        of=of,
        rate=passed / of if of else 0.0,
        ci95=wilson(passed, of),
        falls=int(fell.sum()),
        falls_in_settle=int(np.asarray(out["fell_in_settle"], bool).sum()),
        timeouts=int((~finished & ~fell).sum()),
        progress_mean=mean_of("progress"),
        track_err=mean_of("track_err"),
        saturation=mean_of("saturation"),
        clearance=mean_of("clearance"),
        measured=int(measured.sum()),
        steps_max=int(np.max(out["steps"])) if of else 0,
    )


def result_entry(r: CellResult) -> dict:
    """The JSON form of one cell's result at one speed."""

    def rnd(v):
        return None if v is None else round(float(v), 4)

    return {
        "passed": r.passed,
        "of": r.of,
        "rate": rnd(r.rate),
        "ci95": [rnd(r.ci95[0]), rnd(r.ci95[1])],
        "falls": r.falls,
        "falls_in_settle": r.falls_in_settle,
        "timeouts": r.timeouts,
        "progress_mean": rnd(r.progress_mean),
        "track_err": rnd(r.track_err),
        "saturation": rnd(r.saturation),
        "clearance": rnd(r.clearance),
        "measured": r.measured,
        "steps_max": r.steps_max,
    }


def speed_key(speed: float) -> str:
    return f"{float(speed):g}"


def speed_entries(entry: Mapping) -> dict:
    """The per-speed results of one cell entry, keyed by speed."""
    return {k: v for k, v in entry.items() if isinstance(v, Mapping)}


def gated_pairs(suite: ts.Suite) -> int:
    """The (cell, speed) pairs a full scan of `suite` gates."""
    return sum(1 for c in suite.cells if not c.tracked) * len(suite.speeds)


def absolute_gate(entries: Mapping, expected: int, physics_clean: bool) -> dict:
    """Every gated cell's pass count against its threshold, at every speed
    scanned. `entries` is the report's `cells` block. `expected` is the
    pair count a full scan gates."""
    failures = []
    checked = 0
    for name, entry in entries.items():
        need = entry.get("threshold")
        if need is None:
            continue
        for speed, r in speed_entries(entry).items():
            checked += 1
            if r["passed"] < need:
                failures.append(
                    {
                        "cell": name,
                        "speed": speed,
                        "passed": r["passed"],
                        "of": r["of"],
                        "threshold": need,
                        "provenance": entry["provenance"],
                    }
                )
    if not physics_clean:
        verdict = "invalid"
    elif failures:
        verdict = "fail"
    elif checked < expected:
        verdict = "incomplete"
    else:
        verdict = "pass"
    return {"verdict": verdict, "checked": checked, "expected": expected, "failures": failures}


def cell_key(cell: ts.Cell, eval_seed: int = 0):
    """The PRNG key of `cell`'s runs. Seed 0 is the default stream. Any
    other seed folds into it, a fresh draw of the same course."""
    import jax

    key = jax.random.PRNGKey(cell.row * 1000 + TYPES.index(cell.terrain_type))
    return key if eval_seed == 0 else jax.random.fold_in(key, eval_seed)


def batch_reset_keys(keys, runs_per_cell: int):
    """(C * R, 2) reset keys from (C, 2) cell keys, cell-major. Run r of
    cell c gets `split(keys[c], R)[r]`."""
    import jax

    split = jax.vmap(lambda k: jax.random.split(k, runs_per_cell))(keys)
    return split.reshape(keys.shape[0] * runs_per_cell, -1)


def spawn_table(spec, suite: ts.Suite, cells: Sequence[ts.Cell], ctrl_dt: float,
                speeds: Sequence[float]) -> dict:
    """Per-world starts, cell-major: run r of cell c is world c * R + r.

    Arrays: `cell` (index into `cells`), `row` and `type` (the tile),
    `centre` and `start` (xy), `yaw`, `r_out` and `d0` (the start's
    Chebyshev distance). `deadline` maps each speed key to (N,) step
    deadlines. `r_out` maps each cell name to its radius."""
    course = ts.course(suite)
    runs = len(course)
    yaw = np.array([r.yaw for r in course])
    offset = np.array([r.offset for r in course])
    heading = np.stack([np.cos(yaw), np.sin(yaw)], axis=-1)
    tile_size = spec.params.tile_size
    out = {k: [] for k in ("cell", "row", "type", "centre", "start", "yaw", "r_out", "d0")}
    deadline = {speed_key(s): [] for s in speeds}
    radius = {}
    for c, cell in enumerate(cells):
        tile = ts.cell_tile(spec, cell)
        centre = np.asarray(tile.origin[:2], float)
        r = ts.r_out(tile.feature_radius, tile_size, suite.footprint_reach)
        radius[cell.name] = r
        start = centre + offset[:, None] * heading
        out["cell"].append(np.full(runs, c))
        out["row"].append(np.full(runs, cell.row))
        out["type"].append(np.full(runs, TYPES.index(cell.terrain_type)))
        out["centre"].append(np.repeat(centre[None], runs, axis=0))
        out["start"].append(start)
        out["yaw"].append(yaw)
        out["r_out"].append(np.full(runs, r))
        out["d0"].append(tile_distance(start, centre))
        for s in speeds:
            deadline[speed_key(s)].append(np.asarray(ts.run_deadlines(suite, r, s, ctrl_dt)))
    ints = ("cell", "row", "type")
    table = {k: np.concatenate(v).astype(np.int32 if k in ints else np.float32) for k, v in out.items()}
    table["deadline"] = {k: np.concatenate(v).astype(np.int32) for k, v in deadline.items()}
    table["r_out_by_cell"] = radius
    return table


def command_box_warnings(run: Mapping, speeds: Sequence[float]) -> list[str]:
    """A warning for each speed outside the run's trained forward-speed box.
    A cell at such a speed measures extrapolation."""
    vx = ((run.get("env_config") or {}).get("command") or {}).get("vx")
    if not vx:
        return []
    lo, hi = float(vx[0]), float(vx[1])
    return [
        f"commanded vx {s:g} lies outside the run's trained box [{lo:g}, {hi:g}]. Its cells "
        "measure extrapolation."
        for s in speeds
        if not lo <= s <= hi
    ]


def check_arena(arena: Arena, suite: ts.Suite) -> None:
    """Refuse an arena whose params or fingerprint are not the suite's.

    A generator or default change moves the arena under a fixed params
    record, so scores filed under the suite's fingerprint would describe
    another terrain."""
    params = arena.spec.params
    built = fingerprint(arena)
    if params == suite.arena and built == suite.fingerprint:
        return
    detail = []
    if params != suite.arena:
        detail.append("its params differ from Suite.arena")
    if built != suite.fingerprint:
        detail.append(f"it fingerprints {built}, and the suite pins {suite.fingerprint}")
    raise ScanRefused(
        f"the scan arena is not the {suite.robot} suite's (version {suite.version}): "
        f"{'; '.join(detail)}. The generator is at version {GENERATOR_VERSION}. A changed "
        "arena changes every cell's numbers: bump Suite.version and re-pin its fingerprint."
    )


# -- selection --------------------------------------------------------------------


def run_robot(run: Mapping) -> str:
    """The robot a run trained, from its recorded Hydra config."""
    robot = run["hydra_config"]["robot"]
    return robot.get("name") or Path(robot["dir"]).name


def suite_of(robot: str) -> ts.Suite:
    try:
        return ts.suite_for(robot)
    except ValueError as e:
        raise ScanRefused(str(e)) from None


def select_cells(suite: ts.Suite, names: Sequence[str] | None) -> tuple[ts.Cell, ...]:
    """The suite's cells named by `names`, in suite order. None selects
    every cell. An unknown name is refused."""
    if names is None:
        return suite.cells
    wanted = list(dict.fromkeys(names))
    known = {c.name for c in suite.cells}
    unknown = [n for n in wanted if n not in known]
    if unknown:
        raise ScanRefused(f"unknown cells {unknown}. --list-cells prints the suite's names")
    if not wanted:
        raise ScanRefused("--cells names no cell")
    return tuple(c for c in suite.cells if c.name in wanted)


def select_speeds(suite: ts.Suite, speeds: Sequence[float] | None) -> tuple[float, ...]:
    """The suite's speeds named by `speeds`, in suite order. None selects
    every speed. A speed the suite does not define is refused."""
    if speeds is None:
        return suite.speeds
    unknown = [s for s in speeds if s not in suite.speeds]
    if unknown:
        raise ScanRefused(
            f"speeds {unknown} are not the suite's. --speeds picks from {list(suite.speeds)}"
        )
    if not speeds:
        raise ScanRefused("--speeds names no speed")
    return tuple(s for s in suite.speeds if s in speeds)


def partial_warnings(suite: ts.Suite, cells, speeds) -> list[str]:
    out = []
    if len(cells) < len(suite.cells):
        out.append(f"partial scan: {len(cells)} of {len(suite.cells)} cells")
    if len(speeds) < len(suite.speeds):
        out.append(f"partial scan: speeds {list(speeds)} of {list(suite.speeds)}")
    return out


# -- env and runner -------------------------------------------------------------------


def scan_overrides(suite: ts.Suite, n_worlds: int, backend: str, budgets: Mapping) -> dict:
    """The env overrides the scan adds to the measurement overrides.

    The measurement merge is one level deep, so each terrain sub-block here
    is complete and replaces the run's own. The run's spawn block is not
    kept: the scan places every world itself."""
    spawn = terrain_default_config().terrain.spawn.to_dict()
    spawn.update(mode="pad", yaw=False, pad_jitter=0.0, grace_sec=0.0, level=-1)
    return {
        "terrain": {
            "arena": config_from_params(suite.arena),
            "spawn": spawn,
            "base_contact": {"terminate": True, "tol": suite.base_contact_tol},
            "command_bias": {**terrain_default_config().terrain.command_bias.to_dict(), "enable": False},
        },
        "sim": {
            "backend": backend,
            "num_envs": int(n_worlds),
            "naconmax_per_env": budgets["naconmax_per_env"],
            "njmax": budgets["njmax"],
            "naccdmax_per_env": budgets["naccdmax_per_env"],
        },
    }


def scan_budgets(suite: ts.Suite, naconmax_per_env=None, njmax=None, naccdmax_per_env=None) -> dict:
    """The warp budgets the scan builds with: each flag where given, else
    the suite's."""
    return {
        "naconmax_per_env": suite.naconmax_per_env if naconmax_per_env is None else int(naconmax_per_env),
        "njmax": suite.njmax if njmax is None else int(njmax),
        "naccdmax_per_env": suite.naccdmax_per_env if naccdmax_per_env is None else int(naccdmax_per_env),
    }


def build_scan_env(run_dir: Path, suite: ts.Suite, n_worlds: int, backend: str, budgets: Mapping):
    """(run, env, checkpoint, inference) for scanning `run_dir` on the
    suite's arena with `n_worlds` worlds. All model construction lives
    here."""
    from humanoid_lab.eval.battery import load_checkpoint_policy

    extra = scan_overrides(suite, n_worlds, backend, budgets)
    return load_checkpoint_policy(Path(run_dir), extra, flat=False)


def scan_reset(env, key, start_xy, yaw, centre, row, ttype):
    """One world's start: `env.reset(key)`, then the reset pose on the pad
    at `start_xy` facing `yaw`, at rest, with a zero command. The terrain
    info keys name the scanned tile. The observation is rebuilt from the
    new pose and command."""
    import jax.numpy as jp
    from mujoco import mjx

    state = env.reset(key)
    qpos = env.spawn_qpos(env._reset_qpos, start_xy, yaw, tg.SPAWN_PAD)
    data = state.data.replace(qpos=qpos, qvel=jp.zeros_like(state.data.qvel), ctrl=env._neutral_ctrl)
    data = mjx.forward(env.mjx_model, data)
    r0 = tg.chebyshev(start_xy, centre)
    info = {
        **state.info,
        "command": jp.zeros(3),
        "terrain_type": ttype,
        "terrain_level": row,
        "spawn_xy": start_xy,
        "last_xy": start_xy,
        "spawn_kind": jp.asarray(tg.SPAWN_PAD, jp.int32),
        "tile_origin": centre,
        "cheby_min": r0,
        "cheby_max": r0,
    }
    return state.replace(data=data, obs=env._build_obs(data, info), info=info)


class _Carry(NamedTuple):
    i: object
    state: object
    outcome: Outcome
    saturation: object
    track_err: object
    clearance: object
    peaks: object
    nonfinite: object


def make_batch_runner(env, suite: ts.Suite, inference):
    """A jitted `run(keys, table, speed, deadline, *, budget)` that rolls
    every world of one speed in one `lax.while_loop`.

    `table` holds the spawn table's device arrays. `budget` is the only
    static argument and caps the loop. The loop ends as soon as no run is
    still on its course, and every run stops at its own deadline. Returns
    {"runs": per-world RUN_KEYS, "iterations", "nefc", "nacon",
    "ncollision", "final"}. The counters are 0 off warp. "final" holds
    the last state's qpos and qvel, every world parked."""
    import functools

    import jax
    import jax.numpy as jp

    b = env._base_qadr
    cap = jp.asarray(np.asarray(env.mj_model.actuator_forcerange)[:, 1], jp.float32)
    frac = float(suite.saturation_frac)
    settle = int(suite.settle_steps)
    base_z = arena_top(env._arena) + PARK_LIFT + float(env._z0)
    reset_qpos = env._reset_qpos
    act_key = jax.random.PRNGKey(0)
    reset = jax.vmap(functools.partial(scan_reset, env))
    step = jax.vmap(env.step)
    linvel = jax.vmap(env._local_linvel)
    clearance = jax.vmap(env._foot_clearance)

    @functools.partial(jax.jit, static_argnames=("budget",))
    def run(keys, table, speed, deadline, *, budget: int):
        n = keys.shape[0]
        state = reset(keys, table["start"], table["yaw"], table["centre"], table["row"], table["type"])
        pose = park_pose(reset_qpos, b, table["start"], base_z, xp=jp)
        walk = jp.zeros((n, 3)).at[:, 0].set(speed)
        zero = jp.zeros(n, jp.float32)
        init = _Carry(
            jp.zeros((), jp.int32), state, start_outcome(n, jp), zero, zero, zero,
            jp.zeros(3, jp.int32), jp.zeros(n, bool),
        )

        def cond(c):
            o = c.outcome
            return (c.i < budget) & jp.any(still_running(c.i, o.finished, o.fell, deadline))

        def body(c):
            command = jp.where(c.i >= settle, walk, 0.0)
            state = c.state.replace(info={**c.state.info, "command": command})
            action, _ = inference(state.obs, act_key)
            state = step(state, action)
            data = state.data
            distance = tile_distance(data.qpos[:, b : b + 2], table["centre"], xp=jp)
            o, running, live = track(
                c.i, c.outcome, distance, state.done > 0.5, table["r_out"], table["d0"],
                deadline, settle, xp=jp,
            )
            w = live.astype(jp.float32)
            saturated = jp.mean((jp.abs(data.actuator_force) > frac * cap).astype(jp.float32), axis=-1)
            err = jp.abs(speed - linvel(data)[:, 0])
            clear = jp.mean(clearance(data), axis=-1)
            peaks = jp.maximum(c.peaks, jp.stack(sim_budget.traced_counters(data)))
            nonfinite = c.nonfinite | (running & ~jp.all(jp.isfinite(data.qpos), axis=-1))
            parked = ~still_running(c.i + 1, o.finished, o.fell, deadline)
            qpos, qvel = park(data.qpos, data.qvel, parked, pose, xp=jp)
            state = state.replace(data=data.replace(qpos=qpos, qvel=qvel))
            return _Carry(
                c.i + 1, state, o, c.saturation + w * saturated, c.track_err + w * err,
                c.clearance + w * clear, peaks, nonfinite,
            )

        c = jax.lax.while_loop(cond, body, init)
        o = c.outcome
        per = jp.maximum(o.measured, 1).astype(jp.float32)
        runs = {
            "finished": o.finished,
            "fell": o.fell,
            "fell_in_settle": o.fell_in_settle,
            "steps": o.steps,
            "measured": o.measured,
            "progress": o.progress,
            "saturation": c.saturation / per,
            "track_err": c.track_err / per,
            "clearance": c.clearance / per,
            "nonfinite": c.nonfinite,
        }
        nefc, nacon, ncollision = (c.peaks[k] for k in range(3))
        return {
            "runs": runs,
            "iterations": c.i,
            "nefc": nefc,
            "nacon": nacon,
            "ncollision": ncollision,
            "final": {"qpos": c.state.data.qpos, "qvel": c.state.data.qvel},
        }

    return run


# -- scan ------------------------------------------------------------------------


def _device_table(table: Mapping) -> dict:
    import jax.numpy as jp

    keys = ("row", "type", "centre", "start", "yaw", "r_out", "d0")
    return {k: jp.asarray(table[k]) for k in keys}


def _counter(env, value):
    """A runner counter as a report value: None off warp, which has none."""
    return int(value) if env._backend == "warp" else None


def scan(
    run_dir: Path,
    *,
    suite: ts.Suite | None = None,
    cells: Sequence[str] | None = None,
    speeds: Sequence[float] | None = None,
    backend: str = "auto",
    naconmax_per_env: int | None = None,
    njmax: int | None = None,
    naccdmax_per_env: int | None = None,
    eval_seed: int = 0,
) -> dict:
    """Score `run_dir`'s latest checkpoint on its robot's suite, or on
    `suite`, and return the terrain_scan.json record. `cells` (names) and
    `speeds` pick a subset."""
    import jax
    import jax.numpy as jp

    from humanoid_lab.envs.backend import resolve_backend
    from humanoid_lab.eval.battery import _load_run

    started_at = run_record.utc_now()
    # jax.random.fold_in takes a uint32.
    if not 0 <= eval_seed < 2**32:
        raise ScanRefused(f"--eval-seed must be in [0, 2**32), got {eval_seed}")
    run_dir = Path(run_dir)
    run = _load_run(run_dir)
    if run["task"] not in tasks.TERRAIN_SOURCE_TASKS:
        raise ScanRefused(
            f"the run trained task {run['task']!r}, and the scan rebuilds only "
            f"{list(tasks.TERRAIN_SOURCE_TASKS)} runs"
        )
    robot = run_robot(run)
    if suite is None:
        suite = suite_of(robot)
    if suite.robot != robot:
        raise ScanRefused(f"the run trained {robot}, and the suite is {suite.robot}'s")
    chosen = select_cells(suite, cells)
    chosen_speeds = select_speeds(suite, speeds)
    arena = ts.eval_arena(suite)
    check_arena(arena, suite)
    resolved = resolve_backend(backend)
    boxes = len(arena.boxes) + len(scene.APRON_GEOMS)
    if resolved == "jax" and boxes > JAX_BOX_LIMIT:
        raise ScanRefused(
            f"the suite's arena has {boxes} ground boxes, over the jax limit of {JAX_BOX_LIMIT}. "
            "The suite runs on warp."
        )
    budgets = scan_budgets(suite, naconmax_per_env, njmax, naccdmax_per_env)
    runs = suite.runs_per_cell
    n_worlds = runs * len(chosen)

    run, env, ckpt, inference = build_scan_env(run_dir, suite, n_worlds, resolved, budgets)
    check_arena(env._arena, suite)
    warp = env._backend == "warp"

    table = spawn_table(env._arena.spec, suite, chosen, float(env.dt), chosen_speeds)
    keys = batch_reset_keys(jp.stack([cell_key(c, eval_seed) for c in chosen]), runs)
    device = _device_table(table)
    # Both speeds share the longest deadline as their cap, so they share
    # one compile.
    budget = int(max(int(d.max()) for d in table["deadline"].values()))
    runner = make_batch_runner(env, suite, inference)
    caps = np.asarray(env.mj_model.actuator_forcerange)[:, 1]

    entries = {}
    for cell in chosen:
        need, provenance = ts.threshold(cell, runs)
        entries[cell.name] = {
            "type": cell.terrain_type,
            "row": cell.row,
            "difficulty": cell.difficulty,
            "value": cell.value,
            "unit": cell.unit,
            "r_out": round(table["r_out_by_cell"][cell.name], 6),
            "bar": cell.bar,
            "threshold": need,
            "provenance": provenance,
        }
    contacts, nonfinite, iterations, counted = {}, {}, {}, []
    env_steps = 0
    t0 = time.perf_counter()
    for speed in chosen_speeds:
        key = speed_key(speed)
        deadline = jp.asarray(table["deadline"][key])
        with fd_capture.capture_fd1(warp) as lines:
            out = jax.block_until_ready(runner(keys, device, float(speed), deadline, budget=budget))
        out = jax.tree.map(np.asarray, out)
        if warp:
            counted.append(fd_capture.count_warp_messages("\n".join(lines)))
        per_world = out["runs"]
        for c, cell in enumerate(chosen):
            part = slice(c * runs, (c + 1) * runs)
            result = reduce_runs({k: per_world[k][part] for k in RUN_KEYS})
            entry = result_entry(result)
            if not np.any(caps > 0):
                entry["saturation"] = None
            entries[cell.name][key] = entry
        iterations[key] = int(out["iterations"])
        env_steps += iterations[key] * n_worlds
        nonfinite[key] = int(np.asarray(per_world["nonfinite"]).sum())
        contacts[key] = sim_budget.pool_report(
            env._backend,
            _counter(env, out["nacon"]),
            _counter(env, out["ncollision"]),
            _counter(env, out["nefc"]),
            env._naconmax_per_env,
            env._njmax,
            n_worlds,
        )
    wall = time.perf_counter() - t0

    messages = None
    if warp:
        messages = {k: sum(m.get(k, 0) for m in counted) for k in fd_capture.WARP_MESSAGES}
    physics_clean, physics_warnings = physics_verdict(env._backend, messages, contacts, nonfinite)
    warnings = command_box_warnings(run, chosen_speeds)
    warnings += partial_warnings(suite, chosen, chosen_speeds)
    warnings += physics_warnings

    provenance = run_record.provenance(paths.REPO_ROOT, started_at=started_at)
    versions = provenance["versions"]
    hydra = run["hydra_config"]
    return {
        "schema": SCHEMA,
        "suite": {
            "robot": suite.robot,
            "version": suite.version,
            "fingerprint": suite.fingerprint,
            "generator_version": GENERATOR_VERSION,
        },
        "run": run.get("run_name", run_dir.name),
        "checkpoint": ckpt.name,
        "trained_task": run["task"],
        "robot": robot,
        "preset": hydra["actuators"]["name"],
        "actuator_overrides": hydra["actuators"].get("overrides") or {},
        "action_window": action_window(env),
        "engine": {"backend": env._backend, **{k: versions.get(k) for k in ENGINE_VERSIONS}},
        "protocol": {
            "runs_per_cell_speed": runs,
            "headings": suite.headings,
            "offsets": list(suite.offsets),
            "draws": suite.draws,
            "settle_steps": suite.settle_steps,
            "budget_slack": suite.budget_slack,
            "saturation_frac": suite.saturation_frac,
            "footprint_reach": suite.footprint_reach,
            "base_contact_tol": suite.base_contact_tol,
            "speeds": list(suite.speeds),
            "ctrl_dt": float(env.dt),
        },
        "eval_seed": int(eval_seed),
        "cells": entries,
        "contacts": contacts,
        "nonfinite_runs": nonfinite,
        "messages": messages,
        "physics_clean": physics_clean,
        "gate": {"absolute": absolute_gate(entries, gated_pairs(suite), physics_clean)},
        "warnings": warnings,
        "perf": {
            "wall_s": round(wall, 3),
            "env_steps": env_steps,
            "env_steps_per_s": round(env_steps / wall, 1) if wall > 0 else None,
            "iterations": iterations,
            "num_envs": n_worlds,
        },
        "ccd_scratch": ccd_scratch(env._config.sim, env._ccd_slot_bytes, env._naconmax_per_env),
        "provenance": provenance,
        "timestamp": run_record.utc_now(),
    }


def physics_verdict(backend: str, messages, contacts: Mapping, nonfinite: Mapping) -> tuple[bool, list[str]]:
    """(physics_clean, warnings): no gating MJWarp message, no pool or row
    fill at 1 or more, and no non-finite running world."""
    warnings = []
    hits = {k: n for k, n in (messages or {}).items() if n and k in fd_capture.GATING}
    if hits:
        warnings.append(f"MJWarp overflow messages {hits}. The physics dropped work.")
    for speed, c in contacts.items():
        for name, fill in (("pool", c["fill_pool"]), ("rows", c["fill_rows"])):
            if fill is not None and fill >= 1.0:
                warnings.append(f"speed {speed}: {name} fill {fill:.3f} reached its buffer")
    bad = {speed: n for speed, n in nonfinite.items() if n}
    if bad:
        warnings.append(f"non-finite qpos in running worlds, by speed: {bad}")
    clean = not warnings
    if backend != "warp":
        warnings.append(
            f"the {backend} backend prints no overflow message and has no live counters. "
            "physics_clean covers non-finite states only."
        )
    return clean, warnings


# -- CLI --------------------------------------------------------------------------


def cell_lines(suite: ts.Suite) -> list[str]:
    """`--list-cells`: one line per cell, with its type, row, difficulty,
    value and bar."""
    runs = suite.runs_per_cell
    lines = [
        (
            f"{suite.robot} suite version {suite.version}: {len(suite.cells)} cells, {runs} runs per "
            f"cell and speed, speeds {list(suite.speeds)}"
        )
    ]
    for c in suite.cells:
        need, provenance = ts.threshold(c, runs)
        bar = provenance if need is None else f"{c.bar:.2f} ({need}/{runs}, {provenance})"
        value = f"{c.value:g} {c.unit}"
        lines.append(
            f"  {c.name:32s} {c.terrain_type:24s} row {c.row}  d {c.difficulty:.2f}  {value:8s}  {bar}"
        )
    return lines


def summary(result: Mapping) -> str:
    """The cells x speeds table, the gate and the warnings."""
    speeds = [speed_key(s) for s in result["protocol"]["speeds"]]
    head = f"{'cell':32s} {'need':>6s}" + "".join(f"  {('vx ' + s):>20s}" for s in speeds)
    lines = [
        (
            f"terrain-scan {result['robot']} suite v{result['suite']['version']}  run {result['run']}  "
            f"checkpoint {result['checkpoint']}  backend {result['engine']['backend']}  "
            f"worlds {result['perf']['num_envs']}"
        ),
        head,
    ]
    for name, entry in result["cells"].items():
        need = "-" if entry["threshold"] is None else str(entry["threshold"])
        row = f"{name:32s} {need:>6s}"
        for s in speeds:
            r = entry.get(s)
            text = "" if r is None else f"{r['passed']}/{r['of']} f{r['falls']} t{r['timeouts']}"
            row += f"  {text:>20s}"
        lines.append(row)
    gate = result["gate"]["absolute"]
    lines.append(
        f"absolute gate {gate['verdict']}: {len(gate['failures'])} of {gate['checked']} gated pairs "
        f"below threshold, a full scan gates {gate['expected']}. physics_clean {result['physics_clean']}"
    )
    lines += [f"  {f['cell']} vx {f['speed']}: {f['passed']} < {f['threshold']}" for f in gate["failures"]]
    lines += [f"WARNING: {w}" for w in result["warnings"]]
    perf = result["perf"]
    lines.append(f"{perf['env_steps']} env steps in {perf['wall_s']} s ({perf['env_steps_per_s']}/s)")
    return "\n".join(lines)


def _names(text: str | None) -> list[str] | None:
    if text is None:
        return None
    return [n.strip() for n in text.split(",") if n.strip()]


def _speeds(text: str | None) -> list[float] | None:
    names = _names(text)
    if names is None:
        return None
    try:
        return [float(s) for s in names]
    except ValueError:
        raise ScanRefused(f"--speeds takes comma-separated numbers, got {text!r}") from None


def list_cells(run_dir: Path | None) -> str:
    """The cell listing of `run_dir`'s robot, or of every suite. Reads
    run.json only."""
    if run_dir is None:
        suites = list(ts.SUITES.values())
    else:
        from humanoid_lab.eval.battery import _load_run

        suites = [suite_of(run_robot(_load_run(Path(run_dir))))]
    return "\n".join(line for s in suites for line in cell_lines(s))


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Score a checkpoint on its robot's terrain scan suite.")
    ap.add_argument("--run", type=Path, default=None, help="run directory, required unless --list-cells")
    ap.add_argument("--cells", default=None, help="comma-separated cell names (default: every cell)")
    ap.add_argument("--speeds", default=None, help="comma-separated subset of the suite's speeds")
    ap.add_argument("--backend", choices=["auto", "warp", "jax"], default="auto")
    ap.add_argument("--naconmax-per-env", type=int, default=None, help="contact budget per world (default: the suite's)")
    ap.add_argument("--njmax", type=int, default=None, help="rows per world (default: the suite's)")
    ap.add_argument(
        "--naccdmax-per-env", type=int, default=None,
        help="CCD slots per world (default: the suite's, else the naconmax pool)",
    )
    ap.add_argument("--eval-seed", type=int, default=0, help="observation noise draw (default 0)")
    ap.add_argument("--out", type=Path, default=None, help=f"report path (default: <run>/{OUT_NAME})")
    ap.add_argument("--list-cells", action="store_true", help="print the cells and exit, building nothing")
    return ap


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.list_cells:
            print(list_cells(args.run))
            return 0
        if args.run is None:
            raise ScanRefused("--run is required unless --list-cells")
        result = scan(
            args.run,
            cells=_names(args.cells),
            speeds=_speeds(args.speeds),
            backend=args.backend,
            naconmax_per_env=args.naconmax_per_env,
            njmax=args.njmax,
            naccdmax_per_env=args.naccdmax_per_env,
            eval_seed=args.eval_seed,
        )
    except ScanRefused as e:
        print(f"terrain-scan: {e}", file=sys.stderr)
        return 2
    out = args.out or args.run / OUT_NAME
    out.parent.mkdir(parents=True, exist_ok=True)
    run_record.write_json_atomic(out, result)
    print(summary(result))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
