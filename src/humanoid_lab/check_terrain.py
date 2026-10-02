"""CLI: gate a terrain recipe against MJWarp's contact, CCD and row buffers.

    ./run.sh check-terrain robot=roboto_origin actuators=deploy_pd \\
        ++task.env.sim.naconmax_per_env=N ++task.env.sim.njmax=N \\
        --backend warp --require-warp
    ./run.sh check-terrain experiment=terrain_cpu                # jax on a CPU
    ./run.sh check-terrain robot=roboto_origin --engine mujoco   # C proxy

Leftover arguments are Hydra overrides. `task=terrain` goes first, so an
experiment or a later `task=` can still change it. A composed task that is
not a terrain task exits 2 before anything is built.

Scene. The env is the training env: `registry.env_args_from_config` and
`make_env`, with the recipe's robot, preset, actuator overrides and arena.
Push, the no-progress cut, command resampling and spawn grace are off. The
gate budgets are the recipe's `task.env.sim` values. Where the recipe sets
none, `--naconmax-per-env`, `--njmax` and `--naccdmax-per-env` apply. The
report records which. `--arena eval` swaps in the terrain scan's arena.

Spawns. `spawn_table` sorts the tiles hardest first. World i takes tile
i mod n_tiles and heading k = (i + i // n_tiles) mod 8, at yaw k pi/4. The
base sits on the tile's feature band, at Chebyshev radius
clip((pad_radius + feature_radius) / 2, pad_radius + reach,
tile_size / 2 - reach), along the heading. `reach` is the farthest sole
point from the base. The yaw turns the robot to the heading. The env's
`spawn_qpos` places each world on the dilated spawn grid (kind 1), so no
collider starts inside a box. Flat-row tiles place the base on the pad
(kind 0). No level-foot filter applies: a foot across a riser is a worst
case the gate wants.

Regimes. Each rolls every world `--steps` control steps from its spawn.
Nothing ends a world early:
- stand: zero action. A neutral action holds neither robot up, so the
  robot collapses onto the terrain.
- walk: check_contacts' full-amplitude sinusoid, neighbouring joints in
  antiphase.
- fallen: check_contacts' FALLEN_ATTITUDES, world i taking attitude
  a = (i mod n_tiles + i // n_tiles) mod 3. The base quaternion is
  yaw x tilt x reset quaternion, and the base starts FALLEN_DROP_M above
  the spawn height. Each lap of the tiles turns a tile to its next
  attitude, so a tile sees all three once num_envs reaches 3 x n_tiles.

Recorded at every physics step (n_substeps per control step): warp's
pool-wide `nacon` and `ncollision`, the max over worlds of the per-world
`nefc`, whether every qpos is finite, and on jax the penetrating contacts
per world. Each control step keeps its worst physics step. The rollout
steps the env's model and data with the env's ctrl. With push off, that
is the physics of the env's own step. MJWarp prints its overflow messages
from the device to file descriptor 1, at any physics step. Each regime
runs under `fd_capture.capture_fd1`, which counts them. Every regime
shares one compiled rollout. One warm-up rollout of the first regime pays
the compile, and on warp the kernel load. The report records it as
`compile_s`. Each regime then runs once, and that run is the measurement.

Counters keep counting past their buffers, so one pass at the recipe's
budgets both gates and measures. Three overflows drop work before a
counter sees all of it. A CCD overflow drops a pair before narrowphase
counts its contacts. A broadphase overflow does the same to the candidate
pairs past the pool. Contacts past the pool add no rows to `nefc`. Any of
the three marks the recommendation a lower bound.

jax limits. The jax backend has no per-pair heightfield cap, prints no
message and has no live counters. Its `hfield_convex` sizes the prism
subgrid from the box's largest half-size, so a yawed or rolled box misses
the prisms under its low-side corners. Box colliders get partial
heightfield contact there. The terrain env's `max_contact_points` caps
the contacts jax keeps. jax's active counts are therefore lower bounds,
reported under `jax_lower_bound`. A jax run is `unverified`. Contact
physics of box cells is tested on the C engine only.

C proxy (`--engine mujoco`). Plain MuJoCo steps `env.mj_model` one world
at a time, over the same spawns, regimes and ctrl (the actuator model's
`ctrl_from_action` and the env's clip). The default is one world per
tile. It counts the contacts of each robot geom with `terrain_hfield` at
every physics step. C stops at `mujoco.mjMAXCONPAIR` (50) contacts per
pair, so a count of 50 is a cap hit, and the report names the collider.
MJWarp also collects at most 50 prism hits per heightfield pair. C writes
every hit it collects, MJWarp at most 4. The C count is an early warning,
never a pass. A resting 9.3 x 9.0 cm box gets 17 to 30 contacts in C on
a 4 cm heightfield, depending on where it sits on the grid. Centred on a
node it gets 30. An 18.6 x 27.0 cm box reaches 50 at any placement.
C resets a diverging world to qpos0 inside mj_step and counts a bad
qpos, qvel or qacc warning. That world stops counting at the reset, and
the regime lists it under `diverged`.

Exit rules:
  mjx on warp  fail, exit 1, on a GATING message, fill.pool or fill.rows at
               --max-fill, or a non-finite qpos. Otherwise pass, exit 0.
  mjx on jax   unverified, exit 2, with --require-warp. Otherwise fail,
               exit 1, on a non-finite qpos, else unverified, exit 0.
  mujoco       fail, exit 1, on a C reset (bad qpos, qvel or qacc), with or
               without --strict. Otherwise proxy_at_risk on a collider at
               the cap, exit 1 with --strict, else 0. Otherwise proxy_clear,
               exit 0.
  refused      error, exit 2: an unknown flag or regime, an empty --regimes,
               --steps below 1, a task other than terrain, a missing eval
               arena, an arena over JAX_BOX_LIMIT boxes on jax, or
               --require-warp with --engine mujoco.
  exception    error, exit 1, with the traceback in the report.

The report goes to `--out`, default
`runs/check_terrain/<robot>_<preset>_<arena>_<fingerprint[:12]>_<engine>.json`.
"""

from __future__ import annotations

import os

from humanoid_lab import tasks

# Before jax creates its client: the terrain training process's allocator.
os.environ.update({k: os.environ.get(k, v) for k, v in tasks.TERRAIN_XLA_DEFAULTS.items()})

import argparse
import copy
import math
import sys
import time
import traceback
from collections.abc import Mapping
from typing import NamedTuple

import numpy as np

from humanoid_lab import (
    check_contacts,
    fd_capture,
    paths,
    registry,
    run_record,
    sim_budget,
)
from humanoid_lab.envs import terrain_geometry as tg
from humanoid_lab.envs.terrain_joystick import HFIELD_CONTACTS_PER_PAIR, JAX_BOX_LIMIT
from humanoid_lab.envs.terrain_joystick import default_config as terrain_default_config
from humanoid_lab.terrain import (
    GENERATOR_VERSION,
    TYPES,
    fingerprint,
    params_to_dict,
    scene,
)
from humanoid_lab.terrain.config import (
    arena_for,
    config_from_params,
    params_from_config,
)

# Importing train applies its XLA defaults too, so the gate runs under the
# training process's allocator settings.
from humanoid_lab.train import _apply_ppo_overrides, build_ppo_params

SCHEMA = 1
ENGINES = ("mjx", "mujoco")
REGIMES = ("stand", "walk", "fallen")
ARENAS = ("train", "eval")
HEADINGS = 8

DEFAULT_STEPS = 200
# Worlds per backend when --num-envs is not given. The mujoco engine
# defaults to one world per tile.
DEFAULT_NUM_ENVS = {"warp": 1024, "jax": 16}
# Measurement budgets for a recipe that sets none. nacon and nefc count past
# their buffers, but an UNDERCOUNTING overflow drops work first, so a
# default below the demand makes the recommendation a lower bound.
DEFAULT_NACONMAX_PER_ENV = 512
DEFAULT_NJMAX = 4096
# A buffer this full fails the gate.
DEFAULT_MAX_FILL = 0.9
# Headroom of each recommendation over its measured peak. None is tuned.
POOL_HEADROOM = 2.0
CCD_HEADROOM = 2.0
ROWS_HEADROOM = 2.0
# Command resampling period in control steps, past any rollout here.
# resample_steps is an int field.
RESAMPLE_NEVER = 10_000_000
# The messages that drop pairs or contacts before a counter sees them, so
# the measured peaks undercount the demand.
UNDERCOUNTING = ("ccd_overflow", "broadphase_overflow", "narrowphase_overflow")


class CheckRefused(Exception):
    """A request the gate refuses before it builds anything. Exit 2."""


class SpawnTable(NamedTuple):
    """One start per world."""

    level: np.ndarray  # (N,) arena row
    type: np.ndarray  # (N,) TYPES index
    xy: np.ndarray  # (N, 2) base xy
    yaw: np.ndarray  # (N,) heading, also the base yaw
    kind: np.ndarray  # (N,) tg.SPAWN_PAD on the flat row, tg.SPAWN_FEATURE elsewhere
    attitude: np.ndarray  # (N,) check_contacts.FALLEN_ATTITUDES index for `fallen`


# -- spawns -------------------------------------------------------------------------


def spawn_table(spec, feet_reach: float, n_envs: int) -> SpawnTable:
    """`n_envs` starts over the tiles of `spec` (a TerrainSpec), hardest
    first. Deterministic. See the module docstring for the rule.

    Raises when a tile is too small to keep the feet between the pad and
    the tile edge."""
    if n_envs < 1:
        raise ValueError(f"n_envs must be at least 1, got {n_envs}")
    index = {t: i for i, t in enumerate(TYPES)}
    tiles = sorted(spec.tiles, key=lambda t: (-t.difficulty, -t.row, index[t.terrain_type]))
    half = spec.params.tile_size / 2
    n = len(tiles)
    level = np.zeros(n_envs, np.int32)
    ttype = np.zeros(n_envs, np.int32)
    xy = np.zeros((n_envs, 2), np.float32)
    yaw = np.zeros(n_envs, np.float32)
    kind = np.zeros(n_envs, np.int32)
    attitude = np.zeros(n_envs, np.int32)
    n_attitudes = len(check_contacts.FALLEN_ATTITUDES)
    for i in range(n_envs):
        t = tiles[i % n]
        heading = 2 * math.pi * ((i + i // n) % HEADINGS) / HEADINGS
        level[i], ttype[i], yaw[i] = t.row, index[t.terrain_type], heading
        # A tile's attitude moves on by one each lap. i mod 3 pins a tile to
        # one attitude when the tile count is a multiple of 3, as the eval
        # arena's 48 is. (i + i // n) mod 3 does when it is 2 mod 3, as 80 is.
        attitude[i] = (i % n + i // n) % n_attitudes
        origin = np.asarray(t.origin[:2], np.float64)
        if spec.params.flat_row and t.row == 0:
            xy[i], kind[i] = origin, tg.SPAWN_PAD
            continue
        lo, hi = t.pad_radius + feet_reach, half - feet_reach
        if lo > hi:
            raise ValueError(
                f"a {feet_reach:.3f} m foot reach leaves no feature band between the "
                f"{t.pad_radius} m pad and the edge of a {spec.params.tile_size} m tile"
            )
        radius = min(max((t.pad_radius + t.feature_radius) / 2, lo), hi)
        direction = np.array([math.cos(heading), math.sin(heading)])
        stretch = 1.0 / np.abs(direction).max()
        xy[i], kind[i] = origin + radius * stretch * direction, tg.SPAWN_FEATURE
    return SpawnTable(level, ttype, xy, yaw, kind, attitude)


def tilt_quat(axis: str, degrees: float) -> np.ndarray:
    """(w, x, y, z) of a rotation by `degrees` about the x or y axis."""
    half = math.radians(degrees) / 2
    s = math.sin(half)
    return np.array([math.cos(half), s if axis == "x" else 0.0, s if axis == "y" else 0.0, 0.0])


def start_qpos(env, spawns: SpawnTable, regime: str):
    """(N, nq) start poses for `regime`: the reset pose at each spawn, and
    for `fallen` tilted and dropped."""
    import jax
    import jax.numpy as jp

    attitudes = check_contacts.FALLEN_ATTITUDES
    tilts = jp.asarray(np.stack([tilt_quat(*attitudes[a]) for a in spawns.attitude]))
    b = env._base_qadr
    fallen = regime == "fallen"

    def place(xy, yaw, kind, tilt):
        q = env.spawn_qpos(env._reset_qpos, xy, yaw, kind)
        if not fallen:
            return q
        quat = tg.quat_mul(tg.yaw_quat(yaw), tg.quat_mul(tilt, env._reset_quat))
        return q.at[b + 3 : b + 7].set(quat).at[b + 2].add(check_contacts.FALLEN_DROP_M)

    return jax.jit(jax.vmap(place))(
        jp.asarray(spawns.xy), jp.asarray(spawns.yaw), jp.asarray(spawns.kind), tilts
    )


def regime_actions(env, regime: str, steps: int):
    """(steps, nu) actions for `regime`. All worlds share them."""
    import jax.numpy as jp

    if regime == "walk":
        actions = check_contacts._walk_action(env.action_size, jp.arange(steps)[:, None], env.dt)
    else:
        actions = jp.zeros((steps, env.action_size))
    # One dtype for every regime, so they share one compiled rollout.
    return actions.astype(jp.float32)


def regime_ctrl(env, regime: str, steps: int) -> np.ndarray:
    """(steps, nu) ctrl for `regime`, as the env's step sets it: the
    actuator model's targets for the regime's actions, inside the env's
    clip. Both engines step with it."""
    actions = regime_actions(env, regime, steps)
    return np.clip(
        np.asarray(env._actuator_model.ctrl_from_action(actions, env._default_pose, env._action_scale)),
        np.asarray(env._ctrl_lo),
        np.asarray(env._ctrl_hi),
    )


# -- configuration ------------------------------------------------------------------


def _merge(base: dict, extra: Mapping) -> dict:
    """`extra` merged into `base` in place, recursively, and returned."""
    for key, value in extra.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), Mapping):
            _merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def gate_budgets(cfg: Mapping, args) -> dict:
    """The budgets the gate builds with: the recipe's `task.env.sim` value
    where set, else the flag. `source` says which, per budget."""
    sim = (cfg["task"].get("env") or {}).get("sim") or {}
    flags = {
        "naconmax_per_env": args.naconmax_per_env,
        "naccdmax_per_env": args.naccdmax_per_env,
        "njmax": args.njmax,
    }
    out, source = {}, {}
    for key, flag in flags.items():
        recipe = sim.get(key)
        out[key] = int(recipe) if recipe is not None else (None if flag is None else int(flag))
        source[key] = "recipe" if recipe is not None else "flag"
    return {**out, "source": source}


def check_overrides(env_overrides: Mapping, *, backend: str, num_envs: int, budgets: Mapping,
                    arena: Mapping | None = None) -> dict:
    """The recipe's env overrides plus the gate's: its backend, batch and
    budgets, and push, no-progress, command resampling and spawn grace
    off. `arena`, a full config block, replaces the recipe's arena."""
    out = _merge(copy.deepcopy(dict(env_overrides or {})), {
        "sim": {
            "backend": backend,
            "num_envs": int(num_envs),
            "naconmax_per_env": budgets["naconmax_per_env"],
            "naccdmax_per_env": budgets["naccdmax_per_env"],
            "njmax": budgets["njmax"],
        },
        "push": {"enable": False},
        "no_progress": {"enable": False},
        "command": {"resample_steps": RESAMPLE_NEVER},
        "terrain": {"spawn": {"grace_sec": 0.0}},
    })
    if arena is not None:
        out["terrain"]["arena"] = copy.deepcopy(dict(arena))
    return out


def eval_arena_block() -> dict:
    """The terrain scan's arena as a full config block."""
    try:
        from humanoid_lab.eval.terrain_suite import eval_arena_params
    except ImportError as e:
        raise CheckRefused(
            "--arena eval reads humanoid_lab.eval.terrain_suite.eval_arena_params, "
            "which this checkout does not have"
        ) from e
    return config_from_params(eval_arena_params())


def arena_block(kind: str) -> dict | None:
    """The arena block `--arena kind` swaps in. None keeps the recipe's."""
    if kind == "train":
        return None
    if kind == "eval":
        return eval_arena_block()
    raise CheckRefused(f"--arena must be one of {ARENAS}, got {kind!r}")


def effective_arena(env_overrides: Mapping, arena: Mapping | None):
    """The arena the env will build: the recipe's overrides on the terrain
    default config, or `arena` in their place. Cached, so the env reuses it."""
    cfg = terrain_default_config()
    registry._apply_overrides(cfg, {"terrain": (env_overrides or {}).get("terrain") or {}})
    block = cfg.terrain.arena.to_dict() if arena is None else dict(arena)
    return arena_for(params_from_config(block))


def train_num_envs(cfg: Mapping) -> int:
    """The batch a training run of `cfg` sizes its warp pools for, as
    train.py computes it: the larger of ppo.num_envs and ppo.num_eval_envs."""
    p = build_ppo_params({}, bool(cfg.get("smoke", False)))
    _apply_ppo_overrides(p, cfg["task"].get("ppo") or {})
    _apply_ppo_overrides(p, cfg.get("ppo") or {})
    return int(max(p.num_envs, p.get("num_eval_envs", 0)))


def compose(overrides: list[str]) -> dict:
    """The composed Hydra config as a plain container, `task=terrain` first."""
    from hydra import compose as hydra_compose
    from hydra import initialize_config_dir
    from omegaconf import OmegaConf

    with initialize_config_dir(version_base=None, config_dir=str(paths.CONFIGS_DIR)):
        cfg = hydra_compose(config_name="config", overrides=["task=terrain", *overrides])
    return OmegaConf.to_container(cfg, resolve=True)


def build_check_env(cfg: Mapping, args):
    """The terrain env the gate runs, built as training builds it.

    `args` carries the resolved backend and world count. All model
    construction lives here."""
    ea = registry.env_args_from_config(cfg)
    overrides = check_overrides(
        ea.env_overrides,
        backend=args.backend,
        num_envs=args.num_envs,
        budgets=gate_budgets(cfg, args),
        arena=args.arena_block,
    )
    return registry.make_env(ea.task, ea.robot_dir, ea.preset_name, overrides, ea.actuator_overrides)


# -- rollouts -----------------------------------------------------------------------


def run_mjx(env, spawns: SpawnTable, regimes, steps: int, seed: int) -> tuple[dict, float]:
    """Roll each regime on the env's backend. Returns ({regime: result},
    the warm-up rollout's seconds).

    The rollout steps the env's model and data with the env's ctrl, one
    physics step at a time, and reads the counters after each. With push
    off, that is the physics the env's own step runs. The env's reward,
    observation and termination work is skipped, as nothing here reads
    it."""
    import jax
    import jax.numpy as jp
    from mujoco import mjx

    n = len(spawns.xy)
    warp = env._backend == "warp"
    base = jax.jit(jax.vmap(env.reset))(jax.random.split(jax.random.PRNGKey(seed), n))
    forward = jax.jit(jax.vmap(lambda d: mjx.forward(env.mjx_model, d)))
    physics = jax.vmap(lambda d: mjx.step(env.mjx_model, d))

    def substep(data, _):
        data = physics(data)
        out = {"finite": jp.all(jp.isfinite(data.qpos))}
        if warp:
            out["nefc"], out["nacon"], out["ncollision"] = sim_budget.traced_counters(data)
        else:
            out["active"] = jp.max(jp.sum(sim_budget.contact_dist(data) < 0, axis=-1))
        return data, out

    def body(data, ctrl):
        data = data.replace(ctrl=jp.broadcast_to(ctrl, data.ctrl.shape))
        data, out = jax.lax.scan(substep, data, None, length=env.n_substeps)
        return data, worst_substep(out)

    rollout = jax.jit(lambda data, ctrl: jax.lax.scan(body, data, ctrl)[1])

    def start(regime):
        data = base.data.replace(
            qpos=start_qpos(env, spawns, regime),
            qvel=jp.zeros_like(base.data.qvel),
            ctrl=jp.broadcast_to(env._neutral_ctrl, base.data.ctrl.shape),
        )
        return forward(data), jp.asarray(regime_ctrl(env, regime, steps), base.data.ctrl.dtype)

    # Every regime has the same shapes, so one call compiles the rollout
    # for all of them. It is a real call, not lower().compile(), so warp's
    # kernel load stays out of the measurements too. Its messages are
    # discarded: the first regime's measured run rolls the same start.
    data, ctrl = start(regimes[0])
    with fd_capture.capture_fd1(warp):
        t0 = time.perf_counter()
        jax.block_until_ready(rollout(data, ctrl))
        compile_s = time.perf_counter() - t0

    results = {}
    for i, regime in enumerate(regimes):
        if i:
            data, ctrl = start(regime)
        with fd_capture.capture_fd1(warp) as lines:
            t0 = time.perf_counter()
            out = jax.block_until_ready(rollout(data, ctrl))
            steady = time.perf_counter() - t0
        results[regime] = _regime_result(out, warp, lines, steady, steps, n)
    return results, round(compile_s, 4)


def worst_substep(out: Mapping) -> dict:
    """One control step's record from its physics steps' records, stacked
    on axis 0: each counter's peak, and `finite` only if every step was."""
    import jax.numpy as jp

    return {k: jp.all(v, axis=0) if k == "finite" else jp.max(v, axis=0) for k, v in out.items()}


def _regime_result(out, warp: bool, lines, steady: float, steps: int, n: int) -> dict:
    out = {k: np.asarray(v) for k, v in out.items()}
    if warp:
        nacon, ncoll, nefc = out["nacon"], out["ncollision"], out["nefc"]
        demand = np.maximum(nacon, ncoll)
        peaks = {
            "nacon_pool_max": int(nacon.max()),
            "ncollision_pool_max": int(ncoll.max()),
            "nefc_max": int(nefc.max()),
            "active_max": None,
            "peak_step": int(demand.argmax()),
        }
    else:
        active = out["active"]
        peaks = {
            "nacon_pool_max": None,
            "ncollision_pool_max": None,
            "nefc_max": None,
            "active_max": int(active.max()),
            "peak_step": int(active.argmax()),
        }
    return {
        **peaks,
        "finite": bool(out["finite"].all()),
        "messages": fd_capture.count_warp_messages("\n".join(lines)) if warp else None,
        "steady_s": round(steady, 4),
        "env_steps_per_s": round(steps * n / steady, 1) if steady > 0 else None,
    }


def hfield_pair_counts(m, d, hfield_geom: int) -> np.ndarray:
    """(ngeom,) contacts each geom has with `hfield_geom` in `d`."""
    ncon = int(d.ncon)
    if ncon == 0:
        return np.zeros(m.ngeom, np.int64)
    g = np.asarray(d.contact.geom[:ncon])
    other = np.where(g[:, 0] == hfield_geom, g[:, 1], np.where(g[:, 1] == hfield_geom, g[:, 0], -1))
    return np.bincount(other[other >= 0], minlength=m.ngeom)


def _world_proxy(m, d, ctrl, n_substeps: int, hf: int, cap: int, peak, at_cap, resets) -> int | None:
    """Steps one world through `ctrl` and adds its pair counts to `peak`
    and `at_cap`. Returns the control step of a C reset, else None.

    C resets a diverging world to qpos0 inside mj_step and counts a
    bad-qpos, bad-qvel or bad-qacc warning. That step's contacts and every
    later one come from qpos0, a pose the spawn table never chose, so the
    world stops counting there."""
    import mujoco

    for i in range(len(ctrl)):
        d.ctrl[:] = ctrl[i]
        for _ in range(n_substeps):
            mujoco.mj_step(m, d)
            if any(d.warning[k].number for k in resets):
                return i
            counts = hfield_pair_counts(m, d, hf)
            np.maximum(peak, counts, out=peak)
            at_cap += counts >= cap
    return None


def run_proxy(env, spawns: SpawnTable, regimes, steps: int) -> dict:
    """Step `env.mj_model` in plain MuJoCo, one world at a time, and count
    each robot collider's contacts with the heightfield. Returns
    {regime: result}. `diverged` lists [world, control step] for each
    world C reset, and `finite` is True when it lists none."""
    import mujoco

    m = env.mj_model
    hf = m.geom(scene.HFIELD_GEOM).id
    cap = int(mujoco.mjMAXCONPAIR)
    names = {g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or str(g) for g in range(m.ngeom)}
    neutral = np.asarray(env._neutral_ctrl)
    warn = mujoco.mjtWarning
    resets = tuple(int(w) for w in (warn.mjWARN_BADQPOS, warn.mjWARN_BADQVEL, warn.mjWARN_BADQACC))
    results = {}
    for regime in regimes:
        qpos = np.asarray(start_qpos(env, spawns, regime))
        ctrl = regime_ctrl(env, regime, steps)
        peak = np.zeros(m.ngeom, np.int64)
        at_cap = np.zeros(m.ngeom, np.int64)
        diverged = []
        t0 = time.perf_counter()
        for w in range(len(qpos)):
            d = mujoco.MjData(m)
            d.qpos[:] = qpos[w]
            d.ctrl[:] = neutral
            mujoco.mj_forward(m, d)
            step = _world_proxy(m, d, ctrl, env.n_substeps, hf, cap, peak, at_cap, resets)
            if step is not None:
                diverged.append([w, step])
        results[regime] = {
            "at_cap": {names[g]: int(at_cap[g]) for g in np.flatnonzero(at_cap)},
            "max_pair_count": {names[g]: int(peak[g]) for g in np.flatnonzero(peak)},
            "finite": not diverged,
            "diverged": diverged,
            "steady_s": round(time.perf_counter() - t0, 4),
        }
    return results


def proxy_summary(regimes: Mapping) -> dict:
    """The proxy block: cap hits and peak pair counts over every regime."""
    import mujoco

    at_cap: dict[str, int] = {}
    peak: dict[str, int] = {}
    for r in regimes.values():
        for name, k in r["at_cap"].items():
            at_cap[name] = at_cap.get(name, 0) + k
        for name, k in r["max_pair_count"].items():
            peak[name] = max(peak.get(name, 0), k)
    return {"cap": int(mujoco.mjMAXCONPAIR), "at_cap": at_cap, "max_pair_count": peak}


# -- report -------------------------------------------------------------------------


def recommend(nacon_pool_max, ncollision_pool_max, nefc_max, num_envs: int, n_colliders: int,
              lower_bound, *, pool_headroom: float = POOL_HEADROOM, ccd_headroom: float = CCD_HEADROOM,
              rows_headroom: float = ROWS_HEADROOM) -> dict:
    """Budgets that clear the measured pool and row peaks.

    The pool is shared, so the per-env contact budget covers the batch's
    total demand over the worlds, not the worst world. `ncollision` counts
    every candidate pair, so it bounds each geom-type pair's CCD slot
    demand. MJWarp refuses a CCD pool larger than the contact pool, so
    naccdmax never exceeds naconmax. Unmeasured peaks give None.
    `floor_4x_colliders` is the per-world contact count with every
    ground-pairing collider writing its 4 heightfield contacts."""
    n = int(num_envs)
    peaks = [int(v) for v in (nacon_pool_max, ncollision_pool_max) if v is not None]
    nacon = None
    naccd = None
    if peaks:
        nacon = sim_budget.recommend_budget(math.ceil(max(peaks) / n), pool_headroom, check_contacts.NACON_STEP)
    if ncollision_pool_max is not None:
        naccd = sim_budget.recommend_budget(
            math.ceil(int(ncollision_pool_max) / n), ccd_headroom, check_contacts.NACON_STEP
        )
        naccd = naccd if nacon is None else min(naccd, nacon)
    njmax = None
    if nefc_max is not None:
        njmax = sim_budget.recommend_budget(int(nefc_max), rows_headroom, check_contacts.NJMAX_STEP)
    return {
        "naconmax_per_env": nacon,
        "naccdmax_per_env": naccd,
        "njmax": njmax,
        "floor_4x_colliders": HFIELD_CONTACTS_PER_PAIR * int(n_colliders),
        "pool_headroom": pool_headroom,
        "ccd_headroom": ccd_headroom,
        "rows_headroom": rows_headroom,
        "lower_bound": lower_bound,
    }


def ccd_scratch(bytes_per_slot: int, train_envs: int, recommended, budgets: Mapping) -> dict:
    """The CCD scratch a training run holds outside the XLA pool, at the
    recommended naccdmax_per_env, or at the gate's budget when nothing
    recommends one. The gate's budget is its naccdmax, else its naconmax,
    which MJWarp then sizes the scratch to."""
    if recommended is not None:
        per_env, source = int(recommended), "recommend"
    elif budgets.get("naccdmax_per_env") is not None:
        per_env, source = int(budgets["naccdmax_per_env"]), "budgets.naccdmax_per_env"
    else:
        per_env, source = int(budgets["naconmax_per_env"]), "budgets.naconmax_per_env"
    slots = per_env * int(train_envs)
    return {
        "bytes_per_slot": int(bytes_per_slot),
        "train_num_envs": int(train_envs),
        "naccdmax_per_env": per_env,
        "source": source,
        "slots_at_train": slots,
        "bytes_at_train": slots * int(bytes_per_slot),
    }


def action_window(env) -> dict | None:
    """{joint: [lo, hi]}: the ctrl targets an action in [-1, 1] reaches,
    inside the env's clip. Joint angles for a position-servo preset, None
    for any other actuator model."""
    from humanoid_lab.actuators.models import PositionPD

    if not isinstance(env._actuator_model, PositionPD):
        return None
    ends = [
        np.asarray(env._actuator_model.ctrl_from_action(a, env._default_pose, env._action_scale))
        for a in (-1.0, 1.0)
    ]
    lo = np.maximum(np.minimum(*ends), np.asarray(env._ctrl_lo))
    hi = np.minimum(np.maximum(*ends), np.asarray(env._ctrl_hi))
    joints = env.robot_spec.actuated_joints
    return {j: [round(float(a), 4), round(float(b), 4)] for j, a, b in zip(joints, lo, hi)}


def arena_record(env, kind: str) -> dict:
    a = env._arena
    hf = a.spec.hfield
    return {
        "kind": kind,
        "generator_version": GENERATOR_VERSION,
        "fingerprint": fingerprint(a),
        "params": params_to_dict(a.spec.params),
        "nrow": int(hf.nrow),
        "ncol": int(hf.ncol),
        "cell": float(a.spec.params.cell_size),
        "n_boxes": len(a.boxes),
    }


def model_record(env) -> dict:
    m = env.mj_model
    return {
        "ngeom": int(m.ngeom),
        "ground_geoms": len(scene.ground_geom_ids(m)),
        "robot_colliders": int(env._n_ground_colliders),
        "rows_per_contact": sim_budget.rows_per_contact(int(m.opt.cone), int(np.max(m.geom_condim))),
        "jax_max_contact_points": int(env._jax_contact_cap),
    }


def provenance() -> dict:
    commit, dirty = run_record.git_state(paths.REPO_ROOT)
    return {
        "git_commit": commit,
        "git_dirty": dirty,
        "versions": run_record.versions(),
        "device": run_record.device_info(),
    }


def _sum_messages(regimes: Mapping) -> dict | None:
    counted = [r["messages"] for r in regimes.values() if r.get("messages") is not None]
    if not counted:
        return None
    return {k: sum(c.get(k, 0) for c in counted) for k in fd_capture.WARP_MESSAGES}


def _peak(regimes: Mapping, key: str):
    values = [r[key] for r in regimes.values() if r.get(key) is not None]
    return max(values) if values else None


def verdict(report: Mapping, max_fill: float, require_warp: bool, strict: bool) -> tuple[str, list[str], int]:
    """(status, reasons, exit code) for a finished run. See the module
    docstring's exit rules."""
    if not report["regimes"]:
        # main refuses an empty --regimes. A run that rolled nothing never
        # passes.
        return "error", ["no regime ran"], 2
    reasons: list[str] = []
    if report["engine"] == "mujoco":
        proxy = report["proxy"]
        reset = sorted(name for name, r in report["regimes"].items() if not r["finite"])
        if reset:
            reasons.append(f"C reset diverging worlds in regimes {reset}. Their pair counts stop at the reset.")
        if proxy["at_cap"]:
            reasons.append(
                f"robot colliders reached the {proxy['cap']}-contact heightfield cap in C: "
                f"{sorted(proxy['at_cap'])}"
            )
        # A diverged world fails whatever --strict says, as on jax.
        if reset:
            return "fail", reasons, 1
        if proxy["at_cap"]:
            return "proxy_at_risk", reasons, 1 if strict else 0
        return "proxy_clear", reasons, 0

    broken = sorted(name for name, r in report["regimes"].items() if not r["finite"])
    if broken:
        reasons.append(f"non-finite qpos in regimes {broken}")
    if report["backend"] != "warp":
        if require_warp:
            reasons.append(
                f"--require-warp, and the backend is {report['backend']}: no overflow "
                "message, per-pair cap or live counter exists here"
            )
            return "unverified", reasons, 2
        if broken:
            return "fail", reasons, 1
        reasons.append(
            f"the {report['backend']} backend has no per-pair cap, no overflow message and "
            "no live counters"
        )
        return "unverified", reasons, 0

    hits = {k: n for k, n in (report["messages"] or {}).items() if n and k in fd_capture.GATING}
    if hits:
        reasons.append(f"MJWarp overflow messages: {hits}")
    fill = report["fill"] or {}
    for name, budget in (("pool", "the naconmax pool"), ("rows", "njmax")):
        value = fill.get(name)
        if value is not None and value >= max_fill:
            reasons.append(f"{name} fill {value:.3f} reached --max-fill {max_fill} of {budget}")
    if reasons:
        return "fail", reasons, 1
    return "pass", reasons, 0


def empty_report(args) -> dict:
    """Every report key in order, unset."""
    return {
        "schema": SCHEMA,
        "status": None,
        "reasons": [],
        "engine": args.engine,
        "backend": None,
        "provenance": None,
        "robot": None,
        "preset": None,
        "actuator_overrides": None,
        "action_window": None,
        "arena": None,
        "model": None,
        "num_envs": None,
        "steps": args.steps,
        "seed": args.seed,
        "compile_s": None,
        "budgets": None,
        "regimes": {},
        "jax_lower_bound": None,
        "fill": None,
        "messages": None,
        "recommend": None,
        "ccd_scratch": None,
        "proxy": None,
        "error": None,
        "timestamp": None,
    }


def default_out(report: Mapping, arena_kind: str) -> str:
    fp = ((report.get("arena") or {}).get("fingerprint") or "none")[:12]
    engine = report["engine"] if report["engine"] == "mujoco" else f"mjx-{report.get('backend') or 'none'}"
    name = f"{report.get('robot') or 'none'}_{report.get('preset') or 'none'}_{arena_kind}_{fp}_{engine}.json"
    return str(paths.RUNS_DIR / "check_terrain" / name)


def _resolve_backend(engine: str, backend: str) -> str:
    if engine == "mujoco":
        # The proxy steps C MuJoCo. jax builds the env without budgets or CUDA.
        return "jax"
    from humanoid_lab.envs.backend import resolve_backend

    return resolve_backend(backend)


def run(args, overrides: list[str], report: dict) -> tuple[str, list[str], int]:
    """Compose, build, roll and judge. Fills `report` in place."""
    report["provenance"] = provenance()
    cfg = compose(overrides)
    report["robot"] = cfg["robot"]["name"]
    report["preset"] = (cfg.get("actuators") or {}).get("name")
    report["actuator_overrides"] = (cfg.get("actuators") or {}).get("overrides") or {}
    task = cfg["task"]["name"]
    if not tasks.is_terrain(task):
        raise CheckRefused(f"check-terrain gates a terrain task, and the composed task is {task!r}")

    args.arena_block = arena_block(args.arena)
    args.backend = _resolve_backend(args.engine, args.backend)
    if args.num_envs is None:
        if args.engine == "mujoco":
            arena = effective_arena(registry.env_args_from_config(cfg).env_overrides, args.arena_block)
            args.num_envs = len(arena.spec.tiles)
        else:
            args.num_envs = DEFAULT_NUM_ENVS[args.backend]
    budgets = gate_budgets(cfg, args)

    env = build_check_env(cfg, args)
    boxes = getattr(env, "_n_ground_boxes", 0)
    if args.engine == "mjx" and env._backend == "jax" and boxes > JAX_BOX_LIMIT:
        raise CheckRefused(
            f"the arena has {boxes} ground boxes, over the jax limit of {JAX_BOX_LIMIT}. "
            "Run it on warp, or use --engine mujoco for the C proxy, or experiment=terrain_cpu on jax."
        )
    n = int(args.num_envs)
    report["backend"] = None if args.engine == "mujoco" else env._backend
    report["action_window"] = action_window(env)
    report["arena"] = arena_record(env, args.arena)
    report["model"] = model_record(env)
    report["num_envs"] = n
    report["budgets"] = {
        "naconmax_per_env": budgets["naconmax_per_env"],
        "naccdmax_per_env": budgets["naccdmax_per_env"],
        "njmax": budgets["njmax"],
        "pool": budgets["naconmax_per_env"] * n,
        "source": budgets["source"],
    }
    spawns = spawn_table(env._arena.spec, env._feet_reach, n)

    if args.engine == "mujoco":
        report["regimes"] = run_proxy(env, spawns, args.regimes, args.steps)
        report["proxy"] = proxy_summary(report["regimes"])
        recommended = None
        report["recommend"] = recommend(None, None, None, n, env._n_ground_colliders, None)
    else:
        report["regimes"], report["compile_s"] = run_mjx(env, spawns, args.regimes, args.steps, args.seed)
        warp = env._backend == "warp"
        report["messages"] = _sum_messages(report["regimes"])
        nacon = _peak(report["regimes"], "nacon_pool_max")
        ncoll = _peak(report["regimes"], "ncollision_pool_max")
        nefc = _peak(report["regimes"], "nefc_max")
        pool = sim_budget.pool_report(
            env._backend, nacon, ncoll, nefc, budgets["naconmax_per_env"], budgets["njmax"], n
        )
        report["fill"] = {"pool": pool["fill_pool"], "rows": pool["fill_rows"]}
        lower = None
        if warp:
            lower = bool(any(report["messages"].get(k) for k in UNDERCOUNTING))
        else:
            report["jax_lower_bound"] = {
                "box_colliders": has_box_colliders(env),
                "capped_at": int(env._jax_contact_cap),
            }
        report["recommend"] = recommend(nacon, ncoll, nefc, n, env._n_ground_colliders, lower)
        recommended = report["recommend"]["naccdmax_per_env"]
    report["ccd_scratch"] = ccd_scratch(env._ccd_slot_bytes, train_num_envs(cfg), recommended, budgets)
    return verdict(report, args.max_fill, args.require_warp, args.strict)


def has_box_colliders(env) -> bool:
    """Whether a robot collider is a box, whose low-side heightfield
    prisms jax misses when it is yawed or rolled."""
    import mujoco

    types = np.asarray(env.mj_model.geom_type)[np.unique(env._spawn_owner)]
    return bool(np.any(types == int(mujoco.mjtGeom.mjGEOM_BOX)))


def summary(report: Mapping) -> str:
    lines = [
        (
            f"check-terrain {report['status']}  engine {report['engine']}  backend {report['backend']}  "
            f"robot {report['robot']}  preset {report['preset']}  worlds {report['num_envs']}"
        )
    ]
    for name, r in (report.get("regimes") or {}).items():
        if report["engine"] == "mujoco":
            lines.append(
                f"  {name:<7} at cap {r['at_cap'] or '-'}  finite {r['finite']}  "
                f"diverged {r.get('diverged') or '-'}"
            )
        else:
            lines.append(
                f"  {name:<7} nacon {r['nacon_pool_max']}  ncollision {r['ncollision_pool_max']}  "
                f"nefc {r['nefc_max']}  active {r['active_max']}  finite {r['finite']}  "
                f"{r['env_steps_per_s']} env-steps/s"
            )
    rec = report.get("recommend")
    if rec and rec.get("naconmax_per_env") is not None:
        lines.append(
            f"  recommend naconmax_per_env={rec['naconmax_per_env']} "
            f"naccdmax_per_env={rec['naccdmax_per_env']} njmax={rec['njmax']}"
            + ("  (lower bound)" if rec.get("lower_bound") else "")
        )
    lines += [f"  {reason}" for reason in report.get("reasons") or []]
    return "\n".join(lines)


def parse_regimes(text: str) -> tuple[str, ...]:
    """The regimes `--regimes` names, in order, each once. Refuses an
    unknown name and a list that names none."""
    names = tuple(dict.fromkeys(r.strip() for r in text.split(",") if r.strip()))
    if not names:
        raise CheckRefused(f"--regimes names no regime, have {list(REGIMES)}")
    bad = sorted(set(names) - set(REGIMES))
    if bad:
        raise CheckRefused(f"unknown regimes {bad}, have {list(REGIMES)}")
    return names


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Gate a terrain recipe against MJWarp's buffers. Leftover arguments are Hydra overrides."
    )
    ap.add_argument("--engine", choices=ENGINES, default="mjx")
    ap.add_argument("--backend", choices=["auto", "warp", "jax"], default="auto")
    ap.add_argument("--arena", choices=ARENAS, default="train")
    ap.add_argument(
        "--num-envs", type=int, default=None,
        help=(
            f"worlds (default: {DEFAULT_NUM_ENVS['warp']} on warp, {DEFAULT_NUM_ENVS['jax']} on jax, "
            "one per tile with --engine mujoco)"
        ),
    )
    ap.add_argument("--steps", type=int, default=DEFAULT_STEPS, help="control steps per regime")
    ap.add_argument("--regimes", default=",".join(REGIMES), help="comma-separated subset of " + ",".join(REGIMES))
    ap.add_argument(
        "--naconmax-per-env", type=int, default=DEFAULT_NACONMAX_PER_ENV,
        help="contact budget per world when the recipe sets none",
    )
    ap.add_argument(
        "--naccdmax-per-env", type=int, default=None,
        help="CCD slots per world when the recipe sets none (default: the naconmax pool)",
    )
    ap.add_argument("--njmax", type=int, default=DEFAULT_NJMAX, help="rows per world when the recipe sets none")
    ap.add_argument("--max-fill", type=float, default=DEFAULT_MAX_FILL, help="pool and row fill that fails")
    ap.add_argument("--require-warp", action="store_true", help="exit 2 unless the mjx engine runs on warp")
    ap.add_argument("--strict", action="store_true", help="exit 1 on a C proxy cap hit")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="report path")
    return ap


def main(argv=None) -> int:
    args, overrides = parser().parse_known_args(argv)
    unknown = [a for a in overrides if a.startswith("-")]
    report = empty_report(args)
    try:
        if unknown:
            raise CheckRefused(f"unknown flags {unknown}")
        args.regimes = parse_regimes(args.regimes)
        if args.steps < 1:
            raise CheckRefused(f"--steps must be at least 1, got {args.steps}")
        if args.require_warp and args.engine != "mjx":
            raise CheckRefused("--require-warp gates the mjx engine on warp, and --engine is mujoco")
        status, reasons, code = run(args, overrides, report)
    except CheckRefused as e:
        status, reasons, code = "error", [str(e)], 2
        report["error"] = str(e)
    except Exception:  # noqa: BLE001  # any failure still writes its report
        tb = traceback.format_exc()
        status, reasons, code = "error", [tb.strip().splitlines()[-1]], 1
        report["error"] = tb
        print(tb, file=sys.stderr)
    report["status"], report["reasons"] = status, reasons
    report["timestamp"] = run_record.utc_now()
    out = args.out or default_out(report, args.arena)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    run_record.write_json_atomic(out, report)
    print(summary(report))
    print(f"wrote {out}")
    return code


if __name__ == "__main__":
    sys.exit(main())
