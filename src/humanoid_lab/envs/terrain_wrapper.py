"""The terrain training stack's auto-reset: respawn on a curriculum tile.

Playground's stock auto-reset (`BraxAutoResetWrapper`, full_reset=False)
restores the first reset's cached data and obs on done and leaves info
alone, so a level held in info would never move. Its full reset re-runs
`env.reset` inside every step. This wrapper keeps the cheap restore, then
moves the base: on done it steps the env's level with
`curriculum.curriculum_step`, draws a spawn on the new tile with the env's
`draw_spawn`, and writes `spawn_qpos` into the restored data. Timeouts
exist only above brax's EpisodeWrapper, so the level change lives here, at
the top of the stack.

What a respawn changes:
- data: the first reset's, with the base moved to the new spawn. The
  joints are that first reset's. The yaw composes onto the reset
  quaternion, so yaws never accumulate.
- obs: the first reset's, with `height` and the critic's height scan
  recomputed at the new pose, in whichever list names them. The other
  components read the body frame or the joints, which a translation and a
  turn about the vertical leave unchanged. The command channel is one
  step stale, as in the stock stack. The critic's `contacts`, `linvel` and
  `actuator_force` are the first reset's for one step.
- info: the terrain keys restart at the new spawn. Everything else
  survives, as in the stock stack: the command, the gait phase, air time
  and last contact.

The curriculum reads two distances. The flat row's crossing reads the
displacement from the spawn. The fail test reads `served_dist`, the
env's sum of `curriculum.served_step`.

`reset` stamps `info['curriculum_free']`, False for a pinned env. It
survives every respawn, as all info does. The env's free-level metrics
read it, so the curriculum level in the training metrics leaves pinned
envs out.

`terrain/promoted` and `terrain/demoted` are 1 on the step that respawns
an env that earned the move. The env's step carries the incoming metrics
through, so EpisodeWrapper adds them on the next step: each move counts
once, in the episode it starts.

On warp the wrapper also tracks the batch's peak constraint rows and pool
counters (`sim_budget.traced_counters`), as metrics. Each wrapper prints
one warning per counter, the first time the counter reaches 90% of its
budget. Brax's evaluator resets the eval env before every eval, and the
reset zeroes the traced peaks. The wrapper therefore keeps the counters
it has warned about on the host.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jp
import numpy as np
from brax.envs.wrappers import training as brax_training
from mujoco_playground import wrapper as playground_wrapper
from mujoco_playground._src import mjx_env

from humanoid_lab import sim_budget
from humanoid_lab.envs import curriculum, height_scan
from humanoid_lab.envs import terrain_geometry as tg
from humanoid_lab.envs.terrain_joystick import (
    CURRICULUM_METRICS,
    TELEMETRY_METRICS,
    TELEMETRY_PEAKS,
)

# obs.<list> name -> the observation dict key it fills.
_OBS_KEYS = {"state": "state", "privileged": "privileged_state"}


def wrap_for_terrain_brax_training(
    env: mjx_env.MjxEnv,
    episode_length: int = 1000,
    action_repeat: int = 1,
    randomization_fn=None,
    full_reset: bool = False,
):
    """Playground's training stack with the curriculum auto-reset on top.

    The signature is `wrap_for_brax_training`'s, so brax calls it the same
    way for the training env and the eval env. `full_reset=True` is
    refused: the respawn is this stack's reset."""
    if full_reset:
        raise ValueError(
            "wrap_for_terrain_brax_training does not take full_reset: the curriculum "
            "respawn replaces the reset"
        )
    if randomization_fn is None:
        env = brax_training.VmapWrapper(env)
    else:
        env = playground_wrapper.BraxDomainRandomizationVmapWrapper(env, randomization_fn)
    env = brax_training.EpisodeWrapper(env, episode_length, action_repeat)
    return TerrainAutoResetWrapper(env, episode_length=episode_length)


class TerrainAutoResetWrapper(playground_wrapper.BraxAutoResetWrapper):
    """The stock auto-reset with a curriculum respawn. `reset` and the
    `AutoResetWrapper_*` info keys are the parent's.

    `episode_length` is the length brax's EpisodeWrapper ends episodes at.
    The curriculum projects a short episode's commanded distance onto it."""

    def __init__(self, env: Any, *, episode_length: int):
        super().__init__(env, full_reset=False)
        base = env.unwrapped
        if not hasattr(base, "draw_spawn"):
            raise TypeError(f"{type(base).__name__} has no terrain spawns; use the stock auto-reset")
        cur = base._config.terrain.curriculum
        if int(cur.demote_strikes) < 1:
            raise ValueError(
                f"terrain.curriculum.demote_strikes must be at least 1, got {cur.demote_strikes}"
            )
        # Validates the fractions now, not at the first trace.
        curriculum.pinned_layout(1, float(cur.pinned_frac), float(cur.pinned_flat_frac), 1)
        self._base = base
        self._episode_length = int(episode_length)
        self._demote_fraction = float(cur.demote_fraction)
        self._demote_strikes = int(cur.demote_strikes)
        self._pinned_frac = float(cur.pinned_frac)
        self._pinned_flat_frac = float(cur.pinned_flat_frac)

        # Static splice targets: where `height` and the scan sit in the
        # cached observation, or None when no list names them.
        self._height_slices = {
            key: base.obs_slices(which).get("height") for which, key in _OBS_KEYS.items()
        }
        self._scan_slice = base.obs_slices("privileged").get(height_scan.NAME)

        self._telemetry = base._backend == "warp"
        if self._telemetry:
            self._caps = sim_budget.telemetry_caps(base)
            # Counters this wrapper has warned about, across every reset.
            self._warned: set[str] = set()

    def _pinned_layout(self, n: int):
        """(pinned, level) of a batch of `n` envs, numpy."""
        return curriculum.pinned_layout(
            n, self._pinned_frac, self._pinned_flat_frac, self._base._tables.n_rows
        )

    def reset(self, rng: jax.Array) -> mjx_env.State:
        """The parent's reset, with each env's `curriculum_free` flag."""
        state = super().reset(rng)
        pinned, _ = self._pinned_layout(state.done.shape[0])
        return state.replace(info={**state.info, "curriculum_free": jp.asarray(~pinned)})

    # -- one env's respawn (vmapped) --------------------------------------------
    def _respawn(self, done, level, ttype, spawn_xy, last_xy, commanded, served, steps_lived, rng,
                 strikes, cheby_min, cheby_max, cached_qpos, pinned, pinned_level):
        """The respawn of one env. Every output keeps its old value unless
        `done`."""
        base = self._base
        t = base._tables
        walked = jp.linalg.norm(last_xy - spawn_xy)
        crossed = curriculum.crossed_rule(
            base._on_flat(level), t.feature_r[level, ttype], walked, cheby_min, cheby_max,
            t.pad_radius, t.tile_size,
        )
        new_level, new_strikes, rng_next, promoted, demoted = curriculum.curriculum_step(
            level, served, commanded, steps_lived, rng,
            episode_length=self._episode_length,
            n_rows=t.n_rows,
            demote_fraction=self._demote_fraction,
            strikes=strikes,
            demote_strikes=self._demote_strikes,
            crossed=crossed,
            grace_steps=base._grace_steps,
            pinned=pinned,
        )
        # A pinned env holds its own row from its first respawn on.
        new_level = jp.where(pinned, pinned_level, new_level)
        rng_next, r_spawn = jax.random.split(rng_next)
        xy, yaw, kind = base.draw_spawn(r_spawn, ttype, new_level)
        qpos = base.spawn_qpos(cached_qpos, xy, yaw, kind)
        origin = t.origin_xy[new_level, ttype]
        r0 = tg.chebyshev(xy, origin)

        def keep(new, old):
            return jp.where(done, new, old)

        return {
            "qpos": keep(qpos, cached_qpos),
            "terrain_level": keep(new_level, level),
            "curriculum_strikes": keep(new_strikes, strikes),
            "terrain_rng": keep(rng_next, rng),
            "spawn_xy": keep(xy, spawn_xy),
            "spawn_kind": kind,
            "tile_origin": origin,
            "r0": r0,
            "promoted": promoted & done,
            "demoted": demoted & done,
        }

    # -- observation splice -------------------------------------------------------
    def _resplice(self, obs, done, reset_data, qpos):
        """`height` and the scan of the respawned envs, at the new pose.

        Height is the base above the lookup ground, as the env's `height`
        reads it. The scan's reference is the lowest sole. A respawn moves
        the base and turns it about the vertical, and that leaves every
        point's height above the base unchanged, so the cached first reset's
        lowest sole shifted by the base's own z shift is the new one."""
        base = self._base
        b = base._base_qadr
        xy, z, quat = qpos[:, b : b + 2], qpos[:, b + 2], qpos[:, b + 3 : b + 7]
        obs = dict(obs)

        def splice(vector, where, values):
            return vector.at[:, where].set(jp.where(done[:, None], values, vector[:, where]))

        if any(s is not None for s in self._height_slices.values()):
            height = jax.vmap(base._height_at)(xy, z)[:, None]
            for key, where in self._height_slices.items():
                if where is not None:
                    obs[key] = splice(obs[key], where, height)
        if self._scan_slice is not None:
            ref = jax.vmap(base._sole_ref_z)(reset_data) + (z - reset_data.qpos[:, b + 2])
            scan = jax.vmap(base._scan_at)(xy, quat, ref)
            obs["privileged_state"] = splice(obs["privileged_state"], self._scan_slice, scan)
        return obs

    # -- warp telemetry -----------------------------------------------------------
    def _warn(self, peaks, fire):
        """Prints the fired counters this wrapper has not warned about yet.
        Under pmap each device calls back with its own batch's peaks, and
        the record spans the devices too."""
        fresh = np.array([n not in self._warned for n in sim_budget.TELEMETRY_COUNTERS])
        fire = np.asarray(fire) & fresh
        if not fire.any():
            return
        print(sim_budget.telemetry_warning(peaks, self._caps, fire), flush=True)
        self._warned.update(n for n, f in zip(sim_budget.TELEMETRY_COUNTERS, fire) if f)

    def _track_counters(self, data, info, metrics):
        """Running peaks of the warp counters, in info and as metrics. The
        peaks are batch-wide, the same in every env."""
        counters = sim_budget.traced_counters(data)
        prev = jp.stack([info[k][0] for k in TELEMETRY_PEAKS])
        peaks, fracs, fire = sim_budget.telemetry_step(prev, jp.stack(counters), self._caps)
        jax.lax.cond(
            jp.any(fire),
            lambda p, f: jax.debug.callback(self._warn, p, f),
            lambda p, f: None,
            peaks,
            fire,
        )
        for i, k in enumerate(TELEMETRY_PEAKS):
            info[k] = jp.full_like(info[k], peaks[i])
        nefc_metric, nacon_metric, ncollision_metric = TELEMETRY_METRICS
        n = info[TELEMETRY_PEAKS[0]].shape
        metrics[nefc_metric] = jp.full(n, peaks[0], jp.float32)
        metrics[nacon_metric] = jp.full(n, fracs[1], jp.float32)
        metrics[ncollision_metric] = jp.full(n, fracs[2], jp.float32)

    # -- step ---------------------------------------------------------------------
    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        key = self._info_key
        reset_rng = jax.vmap(jax.random.split)(state.info[f"{key}_rng"])[..., 0]
        reset_data = state.info[f"{key}_first_data"]
        reset_obs = state.info[f"{key}_first_obs"]
        if "steps" in state.info:
            steps = state.info["steps"]
            steps = jp.where(state.done, jp.zeros_like(steps), steps)
            state = state.replace(info={**state.info, "steps": steps})
        state = state.replace(done=jp.zeros_like(state.done))
        state = self.env.step(state, action)
        done = state.done > 0
        info = dict(state.info)
        metrics = dict(state.metrics)

        if self._telemetry:
            self._track_counters(state.data, info, metrics)

        pinned, pinned_level = self._pinned_layout(done.shape[0])
        r = jax.vmap(self._respawn)(
            done,
            info["terrain_level"],
            info["terrain_type"],
            info["spawn_xy"],
            info["last_xy"],
            info["commanded_dist"],
            info["served_dist"],
            info["since_spawn"],
            info["terrain_rng"],
            info["curriculum_strikes"],
            info["cheby_min"],
            info["cheby_max"],
            reset_data.qpos,
            jp.asarray(pinned),
            jp.asarray(pinned_level),
        )

        def where_done(x, y):
            d = done
            if d.shape and d.shape[0] != x.shape[0]:
                return y  # a pool-wide warp field with no env axis
            if d.shape:
                d = jp.reshape(d, [x.shape[0]] + [1] * (len(x.shape) - 1))
            return jp.where(d, x, y)

        data = jax.tree.map(where_done, reset_data.replace(qpos=r["qpos"]), state.data)
        obs = jax.tree.map(where_done, reset_obs, state.obs)
        obs = self._resplice(obs, done, reset_data, r["qpos"])

        for k in ("terrain_level", "curriculum_strikes", "terrain_rng", "spawn_xy"):
            info[k] = r[k]
        info["spawn_kind"] = jp.where(done, r["spawn_kind"], info["spawn_kind"])
        info["tile_origin"] = jp.where(done[:, None], r["tile_origin"], info["tile_origin"])
        info["last_xy"] = jp.where(done[:, None], r["spawn_xy"], info["last_xy"])
        info["cheby_min"] = jp.where(done, r["r0"], info["cheby_min"])
        info["cheby_max"] = jp.where(done, r["r0"], info["cheby_max"])
        for k in ("commanded_dist", "served_dist", "since_spawn"):
            info[k] = jp.where(done, jp.zeros_like(info[k]), info[k])
        promoted_metric, demoted_metric = CURRICULUM_METRICS
        metrics[promoted_metric] = r["promoted"].astype(jp.float32)
        metrics[demoted_metric] = r["demoted"].astype(jp.float32)

        info[f"{key}_done_count"] = info[f"{key}_done_count"] + state.done.astype(int)
        info[f"{key}_rng"] = reset_rng
        return state.replace(data=data, obs=obs, info=info, metrics=metrics)
