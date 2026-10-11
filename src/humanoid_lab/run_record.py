"""The one reader of run.json.

Battery, video (through battery), export and sizing load a run through
`load_run`. The Torch warm start uses it too.

A Brax run comes back as written. A legacy Torch run (one written by the
old wtrain trainer) carries its extra knobs in top-level `warp_*` keys,
outside hydra_config. `load_run` moves the ones this repo still has into
`hydra_config.task.env`, drops the rest of the `warp_*` keys and sets
`env_config` to the migrated task.env. Every check after that reads the
migrated task.env. The old trainer copied env_config from its source run,
so that copy can be stale.

A legacy run is refused when it cannot be rebuilt: it sets a reward term
that PR #11 removed, it carries an unknown `warp_*` key, or it froze the
gait clock without saying which phase observation it trained on.

This module imports only the standard library.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

REQUIRED_KEYS = ("task", "hydra_config", "checkpoint_dir", "ppo_config")

# The four warp_reward terms this repo kept, and the parameters each may
# carry beside its scale. step_length's `target` is not listed: the old
# trainer chose the mode from the presence of target_a, and the migration
# writes the mode out.
KEPT_TERM_PARAMS = {
    "step_length": ("target_a", "target_b", "power", "pair_min"),
    "foot_reach": ("target_a", "target_b", "power", "x_min", "pair_min"),
    "knee_extend": ("flex_lo", "flex_hi", "x_split", "td_min_air", "td_mult", "pair_min"),
    "forward_lean": ("max_deg", "vx_max", "sigma"),
}

# The warp_reward terms PR #11 removed. A run that set one cannot be rebuilt.
DELETED_TERMS = (
    "knee_swing",
    "arm_swing",
    "elbow_hyperext",
    "push_off",
    "hip_swing",
    "swing_back",
    "swing_forward",
    "heading_hold",
    "march",
    "lateral_hold",
    "flight",
    "landing_tax",
    "progress",
    "step_symmetry",
)

# Every warp_* key the migration knows.
LEGACY_KEYS = ("warp_reward", "warp_cmd_ramp", "warp_cmd_straight", "warp_lift")

# envs/joystick.py's FROZEN_PHASE_OBS. It is repeated here so this module
# stays free of jax. test_run_record pins the two equal.
FROZEN_PHASE_OBS = ("cos_sin", "zeros")


def load_run(run_dir) -> dict:
    """Read `<run_dir>/run.json`, check it can rebuild an env, and migrate
    a legacy Torch run.

    The checkpoint directory is not checked here. Callers look for it when
    they need it.
    """
    run_dir = Path(run_dir)
    run_json = run_dir / "run.json"
    if not run_json.exists():
        raise FileNotFoundError(
            f"{run_json} not found -- is {run_dir} a completed run.sh train/smoke output dir?"
        )
    run = json.loads(run_json.read_text(encoding="utf-8"))

    missing = [k for k in REQUIRED_KEYS if k not in run]
    if missing:
        raise ValueError(f"{run_json} is missing required key(s) {missing}")

    hydra = run["hydra_config"]
    for section, key in (("robot", "dir"), ("actuators", "name")):
        if section not in hydra or key not in hydra[section]:
            raise ValueError(
                f"{run_json}: hydra_config.{section}.{key} missing -- cannot rebuild the env"
            )

    if is_legacy(run):
        run = migrate_legacy(run, source=str(run_json))
    return run


def is_legacy(run: dict) -> bool:
    """True for a run the old Torch trainer wrote.

    That trainer wrote `trainer` as a string. This repo's trainers write a
    dict there, or nothing.
    """
    return any(k.startswith("warp_") for k in run) or isinstance(run.get("trainer"), str)


def migrate_legacy(run: dict, source: str = "run.json") -> dict:
    """A new run dict with the legacy `warp_*` knobs moved into
    `hydra_config.task.env` and `env_config` set to a copy of that task.env.

    Raises ValueError for a run that cannot be rebuilt. `source` names the
    run in the messages.
    """
    run = copy.deepcopy(run)
    legacy = {k: run.pop(k) for k in [k for k in run if k.startswith("warp_")]}

    unknown = sorted(set(legacy) - set(LEGACY_KEYS))
    if unknown:
        raise ValueError(
            f"{source}: unknown legacy key(s) {unknown}. load_run migrates only "
            f"{list(LEGACY_KEYS)}, so this run cannot be rebuilt"
        )

    reward = legacy.get("warp_reward") or {}
    deleted = sorted(set(reward) & set(DELETED_TERMS))
    if deleted:
        raise ValueError(
            f"{source}: warp_reward sets {deleted}. PR #11 removed these reward "
            "terms, so this run cannot be rebuilt"
        )
    unknown_terms = sorted(set(reward) - set(KEPT_TERM_PARAMS))
    if unknown_terms:
        raise ValueError(f"{source}: unknown warp_reward term(s) {unknown_terms}")

    hydra = run["hydra_config"]
    task = hydra.get("task") or {}
    hydra["task"] = task
    env = task.get("env") or {}
    task["env"] = env

    for term, allowed in KEPT_TERM_PARAMS.items():
        if term not in reward:
            continue
        spec = dict(reward[term])
        if "scale" not in spec:
            raise ValueError(f"{source}: warp_reward.{term} has no scale")
        scale = spec.pop("scale")
        bad = sorted(set(spec) - set(allowed))
        if bad:
            raise ValueError(
                f"{source}: warp_reward.{term} has unknown parameter(s) {bad}. "
                f"Allowed: {list(allowed)}"
            )
        _put(env, ("reward", "scales", term), scale, source)
        if term == "step_length":
            spec["target"] = "linear" if "target_a" in spec else "clock"
        for key, value in spec.items():
            _put(env, ("reward", term, key), value, source)

    # The old trainer read a falsy ramp, lift or straight value as off.
    for legacy_key, path, keys in (
        ("warp_cmd_ramp", ("command", "ramp"), ("lin", "yaw")),
        ("warp_lift", ("gait", "lift"), ("v_full", "min_frac")),
    ):
        block = legacy.get(legacy_key)
        if not block:
            continue
        bad = sorted(set(block) ^ set(keys))
        if bad:
            raise ValueError(
                f"{source}: {legacy_key} = {block} must carry exactly the keys {list(keys)}"
            )
        for key in keys:
            _put(env, (*path, key), block[key], source)
    if legacy.get("warp_cmd_straight"):
        _put(env, ("command", "straight_prob"), legacy["warp_cmd_straight"], source)

    gait = env.get("gait") or {}
    freq = gait.get("freq")
    if freq is not None and all(float(f) == 0.0 for f in freq) and "frozen_phase_obs" not in gait:
        raise ValueError(
            f"{source}: a legacy Torch run with hydra_config.task.env.gait.freq = {list(freq)} "
            "and no frozen_phase_obs. The Torch trainer observed zeros for a frozen clock "
            "from some version on, so the phase observation this run trained on cannot be "
            "inferred. Add task.env.gait.frozen_phase_obs to its hydra_config, one of "
            f"{list(FROZEN_PHASE_OBS)}"
        )

    run["env_config"] = copy.deepcopy(env)
    return run


def _put(env: dict, path: tuple, value, source: str) -> None:
    """Set env[path] = value, creating dicts on the way. A different value
    already there raises."""
    node = env
    for key in path[:-1]:
        child = node.get(key)
        if child is None:
            child = node[key] = {}
        elif not isinstance(child, dict):
            raise ValueError(
                f"{source}: hydra_config.task.env.{'.'.join(path)}: {key} = {child!r} is not a block"
            )
        node = child
    leaf = path[-1]
    if leaf in node and node[leaf] != value:
        raise ValueError(
            f"{source}: hydra_config.task.env.{'.'.join(path)} = {node[leaf]!r} conflicts with "
            f"the legacy value {value!r}"
        )
    node[leaf] = value
