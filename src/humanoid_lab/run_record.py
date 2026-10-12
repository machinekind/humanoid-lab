"""The one reader and the one writer of run.json.

Both trainers build the record with `build_run_record` and write it with
`write_run`. The Brax trainer writes once, at the end of training.
jobs/train_chain.sh reads a fresh run.json as proof that phase A finished.
The Torch trainer writes at start with null results and again at the end.

Paths in the record are relative to the project dir, in POSIX form, when
they lie inside it. `load_run` resolves a relative checkpoint_dir against
the project dir. The file on disk is never rewritten.

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

At module scope it imports only the standard library and humanoid_lab.paths.
"""

from __future__ import annotations

import copy
import json
import os
from importlib import metadata
from pathlib import Path

from humanoid_lab import paths

# Every key the Brax trainer wrote on origin/main, in its order.
ORIGIN_MAIN_KEYS = (
    "run_name",
    "task",
    "num_timesteps",
    "early_stopped",
    "stopped_at_steps",
    "final_reward",
    "checkpoint_dir",
    "contacts",
    "env_config",
    "ppo_config",
    "hydra_config",
    "actuators",
    "actuator_gains",
)
RECORD_KEYS = ORIGIN_MAIN_KEYS + ("trainer", "warm_start")

# The results build_run_record takes from the trainer. Values may be None.
RESULT_KEYS = ("early_stopped", "stopped_at_steps", "final_reward")

# The source run's results a warm start copies.
SOURCE_RESULT_KEYS = ("final_reward", "stopped_at_steps", "early_stopped", "num_timesteps")

# The packages `versions` reports. It reads their metadata and never imports them.
VERSION_PACKAGES = ("brax", "jax", "torch", "warp-lang", "mujoco", "mujoco-mjx")

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
    ckpt = run["checkpoint_dir"]
    if isinstance(ckpt, str) and not Path(ckpt).is_absolute():
        run["checkpoint_dir"] = resolve_path(ckpt)
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


# -- writing run.json -----------------------------------------------------------


def portable_path(path) -> str:
    """`path` as run.json stores it.

    A relative path is taken relative to the project dir, never the cwd, as
    train.py resolves `restore`. Inside the project dir the result is a
    relative POSIX string. Outside it is the absolute POSIX string.
    """
    p = Path(path)
    if not p.is_absolute():
        p = paths.PROJECT_DIR / p
    p = p.resolve()
    try:
        return p.relative_to(Path(paths.PROJECT_DIR).resolve()).as_posix()
    except ValueError:
        return p.as_posix()


def resolve_path(value) -> str | None:
    """A run.json path as a native path string. A relative value is taken
    relative to the project dir. None stays None."""
    if value is None:
        return None
    p = Path(value)
    if not p.is_absolute():
        p = paths.PROJECT_DIR / p
    return str(p)


def versions() -> dict:
    """The installed version of each package in VERSION_PACKAGES, None when
    it is not installed."""
    out = {}
    for name in VERSION_PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def trainer_block(name: str, physics: str, device, **extra) -> dict:
    """run.json's `trainer` block: the trainer, its physics backend, the
    package versions and the device. `extra` holds trainer-only fields."""
    return {"name": name, "physics": physics, "versions": versions(), "device": str(device), **extra}


def actuator_gains(env, hydra: dict) -> dict:
    """The `actuator_gains` block, read off the built model."""
    from humanoid_lab.robot.presets import effective_gains, load_actuator_preset

    robot_dir = paths.REPO_ROOT / hydra["robot"]["dir"]
    preset = hydra["actuators"]["name"]
    overrides = (hydra["actuators"] or {}).get("overrides") or {}
    # Gains as the compiled model holds them. Read here rather than from
    # hydra's actuators block: the yaml is what was asked for, and
    # actuators.overrides merges on top of it inside load_actuator_preset.
    # The preset is re-loaded through that same choke point purely for its
    # model name, which the mjModel does not carry.
    return effective_gains(
        env.mj_model.actuator_gainprm,
        env.mj_model.actuator_biasprm,
        env.robot_spec.actuated_joints,
        model=load_actuator_preset(robot_dir, preset, overrides).model,
        preset=preset,
    )


def _section(tree, dotted: str):
    node = tree
    for key in dotted.split("."):
        if not isinstance(node, dict) or key not in node:
            return {}
        node = node[key]
    return node


def _leaves(node, prefix: str, out: dict) -> dict:
    if isinstance(node, dict):
        for key, value in node.items():
            _leaves(value, f"{prefix}.{key}", out)
    else:
        out[prefix] = list(node) if isinstance(node, (list, tuple)) else node
    return out


def config_diff(old: dict, new: dict, sections=("task.env", "actuators", "ppo")) -> dict:
    """The leaves of `sections` that differ between two hydra configs.

    Keys are dotted paths, for example "task.env.reward.scales.feet_air".
    Each value is [old, new]. A leaf on one side only has None on the other.
    Lists and tuples are leaves and compare as lists.
    """
    diff = {}
    for section in sections:
        a = _leaves(_section(old, section), section, {})
        b = _leaves(_section(new, section), section, {})
        for key in a.keys() | b.keys():
            if key not in a or key not in b or a[key] != b[key]:
                diff[key] = [a.get(key), b.get(key)]
    return dict(sorted(diff.items()))


def warm_start_block(restore, composed_hydra: dict, strict: bool) -> dict:
    """run.json's `warm_start` block for a run that restores `restore`.

    The source run is the run dir above a `checkpoints/<step>` restore path.
    When it cannot be read, `strict` raises ValueError and non-strict sets
    source_run, source_results and config_diff to None.
    """
    block = {
        "checkpoint": portable_path(restore),
        "source_run": None,
        "source_results": None,
        "config_diff": None,
    }
    ckpt = Path(restore)
    if not ckpt.is_absolute():
        ckpt = paths.PROJECT_DIR / ckpt
    ckpt = ckpt.resolve()
    try:
        if ckpt.parent.name != "checkpoints":
            raise ValueError(f"{ckpt} is not a step dir under a run's checkpoints dir")
        run_dir = ckpt.parent.parent
        run = load_run(run_dir)
    except (FileNotFoundError, ValueError) as e:
        if strict:
            raise ValueError(f"cannot read the source run of checkpoint {ckpt}: {e}") from e
        return block
    block["source_run"] = portable_path(run_dir)
    block["source_results"] = {k: run.get(k) for k in SOURCE_RESULT_KEYS}
    block["config_diff"] = config_diff(run["hydra_config"], composed_hydra)
    return block


def build_run_record(
    env,
    hydra: dict,
    ppo_params,
    *,
    run_name: str,
    checkpoint_dir,
    contacts: dict,
    env_config: dict,
    actuator_gains: dict,
    trainer: dict,
    results: dict,
    warm_start: dict | None,
) -> dict:
    """The run.json record, keys in RECORD_KEYS order.

    `hydra` is the resolved composed config. `results` holds early_stopped,
    stopped_at_steps and final_reward. Each may be None. `warm_start` is
    None for a run without a restore. `env` is the built env. The trainer
    reads env_config and actuator_gains off it.
    """
    missing = [k for k in RESULT_KEYS if k not in results]
    if missing:
        raise ValueError(f"results is missing {missing}")
    hydra_config = copy.deepcopy(hydra)
    if hydra_config.get("restore"):
        hydra_config["restore"] = portable_path(hydra_config["restore"])
    record = {
        "run_name": run_name,
        "task": hydra["task"]["name"],
        "num_timesteps": int(ppo_params.num_timesteps),
        "early_stopped": results["early_stopped"],
        # Brax: the last eval's step count. On a normal run that is the
        # final eval, so this equals the budget brax actually ran.
        "stopped_at_steps": results["stopped_at_steps"],
        "final_reward": results["final_reward"],
        "checkpoint_dir": portable_path(checkpoint_dir),
        # Warp contact/constraint budgets and the preflight's peaks. Same
        # schema as battery.json's, on every backend, so a GPU run and a
        # local one diff without branching.
        "contacts": contacts,
        "env_config": env_config,
        "ppo_config": ppo_params.to_dict(),
        "hydra_config": hydra_config,
        "actuators": copy.deepcopy(hydra["actuators"]),
        # What the BUILT model got, next to the `actuators` block above,
        # which is what the config asked for. The two differ whenever
        # actuators.overrides patches a gain: the override never appears in
        # the preset yaml.
        "actuator_gains": actuator_gains,
        "trainer": trainer,
        "warm_start": warm_start,
    }
    return record


def write_run(run_dir, record: dict) -> Path:
    """Write `record` to `<run_dir>/run.json` through a temp file and
    os.replace, so a reader never sees a half-written file. NaN is written
    as NaN, as before."""
    run_dir = Path(run_dir)
    out = run_dir / "run.json"
    tmp = run_dir / "run.json.tmp"
    tmp.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, out)
    return out
