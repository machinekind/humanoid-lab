"""Task registry: name -> (env class, default config), plus override merge
and the `make_env` arguments a composed Hydra config names.

Env configs are ml_collections ConfigDicts built in code; Hydra yaml carries
plain dicts/lists. _apply_overrides merges the latter onto the former with
tuple/float coercion so yaml `vx: [-0.8, 1.8]` lands on a tuple field.
"""

import copy
from collections.abc import Mapping
from pathlib import Path
from typing import NamedTuple

from ml_collections import config_dict

from humanoid_lab import paths, tasks
from humanoid_lab.envs.joystick import Joystick
from humanoid_lab.envs.joystick import default_config as joystick_default_config
from humanoid_lab.envs.sizing import Sizing
from humanoid_lab.envs.sizing import default_config as sizing_default_config
from humanoid_lab.envs.terrain_joystick import TerrainJoystick
from humanoid_lab.envs.terrain_joystick import default_config as terrain_default_config

# Tasks register themselves here as their env classes land (build order
# step 6: joystick/velocity; step 7: sizing).
TASKS = {
    "joystick": (Joystick, joystick_default_config),
    "sizing": (Sizing, sizing_default_config),
    "terrain": (TerrainJoystick, terrain_default_config),
}


def _apply_overrides(cfg: config_dict.ConfigDict, overrides: dict) -> None:
    for key, value in (overrides or {}).items():
        current = getattr(cfg, key)
        if isinstance(current, config_dict.ConfigDict):
            _apply_overrides(current, value)
            continue
        if isinstance(current, tuple) and isinstance(value, (list, tuple)):
            value = tuple(value)
        if isinstance(current, float) and isinstance(value, int):
            value = float(value)
        # scalar defaults may be overridden with per-element vectors
        # (e.g. action_scale: [0.2, 0.5, 0.5]); bypass the type lock
        if isinstance(current, float) and isinstance(value, (list, tuple)):
            with cfg.ignore_type():
                setattr(cfg, key, tuple(float(v) for v in value))
            continue
        setattr(cfg, key, value)


class EnvArgs(NamedTuple):
    """`make_env`'s positional arguments, in order."""

    task: str
    robot_dir: Path
    preset_name: str
    env_overrides: dict
    actuator_overrides: dict


def env_args_from_config(cfg: Mapping) -> EnvArgs:
    """`make_env`'s arguments from a composed Hydra config, given as a
    plain container (`OmegaConf.to_container(cfg, resolve=True)`).

    The overrides are copies, so a caller may edit them."""
    actuators = cfg.get("actuators") or {}
    return EnvArgs(
        task=cfg["task"]["name"],
        robot_dir=paths.REPO_ROOT / cfg["robot"]["dir"],
        preset_name=actuators["name"],
        env_overrides=copy.deepcopy(cfg["task"].get("env") or {}),
        actuator_overrides=copy.deepcopy(actuators.get("overrides") or {}),
    )


def flat_counterpart(task: str, env_overrides: dict | None) -> tuple[str, dict]:
    """The task and overrides that rebuild a `task` run on the flat floor.

    A terrain task's config is its flat task's plus one `terrain` block, so
    its flat rebuild drops that block and keeps every other override. Any
    other task is its own counterpart. The overrides are copies."""
    overrides = copy.deepcopy(dict(env_overrides or {}))
    flat = tasks.FLAT_COUNTERPART.get(task)
    if flat is None:
        return task, overrides
    overrides.pop("terrain", None)
    return flat, overrides


def terrain_counterpart(task: str, env_overrides: dict | None) -> tuple[str, dict]:
    """The task and overrides that rebuild a `task` run on a terrain arena.

    A joystick or terrain run keeps every override. A joystick run gets the
    terrain block's defaults. Any other task raises. The overrides are
    copies."""
    if task not in tasks.TERRAIN_SOURCE_TASKS:
        raise ValueError(
            f"task '{task}' has no terrain counterpart. Tasks that have one: "
            f"{list(tasks.TERRAIN_SOURCE_TASKS)}"
        )
    return tasks.TERRAIN_TASK, copy.deepcopy(dict(env_overrides or {}))


def make_env(
    task: str,
    robot_dir,
    preset_name: str,
    env_overrides: dict | None = None,
    actuator_overrides: dict | None = None,
):
    if task not in TASKS:
        raise KeyError(f"unknown task '{task}', have {sorted(TASKS)}")
    cls, default_config = TASKS[task]
    cfg = default_config()
    _apply_overrides(cfg, env_overrides or {})
    return cls(robot_dir, preset_name, cfg, actuator_overrides=actuator_overrides)
