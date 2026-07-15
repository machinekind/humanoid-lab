"""Task registry: name -> (env class, default config), plus override merge.

Env configs are ml_collections ConfigDicts built in code; Hydra yaml carries
plain dicts/lists. _apply_overrides merges the latter onto the former with
tuple/float coercion so yaml `vx: [-0.8, 1.8]` lands on a tuple field.
"""

from ml_collections import config_dict

from humanoid_lab.envs.joystick import Joystick
from humanoid_lab.envs.joystick import default_config as joystick_default_config
from humanoid_lab.envs.sizing import Sizing
from humanoid_lab.envs.sizing import default_config as sizing_default_config

# Tasks register themselves here as their env classes land (build order
# step 6: joystick/velocity; step 7: sizing).
TASKS = {
    "joystick": (Joystick, joystick_default_config),
    "sizing": (Sizing, sizing_default_config),
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
