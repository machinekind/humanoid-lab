"""Guardrails so configs/experiment/*.yaml stay valid as robot/task bases
drift: every experiment must compose, pin its robot and task, and produce
reward and actuator overrides that resolve cleanly, all before any GPU time
is spent on it.
"""

import pytest
import yaml
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from humanoid_lab import paths
from humanoid_lab.registry import TASKS, _apply_overrides
from humanoid_lab.robot.presets import load_actuator_preset, resolve
from humanoid_lab.robot.spec import load_robot_spec

EXPERIMENT_FILES = sorted(
    p for p in (paths.CONFIGS_DIR / "experiment").glob("*.yaml") if not p.stem.startswith("_")
)
EXPERIMENT_IDS = [p.stem for p in EXPERIMENT_FILES]


def _compose(overrides):
    with initialize_config_dir(version_base=None, config_dir=str(paths.CONFIGS_DIR)):
        return compose(config_name="config", overrides=overrides)


@pytest.mark.parametrize("path", EXPERIMENT_FILES, ids=EXPERIMENT_IDS)
def test_experiment_composes(path):
    _compose([f"experiment={path.stem}"])


@pytest.mark.parametrize("path", EXPERIMENT_FILES, ids=EXPERIMENT_IDS)
def test_experiment_pins_robot_and_task(path):
    with path.open() as f:
        raw = yaml.safe_load(f)
    keys = {key for entry in raw.get("defaults", []) if isinstance(entry, dict) for key in entry}
    assert "override /robot" in keys and "override /task" in keys, (
        f"{path.name}: an experiment must pin robot and task via "
        "`defaults: [override /robot: ..., override /task: ...]` so it "
        "composes identically as config.yaml's own defaults drift"
    )


@pytest.mark.parametrize("path", EXPERIMENT_FILES, ids=EXPERIMENT_IDS)
def test_experiment_env_overrides_are_valid(path):
    cfg = _compose([f"experiment={path.stem}"])
    assert cfg.task.name in TASKS

    _, default_config = TASKS[cfg.task.name]
    env_cfg = default_config()
    env_overrides = OmegaConf.to_container(cfg.task.env, resolve=True)
    _apply_overrides(env_cfg, env_overrides)


@pytest.mark.parametrize("path", EXPERIMENT_FILES, ids=EXPERIMENT_IDS)
def test_experiment_actuator_preset_is_valid(path):
    cfg = _compose([f"experiment={path.stem}"])
    robot_dir = paths.REPO_ROOT / cfg.robot.dir

    spec = load_robot_spec(robot_dir)
    overrides = OmegaConf.to_container(cfg.actuators.overrides, resolve=True)
    preset = load_actuator_preset(robot_dir, cfg.actuators.name, overrides)
    resolve(preset, spec)


def test_actuator_overrides_flow_from_the_experiment_cli_override():
    cfg = _compose(
        ["experiment=asimov_gentle_penalties", "+actuators.overrides.groups.knee.kp=99"]
    )
    assert cfg.actuators.overrides.groups.knee.kp == 99


def test_actuator_preset_rejects_a_typo_d_override_key():
    robot_dir = paths.ROBOTS_DIR / "asimov_v1"
    with pytest.raises(ValueError):
        load_actuator_preset(robot_dir, "sizing_ideal", {"groups": {"knee": {"kp_": 1.0}}})
