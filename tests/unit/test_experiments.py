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


def test_roboto_walk_v5_arms_the_cut_style_package_on_top_of_the_v4_recipe():
    cfg = _compose(["experiment=roboto_walk_v5"])

    assert cfg.robot.name == "roboto_origin"
    assert cfg.actuators.name == "deploy_pd"
    assert cfg.domain_rand is True
    for field in ("joint_gains", "com_offset", "dof", "foot_friction", "base_mass_add"):
        assert cfg.dr[field].enable is True, field
    # No-op decouple: upstream does not randomize effort limits.
    assert cfg.dr.motor_strength.enable is True
    assert cfg.dr.motor_strength.range == [1.0, 1.0]

    # rpo_agent_cfg.py values, unchanged from v4.
    assert cfg.ppo.discounting == 0.994
    assert cfg.ppo.gae_lambda == 0.9
    assert cfg.ppo.entropy_cost == 0.005
    assert cfg.ppo.learning_rate == 1.0e-4
    assert cfg.ppo.num_timesteps == 1.2e9

    # The cut style package (gate-1 FAIL applied the pre-committed cut):
    # gait_symmetry armed, energy tripled over the roboto_origin overlay's
    # ported -1e-4 (the experiment overlay composes after the robot
    # overlay, so this file must win). knee_stance and the clock change
    # are gone: no overrides at all, so the env keeps its Python defaults
    # (knee_stance 0.0 = off, freq 1.0-2.0, threshold 0.4).
    scales = cfg.task.env.reward.scales
    assert scales.gait_symmetry == -1.0
    assert scales.energy == -3.0e-4
    assert "knee_stance" not in scales
    assert "gait" not in cfg.task.env
    assert "biped_air_time_threshold" not in cfg.task.env.reward


def _reward_and_ppo_split(experiment):
    """The composed config as a plain dict, with reward.scales and ppo popped."""
    cfg = OmegaConf.to_container(_compose([f"experiment={experiment}"]), resolve=True)
    reward = cfg["task"]["env"]["reward"]
    scales = reward.pop("scales")
    return cfg, reward, scales, cfg.pop("ppo")


def test_yolo_income_long_is_yolo_chain_income_b_with_feet_apex_doubled():
    ours, ours_reward, ours_scales, ours_ppo = _reward_and_ppo_split("yolo_income_long")
    base, base_reward, base_scales, base_ppo = _reward_and_ppo_split("yolo_chain_income_b")

    assert ours == base
    assert ours_reward == base_reward
    assert ours_scales.pop("feet_apex") == 10.0
    assert base_scales.pop("feet_apex") == 5.0
    assert ours_scales == base_scales
    assert ours_scales["gait_symmetry_income"] == 0.5
    assert ours_scales["energy"] == -3.0e-4
    # The term clips at the target; the overlay's 0.08 stays.
    assert ours_reward["apex_target"] == 0.08

    assert ours_ppo.pop("num_timesteps") == pytest.approx(1.5e9)
    base_ppo.pop("num_timesteps")
    assert ours_ppo == base_ppo
    assert ours_ppo["num_evals"] == 24


SMOOTHNESS_SCALES = {
    "action_rate": -0.05,
    "action_accel": -0.05,
    "torque_rate": -3.0e-7,
    "joint_acc": -7.5e-7,
}


def test_yolo_income_smooth_is_yolo_income_long_plus_the_smoothness_package():
    ours, ours_reward, ours_scales, ours_ppo = _reward_and_ppo_split("yolo_income_smooth")
    base, base_reward, base_scales, base_ppo = _reward_and_ppo_split("yolo_income_long")

    assert ours == base
    assert ours_reward.pop("stand_still_vel_weight") == 0.5
    assert "stand_still_vel_weight" not in base_reward
    assert ours_reward == base_reward
    for key, value in SMOOTHNESS_SCALES.items():
        assert ours_scales.pop(key) == pytest.approx(value), key
        assert base_scales.pop(key, 0.0) != pytest.approx(value), key
    assert ours_scales == base_scales

    assert ours_ppo.pop("num_timesteps") == pytest.approx(1.0e9)
    base_ppo.pop("num_timesteps")
    assert ours_ppo == base_ppo


def test_yolo_income_smooth_reaches_the_env_config():
    cfg = _compose(["experiment=yolo_income_smooth"])
    _, default_config = TASKS[cfg.task.name]
    env_cfg = default_config()
    assert env_cfg.reward.stand_still_vel_weight == 0.2
    _apply_overrides(env_cfg, OmegaConf.to_container(cfg.task.env, resolve=True))
    assert env_cfg.reward.stand_still_vel_weight == 0.5
    for key, value in SMOOTHNESS_SCALES.items():
        assert env_cfg.reward.scales[key] == pytest.approx(value), key
    assert env_cfg.reward.scales.feet_apex == 10.0
