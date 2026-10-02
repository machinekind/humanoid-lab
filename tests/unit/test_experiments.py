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


def _env_config(cfg):
    """The task's default config with the composed `task.env` applied."""
    _, default_config = TASKS[cfg.task.name]
    env_cfg = default_config()
    _apply_overrides(env_cfg, OmegaConf.to_container(cfg.task.env, resolve=True))
    return env_cfg


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
    _env_config(cfg)


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
    """The composed config as a plain dict, with task.env.reward (and its
    scales) and ppo popped out of it."""
    cfg = OmegaConf.to_container(_compose([f"experiment={experiment}"]), resolve=True)
    reward = cfg["task"]["env"].pop("reward")
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
    "torque_rate": -3.0e-4,
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


def test_yolo_apex_min_is_yolo_income_sym_with_feet_apex_swapped_for_feet_apex_min():
    ours, ours_reward, ours_scales, ours_ppo = _reward_and_ppo_split("yolo_apex_min")
    base, base_reward, base_scales, base_ppo = _reward_and_ppo_split("yolo_income_sym")

    assert ours == base
    assert ours_reward == base_reward
    assert ours_ppo == base_ppo
    assert ours_scales.pop("feet_apex") == 0.0
    assert base_scales.pop("feet_apex") == 10.0
    assert ours_scales.pop("feet_apex_min") == 10.0
    assert base_scales.pop("feet_apex_min", 0.0) == 0.0
    assert ours_scales == base_scales


def test_yolo_apex_min_reaches_the_env_config():
    cfg = _compose(["experiment=yolo_apex_min"])
    _, default_config = TASKS[cfg.task.name]
    env_cfg = default_config()
    assert env_cfg.reward.scales.feet_apex_min == 0.0
    _apply_overrides(env_cfg, OmegaConf.to_container(cfg.task.env, resolve=True))
    assert env_cfg.reward.scales.feet_apex_min == 10.0
    assert env_cfg.reward.scales.feet_apex == 0.0


MIRROR_BLOCK = {"enable": True, "mirror_prob": 0.5}


def test_yolo_mirror_is_yolo_income_long_plus_the_mirror_augmentation():
    ours, ours_reward, ours_scales, ours_ppo = _reward_and_ppo_split("yolo_mirror")
    base, base_reward, base_scales, base_ppo = _reward_and_ppo_split("yolo_income_long")

    assert ours["task"]["env"].pop("symmetry") == MIRROR_BLOCK
    assert "symmetry" not in base["task"]["env"]
    assert ours == base
    assert ours_reward == base_reward
    assert ours_scales == base_scales
    assert ours_scales["gait_symmetry_income"] == 0.5
    assert ours_scales["energy"] == -3.0e-4
    assert ours_scales["feet_apex"] == 10.0

    assert ours_ppo.pop("num_timesteps") == pytest.approx(1.0e9)
    base_ppo.pop("num_timesteps")
    assert ours_ppo == base_ppo


def test_yolo_v4_mirror_is_yolo_v4_plus_the_mirror_augmentation():
    ours, ours_reward, ours_scales, ours_ppo = _reward_and_ppo_split("yolo_v4_mirror")
    base, base_reward, base_scales, base_ppo = _reward_and_ppo_split("yolo_v4")

    assert ours["task"]["env"].pop("symmetry") == MIRROR_BLOCK
    assert "symmetry" not in base["task"]["env"]
    assert ours == base
    assert ours_reward == base_reward
    assert ours_scales == base_scales

    assert ours_ppo.pop("num_timesteps") == pytest.approx(1.5e9)
    base_ppo.pop("num_timesteps")
    assert ours_ppo == base_ppo


@pytest.mark.parametrize("experiment", ["yolo_mirror", "yolo_v4_mirror"])
def test_the_mirror_presets_reach_the_env_config(experiment):
    cfg = _compose([f"experiment={experiment}"])
    _, default_config = TASKS[cfg.task.name]
    env_cfg = default_config()
    assert env_cfg.symmetry.enable is False
    _apply_overrides(env_cfg, OmegaConf.to_container(cfg.task.env, resolve=True))
    assert env_cfg.symmetry.enable is True
    assert env_cfg.symmetry.mirror_prob == 0.5


def test_terrain_cpu_experiment_is_the_cpu_arena_and_logs_training_metrics():
    """The CPU smoke's arena is CPU_ARENA. It logs training metrics before
    it ends, so the curriculum level shows, and it passes the terrain
    guards."""
    from humanoid_lab.terrain.config import CPU_ARENA
    from humanoid_lab.train import build_ppo_params, check_terrain_training

    cfg = _compose(["experiment=terrain_cpu"])
    assert cfg.task.name == "terrain"
    assert cfg.robot.name == "roboto_origin"
    assert OmegaConf.to_container(cfg.task.env.terrain.arena) == CPU_ARENA
    p = build_ppo_params({}, smoke=True)
    p.update(OmegaConf.to_container(cfg.task.ppo))
    p.update(OmegaConf.to_container(cfg.ppo))
    assert p.log_training_metrics is True
    assert p.training_metrics_steps < p.num_timesteps
    assert p.num_eval_envs == p.num_envs
    assert p.batch_size * p.num_minibatches % p.num_envs == 0
    check_terrain_training(cfg.task.name, p, cfg.early_stop)


def test_roboto_terrain_v1_is_the_default_arena_with_its_flat_row():
    """The recipe sets the arena's flat row and two curriculum knobs, and
    pins pad spawns. Every other terrain key keeps its default, and the
    terrain guards pass at the task's own PPO settings."""
    from humanoid_lab.envs.terrain_joystick import (
        default_config as terrain_default_config,
    )
    from humanoid_lab.terrain import ArenaParams
    from humanoid_lab.terrain.config import config_from_params
    from humanoid_lab.train import build_ppo_params, check_terrain_training

    cfg = _compose(["experiment=roboto_terrain_v1"])
    assert (cfg.task.name, cfg.robot.name) == ("terrain", "roboto_origin")
    env_cfg = _env_config(cfg)
    expected = terrain_default_config().terrain.to_dict()
    expected["arena"] = config_from_params(ArenaParams(flat_row=True))
    # The recipe pins pad spawns, whatever the code default is.
    expected["spawn"]["mode"] = "pad"
    expected["curriculum"].update(demote_strikes=3, pinned_frac=0.2)
    assert env_cfg.terrain.to_dict() == expected
    p = build_ppo_params({}, smoke=False)
    p.update(OmegaConf.to_container(cfg.task.ppo))
    p.update(OmegaConf.to_container(cfg.ppo))
    check_terrain_training(cfg.task.name, p, cfg.early_stop)


def test_roboto_terrain_v1_leaves_the_warp_budgets_and_the_preset_open():
    """No budget: a warp build refuses until check-terrain measures them.
    No actuator pin: the action window is an open decision."""
    from humanoid_lab.terrain.config import require_terrain_budgets

    path = paths.CONFIGS_DIR / "experiment" / "roboto_terrain_v1.yaml"
    with path.open() as f:
        raw = yaml.safe_load(f)
    keys = {key for entry in raw["defaults"] if isinstance(entry, dict) for key in entry}
    assert "override /actuators" not in keys
    assert "actuators" not in raw
    assert "sim" not in raw["task"]["env"]
    env_cfg = _env_config(_compose(["experiment=roboto_terrain_v1"]))
    with pytest.raises(ValueError, match="check-terrain"):
        require_terrain_budgets("warp", env_cfg.sim, ccd_slot_bytes=5480)
