"""Per-robot training-tuning overlays (configs/robot/*.yaml) compose as
`@package _global_` sections that patch task/dr onto the robot, and CLI
overrides win over them. See configs/config.yaml's defaults comment for
the composition order this pins down.
"""

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from humanoid_lab import paths
from humanoid_lab.envs.joystick import default_config as joystick_default_config
from humanoid_lab.registry import _apply_overrides

# Reward weights ported from RPORewardCfg (rpo_env_cfg.py); see
# configs/robot/roboto_origin.yaml's per-entry provenance comments.
ROBOTO_PORTED_SCALES = {
    "tracking_lin_vel": 1.0,
    "tracking_ang_vel": 1.0,
    "lin_vel_z": -0.2,
    "ang_vel_xy": -0.1,
    "orientation": -1.0,
    "torques": -1e-5,
    "action_rate": -2e-2,
    "action_accel": -2e-2,
    "energy": -1e-4,
    "termination": -200.0,
}


def _compose(overrides):
    with initialize_config_dir(version_base=None, config_dir=str(paths.CONFIGS_DIR)):
        return compose(config_name="config", overrides=overrides)


def test_asimov_v1_resolves_its_own_command_and_obs_noise():
    """asimov's overlay values land in the resolved config. The values equal
    the task base's, so this proves the overlay resolves, not that it wins
    over the base; precedence is proven by the roboto_origin test below,
    whose values differ from the base."""
    cfg = _compose(["robot=asimov_v1"])

    assert cfg.task.env.command.vx == [-0.8, 0.8]
    assert cfg.task.env.command.vy == [-0.6, 0.6]
    assert cfg.task.env.command.wz == [-0.6, 0.6]
    assert cfg.task.env.obs_noise.gyro == 0.01
    assert cfg.task.env.obs_noise.joint_pos == 0.01
    assert cfg.task.env.obs_noise.joint_vel == 0.1


def test_roboto_origin_overlay_wins_over_the_task_and_dr_bases():
    cfg = _compose(["robot=roboto_origin", "actuators=deploy_pd"])

    # CommandRangesCfg (base_config.py).
    assert cfg.task.env.command.vx == [-0.6, 1.0]
    assert cfg.task.env.command.vy == [-0.5, 0.5]
    assert cfg.task.env.command.wz == [-1.57, 1.57]

    # RPOFlatEnvCfg.__post_init__'s noise_scales override (rpo_env_cfg.py);
    # gyro is inherited from the task base.
    assert cfg.task.env.obs_noise.joint_vel == 1.75
    assert cfg.task.env.obs_noise.joint_pos == 0.03
    assert cfg.task.env.obs_noise.gyro == 0.01

    # The overlay's reward block carries exactly the ported scales, nothing
    # else (partial overlay; the unlisted entries come from the env's
    # default_config, exercised by the merge-path test below).
    assert OmegaConf.to_container(cfg.task.env.reward.scales) == ROBOTO_PORTED_SCALES
    assert list(cfg.task.env.reward.keys()) == ["scales"]

    # Mapped DR ranges: EventCfg.scale_joint_parameters (armature),
    # EventCfg.scale_actuator_gains (joint_gains), EventCfg.
    # randomize_rigid_body_com (com_offset) (base_config.py).
    assert cfg.dr.dof.armature == [0.5, 1.5]
    assert cfg.dr.joint_gains.gain_pct == 0.1
    assert cfg.dr.joint_gains.kd_pct == 0.1
    assert cfg.dr.com_offset.xy == 0.025
    assert cfg.dr.com_offset.z == 0.05

    # Untouched DR sub-fields still come from the base configs/dr/default.yaml.
    assert cfg.dr.dof.damping == [0.9, 1.1]
    assert cfg.dr.dof.frictionloss == [0.9, 1.1]


def test_roboto_origin_reward_overlay_survives_the_env_merge_path():
    """train.py feeds the resolved cfg.task.env through make_env, which
    deep-merges it onto the task's default_config (registry.
    _apply_overrides). The partial reward overlay must change exactly the
    ported entries and leave every unlisted reward field at its Python
    default."""
    cfg = _compose(["robot=roboto_origin", "actuators=deploy_pd"])
    env_overrides = OmegaConf.to_container(cfg.task.env, resolve=True)

    env_cfg = joystick_default_config()
    _apply_overrides(env_cfg, env_overrides)

    for key, value in ROBOTO_PORTED_SCALES.items():
        assert env_cfg.reward.scales[key] == value, key

    # Unlisted reward fields keep default_config's values.
    defaults = joystick_default_config()
    assert env_cfg.reward.tracking_sigma == defaults.reward.tracking_sigma
    assert env_cfg.reward.phase_sigma == defaults.reward.phase_sigma
    assert env_cfg.reward.torque_limit_frac == defaults.reward.torque_limit_frac
    assert env_cfg.reward.scales.pose == defaults.reward.scales.pose
    assert env_cfg.reward.scales.feet_air_time == defaults.reward.scales.feet_air_time
    assert env_cfg.reward.scales.feet_slip == defaults.reward.scales.feet_slip
    assert env_cfg.reward.scales.stand_still == defaults.reward.scales.stand_still
    assert len(env_cfg.reward.scales) == len(defaults.reward.scales)

    # The overlay's non-reward values landed on the merged env config too.
    assert env_cfg.obs_noise.joint_vel == 1.75
    assert env_cfg.command.vx == (-0.6, 1.0)


def test_cli_override_beats_the_roboto_origin_overlay():
    cfg = _compose(
        [
            "robot=roboto_origin",
            "actuators=deploy_pd",
            "task.env.obs_noise.joint_vel=0.5",
        ]
    )

    assert cfg.task.env.obs_noise.joint_vel == 0.5
    # The overlay's other mapped value is untouched by the one-key CLI override.
    assert cfg.task.env.obs_noise.joint_pos == 0.03
