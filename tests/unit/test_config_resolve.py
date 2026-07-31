"""The full Hydra defaults tree (repo-root configs/) composes cleanly."""

from hydra import compose, initialize_config_dir

from humanoid_lab import paths


def test_full_defaults_tree_composes():
    with initialize_config_dir(version_base=None, config_dir=str(paths.CONFIGS_DIR)):
        cfg = compose(config_name="config")

    assert cfg.task.name == "joystick"
    assert cfg.robot.name == "asimov_v1"
    assert cfg.actuators.name == "sizing_ideal"
    assert "com_offset" in cfg.dr
    assert cfg.wandb.project == "humanoid-lab"


def test_config_groups_are_overridable_from_the_cli():
    with initialize_config_dir(version_base=None, config_dir=str(paths.CONFIGS_DIR)):
        cfg = compose(config_name="config", overrides=["network=large", "smoke=true"])

    assert cfg.network.policy_hidden_layer_sizes == [512, 256, 128]
    assert cfg.smoke is True
