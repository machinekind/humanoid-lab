"""The key ledger of the fail-closed deploy contract.

These are the tests that ARE the feature: an env option nobody classified
must not be able to ship. The completeness walk below fails the moment any
registered task's `default_config()` grows a key that `deploy_contract.py`
does not name, and `check_config_covered` refuses a run config carrying
one.

Unit-suite safe: `default_config()` is a plain ConfigDict builder, so
nothing here compiles a spec or constructs an env (see
tests/unit/test_suite_split.py).
"""

from __future__ import annotations

import pytest

from humanoid_lab import deploy_contract as dc
from humanoid_lab.envs.joystick import default_config
from humanoid_lab.envs.terrain_joystick import default_config as terrain_default_config
from humanoid_lab.registry import TASKS


def leaf_paths(config, prefix: str = "") -> set[str]:
    """Every dotted path to a non-mapping value, the ledger's key form."""
    paths: set[str] = set()
    for key, value in config.items():
        path = f"{prefix}{key}"
        if hasattr(value, "items"):
            paths |= leaf_paths(value, path + ".")
        else:
            paths.add(path)
    return paths


def every_task_key() -> set[str]:
    """The union of every registered task's default config leaf paths."""
    keys: set[str] = set()
    for _cls, task_default_config in TASKS.values():
        keys |= leaf_paths(task_default_config().to_dict())
    return keys


@pytest.mark.parametrize("task", sorted(TASKS))
def test_every_default_config_key_is_classified(task):
    """The completeness walk. A new env option lands here before it ships."""
    keys = leaf_paths(TASKS[task][1]().to_dict())

    unclassified = sorted(k for k in keys if not dc.is_classified(k))
    assert not unclassified, (
        f"{task} env config key(s) {unclassified} are neither in the ledger "
        "nor covered by a prefix rule -- classify them in "
        "src/humanoid_lab/deploy_contract.py"
    )
    stale = sorted((dc.CONSUMED_KEYS | dc.TRAINING_ONLY_KEYS) - every_task_key())
    assert not stale, (
        f"ledger entries {stale} name keys no task's default_config() has -- "
        "the ledger describes a config that does not exist"
    )


def test_the_two_sets_are_disjoint():
    overlap = sorted(dc.CONSUMED_KEYS & dc.TRAINING_ONLY_KEYS)
    assert not overlap, f"{overlap} are classified both ways"


def test_the_headline_classifications_are_the_documented_ones():
    """Pins the split the deploy runtime is written against."""
    for key in (
        "ctrl_dt",
        "reset_keyframe",
        "obs.state",
        "command.vx",
        "command.vy",
        "command.wz",
        "gait.freq",
    ):
        assert key in dc.CONSUMED_KEYS, f"{key} should reach the robot"
    for key in (
        "sim_dt",
        "sim.backend",
        "episode_length",
        "real_pose_ref",
        "obs.privileged",
        "obs_noise.gyro",
        "push.enable",
        "no_progress.enable",
        "fall.min_height",
        "gait.swing_height",
        "gait.duty",
        "command.zero_prob",
        "command.pure_back_prob",
    ):
        assert key in dc.TRAINING_ONLY_KEYS, f"{key} should stay in training"
    for key in ("reward.scales.tracking_lin_vel", "reward.tracking_product"):
        assert dc.is_classified(key), f"{key} should fall under the reward. prefix rule"


def test_a_new_reward_key_needs_no_ledger_edit():
    """The reward prefix rule: reward terms shape the network, the network
    ships, so adding one is not a two-file edit."""
    config = default_config().to_dict()
    config["reward"]["scales"]["brand_new_term"] = -0.1
    config["reward"]["brand_new_knob"] = 0.5
    dc.check_config_covered(config)


def test_the_default_config_is_covered():
    dc.check_config_covered(default_config().to_dict())


def test_an_unclassified_top_level_key_raises():
    config = default_config().to_dict()
    config["knee_snap_guard"] = True
    with pytest.raises(ValueError, match="knee_snap_guard"):
        dc.check_config_covered(config)


def test_an_unclassified_nested_key_raises_with_its_full_path():
    config = default_config().to_dict()
    config["command"]["pure_diagonal_prob"] = 0.1
    with pytest.raises(ValueError, match=r"command\.pure_diagonal_prob"):
        dc.check_config_covered(config)


def test_a_partial_config_of_known_keys_is_covered():
    """Hydra `task.env` overrides are partial; only unknown keys fail."""
    dc.check_config_covered({"command": {"vx": (-1.0, 1.0)}, "ctrl_dt": 0.02})


def test_an_armed_pure_draw_is_covered():
    """The curriculum keys stay training-only whatever their values; the
    out-of-box refusal lives at env construction (see
    tests/unit/test_pure_draw_ranges.py)."""
    config = default_config().to_dict()
    config["command"]["pure_back_prob"] = 0.3
    dc.check_config_covered(config)


def test_the_terrain_task_is_deployable():
    """A terrain run exports from its flat rebuild. Its terrain block is
    training-only whole, and sizing still has no runtime."""
    assert {"joystick", "terrain"} <= dc.DEPLOYABLE_TASKS
    assert "sizing" not in dc.DEPLOYABLE_TASKS
    dc.check_config_covered(terrain_default_config().to_dict())
    for key in (
        "terrain.arena.seed",
        "terrain.spawn.mode",
        "terrain.command_bias.zero_prob",
    ):
        assert dc.is_classified(key) and key not in dc.CONSUMED_KEYS


def test_a_new_terrain_key_needs_no_ledger_edit():
    config = terrain_default_config().to_dict()
    config["terrain"]["spawn"]["brand_new_knob"] = 1
    config["terrain"]["brand_new_block"] = {"x": 0.5}
    dc.check_config_covered(config)


def test_a_terrain_prefix_does_not_cover_a_lookalike_key():
    config = default_config().to_dict()
    config["terrain_gate"] = 0.2
    with pytest.raises(ValueError, match="terrain_gate"):
        dc.check_config_covered(config)


def _bias(**probs) -> dict:
    """A run.json `env_config` with the terrain command bias on."""
    bias = terrain_default_config().terrain.command_bias.to_dict()
    bias.update(enable=True, **probs)
    return {"terrain": {"command_bias": bias}}


def test_an_in_box_command_bias_is_covered():
    """The default bias arms no draw outside joystick's box."""
    dc.check_terrain_command_bias(_bias(), default_config().command)
    dc.check_terrain_command_bias({}, default_config().command)


def test_an_out_of_box_command_bias_is_refused():
    """roboto_origin's vx box is [-0.6, 1.0], and back_vx (-0.8, -0.2) lies
    outside it. The flat rebuild carries no bias, so the contract checks the
    run's own."""
    command = default_config().command
    command.vx = (-0.6, 1.0)
    with pytest.raises(ValueError, match=r"terrain\.command_bias.*pure_back_prob"):
        dc.check_terrain_command_bias(_bias(pure_back_prob=0.1), command)
    off = _bias(pure_back_prob=0.1)
    off["terrain"]["command_bias"]["enable"] = False
    dc.check_terrain_command_bias(off, command)
