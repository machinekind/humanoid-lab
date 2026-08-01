"""The key ledger of the fail-closed deploy contract.

These are the tests that ARE the feature: an env option nobody classified
must not be able to ship. The completeness walk below fails the moment
`envs/joystick.py::default_config()` grows a key that
`deploy_contract.py` does not name, and `check_config_covered` refuses a
run config carrying one.

Unit-suite safe: `default_config()` is a plain ConfigDict builder, so
nothing here compiles a spec or constructs an env (see
tests/unit/test_suite_split.py).
"""

from __future__ import annotations

import pytest

from humanoid_lab import deploy_contract as dc
from humanoid_lab.envs.joystick import default_config


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


def test_every_default_config_key_is_classified():
    """The completeness walk. A new env option lands here before it ships."""
    classified = dc.CONSUMED_KEYS | dc.TRAINING_ONLY_KEYS
    keys = leaf_paths(default_config().to_dict())

    unclassified = sorted(keys - classified)
    assert not unclassified, (
        f"env config key(s) {unclassified} are in neither CONSUMED_KEYS nor "
        "TRAINING_ONLY_KEYS -- classify them in src/humanoid_lab/deploy_contract.py"
    )
    stale = sorted(classified - keys)
    assert not stale, (
        f"ledger entries {stale} name keys default_config() no longer has -- "
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
        "reward.scales.tracking_lin_vel",
        "reward.tracking_product",
    ):
        assert key in dc.TRAINING_ONLY_KEYS, f"{key} should stay in training"


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


