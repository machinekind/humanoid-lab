"""Preset loading and action-scale derivation (robot/presets.py), on the
shipped yaml files alone -- no model is built here.
"""

from humanoid_lab import paths
from humanoid_lab.robot.presets import action_scale, load_actuator_preset
from humanoid_lab.robot.spec import load_robot_spec

ROBOTO_DIR = paths.ROBOTS_DIR / "roboto_origin"


def test_deploy_pd_uses_the_flat_upstream_action_scale():
    """RoboParty's scheme: 0.25 rad per unit action, every joint the same."""
    preset = load_actuator_preset(ROBOTO_DIR, "deploy_pd")
    spec = load_robot_spec(ROBOTO_DIR)

    assert preset.action_scale_rad == 0.25
    scales = action_scale(preset, spec)
    assert set(scales) == set(spec.actuated_joints)
    assert all(s == 0.25 for s in scales.values())


def test_sizing_ideal_keeps_the_formula_path():
    preset = load_actuator_preset(ROBOTO_DIR, "sizing_ideal")
    assert preset.action_scale_rad is None
    # ideal_torque: action_scale is the effort limit, not a flat angle.
    scales = action_scale(preset, load_robot_spec(ROBOTO_DIR))
    assert len(set(scales.values())) > 1


def test_a_non_pd_model_ignores_action_scale_rad():
    """Like action_scale_factor: an ideal-torque ctrl is not an angle. This
    keeps `--set model=ideal_torque` comparisons working on deploy_pd."""
    preset = load_actuator_preset(
        ROBOTO_DIR, "deploy_pd", overrides={"model": "ideal_torque"}
    )
    scales = action_scale(preset, load_robot_spec(ROBOTO_DIR))
    assert all(s != 0.25 for s in scales.values())


def test_action_scale_rad_can_arrive_as_an_override():
    preset = load_actuator_preset(
        ROBOTO_DIR, "deploy_pd", overrides={"action_scale_rad": 0.3}
    )
    scales = action_scale(preset, load_robot_spec(ROBOTO_DIR))
    assert all(s == 0.3 for s in scales.values())
