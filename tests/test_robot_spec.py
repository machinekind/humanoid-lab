import dataclasses
from pathlib import Path

import mujoco
import pytest

from humanoid_lab.robot.spec import load_robot_spec, validate_against_model

TOY_ROBOT_DIR = Path(__file__).parent / "data" / "toy_robot"


def _toy_model() -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(str(TOY_ROBOT_DIR / "source" / "xmls" / "toy_biped.xml"))
    return spec.compile()


def test_load_robot_spec_parses_toy_fixture():
    spec = load_robot_spec(TOY_ROBOT_DIR)

    assert spec.name == "toy_biped"
    assert spec.model_xml == "source/xmls/toy_biped.xml"
    assert spec.model_xml_path == TOY_ROBOT_DIR / "source/xmls/toy_biped.xml"

    assert spec.actuated_joints == ["hip_pitch_joint", "knee_joint"]
    assert spec.joint_groups == {
        "hip_pitch": ["hip_pitch_joint"],
        "knee": ["knee_joint"],
    }
    assert spec.group_of("hip_pitch_joint") == "hip_pitch"
    assert spec.group_of("knee_joint") == "knee"

    assert spec.passive_joints["toe_joint_a"].stiffness == 5.0
    assert spec.passive_joints["toe_joint_a"].damping == 0.3
    assert spec.passive_joints["toe_joint_b"].stiffness == 5.0

    assert spec.foot_sites == ["foot_site"]
    assert spec.foot_geoms == ["foot_geom"]
    assert spec.termination_bodies == ["torso"]
    assert spec.symmetry == {}

    assert set(spec.keyframes) == {"standing"}
    kf = spec.keyframes["standing"]
    assert kf.base_pos == (0.0, 0.0, 0.8)
    assert kf.base_quat == (1.0, 0.0, 0.0, 0.0)
    assert kf.joints == {"hip_pitch_joint": 0.2, "knee_joint": -0.4}


def test_load_robot_spec_missing_group_yaml_raises_clear_error(tmp_path):
    robot_dir = tmp_path / "broken_ungrouped"
    robot_dir.mkdir()
    (robot_dir / "robot.yaml").write_text(
        """
name: broken
model_xml: does/not/matter.xml
actuated_joints:
  - hip_pitch_joint
  - knee_joint
joint_groups:
  hip_pitch: [hip_pitch_joint]
passive_joints: {}
foot_sites: []
foot_geoms: []
"""
    )

    with pytest.raises(ValueError, match="knee_joint"):
        load_robot_spec(robot_dir)


def test_load_robot_spec_group_referencing_unknown_joint_raises_clear_error(tmp_path):
    robot_dir = tmp_path / "broken_phantom"
    robot_dir.mkdir()
    (robot_dir / "robot.yaml").write_text(
        """
name: broken
model_xml: does/not/matter.xml
actuated_joints:
  - hip_pitch_joint
joint_groups:
  hip_pitch: [hip_pitch_joint, phantom_joint]
passive_joints: {}
foot_sites: []
foot_geoms: []
"""
    )

    with pytest.raises(ValueError, match="phantom_joint"):
        load_robot_spec(robot_dir)


def test_load_robot_spec_missing_required_key_raises_clear_error(tmp_path):
    robot_dir = tmp_path / "broken_no_foot_sites"
    robot_dir.mkdir()
    (robot_dir / "robot.yaml").write_text(
        """
name: broken
model_xml: does/not/matter.xml
actuated_joints: []
joint_groups: {}
passive_joints: {}
foot_geoms: []
"""
    )

    with pytest.raises(ValueError, match="foot_sites"):
        load_robot_spec(robot_dir)


def test_validate_against_model_passes_for_valid_spec():
    spec = load_robot_spec(TOY_ROBOT_DIR)
    model = _toy_model()
    validate_against_model(spec, model)  # must not raise


def test_validate_against_model_catches_bogus_site():
    spec = load_robot_spec(TOY_ROBOT_DIR)
    bogus_spec = dataclasses.replace(spec, foot_sites=["not_a_real_site"])
    model = _toy_model()

    with pytest.raises(ValueError, match="not_a_real_site"):
        validate_against_model(bogus_spec, model)


def test_validate_against_model_catches_actuated_passive_overlap():
    spec = load_robot_spec(TOY_ROBOT_DIR)
    bogus_spec = dataclasses.replace(
        spec,
        passive_joints={
            **spec.passive_joints,
            "hip_pitch_joint": spec.passive_joints["toe_joint_a"],
        },
    )
    model = _toy_model()

    with pytest.raises(ValueError, match="hip_pitch_joint"):
        validate_against_model(bogus_spec, model)
