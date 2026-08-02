from pathlib import Path

import mujoco
import pytest

from humanoid_lab.robot.build import build_spec, compile_spec
from humanoid_lab.robot.presets import (
    action_scale,
    effective_gains,
    load_actuator_preset,
    resolve,
)
from humanoid_lab.robot.spec import load_robot_spec

TOY_ROBOT_DIR = Path(__file__).parent.parent / "data" / "toy_robot"


def _free_joint_qpos_addr(model: mujoco.MjModel) -> int:
    for i in range(model.njnt):
        if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE:
            return int(model.jnt_qposadr[i])
    raise AssertionError("no free joint in toy model")


def test_build_spec_pd_compiles_and_matches_preset():
    spec = build_spec(TOY_ROBOT_DIR, "pd_test")
    model = compile_spec(spec)

    assert model.nu == 4
    actuator_names = [model.actuator(i).name for i in range(model.nu)]
    assert actuator_names == [  # canonical actuated_joints order
        "left_hip_pitch_joint",
        "left_knee_joint",
        "right_hip_pitch_joint",
        "right_knee_joint",
    ]

    hip = model.actuator("left_hip_pitch_joint")
    assert hip.gainprm[0] == pytest.approx(50.0)
    assert hip.biasprm[1] == pytest.approx(-50.0)
    assert hip.biasprm[2] == pytest.approx(-2.0)
    assert tuple(hip.forcerange) == pytest.approx((-40.0, 40.0))
    # gaintype/biastype make gainprm/biasprm mean what the PD math assumes; a
    # regression to biastype=NONE would pass every prm assertion while killing
    # all position/velocity feedback.
    assert hip.gaintype[0] == mujoco.mjtGain.mjGAIN_FIXED
    assert hip.biastype[0] == mujoco.mjtBias.mjBIAS_AFFINE
    # The PD setpoint is deliberately unclamped (asimov-mjlab convention): the
    # policy may command past the kinematic limit.
    assert hip.ctrllimited[0] == 0

    knee = model.actuator("left_knee_joint")
    assert knee.gainprm[0] == pytest.approx(60.0)
    assert knee.biasprm[1] == pytest.approx(-60.0)
    assert knee.biasprm[2] == pytest.approx(-3.0)
    assert tuple(knee.forcerange) == pytest.approx((-55.0, 55.0))
    assert knee.ctrllimited[0] == 0

    # A group's params reach every joint in it, not just the first one.
    assert model.actuator("right_hip_pitch_joint").gainprm[0] == pytest.approx(50.0)
    assert model.actuator("right_knee_joint").gainprm[0] == pytest.approx(60.0)

    # Armature overridden from the preset, NOT the XML's 99.0 / 88.0 placeholders.
    assert model.joint("left_hip_pitch_joint").armature[0] == pytest.approx(0.05)
    assert model.joint("left_knee_joint").armature[0] == pytest.approx(0.04)
    assert model.joint("left_hip_pitch_joint").frictionloss[0] == pytest.approx(0.1)
    # The preset omits knee frictionloss, so the XML's own 0.25 must survive.
    assert model.joint("left_knee_joint").frictionloss[0] == pytest.approx(0.25)

    # Passive joints got stiffness/damping from robot.yaml, independent of the preset.
    assert model.joint("toe_joint_a").stiffness[0] == pytest.approx(5.0)
    assert model.joint("toe_joint_a").damping[0] == pytest.approx(0.3)
    assert model.joint("toe_joint_b").stiffness[0] == pytest.approx(5.0)
    assert model.joint("toe_joint_b").damping[0] == pytest.approx(0.3)

    # Keyframe baked into the final compiled model.
    assert model.nkey == 1
    key = model.key("standing")
    base_addr = _free_joint_qpos_addr(model)
    assert list(key.qpos[base_addr : base_addr + 3]) == pytest.approx([0.0, 0.0, 0.7074])
    assert list(key.qpos[base_addr + 3 : base_addr + 7]) == pytest.approx([1.0, 0.0, 0.0, 0.0])
    for joint_name, angle in (
        ("left_hip_pitch_joint", 0.2),
        ("left_knee_joint", -0.4),
        ("right_hip_pitch_joint", 0.2),
        ("right_knee_joint", -0.4),
    ):
        assert key.qpos[model.joint(joint_name).qposadr[0]] == pytest.approx(angle), joint_name


def test_build_spec_ideal_torque_compiles_and_matches_preset():
    spec = build_spec(TOY_ROBOT_DIR, "ideal_test")
    model = compile_spec(spec)

    assert model.nu == 4

    hip = model.actuator("left_hip_pitch_joint")
    assert hip.gainprm[0] == pytest.approx(1.0)
    assert tuple(hip.forcerange) == pytest.approx((-40.0, 40.0))
    assert tuple(hip.ctrlrange) == pytest.approx((-40.0, 40.0))

    knee = model.actuator("left_knee_joint")
    assert knee.gainprm[0] == pytest.approx(1.0)
    assert tuple(knee.forcerange) == pytest.approx((-55.0, 55.0))
    assert tuple(knee.ctrlrange) == pytest.approx((-55.0, 55.0))


def test_pd_force_matches_position_servo_law():
    spec = build_spec(TOY_ROBOT_DIR, "pd_test")
    model = compile_spec(spec)
    data = mujoco.MjData(model)

    hip_qadr = model.joint("left_hip_pitch_joint").qposadr[0]
    hip_vadr = model.joint("left_hip_pitch_joint").dofadr[0]
    data.qpos[hip_qadr] = 0.3
    data.qvel[hip_vadr] = -0.5
    data.ctrl[0] = 1.0  # left hip actuator is index 0 (canonical order)
    mujoco.mj_forward(model, data)

    # force = kp*(ctrl - q) - kd*qvel, well inside the ±40 force clamp here
    expected = 50.0 * (1.0 - 0.3) - 2.0 * (-0.5)
    assert data.actuator_force[0] == pytest.approx(expected)


def test_resolve_carries_velocity_limit():
    # velocity_limit is not enforceable in the XML, but later actuator models
    # (DC motor) and the sizing report read it; resolve() must carry it.
    robot_spec = load_robot_spec(TOY_ROBOT_DIR)
    preset = load_actuator_preset(TOY_ROBOT_DIR, "pd_test")

    params = resolve(preset, robot_spec)

    assert params["left_hip_pitch_joint"].velocity_limit == pytest.approx(10.0)
    assert params["right_hip_pitch_joint"].velocity_limit == pytest.approx(10.0)
    assert params["left_knee_joint"].velocity_limit == pytest.approx(12.0)
    assert params["right_knee_joint"].velocity_limit == pytest.approx(12.0)


def test_action_scale_pd():
    robot_spec = load_robot_spec(TOY_ROBOT_DIR)
    preset = load_actuator_preset(TOY_ROBOT_DIR, "pd_test")

    scales = action_scale(preset, robot_spec)

    assert scales["left_hip_pitch_joint"] == pytest.approx(0.3 * 40.0 / 50.0)
    assert scales["left_knee_joint"] == pytest.approx(0.3 * 55.0 / 60.0)


def test_action_scale_ideal_torque():
    robot_spec = load_robot_spec(TOY_ROBOT_DIR)
    preset = load_actuator_preset(TOY_ROBOT_DIR, "ideal_test")

    scales = action_scale(preset, robot_spec)

    assert scales["left_hip_pitch_joint"] == pytest.approx(40.0)
    assert scales["left_knee_joint"] == pytest.approx(55.0)


def test_unknown_actuator_model_key_raises_clear_error(tmp_path):
    robot_dir = tmp_path
    (robot_dir / "actuators").mkdir()
    (robot_dir / "actuators" / "bad.yaml").write_text(
        """
model: not_a_real_model
groups: {}
"""
    )

    with pytest.raises(ValueError, match="not_a_real_model"):
        load_actuator_preset(robot_dir, "bad")


def test_preset_missing_group_params_raises_clear_error(tmp_path):
    robot_dir = tmp_path
    (robot_dir / "actuators").mkdir()
    (robot_dir / "actuators" / "incomplete.yaml").write_text(
        """
model: pd
groups:
  hip_pitch: {kp: 50.0, kd: 2.0, effort_limit: 40.0}
"""
    )

    robot_spec = load_robot_spec(TOY_ROBOT_DIR)
    preset = load_actuator_preset(robot_dir, "incomplete")

    with pytest.raises(ValueError, match="knee"):
        resolve(preset, robot_spec)


# -- the effective-gain stamp ------------------------------------------------


def test_effective_gains_report_what_the_built_model_got():
    """The point of reading gains back off the model: an `actuators.overrides`
    entry never appears in the preset yaml, so a stamp taken from the yaml
    would record numbers the physics never used."""
    overrides = {"groups": {"knee": {"kp": 999.0, "kd": 42.0}}}
    model = compile_spec(build_spec(TOY_ROBOT_DIR, "pd_test", overrides))
    joints = load_robot_spec(TOY_ROBOT_DIR).actuated_joints

    block = effective_gains(
        model.actuator_gainprm, model.actuator_biasprm, joints, model="pd", preset="pd_test"
    )

    assert block["joints"] == [
        "left_hip_pitch_joint",
        "left_knee_joint",
        "right_hip_pitch_joint",
        "right_knee_joint",
    ]
    # yaml says 60.0 / 3.0 for the knee group; the override wins on both knees
    assert block["kp"] == pytest.approx([50.0, 999.0, 50.0, 999.0])
    assert block["kd"] == pytest.approx([2.0, 42.0, 2.0, 42.0])


def test_effective_gains_stamp_an_ideal_torque_preset_too():
    """Its gain params are not PD gains; the model name is what says so."""
    mj_model = compile_spec(build_spec(TOY_ROBOT_DIR, "ideal_test"))
    joints = load_robot_spec(TOY_ROBOT_DIR).actuated_joints

    block = effective_gains(
        mj_model.actuator_gainprm, mj_model.actuator_biasprm, joints,
        model="ideal_torque", preset="ideal_test",
    )

    assert block["model"] == "ideal_torque"
    assert block["kp"] == pytest.approx([1.0] * len(joints))
    assert block["kd"] == pytest.approx([0.0] * len(joints))
