"""Build-order step 4 gate: robots/asimov_v1 parses, injects, and steps clean.

CPU-fast on purpose (module-scoped fixtures, no MJX): this is the per-robot
compile/NaN smoke test PLAN.md's repo shape calls out for CI, not a training
or sizing check.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from humanoid_lab import paths
from humanoid_lab.robot.build import build_spec, compile_spec
from humanoid_lab.robot.spec import load_robot_spec, validate_against_model

ROBOT_DIR = paths.ROBOTS_DIR / "asimov_v1"

ARM_JOINTS = (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_yaw_joint",
)


@pytest.fixture(scope="module")
def robot_spec():
    return load_robot_spec(ROBOT_DIR)


@pytest.fixture(scope="module")
def source_model():
    spec = mujoco.MjSpec.from_file(str(ROBOT_DIR / "source" / "xmls" / "asimov.xml"))
    return spec.compile()


@pytest.fixture(scope="module")
def built_model():
    spec = build_spec(ROBOT_DIR, "sizing_ideal")
    return compile_spec(spec)


def test_load_robot_spec_parses_and_validates_against_source_xml(robot_spec, source_model):
    assert robot_spec.name == "asimov_v1"
    assert robot_spec.model_xml == "source/xmls/asimov.xml"
    assert len(robot_spec.actuated_joints) == 12
    validate_against_model(robot_spec, source_model)  # must not raise


def test_build_spec_sizing_ideal_compiles(robot_spec, built_model):
    model = built_model

    assert model.nu == 12
    actuator_names = [model.actuator(i).name for i in range(model.nu)]
    assert actuator_names == robot_spec.actuated_joints  # canonical action/obs order

    # Locks the compiled armature to PLAN.md's v1 table. This cannot prove
    # the preset override ran (every sizing_ideal armature equals the XML
    # default, so pass-through would look identical); the override mechanism
    # itself is proven by tests/test_injection.py, where the toy XML carries
    # 99.0/88.0 placeholders that the preset replaces.
    hip_pitch_dof = model.joint("left_hip_pitch_joint").dofadr[0]
    assert model.dof_armature[hip_pitch_dof] == pytest.approx(0.095625)

    for joint_name in ARM_JOINTS:
        joint = model.joint(joint_name)
        assert joint.stiffness[0] == pytest.approx(80.0)
        assert joint.damping[0] == pytest.approx(15.0)

    assert model.nkey >= 1
    key = model.key("home")
    free_addr = next(
        model.jnt_qposadr[i]
        for i in range(model.njnt)
        if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE
    )
    assert key.qpos[free_addr + 2] == pytest.approx(0.75)


def test_home_keyframe_steps_without_nan(robot_spec, built_model):
    model = built_model
    data = mujoco.MjData(model)
    key = model.key("home")
    home = robot_spec.keyframes["home"]

    data.qpos[:] = key.qpos
    ctrl = np.zeros(model.nu)
    for i, joint_name in enumerate(robot_spec.actuated_joints):
        ctrl[i] = home.joints.get(joint_name, 0.0)
    data.ctrl[:] = ctrl
    mujoco.mj_forward(model, data)

    for _ in range(100):
        mujoco.mj_step(model, data)

    assert np.all(np.isfinite(data.qpos))
    assert np.all(np.isfinite(data.qvel))
