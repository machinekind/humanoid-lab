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


def _capsule_bottom(model: mujoco.MjModel, data: mujoco.MjData, geom_id: int) -> float:
    """Exact world-frame bottom of a capsule geom.

    A capsule is the Minkowski sum of a line segment and a ball of
    `radius`, so its lowest point is always (the lower of its two
    world-frame end-cap centers) minus `radius`, regardless of orientation.
    This is exact where `model.geom_rbound` (bounding-sphere radius) is not:
    rbound over-estimates badly for a capsule lying flat.
    """
    xpos = data.geom_xpos[geom_id]
    xmat = data.geom_xmat[geom_id].reshape(3, 3)
    radius, half_len = model.geom_size[geom_id, 0], model.geom_size[geom_id, 1]
    local_z = xmat[:, 2]
    end1 = xpos + local_z * half_len
    end2 = xpos - local_z * half_len
    return min(end1[2], end2[2]) - radius


def test_keyframe_base_z_matches_robot_yaml(robot_spec, built_model):
    """robot.yaml is the single source of truth for keyframe base height
    (its own comment records how the value was measured); lock the
    compiled model's keyframe to whatever robot.yaml currently says instead
    of a hardcoded literal that can silently drift from it.
    """
    free_addr = next(
        built_model.jnt_qposadr[i]
        for i in range(built_model.njnt)
        if built_model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE
    )
    for kf_name, kf in robot_spec.keyframes.items():
        key = built_model.key(kf_name)
        assert key.qpos[free_addr + 2] == pytest.approx(kf.base_pos[2])


def test_keyframe_feet_touch_the_floor(robot_spec, built_model):
    """Regression guard for the keyframe-height bug (PLAN.md step 6 review):
    every keyframe's lowest foot-geom bottom must sit just above the floor
    -- not floating the robot in the air (the original 0.75/0.70 values, a
    copy of v0's height plus an unmeasured margin) and not clipping through
    it.
    """
    model = built_model
    foot_geom_ids = [model.geom(n).id for n in robot_spec.foot_geoms]
    data = mujoco.MjData(model)
    for kf_name in robot_spec.keyframes:
        key = model.key(kf_name)
        data.qpos[:] = key.qpos
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)
        lowest = min(_capsule_bottom(model, data, gid) for gid in foot_geom_ids)
        assert 0.0 <= lowest <= 0.02, f"keyframe '{kf_name}': lowest foot bottom {lowest:.4f} m"


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
