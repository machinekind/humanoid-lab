"""Build-order step 5 gate: robots/roboto_origin parses, injects, and steps clean.

CPU-fast on purpose (module-scoped fixtures, no MJX): this is the per-robot
compile/NaN smoke test PLAN.md's repo shape calls out for CI, not a training
or sizing check.

Deviates from tests/test_asimov_v1.py in one respect: asimov validates its
RobotSpec against the compiled *source* XML, because asimov's source XML
already carries the foot sites/geoms robot.yaml names. RPO's source XML
(source/mjcf/rpo.xml) has none of that -- foot_sites/foot_geoms only exist
once model_patches injects them at build time -- so validate_against_model
here runs against the *built* model instead.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from humanoid_lab import paths
from humanoid_lab.robot.build import build_spec, compile_spec
from humanoid_lab.robot.spec import load_robot_spec, validate_against_model

ROBOT_DIR = paths.ROBOTS_DIR / "roboto_origin"


@pytest.fixture(scope="module")
def robot_spec():
    return load_robot_spec(ROBOT_DIR)


@pytest.fixture(scope="module")
def built_model():
    spec = build_spec(ROBOT_DIR, "deploy_pd")
    return compile_spec(spec)


def test_load_robot_spec_parses_and_validates_against_built_model(robot_spec, built_model):
    assert robot_spec.name == "roboto_origin"
    assert robot_spec.model_xml == "source/mjcf/rpo.xml"
    assert len(robot_spec.actuated_joints) == 23
    validate_against_model(robot_spec, built_model)  # must not raise


def test_build_spec_deploy_pd_compiles(robot_spec, built_model):
    model = built_model

    assert model.nu == 23
    actuator_names = [model.actuator(i).name for i in range(model.nu)]
    assert actuator_names == robot_spec.actuated_joints  # canonical action/obs order

    assert model.nq == 30  # 7 (free joint) + 23 hinge joints

    # 69 source actuator sensors (23 actuatorpos/vel/frc each) survive
    # build_spec's strip because injection reuses every one of their joint
    # names, plus the 6 IMU-site sensors (orientation, position,
    # angular-velocity, linear-velocity, linear-acceleration, magnetometer).
    assert model.nsensor == 75

    # rpo.xml ships solver="PGS", which MJX cannot run; model_patches.options
    # overrides it to newton.
    assert model.opt.solver == mujoco.mjtSolver.mjSOL_NEWTON

    # model_patches.mesh_collisions: visual makes every mesh geom (all of
    # the source XML's robot-body geometry) collision-inert; the six
    # injected foot capsules and the source's ground plane are the only
    # remaining collision surfaces.
    for i in range(model.ngeom):
        if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH:
            assert model.geom_contype[i] == 0
            assert model.geom_conaffinity[i] == 0

    for name in robot_spec.foot_geoms:
        # model.geom(name) raises KeyError if the capsule was not injected.
        assert model.geom(name).contype[0] == 1

    assert 33 < model.body_mass.sum() < 34.5

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
    """Regression guard for the keyframe-height bug docs/adding-a-robot.md
    warns about: every keyframe's lowest foot-geom bottom must sit just
    above the floor -- not floating the robot in the air and not clipping
    through it.
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

    for _ in range(200):
        mujoco.mj_step(model, data)

    assert np.all(np.isfinite(data.qpos))
    assert np.all(np.isfinite(data.qvel))

    # The PD gains should hold the pose, not let the robot collapse.
    free_addr = next(
        model.jnt_qposadr[i]
        for i in range(model.njnt)
        if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE
    )
    assert data.qpos[free_addr + 2] > 0.5
