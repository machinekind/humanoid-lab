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

# The ten body primitives robot.yaml's model_patches.geoms injects on top of
# the six foot capsules, so a fallen robot rests on the floor instead of
# passing through it.
BODY_COLLISION_GEOMS = (
    "base_collision",
    "torso_collision",
    "left_thigh_collision",
    "right_thigh_collision",
    "left_shin_collision",
    "right_shin_collision",
    "left_upper_arm_collision",
    "right_upper_arm_collision",
    "left_forearm_collision",
    "right_forearm_collision",
)


def _home_pd_hold(robot_spec, model, data):
    """Set qpos to the home keyframe and ctrl to the PD targets that hold it."""
    key = model.key("home")
    home = robot_spec.keyframes["home"]
    data.qpos[:] = key.qpos
    data.qvel[:] = 0
    for i in range(model.nu):
        data.ctrl[i] = home.joints.get(model.actuator(i).name, 0.0)


def _free_joint_qpos_addr(model):
    return next(
        int(model.jnt_qposadr[i])
        for i in range(model.njnt)
        if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE
    )


@pytest.fixture(scope="module")
def robot_spec():
    return load_robot_spec(ROBOT_DIR)


@pytest.fixture(scope="module")
def built_model():
    spec = build_spec(ROBOT_DIR, "deploy_pd")
    return compile_spec(spec)


@pytest.fixture(scope="module")
def sizing_ideal_model():
    spec = build_spec(ROBOT_DIR, "sizing_ideal")
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
    # the source XML's robot-body geometry) collision-inert; the injected
    # primitives (six foot capsules plus the ten body primitives below)
    # and the source's ground plane are the only remaining collision
    # surfaces.
    for i in range(model.ngeom):
        if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH:
            assert model.geom_contype[i] == 0
            assert model.geom_conaffinity[i] == 0

    for name in robot_spec.foot_geoms:
        # model.geom(name) raises KeyError if the capsule was not injected.
        assert model.geom(name).contype[0] == 1

    # The body primitives from the fell-through-the-floor fix (robot.yaml's
    # model_patches.geoms comment): without them, only the feet collide and
    # a toppled robot passes through the floor.
    for name in BODY_COLLISION_GEOMS:
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


def test_home_settle_contacts_floor_and_feet_only(robot_spec, built_model):
    """Standing at home under PD hold, every contact must be a foot capsule
    against the floor: no body primitive may touch the ground (a polluted
    stance would corrupt any contact-derived signal), and no robot-robot
    pair may exist at all (the source XML's contype=1/conaffinity=0 scheme
    promises primitives only ever pair with the floor).
    """
    model = built_model
    data = mujoco.MjData(model)
    _home_pd_hold(robot_spec, model, data)
    mujoco.mj_forward(model, data)

    floor_id = model.geom("ground").id
    foot_ids = {model.geom(n).id for n in robot_spec.foot_geoms}
    for _ in range(500):
        mujoco.mj_step(model, data)
        for c in data.contact[: data.ncon]:
            pair = {c.geom1, c.geom2}
            assert floor_id in pair, (
                f"robot-robot contact {model.geom(c.geom1).name} <-> {model.geom(c.geom2).name}"
            )
            other = (pair - {floor_id}).pop()
            assert other in foot_ids, (
                f"non-foot geom {model.geom(other).name} touches the floor at the home pose"
            )


def test_toppled_robot_rests_on_the_floor(robot_spec, built_model):
    """Regression for the fell-through-the-floor bug (2026-07-15): with
    only foot capsules as collision geometry, a lateral shove at the home
    pose ended with the base at z = -0.73 m, the whole body below the
    ground plane and dangling from the feet. With the body primitives the
    same shove must end with the robot lying ON the floor.
    """
    model = built_model
    data = mujoco.MjData(model)
    _home_pd_hold(robot_spec, model, data)
    mujoco.mj_forward(model, data)

    free_addr = _free_joint_qpos_addr(model)
    base_body_id = model.body("base_link").id
    n_settle = int(1.0 / model.opt.timestep)
    n_shove = int(0.2 / model.opt.timestep)
    n_watch = int(3.0 / model.opt.timestep)

    min_base_z = np.inf
    for step in range(n_settle + n_shove + n_watch):
        if n_settle <= step < n_settle + n_shove:
            data.xfrc_applied[base_body_id, :3] = [120.0, 60.0, 0.0]
        else:
            data.xfrc_applied[base_body_id, :3] = 0.0
        mujoco.mj_step(model, data)
        min_base_z = min(min_base_z, data.qpos[free_addr + 2])

    assert np.all(np.isfinite(data.qpos))
    # The base may dip briefly while limbs absorb the fall, but must never
    # sink meaningfully below the plane, and must come to rest on top of it
    # (lying flat it sits at ~0.054 m, roughly the base box's half-height).
    assert min_base_z > -0.02, f"base passed through the floor (min z {min_base_z:.3f} m)"
    assert data.qpos[free_addr + 2] > 0.02, (
        f"base ended below the floor (z {data.qpos[free_addr + 2]:.3f} m)"
    )


def test_build_spec_sizing_ideal_compiles(robot_spec, sizing_ideal_model):
    model = sizing_ideal_model

    assert model.nu == 23
    actuator_names = [model.actuator(i).name for i in range(model.nu)]
    assert actuator_names == robot_spec.actuated_joints  # canonical action/obs order

    # Lock the preset's headline numbers into the compiled model: one
    # actuator from the 4340P ankle group and one from the PROVISIONAL
    # 10010L thigh group (see the preset's header for both derivations).
    ankle = model.actuator("left_ankle_pitch_joint")
    assert ankle.gainprm[0] == pytest.approx(252.7)
    assert list(ankle.forcerange) == pytest.approx([-54.0, 54.0])

    thigh = model.actuator("left_thigh_pitch_joint")
    assert thigh.gainprm[0] == pytest.approx(120.8)
    assert list(thigh.forcerange) == pytest.approx([-240.0, 240.0])

    ankle_dof = model.joint("left_ankle_pitch_joint").dofadr[0]
    assert model.dof_armature[ankle_dof] == pytest.approx(0.064)
    thigh_dof = model.joint("left_thigh_pitch_joint").dofadr[0]
    assert model.dof_armature[thigh_dof] == pytest.approx(0.0306)


def test_sizing_ideal_home_keyframe_steps_without_nan(robot_spec, sizing_ideal_model):
    """sizing_ideal's kp reaches 252.7. deploy_pd's kp reaches 150.0 at
    most. This test reruns test_home_keyframe_steps_without_nan's hold
    check against sizing_ideal's own compiled model. It confirms the
    stiffer gains still hold the home pose instead of letting the robot
    collapse.
    """
    model = sizing_ideal_model
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

    free_addr = next(
        model.jnt_qposadr[i]
        for i in range(model.njnt)
        if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE
    )
    assert data.qpos[free_addr + 2] > 0.5
