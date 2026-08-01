"""The conformance suite every robot in the lab has to pass.

Discovery-driven on purpose. Adding a robot means adding a directory under
`robots/`; it must never mean writing a test file. This module globs
`robots/*/robot.yaml` (skipping `_template`, which ships templates rather
than a robot) and parametrizes over each robot's `actuators/*.yaml` presets.
A new robot directory is picked up by collection with no edit here.

Everything asserted here is either derived from the robot's own yaml (the
compiled model must agree with what robot.yaml and the preset declare) or a
property every lab robot must have whatever its geometry. Nothing in this
file names a joint, site, geom or body. The two per-robot NUMBERS that
survived the robot-named test files this replaced live in the optional
`robots/<name>/conformance.yaml` next to robot.yaml, not in code.

Cost: one build+compile and one env per (robot, preset), cached module-wide.
The plain-MuJoCo rollouts run on the CPU model and are cheap; the env fixture
is the expensive one (MJX put_model plus a jit), which is why it is separate
and why the checks that do not need an env do not ask for it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import jax
import mujoco
import numpy as np
import pytest
import yaml

from humanoid_lab import paths
from humanoid_lab.envs import symmetry
from humanoid_lab.envs.joystick import Joystick, default_config
from humanoid_lab.robot.build import build_spec, compile_spec
from humanoid_lab.robot.presets import action_scale, load_actuator_preset, resolve
from humanoid_lab.robot.spec import RobotSpec, load_robot_spec, validate_against_model

# Keyframe foot-clearance band, docs/adding-a-robot.md's measurement rule:
# the lowest foot-geom bottom must sit just above the floor, neither floating
# the robot in the air nor clipping it through the ground.
FOOT_CLEARANCE_BAND = (0.0, 0.02)

# Physical durations, not step counts: the robots run different timesteps
# (asimov_v1 0.002 s, roboto_origin 0.001 s) and a step count would mean a
# different amount of physics per robot.
HOLD_SECONDS = 0.5  # PD hold at the reset keyframe must survive this
NAN_SECONDS = 2.0  # ... and stay finite this long, standing or not
SETTLE_SECONDS = 1.0  # shove test: settle before the push
SHOVE_SECONDS = 0.2  # ... push for this long ...
WATCH_SECONDS = 3.0  # ... then watch it land

# Shove impulse, per kilogram of robot, applied to the floating base as a
# lateral+forward force. Scaled by mass so the same test means the same
# thing on a 33 kg humanoid and on the toy fixture: at roboto_origin's
# 33.76 kg this is (120.2, 60.1) N, reproducing the (120, 60) N shove the
# original fell-through-the-floor regression used.
SHOVE_N_PER_KG = (3.56, 1.78, 0.0)

# Largest base-height dip tolerated while the robot's limbs absorb a fall,
# and the height it has to come to rest above. Both are "did it end up on
# top of the floor or under it" bounds, not fit values.
FALL_MIN_BASE_Z = -0.02
FALL_REST_BASE_Z = 0.02

# The PD hold at the reset keyframe must not lose more than this fraction of
# the keyframe's own base height. Neither robot's home keyframe is a lasting
# equilibrium (both are over by ~1.5 s under their own preset gains, which is
# why envs/base.py's real_pose_ref gate exists); this catches gains that let
# the robot collapse immediately, which is what the robot-named tests it
# replaces checked at 0.2 s.
HOLD_MIN_HEIGHT_FRACTION = 0.85


# -- discovery ---------------------------------------------------------------


@dataclass(frozen=True)
class RobotCase:
    """One (robot directory, actuator preset) pair."""

    robot_dir: Path
    preset: str

    @property
    def id(self) -> str:
        return f"{self.robot_dir.name}__{self.preset}"


def _robot_dirs() -> list[Path]:
    """Every robot directory in the lab.

    `robots/_template` is skipped: it ships `robot.yaml.template`, not a
    `robot.yaml`, so the glob already misses it -- the explicit filter is
    here so a future template that does carry a real robot.yaml still stays
    out of the suite.
    """
    return [
        p.parent
        for p in sorted(paths.ROBOTS_DIR.glob("*/robot.yaml"))
        if p.parent.name != "_template"
    ]


def _cases() -> list[RobotCase]:
    cases = []
    for robot_dir in _robot_dirs():
        presets = sorted(p.stem for p in (robot_dir / "actuators").glob("*.yaml"))
        assert presets, f"{robot_dir} declares no actuator preset in actuators/*.yaml"
        cases.extend(RobotCase(robot_dir, preset) for preset in presets)
    return cases


ROBOT_DIRS = _robot_dirs()
ROBOT_IDS = [d.name for d in ROBOT_DIRS]
CASES = _cases()
CASE_IDS = [c.id for c in CASES]


# -- module-scoped caches ----------------------------------------------------
#
# One RobotSpec per robot, one compiled model and one env per (robot,
# preset). Plain dicts behind a module-scoped fixture so everything is
# released at module teardown rather than living for the whole session.


@pytest.fixture(scope="module")
def _cache():
    return {"spec": {}, "model": {}, "env": {}, "expect": {}}


def robot_spec(_cache, robot_dir: Path) -> RobotSpec:
    if robot_dir not in _cache["spec"]:
        _cache["spec"][robot_dir] = load_robot_spec(robot_dir)
    return _cache["spec"][robot_dir]


def built_model(_cache, case: RobotCase) -> mujoco.MjModel:
    if case.id not in _cache["model"]:
        _cache["model"][case.id] = compile_spec(build_spec(case.robot_dir, case.preset))
    return _cache["model"][case.id]


def built_env(_cache, case: RobotCase) -> Joystick:
    """A stock Joystick env for this case.

    The only override is `reset_keyframe`, and only when the robot has no
    keyframe by that config default's name (see `_reset_keyframe`): the
    point is to construct the env every training run would build.
    """
    if case.id not in _cache["env"]:
        spec = robot_spec(_cache, case.robot_dir)
        config = default_config()
        config.reset_keyframe = _reset_keyframe(spec)
        _cache["env"][case.id] = Joystick(case.robot_dir, case.preset, config)
    return _cache["env"][case.id]


def expectations(_cache, robot_dir: Path) -> dict:
    """The robot's optional `conformance.yaml`, or {} when it has none.

    The escape hatch for a genuinely per-robot NUMBER (a measured total
    mass), kept as data next to robot.yaml rather than as a branch in this
    file. Absent file = no extra expectations, which is the toy fixture's
    case and the case any new robot starts in.
    """
    if robot_dir not in _cache["expect"]:
        path = robot_dir / "conformance.yaml"
        raw = yaml.safe_load(path.read_text()) if path.exists() else None
        _cache["expect"][robot_dir] = raw or {}
    return _cache["expect"][robot_dir]


# -- helpers -----------------------------------------------------------------


def _reset_keyframe(spec: RobotSpec) -> str:
    """The keyframe this robot resets from.

    `envs/base.py` reads the task config's `reset_keyframe` (default "home").
    A robot that has that keyframe uses it. A robot with exactly one keyframe
    uses that one whatever it is called, which is what lets the toy fixture
    keep its own `standing` name without this suite knowing about it. Any
    other robot has to say which, and there is no way to say it yet -- so it
    fails loudly here instead of picking one.
    """
    default = default_config().reset_keyframe
    if default in spec.keyframes:
        return default
    if len(spec.keyframes) == 1:
        return next(iter(spec.keyframes))
    raise AssertionError(
        f"robot '{spec.name}' has no '{default}' keyframe and declares more than one "
        f"({sorted(spec.keyframes)}), so this suite cannot tell which one it resets from"
    )


def _free_joint(model: mujoco.MjModel) -> tuple[int, int, int]:
    """(qpos addr, dof addr, body id) of the model's one free joint.

    Found by joint TYPE, never assumed to be joint 0 / qpos[0:7]: the same
    rule envs/base.py's `_free_joint_addr` follows.
    """
    free = [i for i in range(model.njnt) if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE]
    assert len(free) == 1, (
        f"expected exactly one free joint (the floating base), found {len(free)}"
    )
    i = free[0]
    return int(model.jnt_qposadr[i]), int(model.jnt_dofadr[i]), int(model.jnt_bodyid[i])


def _geom_bottom(model: mujoco.MjModel, data: mujoco.MjData, geom_id: int) -> float:
    """Exact world-frame lowest point of a primitive geom.

    Exact where `model.geom_rbound` (the bounding-sphere radius) is not:
    rbound over-estimates badly for a capsule lying flat. Per type:

    - sphere: centre minus radius, orientation-free.
    - capsule: the Minkowski sum of a segment and a ball, so the lowest point
      is the lower of the two world-frame end-cap centres minus the radius,
      at any orientation.
    - box: the lowest of the eight corners, which is the centre minus the
      sum of |R[2,i]| * half_extent_i.

    Anything else fails rather than guessing. A foot geom that is a mesh or a
    plane cannot have its sole measured this way, and silently skipping it
    would turn the keyframe-height gate into a no-op.
    """
    xpos = data.geom_xpos[geom_id]
    xmat = data.geom_xmat[geom_id].reshape(3, 3)
    size = model.geom_size[geom_id]
    geom_type = model.geom_type[geom_id]

    if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
        return float(xpos[2] - size[0])
    if geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
        local_z = xmat[:, 2]
        ends = (xpos + local_z * size[1], xpos - local_z * size[1])
        return float(min(e[2] for e in ends) - size[0])
    if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
        return float(xpos[2] - np.abs(xmat[2, :3]) @ size[:3])
    raise AssertionError(
        f"geom '{model.geom(geom_id).name}' is type {mujoco.mjtGeom(geom_type).name}; the "
        "keyframe foot-height rule needs a sphere, capsule or box to measure a sole against"
    )


def _pd_hold(spec: RobotSpec, model: mujoco.MjModel) -> mujoco.MjData:
    """Fresh MjData at the reset keyframe, ctrl set to hold that pose.

    ctrl carries the keyframe's own joint angles. For a `pd` preset those are
    position targets and this really is a hold; for an `ideal_torque` preset
    ctrl is a torque and the same numbers are a small constant torque. That
    asymmetry is deliberate and matches what the robot-named tests did: the
    rollout is a "does it explode" probe over the robot's own reset pose, not
    a controller.
    """
    data = mujoco.MjData(model)
    keyframe = model.key(_reset_keyframe(spec))
    data.qpos[:] = keyframe.qpos
    data.qvel[:] = 0.0
    reset_kf = spec.keyframes[_reset_keyframe(spec)]
    for i in range(model.nu):
        data.ctrl[i] = reset_kf.joints.get(model.actuator(i).name, 0.0)
    mujoco.mj_forward(model, data)
    return data


def _floor_geom_id(model: mujoco.MjModel) -> int:
    """The ground plane, found by geom type rather than by name.

    asimov_v1 calls it `floor` and roboto_origin calls it `ground`; both
    models have exactly one plane geom, which is what "the floor" means.
    """
    planes = [i for i in range(model.ngeom) if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE]
    assert len(planes) == 1, (
        f"expected exactly one plane geom (the floor), found {len(planes)}: "
        f"{[model.geom(i).name for i in planes]}"
    )
    return planes[0]


def _collidable(model: mujoco.MjModel, geom_id: int) -> bool:
    return bool(model.geom_contype[geom_id]) or bool(model.geom_conaffinity[geom_id])


# -- robot-level checks (no actuator preset involved) -------------------------


@pytest.mark.parametrize("robot_dir", ROBOT_DIRS, ids=ROBOT_IDS)
def test_robot_yaml_parses_and_points_at_a_real_model(_cache, robot_dir):
    """The entry point every downstream consumer starts from."""
    spec = robot_spec(_cache, robot_dir)

    assert spec.name, f"{robot_dir}/robot.yaml declares no name"
    assert spec.model_xml_path.exists(), (
        f"robot '{spec.name}' points model_xml at {spec.model_xml_path}, which does not exist"
    )
    assert spec.actuated_joints, "a robot with no actuated joints has nothing to train"
    assert spec.foot_sites, "foot_sites is what every gait signal is measured at"
    assert spec.foot_geoms, "foot_geoms is what every contact signal is read from"
    assert spec.keyframes, (
        "a robot needs at least one keyframe: envs/base.py resets from one and anchors the "
        "PD default pose on it"
    )
    # _reset_keyframe raises with its own message if the robot is ambiguous.
    assert _reset_keyframe(spec) in spec.keyframes


@pytest.mark.parametrize("robot_dir", ROBOT_DIRS, ids=ROBOT_IDS)
def test_every_actuator_preset_resolves_onto_every_actuated_joint(_cache, robot_dir):
    """Each preset covers the robot's groups, and each joint gets a scale.

    `resolve` raises if a preset names a group the robot does not have or
    leaves an actuated joint's group without params; `action_scale` is what
    envs/base.py turns a policy action into ctrl with, so a joint missing
    from it would be an env that cannot be built.
    """
    spec = robot_spec(_cache, robot_dir)
    presets = sorted(p.stem for p in (robot_dir / "actuators").glob("*.yaml"))

    for preset_name in presets:
        preset = load_actuator_preset(robot_dir, preset_name)
        params = resolve(preset, spec)
        assert sorted(params) == sorted(spec.actuated_joints), preset_name

        scales = action_scale(preset, spec)
        assert sorted(scales) == sorted(spec.actuated_joints), preset_name
        for joint_name, scale in scales.items():
            assert np.isfinite(scale) and scale > 0.0, (preset_name, joint_name, scale)


@pytest.mark.parametrize("robot_dir", ROBOT_DIRS, ids=ROBOT_IDS)
def test_symmetry_map_derives_a_valid_mirror(_cache, robot_dir):
    """A declared `symmetry` map has to survive the numeric derivation.

    envs/symmetry.py does not read signs off axis conventions: it perturbs
    each joint of a pair and asks which sign of the partner reproduces the
    mirrored motion, requiring the winner to fit within MAX_FIT_RESIDUAL and
    to beat the loser by MIN_DISCRIMINATION. Passing therefore proves the
    compiled model really IS mirror-symmetric under robot.yaml's pairing --
    which is the only condition under which the augmentation means anything.

    A robot with no symmetry map skips: the map is optional, and
    `symmetry.enable` is unavailable to that robot until it has one.
    """
    spec = robot_spec(_cache, robot_dir)
    if not spec.symmetry:
        pytest.skip(f"robot '{spec.name}' declares no symmetry map")

    presets = sorted(p.stem for p in (robot_dir / "actuators").glob("*.yaml"))
    model = built_model(_cache, RobotCase(robot_dir, presets[0]))
    qpos = np.array(model.key(_reset_keyframe(spec)).qpos)

    tables = symmetry.derive(model, spec, qpos)

    n_joints, n_feet = len(spec.actuated_joints), len(spec.foot_sites)
    assert tables.joint_perm.shape == (n_joints,)
    assert tables.joint_sign.shape == (n_joints,)
    assert tables.foot_perm.shape == (n_feet,)
    assert set(np.unique(tables.joint_sign)) <= {-1.0, 1.0}
    # An involution: mirroring twice is the identity, on joints and on feet.
    assert tables.joint_perm[tables.joint_perm].tolist() == list(range(n_joints))
    assert tables.foot_perm[tables.foot_perm].tolist() == list(range(n_feet))


# -- build/compile checks (per robot x preset) --------------------------------


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_model_compiles_and_the_spec_validates_against_it(_cache, case):
    """Every name robot.yaml uses exists in the model the robot trains on.

    Validated against the BUILT model rather than the source XML, because
    the built model is the one every env, eval and export step sees, and
    because a robot may only acquire its foot sites and geoms at build time
    (roboto_origin's source MJCF ships neither; robot.yaml's model_patches
    inject both).
    """
    spec = robot_spec(_cache, case.robot_dir)
    model = built_model(_cache, case)
    validate_against_model(spec, model)  # must not raise


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_actuators_are_the_actuated_joints_in_canonical_order(_cache, case):
    """The action/obs contract: one actuator per actuated joint, in order.

    robot.yaml's `actuated_joints` order IS the policy's action vector order
    and its joint observation order. A model whose actuator table disagrees
    would silently permute every action a checkpoint emits.
    """
    spec = robot_spec(_cache, case.robot_dir)
    model = built_model(_cache, case)

    assert model.nu == len(spec.actuated_joints)
    assert [model.actuator(i).name for i in range(model.nu)] == spec.actuated_joints


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_the_model_has_exactly_one_floating_base(_cache, case):
    """One free joint, everything else a 1-dof hinge or slide.

    envs/base.py, envs/symmetry.py and this file all locate the base by
    finding the single free joint; a second one (or none) breaks all three.
    The qpos-width check is the same statement from the other side: 7 for the
    free joint plus one per remaining joint means nothing hides a ball joint.
    """
    model = built_model(_cache, case)
    _free_joint(model)  # asserts there is exactly one

    one_dof = (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)
    others = [i for i in range(model.njnt) if model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE]
    assert all(model.jnt_type[i] in one_dof for i in others)
    assert model.nq == 7 + len(others)


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_preset_injects_the_values_its_yaml_declares(_cache, case):
    """Every number the preset yaml carries reaches the compiled model.

    Derived from the preset, never hardcoded: `resolve` maps the yaml's
    per-group params onto every actuated joint, and this asserts the compiled
    actuator holds exactly those. The branch is on the ACTUATOR MODEL, not on
    the robot -- `pd` and `ideal_torque` inject structurally different
    actuators (actuators/models.py), and checking gainprm without checking
    gaintype/biastype would pass a regression that killed all feedback.
    """
    spec = robot_spec(_cache, case.robot_dir)
    model = built_model(_cache, case)
    preset = load_actuator_preset(case.robot_dir, case.preset)
    params = resolve(preset, spec)

    for joint_name in spec.actuated_joints:
        p = params[joint_name]
        where = (case.id, joint_name)
        actuator = model.actuator(joint_name)

        effort_pair = (-p.effort_limit, p.effort_limit)
        assert tuple(actuator.forcerange) == pytest.approx(effort_pair), where
        assert actuator.forcelimited[0] == 1, where

        if preset.model == "pd":
            assert actuator.gainprm[0] == pytest.approx(p.kp), where
            assert actuator.biasprm[1] == pytest.approx(-p.kp), where
            assert actuator.biasprm[2] == pytest.approx(-p.kd), where
            assert actuator.gaintype[0] == mujoco.mjtGain.mjGAIN_FIXED, where
            assert actuator.biastype[0] == mujoco.mjtBias.mjBIAS_AFFINE, where
            # Deliberately unclamped (asimov-mjlab convention): the policy may
            # command a setpoint past the kinematic limit.
            assert actuator.ctrllimited[0] == 0, where
        elif preset.model == "ideal_torque":
            assert actuator.gainprm[0] == pytest.approx(1.0), where
            assert actuator.biastype[0] == mujoco.mjtBias.mjBIAS_NONE, where
            assert tuple(actuator.ctrlrange) == pytest.approx(effort_pair), where
            assert actuator.ctrllimited[0] == 1, where
        else:
            pytest.fail(f"{case.id}: no conformance rule for actuator model '{preset.model}'")

        joint = model.joint(joint_name)
        if p.armature is not None:
            assert model.dof_armature[joint.dofadr[0]] == pytest.approx(p.armature), where
        if p.frictionloss is not None:
            assert joint.frictionloss[0] == pytest.approx(p.frictionloss), where


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_passive_joints_get_the_spring_robot_yaml_declares(_cache, case):
    """robot.yaml's passive_joints, applied at build time and independent of
    the actuator preset. No-op for a fully actuated robot (roboto_origin)."""
    spec = robot_spec(_cache, case.robot_dir)
    model = built_model(_cache, case)

    for joint_name, passive in spec.passive_joints.items():
        joint = model.joint(joint_name)
        assert joint.stiffness[0] == pytest.approx(passive.stiffness), (case.id, joint_name)
        assert joint.damping[0] == pytest.approx(passive.damping), (case.id, joint_name)


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_model_patches_land_in_the_compiled_model(_cache, case):
    """robot.yaml's optional model_patches section, read back off the model.

    No-op for a robot whose source XML is MJX-ready as vendored. For one that
    is not, this is the only place the patches are checked end to end: an
    injected site or geom that silently failed to apply would otherwise
    surface as a training run with no feet.
    """
    spec = robot_spec(_cache, case.robot_dir)
    model = built_model(_cache, case)
    patches = spec.model_patches

    options = patches.options
    if options.solver is not None:
        expected = {
            "pgs": mujoco.mjtSolver.mjSOL_PGS,
            "cg": mujoco.mjtSolver.mjSOL_CG,
            "newton": mujoco.mjtSolver.mjSOL_NEWTON,
        }[options.solver]
        assert model.opt.solver == expected, case.id
    if options.iterations is not None:
        assert model.opt.iterations == options.iterations, case.id
    if options.timestep is not None:
        assert model.opt.timestep == pytest.approx(options.timestep), case.id

    for name, site in patches.sites.items():
        assert model.body(model.site_bodyid[model.site(name).id]).name == site.body, name

    for name in patches.geoms:
        geom = model.geom(name)  # KeyError here means the geom never got injected
        assert _collidable(model, geom.id), (
            f"{case.id}: injected collision primitive '{name}' is not collidable"
        )

    if patches.mesh_collisions == "visual":
        mesh_ids = [
            i for i in range(model.ngeom) if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH
        ]
        assert mesh_ids, f"{case.id}: mesh_collisions is set but the model has no mesh geoms"
        for i in mesh_ids:
            assert not _collidable(model, i), (
                f"{case.id}: mesh geom '{model.geom(i).name}' should be collision-inert"
            )


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_every_actuator_sensor_resolves_to_an_actuated_joint(_cache, case):
    """build_spec's strip promise, checked on the built model.

    A vendored source XML may ship actuatorpos/actuatorvel/actuatorfrc
    sensors for joints this robot does not actuate. build_spec deletes those
    (they would dangle once the source actuators are stripped) and keeps the
    ones injection recreates an actuator for. Every survivor must therefore
    name an actuated joint.
    """
    spec = robot_spec(_cache, case.robot_dir)
    model = built_model(_cache, case)
    actuator_sensors = (
        mujoco.mjtSensor.mjSENS_ACTUATORPOS,
        mujoco.mjtSensor.mjSENS_ACTUATORVEL,
        mujoco.mjtSensor.mjSENS_ACTUATORFRC,
    )

    actuated = set(spec.actuated_joints)
    for i in range(model.nsensor):
        if model.sensor_type[i] not in actuator_sensors:
            continue
        name = model.actuator(int(model.sensor_objid[i])).name
        assert name in actuated, (
            f"{case.id}: sensor '{model.sensor(i).name}' points at actuator '{name}', which is "
            "not one of robot.yaml's actuated joints"
        )


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_every_foot_geom_belongs_to_a_declared_foot(_cache, case):
    """foot_geoms and foot_sites have to agree about what a foot is.

    envs/base.py aggregates contact and velocity per LOGICAL foot by walking
    each foot geom's body up the kinematic tree to a foot_sites body (a robot
    may carry ten geoms per foot). A geom that reaches the world body instead
    makes the env raise at construction; this catches it at the model.
    """
    spec = robot_spec(_cache, case.robot_dir)
    model = built_model(_cache, case)

    site_bodies = [int(model.site_bodyid[model.site(n).id]) for n in spec.foot_sites]
    owners = set()
    for geom_name in spec.foot_geoms:
        body = int(model.geom_bodyid[model.geom(geom_name).id])
        seen = set()
        while body not in site_bodies:
            assert body != 0 and body not in seen, (
                f"{case.id}: foot geom '{geom_name}' is not under any foot_sites body"
            )
            seen.add(body)
            body = int(model.body_parentid[body])
        owners.add(site_bodies.index(body))
        assert _collidable(model, model.geom(geom_name).id), (
            f"{case.id}: foot geom '{geom_name}' is not collidable, so it can never register "
            "a ground contact"
        )

    assert owners == set(range(len(spec.foot_sites))), (
        f"{case.id}: foot site(s) "
        f"{[spec.foot_sites[i] for i in sorted(set(range(len(spec.foot_sites))) - owners)]} own "
        "no foot geom"
    )


# -- keyframe checks ---------------------------------------------------------


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_keyframes_are_baked_into_the_model_as_robot_yaml_writes_them(_cache, case):
    """robot.yaml is the single source of truth for every keyframe.

    Base position, base orientation and each named joint angle are read back
    off the compiled key, so a value that drifts between the yaml and the
    model fails here rather than shifting a reset pose silently.
    """
    spec = robot_spec(_cache, case.robot_dir)
    model = built_model(_cache, case)
    base_addr, _, _ = _free_joint(model)

    assert model.nkey == len(spec.keyframes)
    for kf_name, kf in spec.keyframes.items():
        key = model.key(kf_name)
        base = list(key.qpos[base_addr : base_addr + 3])
        assert base == pytest.approx(list(kf.base_pos)), kf_name
        assert list(key.qpos[base_addr + 3 : base_addr + 7]) == pytest.approx(
            list(kf.base_quat)
        ), kf_name
        for joint_name, angle in kf.joints.items():
            addr = model.joint(joint_name).qposadr[0]
            assert key.qpos[addr] == pytest.approx(angle), (kf_name, joint_name)


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_every_keyframe_puts_the_feet_on_the_floor(_cache, case):
    """docs/adding-a-robot.md's measurement rule, as a gate.

    asimov_v1's original keyframes were copied from a previous revision's
    standing height plus a margin and floated the robot 0.119 m clear of the
    ground; nothing noticed until this was measured. Every keyframe's lowest
    foot-geom bottom must sit in the FOOT_CLEARANCE_BAND: neither in the air
    nor clipped through the floor.
    """
    spec = robot_spec(_cache, case.robot_dir)
    model = built_model(_cache, case)
    foot_geom_ids = [model.geom(n).id for n in spec.foot_geoms]
    data = mujoco.MjData(model)
    lo, hi = FOOT_CLEARANCE_BAND

    for kf_name in spec.keyframes:
        data.qpos[:] = model.key(kf_name).qpos
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        lowest = min(_geom_bottom(model, data, gid) for gid in foot_geom_ids)
        assert lo <= lowest <= hi, (
            f"{case.id} keyframe '{kf_name}': lowest foot-geom bottom is {lowest:.4f} m, outside "
            f"the {lo}-{hi} m band"
        )


# -- plain-MuJoCo rollouts ---------------------------------------------------


@pytest.fixture(scope="module")
def _rollouts():
    return {}


def keyframe_rollout(_cache, _rollouts, case: RobotCase) -> dict:
    """One NAN_SECONDS rollout from the reset keyframe under a PD hold.

    Recorded once per case and read by three tests: whether the physics stays
    finite for the whole run, how much base height survived HOLD_SECONDS, and
    whether anything other than a foot touched the floor in that window.
    """
    if case.id in _rollouts:
        return _rollouts[case.id]

    spec = robot_spec(_cache, case.robot_dir)
    model = built_model(_cache, case)
    data = _pd_hold(spec, model)
    base_addr, _, _ = _free_joint(model)
    floor_id = _floor_geom_id(model)
    foot_ids = {model.geom(n).id for n in spec.foot_geoms}

    hold_steps = int(round(HOLD_SECONDS / model.opt.timestep))
    impure_contacts, finite = [], True
    for step in range(int(round(NAN_SECONDS / model.opt.timestep))):
        mujoco.mj_step(model, data)
        finite = finite and bool(np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel)))
        if step == hold_steps - 1:
            hold_base_z = float(data.qpos[base_addr + 2])
        if step < hold_steps:
            for c in data.contact[: data.ncon]:
                pair = {int(c.geom1), int(c.geom2)}
                if floor_id not in pair:
                    impure_contacts.append(
                        f"robot-robot contact {model.geom(c.geom1).name} <-> "
                        f"{model.geom(c.geom2).name} at step {step}"
                    )
                    continue
                other = (pair - {floor_id}).pop()
                if other not in foot_ids:
                    impure_contacts.append(
                        f"non-foot geom {model.geom(other).name} touches the floor at step {step}"
                    )

    _rollouts[case.id] = {
        "finite": finite,
        "hold_base_z": hold_base_z,
        "keyframe_base_z": spec.keyframes[_reset_keyframe(spec)].base_pos[2],
        "impure_contacts": impure_contacts,
    }
    return _rollouts[case.id]


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_the_reset_keyframe_steps_without_nan(_cache, _rollouts, case):
    """The per-robot compile/NaN smoke test, in plain MuJoCo on the CPU model.

    A model whose keyframe, gains or armature are wrong enough to blow the
    integrator up does it well inside NAN_SECONDS.
    """
    result = keyframe_rollout(_cache, _rollouts, case)
    assert result["finite"], f"{case.id}: qpos or qvel went non-finite within {NAN_SECONDS} s"


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_the_reset_keyframe_holds_the_robot_up(_cache, _rollouts, case):
    """The preset's gains hold the reset pose rather than dropping it.

    Not a claim that the keyframe is a lasting equilibrium -- neither robot's
    is, which is exactly why envs/base.py's real_pose_ref settle exists. It
    catches the collapse: gains or a keyframe wrong enough that the robot is
    already on its way down when the episode starts.
    """
    result = keyframe_rollout(_cache, _rollouts, case)
    floor = HOLD_MIN_HEIGHT_FRACTION * result["keyframe_base_z"]
    assert result["hold_base_z"] > floor, (
        f"{case.id}: base fell to {result['hold_base_z']:.4f} m within {HOLD_SECONDS} s, below "
        f"{HOLD_MIN_HEIGHT_FRACTION:g} of the keyframe's own {result['keyframe_base_z']:.4f} m"
    )


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_only_the_feet_touch_the_floor_at_the_reset_keyframe(_cache, _rollouts, case):
    """A clean stance: every contact is a foot geom against the ground.

    Two failures are worth separating and this catches both. A body primitive
    resting on the floor at the reset pose pollutes every contact-derived
    signal the env computes. A robot-robot contact means the model's
    contype/conaffinity scheme is not what robot.yaml assumes -- injected
    primitives are supposed to pair with the floor only.
    """
    result = keyframe_rollout(_cache, _rollouts, case)
    assert not result["impure_contacts"], f"{case.id}: {result['impure_contacts'][:5]}"


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_a_shoved_robot_lands_on_the_floor_not_through_it(_cache, case):
    """Regression for the fell-through-the-floor bug (2026-07-15).

    roboto_origin's collision geometry was feet only, so a lateral shove at
    the reset pose ended with the base at z = -0.73 m -- the whole body below
    the ground plane, dangling from the feet. Fall detection is base height
    and tilt, but the fall itself still has to resolve physically, which
    needs collision geometry on the body segments.

    The shove is scaled by the robot's own mass (SHOVE_N_PER_KG) so the test
    means the same thing whatever the robot weighs.
    """
    spec = robot_spec(_cache, case.robot_dir)
    model = built_model(_cache, case)
    data = _pd_hold(spec, model)
    base_addr, _, base_body = _free_joint(model)

    force = np.array(SHOVE_N_PER_KG) * float(model.body_mass.sum())
    dt = model.opt.timestep
    n_settle = int(round(SETTLE_SECONDS / dt))
    n_shove = int(round(SHOVE_SECONDS / dt))
    n_watch = int(round(WATCH_SECONDS / dt))

    min_base_z = np.inf
    for step in range(n_settle + n_shove + n_watch):
        data.xfrc_applied[base_body, :3] = force if n_settle <= step < n_settle + n_shove else 0.0
        mujoco.mj_step(model, data)
        min_base_z = min(min_base_z, float(data.qpos[base_addr + 2]))

    assert np.all(np.isfinite(data.qpos)), f"{case.id}: the shove diverged"
    assert min_base_z > FALL_MIN_BASE_Z, (
        f"{case.id}: base passed through the floor (min z {min_base_z:.3f} m)"
    )
    assert float(data.qpos[base_addr + 2]) > FALL_REST_BASE_Z, (
        f"{case.id}: base came to rest at z {float(data.qpos[base_addr + 2]):.3f} m, below the "
        "floor"
    )


# -- env-level checks --------------------------------------------------------


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_the_env_constructs_and_a_jitted_reset_and_step_run(_cache, case):
    """The robot is trainable: a stock task env builds and steps on MJX.

    Env construction is where most of a bad robot.yaml surfaces -- the
    validate-against-model pass, the reset-keyframe lookup, the foot geom to
    foot site mapping, the action-scale table and the sensor addresses all
    resolve there. Then one jitted step proves the model survives MJX, not
    just plain MuJoCo.
    """
    env = built_env(_cache, case)

    state = jax.jit(env.reset)(jax.random.PRNGKey(0))
    assert np.all(np.isfinite(state.obs["state"]))
    assert np.all(np.isfinite(state.obs["privileged_state"]))

    state = jax.jit(env.step)(state, np.zeros(env.action_size, dtype=np.float32))
    assert np.all(np.isfinite(state.obs["state"]))
    assert np.all(np.isfinite(state.obs["privileged_state"]))
    assert np.isfinite(state.reward)


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_observation_widths_match_the_declared_component_lists(_cache, case):
    """The obs vectors are exactly what the config's ordered lists build.

    Sizes are measured on THIS env's own catalog, never a static table -- the
    same rule envs/symmetry.py's docstring insists on, and for the same
    reason: a table cannot notice a robot whose joint or foot count differs
    from it. The per-component identities below are what tie the catalog back
    to the robot: one entry per actuated joint, one per foot.
    """
    spec = robot_spec(_cache, case.robot_dir)
    env = built_env(_cache, case)

    catalog = env._obs_catalog(env._make_data(), env._mirror_probe_info())
    sizes = {name: int(np.prod(value.shape)) for name, value in catalog.items()}

    state = jax.jit(env.reset)(jax.random.PRNGKey(0))
    actor_names = env.actor_obs_names
    priv_names = list(env._config.obs.privileged)
    assert state.obs["state"].shape == (sum(sizes[n] for n in actor_names),)
    assert state.obs["privileged_state"].shape == (sum(sizes[n] for n in priv_names),)

    n_joints = len(spec.actuated_joints)
    assert env.action_size == n_joints
    for name in ("joint_pos", "joint_vel", "last_action", "actuator_force"):
        assert sizes[name] == n_joints, (case.id, name)
    assert sizes["contacts"] == len(spec.foot_sites), case.id


# -- per-robot measured expectations -----------------------------------------


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_total_mass_matches_the_robots_recorded_measurement(_cache, case):
    """`robots/<name>/conformance.yaml`'s `total_mass_kg` band, if it has one.

    The one genuinely per-robot NUMBER left from the robot-named test files
    this suite replaces, and it is data rather than code. It guards the
    vendored source: a meshdir or inertial regression that silently zeroes or
    doubles a link's mass shows up nowhere else in this suite, because every
    other check is derived from robot.yaml and would move with it.
    """
    expected = expectations(_cache, case.robot_dir).get("total_mass_kg")
    if expected is None:
        pytest.skip(f"{case.robot_dir.name} records no total_mass_kg expectation")

    model = built_model(_cache, case)
    total = float(model.body_mass.sum())
    lo, hi = float(expected["min"]), float(expected["max"])
    assert lo < total < hi, f"{case.id}: total mass {total:.4f} kg is outside {lo}-{hi} kg"
