"""Pure library functions that assemble a compiled model from robot data.

build_spec loads a robot's robot.yaml and a named actuator preset, applies
robot.yaml's optional model_patches section (compiler <option> overrides,
then an unconditional strip of every actuator and dangling actuator sensor
already present in the source XML, then injected sites, injected collision
geoms, then mesh-collision stripping), injects actuators for every actuated
joint (in the RobotSpec's canonical order), overrides armature/frictionloss
per the preset, applies passive-joint spring/damper params, and bakes the
robot's keyframes into the spec. The actuator/actuator-sensor strip runs for
every robot whether or not model_patches is present: the preset is always
the source of truth for actuator params, and injection always names an
actuator after its joint, so a source XML that ships its own <actuator>
block would otherwise collide with injection. compile_spec just compiles.
No CLI here; run.sh's `build` verb (step 4) calls these.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from humanoid_lab.actuators.models import ACTUATOR_MODELS
from humanoid_lab.robot.presets import load_actuator_preset, resolve
from humanoid_lab.robot.spec import (
    ModelPatches,
    RobotSpec,
    load_robot_spec,
    validate_against_model,
)

_SOLVER_BY_NAME = {
    "pgs": mujoco.mjtSolver.mjSOL_PGS,
    "cg": mujoco.mjtSolver.mjSOL_CG,
    "newton": mujoco.mjtSolver.mjSOL_NEWTON,
}

_GEOM_TYPE_BY_NAME = {
    "box": mujoco.mjtGeom.mjGEOM_BOX,
    "capsule": mujoco.mjtGeom.mjGEOM_CAPSULE,
    "sphere": mujoco.mjtGeom.mjGEOM_SPHERE,
}

# The three sensor types that reference an actuator by name (sensor.objname).
_ACTUATOR_SENSOR_TYPES = (
    mujoco.mjtSensor.mjSENS_ACTUATORPOS,
    mujoco.mjtSensor.mjSENS_ACTUATORVEL,
    mujoco.mjtSensor.mjSENS_ACTUATORFRC,
)


def build_spec(robot_dir: Path, preset_name: str) -> mujoco.MjSpec:
    """Assemble the mujoco.MjSpec for `robot_dir` under actuator preset `preset_name`.

    Applies robot.yaml's model_patches (if any) and strips every source-XML
    actuator and dangling actuator sensor before injecting actuators. See
    this module's docstring for the full sequence.
    """
    robot_dir = Path(robot_dir)
    robot_spec = load_robot_spec(robot_dir)
    preset = load_actuator_preset(robot_dir, preset_name)
    params_by_joint = resolve(preset, robot_spec)
    actuator_model = ACTUATOR_MODELS[preset.model]

    spec = mujoco.MjSpec.from_file(str(robot_spec.model_xml_path))

    # model_patches order: options, actuator strip, sites, geoms, mesh_collisions.
    # Sites/geoms must exist before validate_against_model runs below (it
    # validates foot_sites/foot_geoms).
    _apply_options_patch(spec, robot_spec.model_patches)
    _strip_source_actuators(spec, robot_spec)
    _apply_sites_patch(spec, robot_spec)
    _apply_geoms_patch(spec, robot_spec)
    _apply_mesh_collisions_patch(spec, robot_spec.model_patches)

    for joint_name in robot_spec.actuated_joints:  # canonical order: the action/obs contract
        params = params_by_joint[joint_name]
        actuator_model.inject(spec, joint_name, params, soft_limit_factor=preset.soft_limit_factor)

        # Injection overrides the XML's per-joint armature (and frictionloss, if
        # set in the preset): the base MJCF's value is a placeholder, the motor
        # choice captured in the actuator preset is the source of truth.
        joint = spec.joint(joint_name)
        if params.armature is not None:
            joint.armature = params.armature
        if params.frictionloss is not None:
            joint.frictionloss = params.frictionloss

    for joint_name, passive in robot_spec.passive_joints.items():
        joint = spec.joint(joint_name)
        if joint is None:
            raise ValueError(
                f"passive joint '{joint_name}' from robot.yaml not found in '{robot_spec.model_xml}'"
            )
        # MjsJoint.stiffness/damping are length-3 buffers (shared with ball-joint
        # storage); only index 0 is used for a hinge/slide joint, and it must be
        # mutated in place rather than reassigned as a bare scalar.
        joint.stiffness[0] = passive.stiffness
        joint.damping[0] = passive.damping

    # Full spec-vs-model validation (existence of every referenced name plus
    # the actuated/passive disjointness check) so a bad robot.yaml fails here
    # with a named error instead of building a silently wrong model.
    validate_against_model(robot_spec, spec.compile())

    if robot_spec.keyframes:
        _add_keyframes(spec, robot_spec)

    return spec


def _apply_options_patch(spec: mujoco.MjSpec, patches: ModelPatches) -> None:
    """Override <option> solver/iterations/timestep per model_patches.options.

    A field left unset (None) leaves the source XML's own value untouched.
    """
    options = patches.options
    if options.solver is not None:
        spec.option.solver = _SOLVER_BY_NAME[options.solver]
    if options.iterations is not None:
        spec.option.iterations = options.iterations
    if options.timestep is not None:
        spec.option.timestep = options.timestep


def _strip_source_actuators(spec: mujoco.MjSpec, robot_spec: RobotSpec) -> None:
    """Delete every actuator, and every dangling actuator sensor, already in `spec`.

    Runs unconditionally for every robot. The actuator preset is the source
    of truth for actuator params, and the injection loop below names each
    actuator after its joint, so a source-XML actuator on the same joint
    would collide with it. A sensor of type actuatorpos/actuatorvel/
    actuatorfrc whose objname is in actuated_joints keeps resolving once
    injection recreates that same-named actuator; any other actuator sensor
    would dangle and fail spec.compile() with "unrecognized name ... of
    sensorized object", so it is deleted too. For a source XML with no
    actuators and no actuator sensors, both loops are no-ops.
    """
    for actuator in list(spec.actuators):
        spec.delete(actuator)

    actuated = set(robot_spec.actuated_joints)
    for sensor in list(spec.sensors):
        if sensor.type in _ACTUATOR_SENSOR_TYPES and sensor.objname not in actuated:
            spec.delete(sensor)


def _apply_sites_patch(spec: mujoco.MjSpec, robot_spec: RobotSpec) -> None:
    """Inject model_patches.sites into their named bodies."""
    for name, site in robot_spec.model_patches.sites.items():
        body = spec.body(site.body)
        if body is None:
            raise ValueError(
                f"model_patches.sites['{name}'] references body '{site.body}', which is "
                f"not in '{robot_spec.model_xml}'"
            )
        body.add_site(name=name, pos=site.pos, quat=site.quat)


def _apply_geoms_patch(spec: mujoco.MjSpec, robot_spec: RobotSpec) -> None:
    """Inject model_patches.geoms collision primitives into their named bodies.

    No explicit contype/conaffinity is set on the injected geom: it inherits
    whichever default class applies to its body in the source XML. Group is
    forced to 3, the collision-geom convention renderers hide by default;
    inheriting a visible group draws the primitives over the visual meshes
    (roboto_origin rendered as its capsules until this was set).
    """
    for name, geom in robot_spec.model_patches.geoms.items():
        body = spec.body(geom.body)
        if body is None:
            raise ValueError(
                f"model_patches.geoms['{name}'] references body '{geom.body}', which is "
                f"not in '{robot_spec.model_xml}'"
            )
        kwargs = dict(
            name=name,
            type=_GEOM_TYPE_BY_NAME[geom.type],
            size=list(geom.size),
            quat=geom.quat,
            group=3,
        )
        if geom.pos is not None:
            kwargs["pos"] = geom.pos
        if geom.fromto is not None:
            kwargs["fromto"] = geom.fromto
        body.add_geom(**kwargs)


def _apply_mesh_collisions_patch(spec: mujoco.MjSpec, patches: ModelPatches) -> None:
    """Zero contype/conaffinity on every mesh geom, if mesh_collisions is "visual".

    For a source XML whose only collision geometry is its full visual
    meshes, this turns them collision-inert so model_patches.geoms's named
    primitives become the only collision surface.
    """
    if patches.mesh_collisions != "visual":
        return
    for geom in spec.geoms:
        if geom.type == mujoco.mjtGeom.mjGEOM_MESH:
            geom.contype = 0
            geom.conaffinity = 0


def compile_spec(spec: mujoco.MjSpec) -> mujoco.MjModel:
    """Compile an assembled spec into an MjModel."""
    return spec.compile()


def _add_keyframes(spec: mujoco.MjSpec, robot_spec: RobotSpec) -> None:
    # Compile once, off to the side, purely to resolve joint -> qpos addresses.
    # `spec` (actuators injected, armature/frictionloss/passive params already
    # applied) is what the caller actually recompiles once the keys below are
    # attached to it.
    addr_model = spec.compile()
    free_addr = _free_joint_qpos_addr(addr_model)

    for kf_name, kf in robot_spec.keyframes.items():
        qpos = np.zeros(addr_model.nq)
        qpos[free_addr : free_addr + 3] = kf.base_pos
        qpos[free_addr + 3 : free_addr + 7] = kf.base_quat
        # Every other joint (actuated or passive) defaults to 0.0 unless the
        # keyframe names it explicitly.
        for joint_name, angle in kf.joints.items():
            try:
                qpos_addr = addr_model.joint(joint_name).qposadr[0]
            except KeyError as e:
                raise ValueError(
                    f"keyframe '{kf_name}' references unknown joint '{joint_name}'"
                ) from e
            qpos[qpos_addr] = angle
        spec.add_key(name=kf_name, qpos=qpos)


def _free_joint_qpos_addr(model: mujoco.MjModel) -> int:
    for i in range(model.njnt):
        if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE:
            return int(model.jnt_qposadr[i])
    raise ValueError("model has no free joint; cannot place a keyframe base pose")
