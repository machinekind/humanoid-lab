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
    ModelPatchGeom,
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

# Taken off each half-size of every split-box cell, in metres, so each
# cell face sits this far inside its grid cell. Diagonal neighbours in a
# chessboard share an edge or a corner. Two touching cells would both
# report a contact at the shared point. Shrunk, any two filled cells are at
# least 2x this apart along some axis.
SPLIT_CELL_SHRINK = 0.0005

# The three sensor types that reference an actuator by name (sensor.objname).
_ACTUATOR_SENSOR_TYPES = (
    mujoco.mjtSensor.mjSENS_ACTUATORPOS,
    mujoco.mjtSensor.mjSENS_ACTUATORVEL,
    mujoco.mjtSensor.mjSENS_ACTUATORFRC,
)


def build_spec(
    robot_dir: Path, preset_name: str, actuator_overrides: dict | None = None
) -> mujoco.MjSpec:
    """Assemble the mujoco.MjSpec for `robot_dir` under actuator preset `preset_name`.

    Applies robot.yaml's model_patches (if any) and strips every source-XML
    actuator and dangling actuator sensor before injecting actuators. See
    this module's docstring for the full sequence. `actuator_overrides` is
    forwarded to load_actuator_preset (see robot/presets.py) so a CLI/
    experiment override lands in the built spec too.
    """
    robot_dir = Path(robot_dir)
    robot_spec = load_robot_spec(robot_dir)
    preset = load_actuator_preset(robot_dir, preset_name, actuator_overrides)
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

    A box with `split` is injected as its chessboard cells (split_box_cells)
    instead of whole. Each cell goes through the same add_geom call the
    whole box would, differing only in name, pos and size, so it inherits
    exactly the contact attributes the box would have.
    """
    for name, geom in robot_spec.model_patches.geoms.items():
        body = spec.body(geom.body)
        if body is None:
            raise ValueError(
                f"model_patches.geoms['{name}'] references body '{geom.body}', which is "
                f"not in '{robot_spec.model_xml}'"
            )
        if geom.split is None:
            parts = [(name, geom.pos, geom.size)]
        else:
            _require_geom_free_inertia(spec, body, name)
            parts = split_box_cells(name, geom)
        for part_name, pos, size in parts:
            kwargs = dict(
                name=part_name,
                type=_GEOM_TYPE_BY_NAME[geom.type],
                size=list(size),
                quat=geom.quat,
                group=3,
            )
            if pos is not None:
                kwargs["pos"] = list(pos)
            if geom.fromto is not None:
                kwargs["fromto"] = geom.fromto
            body.add_geom(**kwargs)


def split_box_cells(
    name: str, geom: ModelPatchGeom
) -> list[tuple[str, tuple[float, float, float], tuple[float, float, float]]]:
    """(name, body-frame centre, half-sizes) of each filled cell of a split box.

    The box is cut into an nx x ny x nz grid in its own frame and cell
    (i, j, k) is filled when i + j + k is even, so no two filled cells share
    a face. A cell's half-sizes are its grid cell's less SPLIT_CELL_SHRINK,
    so each face moves in by SPLIT_CELL_SHRINK. It keeps the box's quat and
    is named f"{name}_{i}{j}{k}". The outermost cells reach every face of the
    box to within SPLIT_CELL_SHRINK, and every grid layer along every axis
    holds a cell. A split that breaks either raises.
    """
    nx, ny, nz = geom.split
    hx, hy, hz = geom.size
    half = (hx / nx - SPLIT_CELL_SHRINK, hy / ny - SPLIT_CELL_SHRINK, hz / nz - SPLIT_CELL_SHRINK)
    if min(half) <= 0.0:
        raise ValueError(
            f"model_patches.geoms['{name}'] split {list(geom.split)} leaves a cell of "
            f"half-size {half} once shrunk by {SPLIT_CELL_SHRINK}; use fewer cells"
        )
    filled = [
        (i, j, k)
        for i in range(nx)
        for j in range(ny)
        for k in range(nz)
        if (i + j + k) % 2 == 0
    ]
    _require_full_extent(name, geom, filled, half)

    quat = np.asarray(geom.quat, dtype=np.float64)
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, quat / np.linalg.norm(quat))
    rot = rot.reshape(3, 3)
    centre = np.zeros(3) if geom.pos is None else np.asarray(geom.pos, dtype=np.float64)

    cells = []
    for i, j, k in filled:
        offset = np.array(
            [
                -hx + (2 * i + 1) * hx / nx,
                -hy + (2 * j + 1) * hy / ny,
                -hz + (2 * k + 1) * hz / nz,
            ]
        )
        pos = tuple(float(x) for x in centre + rot @ offset)
        cells.append((f"{name}_{i}{j}{k}", pos, half))
    return cells


def _require_full_extent(
    name: str,
    geom: ModelPatchGeom,
    filled: list[tuple[int, int, int]],
    half: tuple[float, float, float],
) -> None:
    """Refuse a split whose filled cells fall short of the box.

    Each of the six faces needs a cell within SPLIT_CELL_SHRINK of it, and
    each grid layer along each axis needs a cell, or a gap runs through the
    box's whole cross-section. A grid cut along one axis only keeps just its
    even layers: an even count misses a face, an odd count leaves gaps.
    """
    where = f"model_patches.geoms['{name}'] split {list(geom.split)}"
    for axis, (h, n, cell_half) in enumerate(zip(geom.size, geom.split, half)):
        centres = [-h + (2 * cell[axis] + 1) * h / n for cell in filled]
        reach = h - SPLIT_CELL_SHRINK - 1e-12
        if max(centres) + cell_half < reach or min(centres) - cell_half > -reach:
            raise ValueError(
                f"{where} leaves a {'xyz'[axis]} face of the box with no cell on it; "
                "split at least two axes"
            )
        empty = sorted(set(range(n)) - {cell[axis] for cell in filled})
        if empty:
            raise ValueError(
                f"{where} leaves {'xyz'[axis]} layer(s) {empty} with no cell, a gap through "
                "the whole box; split at least two axes"
            )


def _require_geom_free_inertia(spec: mujoco.MjSpec, body: mujoco.MjsBody, name: str) -> None:
    """Refuse a split on a body whose mass or inertia the compiler takes from its geoms.

    The split is a collision detail. On a body without an explicit
    <inertial>, or under compiler inertiafromgeom="true", the cells would
    carry half the box's volume in different places and quietly change the
    body's mass and inertia.
    """
    if not body.explicitinertial:
        raise ValueError(
            f"model_patches.geoms['{name}'] splits a box on body '{body.name}', which has no "
            "explicit <inertial>; its mass would come from the cells"
        )
    if spec.compiler.inertiafromgeom == mujoco.mjtInertiaFromGeom.mjINERTIAFROMGEOM_TRUE:
        raise ValueError(
            f"model_patches.geoms['{name}'] splits a box, but compiler inertiafromgeom='true' "
            "takes every body's inertia from its geoms"
        )


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
