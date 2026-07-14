"""Pure library functions that assemble a compiled model from robot data.

build_spec loads a robot's robot.yaml and a named actuator preset, injects
actuators for every actuated joint (in the RobotSpec's canonical order),
overrides armature/frictionloss per the preset, applies passive-joint
spring/damper params, and bakes the robot's keyframes into the spec.
compile_spec just compiles. No CLI here; run.sh's `build` verb (step 4) calls
these.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from humanoid_lab.actuators.models import ACTUATOR_MODELS
from humanoid_lab.robot.presets import load_actuator_preset, resolve
from humanoid_lab.robot.spec import RobotSpec, load_robot_spec, validate_against_model


def build_spec(robot_dir: Path, preset_name: str) -> mujoco.MjSpec:
    """Assemble the mujoco.MjSpec for `robot_dir` under actuator preset `preset_name`."""
    robot_dir = Path(robot_dir)
    robot_spec = load_robot_spec(robot_dir)
    preset = load_actuator_preset(robot_dir, preset_name)
    params_by_joint = resolve(preset, robot_spec)
    actuator_model = ACTUATOR_MODELS[preset.model]

    spec = mujoco.MjSpec.from_file(str(robot_spec.model_xml_path))

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
