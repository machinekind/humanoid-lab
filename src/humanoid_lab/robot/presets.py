"""Actuator presets: a Hydra axis separate from the robot.

A preset (`<robot_dir>/actuators/<name>.yaml`) picks an actuator model and
carries per-joint-group parameters (kp/kd, effort/velocity limits, armature,
friction). resolve() maps those group params onto every actuated joint via
the robot's joint_groups; action_scale() derives the per-joint action scale
for the preset's model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from humanoid_lab.actuators.models import ACTUATOR_MODELS, JointActuatorParams
from humanoid_lab.robot.spec import RobotSpec

_OPTIONAL_PARAM_KEYS = ("kp", "kd", "velocity_limit", "armature", "frictionloss")


@dataclass(frozen=True)
class ActuatorPreset:
    """Parsed contents of an actuators/<name>.yaml preset file."""

    model: str
    soft_limit_factor: float
    action_scale_factor: float
    groups: dict[str, dict[str, float]]


def load_actuator_preset(robot_dir: Path, name: str) -> ActuatorPreset:
    """Load `<robot_dir>/actuators/<name>.yaml` into an ActuatorPreset."""
    robot_dir = Path(robot_dir)
    yaml_path = robot_dir / "actuators" / f"{name}.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"actuator preset not found at {yaml_path}")

    with yaml_path.open() as f:
        raw = yaml.safe_load(f) or {}

    missing_keys = [k for k in ("model", "groups") if k not in raw]
    if missing_keys:
        raise ValueError(f"{yaml_path}: missing required key(s): {missing_keys}")

    model = raw["model"]
    if model not in ACTUATOR_MODELS:
        raise ValueError(
            f"{yaml_path}: unknown actuator model '{model}'; known models: "
            f"{sorted(ACTUATOR_MODELS)}"
        )

    return ActuatorPreset(
        model=model,
        soft_limit_factor=float(raw.get("soft_limit_factor", 0.9)),
        action_scale_factor=float(raw.get("action_scale_factor", 0.3)),
        groups={group: dict(params) for group, params in raw["groups"].items()},
    )


def resolve(preset: ActuatorPreset, robot_spec: RobotSpec) -> dict[str, JointActuatorParams]:
    """Map preset group params onto each of robot_spec's actuated joints.

    Errors if the preset names a group the robot doesn't have, or if an
    actuated joint's group has no params in the preset.
    """
    for group_name in preset.groups:
        if group_name not in robot_spec.joint_groups:
            raise ValueError(
                f"actuator preset references unknown joint group '{group_name}'; "
                f"robot defines groups: {sorted(robot_spec.joint_groups)}"
            )

    result: dict[str, JointActuatorParams] = {}
    for joint_name in robot_spec.actuated_joints:
        group_name = robot_spec.group_of(joint_name)
        if group_name not in preset.groups:
            raise ValueError(
                f"actuated joint '{joint_name}' belongs to group '{group_name}', which has "
                "no params in this actuator preset"
            )
        group_params = preset.groups[group_name]
        if "effort_limit" not in group_params:
            raise ValueError(f"actuator preset group '{group_name}' is missing 'effort_limit'")

        result[joint_name] = JointActuatorParams(
            effort_limit=float(group_params["effort_limit"]),
            **{
                key: float(group_params[key])
                for key in _OPTIONAL_PARAM_KEYS
                if key in group_params
            },
        )
    return result


def action_scale(preset: ActuatorPreset, robot_spec: RobotSpec) -> dict[str, float]:
    """The per-joint action_scale for this preset's model.

    pd: action_scale_factor * effort_limit / kp. ideal_torque: effort_limit
    (action_scale_factor does not apply). See ActuatorModel.action_scale.
    """
    model = ACTUATOR_MODELS[preset.model]
    params_by_joint = resolve(preset, robot_spec)
    return {
        joint_name: model.action_scale(params, preset.action_scale_factor)
        for joint_name, params in params_by_joint.items()
    }
