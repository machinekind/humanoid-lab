"""RobotSpec: the per-robot contract every env, actuator preset, and export
step reads instead of naming joints/sites/bodies directly.

A robot directory (e.g. robots/asimov_v1/) carries a single robot.yaml that
this module parses into a RobotSpec. Everything downstream (actuator
injection, observations, terminations, symmetry augmentation, export) reads
the RobotSpec; nothing downstream should hardcode a joint name.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mujoco
import yaml

_IDENTITY_QUAT: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

_REQUIRED_ROBOT_YAML_KEYS = (
    "name",
    "model_xml",
    "actuated_joints",
    "joint_groups",
    "passive_joints",
    "foot_sites",
    "foot_geoms",
)


@dataclass(frozen=True)
class PassiveJointParams:
    """Spring/damper parameters applied to a passive joint at build time."""

    stiffness: float
    damping: float


@dataclass(frozen=True)
class Keyframe:
    """A named pose: floating-base pos/quat plus a sparse joint-angle map.

    Actuated joints not listed in `joints` default to 0.0.
    """

    base_pos: tuple[float, float, float]
    base_quat: tuple[float, float, float, float] = _IDENTITY_QUAT
    joints: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class RobotSpec:
    """Parsed, schema-validated contents of a robot's robot.yaml."""

    name: str
    robot_dir: Path
    model_xml: str
    actuated_joints: list[str]
    joint_groups: dict[str, list[str]]
    passive_joints: dict[str, PassiveJointParams]
    foot_sites: list[str]
    foot_geoms: list[str]
    symmetry: dict[str, str] = field(default_factory=dict)
    keyframes: dict[str, Keyframe] = field(default_factory=dict)
    termination_bodies: list[str] = field(default_factory=list)
    obs_layout: dict[str, Any] = field(default_factory=dict)

    @property
    def model_xml_path(self) -> Path:
        """Absolute path to the model XML, resolved against robot_dir."""
        return self.robot_dir / self.model_xml

    def group_of(self, joint_name: str) -> str:
        """The joint_groups key that `joint_name` belongs to.

        load_robot_spec already validated that every actuated joint belongs to
        exactly one group, so this should never raise for an actuated joint.
        """
        for group_name, joints in self.joint_groups.items():
            if joint_name in joints:
                return group_name
        raise KeyError(f"joint '{joint_name}' is not assigned to any group in joint_groups")


def load_robot_spec(robot_dir: Path) -> RobotSpec:
    """Load and validate `<robot_dir>/robot.yaml` into a RobotSpec.

    Validation performed here is schema-level only (no mujoco model involved):
    every actuated joint must belong to exactly one joint_groups entry, and
    every joint named in joint_groups must be an actuated joint. Cross-checks
    against a compiled model (do the named joints/sites/geoms/bodies actually
    exist, do actuated and passive joints overlap) are validate_against_model's
    job, once a model is available.
    """
    robot_dir = Path(robot_dir)
    yaml_path = robot_dir / "robot.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"robot.yaml not found at {yaml_path}")

    with yaml_path.open() as f:
        raw = yaml.safe_load(f) or {}

    missing_keys = [k for k in _REQUIRED_ROBOT_YAML_KEYS if k not in raw]
    if missing_keys:
        raise ValueError(f"{yaml_path}: missing required key(s): {missing_keys}")

    actuated_joints = list(raw["actuated_joints"])
    joint_groups = {group: list(joints) for group, joints in raw["joint_groups"].items()}
    passive_joints = {
        joint_name: PassiveJointParams(
            stiffness=float(params["stiffness"]), damping=float(params["damping"])
        )
        for joint_name, params in raw["passive_joints"].items()
    }
    keyframes = _parse_keyframes(raw.get("keyframes") or {}, yaml_path)

    _validate_joint_groups(joint_groups, actuated_joints, yaml_path)

    return RobotSpec(
        name=raw["name"],
        robot_dir=robot_dir,
        model_xml=raw["model_xml"],
        actuated_joints=actuated_joints,
        joint_groups=joint_groups,
        passive_joints=passive_joints,
        foot_sites=list(raw["foot_sites"]),
        foot_geoms=list(raw["foot_geoms"]),
        symmetry=dict(raw.get("symmetry") or {}),
        keyframes=keyframes,
        termination_bodies=list(raw.get("termination_bodies") or []),
        obs_layout=dict(raw.get("obs_layout") or {}),
    )


def _parse_keyframes(raw_keyframes: dict[str, Any], yaml_path: Path) -> dict[str, Keyframe]:
    keyframes: dict[str, Keyframe] = {}
    for kf_name, kf_raw in raw_keyframes.items():
        if "base_pos" not in kf_raw:
            raise ValueError(f"{yaml_path}: keyframe '{kf_name}' is missing required 'base_pos'")
        base_pos = tuple(float(x) for x in kf_raw["base_pos"])
        if len(base_pos) != 3:
            raise ValueError(f"{yaml_path}: keyframe '{kf_name}' base_pos must have 3 elements")
        base_quat = tuple(float(x) for x in kf_raw.get("base_quat", _IDENTITY_QUAT))
        if len(base_quat) != 4:
            raise ValueError(f"{yaml_path}: keyframe '{kf_name}' base_quat must have 4 elements")
        joints = {jn: float(v) for jn, v in (kf_raw.get("joints") or {}).items()}
        keyframes[kf_name] = Keyframe(base_pos=base_pos, base_quat=base_quat, joints=joints)
    return keyframes


def _validate_joint_groups(
    joint_groups: dict[str, list[str]], actuated_joints: list[str], yaml_path: Path
) -> None:
    actuated_set = set(actuated_joints)
    membership_count: dict[str, int] = {}
    for group_name, joints in joint_groups.items():
        for joint_name in joints:
            if joint_name not in actuated_set:
                raise ValueError(
                    f"{yaml_path}: joint_groups['{group_name}'] references '{joint_name}', "
                    "which is not listed in actuated_joints"
                )
            membership_count[joint_name] = membership_count.get(joint_name, 0) + 1

    ungrouped = [j for j in actuated_joints if membership_count.get(j, 0) == 0]
    if ungrouped:
        raise ValueError(
            f"{yaml_path}: actuated joint(s) not assigned to any joint_groups entry: {ungrouped}"
        )

    overgrouped = [j for j, count in membership_count.items() if count > 1]
    if overgrouped:
        raise ValueError(
            f"{yaml_path}: actuated joint(s) assigned to more than one joint_groups entry: "
            f"{overgrouped}"
        )


def validate_against_model(spec: RobotSpec, model: mujoco.MjModel) -> None:
    """Check every joint/site/geom/body the spec names actually exists in `model`.

    Also checks that no joint is listed as both actuated and passive. Raises
    ValueError naming the first offending item found.
    """
    overlap = sorted(set(spec.actuated_joints) & set(spec.passive_joints))
    if overlap:
        raise ValueError(f"joint(s) listed as both actuated and passive: {overlap}")

    _check_names_exist(spec.actuated_joints, "joint", model.joint)
    _check_names_exist(spec.passive_joints.keys(), "joint", model.joint)
    _check_names_exist(spec.foot_sites, "site", model.site)
    _check_names_exist(spec.foot_geoms, "geom", model.geom)
    _check_names_exist(spec.termination_bodies, "body", model.body)

    for left, right in spec.symmetry.items():
        _check_names_exist([left, right], "joint", model.joint)

    for kf_name, kf in spec.keyframes.items():
        try:
            _check_names_exist(kf.joints.keys(), "joint", model.joint)
        except ValueError as e:
            raise ValueError(f"keyframe '{kf_name}': {e}") from e


def _check_names_exist(names, kind: str, getter) -> None:
    for name in names:
        try:
            getter(name)
        except KeyError as e:
            raise ValueError(
                f"{kind} '{name}' referenced in robot.yaml but not found in the compiled model"
            ) from e
