"""RobotSpec: the per-robot contract every env, actuator preset, and export
step reads instead of naming joints/sites/bodies directly.

A robot directory (e.g. robots/asimov_v1/) carries a single robot.yaml that
this module parses into a RobotSpec. Everything downstream (actuator
injection, observations, terminations, export) reads the RobotSpec;
nothing downstream should hardcode a joint name.
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

_ALLOWED_MODEL_PATCH_KEYS = ("options", "mesh_collisions", "sites", "geoms")
_ALLOWED_MODEL_PATCH_OPTION_KEYS = ("solver", "iterations", "timestep")
_ALLOWED_SOLVERS = ("pgs", "cg", "newton")
_ALLOWED_MESH_COLLISIONS_VALUES = ("visual",)
_ALLOWED_MODEL_PATCH_SITE_KEYS = ("body", "pos", "quat")
_ALLOWED_MODEL_PATCH_GEOM_KEYS = ("body", "type", "size", "pos", "fromto", "quat")
_ALLOWED_GEOM_TYPES = ("box", "capsule", "sphere")


@dataclass(frozen=True)
class PassiveJointParams:
    """Spring/damper parameters applied to a passive joint at build time."""

    stiffness: float
    damping: float


@dataclass(frozen=True)
class ModelPatchOptions:
    """<option> overrides from robot.yaml's model_patches.options.

    Any field left unset (None) leaves the source XML's own <option> value
    untouched.
    """

    solver: str | None = None
    iterations: int | None = None
    timestep: float | None = None


@dataclass(frozen=True)
class ModelPatchSite:
    """A site to inject into a named body, from a model_patches.sites entry."""

    body: str
    pos: tuple[float, float, float]
    quat: tuple[float, float, float, float] = _IDENTITY_QUAT


@dataclass(frozen=True)
class ModelPatchGeom:
    """A collision primitive to inject into a named body, from a
    model_patches.geoms entry. `type` is one of box/capsule/sphere; `size`
    follows mujoco's per-type geom size semantics. `pos` and `fromto` are
    both optional but mutually exclusive, matching MJCF geom semantics.
    """

    body: str
    type: str
    size: tuple[float, ...]
    pos: tuple[float, float, float] | None = None
    fromto: tuple[float, float, float, float, float, float] | None = None
    quat: tuple[float, float, float, float] = _IDENTITY_QUAT


@dataclass(frozen=True)
class ModelPatches:
    """Parsed robot.yaml `model_patches` section. The section and every
    sub-key inside it are optional; an absent robot.yaml key parses to this
    dataclass's all-empty defaults, which build_spec applies as no-ops.
    """

    options: ModelPatchOptions = field(default_factory=ModelPatchOptions)
    mesh_collisions: str | None = None
    sites: dict[str, ModelPatchSite] = field(default_factory=dict)
    geoms: dict[str, ModelPatchGeom] = field(default_factory=dict)


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
    model_patches: ModelPatches = field(default_factory=ModelPatches)
    termination_bodies: list[str] = field(default_factory=list)
    obs_layout: dict[str, Any] = field(default_factory=dict)
    # Optional MJCF sensor names the env reads instead of computing the
    # equivalent signal from qpos/qvel. Recognized keys: gyro (3-vector
    # angular velocity), quat (4-vector base orientation), linvel (3-vector
    # local-frame linear velocity), acc (3-vector local-frame linear
    # acceleration). Any subset may be present; envs fall back to a
    # qpos/qvel-derived computation for keys not listed here.
    sensors: dict[str, str] = field(default_factory=dict)
    # Warp contact/constraint budgets measured for THIS robot's collision
    # geometry (`./run.sh check-contacts`). Recognized keys:
    # naconmax_per_env, njmax. A robot without a measurement omits the
    # block; a warp run then refuses at env construction (envs/base.py)
    # instead of dropping contacts silently.
    sim_budget: dict[str, int] = field(default_factory=dict)

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
    model_patches = _parse_model_patches(raw.get("model_patches") or {}, yaml_path)

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
        model_patches=model_patches,
        termination_bodies=list(raw.get("termination_bodies") or []),
        obs_layout=dict(raw.get("obs_layout") or {}),
        sensors=dict(raw.get("sensors") or {}),
        sim_budget=_parse_sim_budget(raw.get("sim_budget") or {}, yaml_path),
    )


def _parse_sim_budget(raw: dict, yaml_path: Path) -> dict[str, int]:
    known = {"naconmax_per_env", "njmax"}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(
            f"{yaml_path}: sim_budget has unknown key(s) {unknown}; known: {sorted(known)}"
        )
    out = {}
    for key, value in raw.items():
        if int(value) <= 0:
            raise ValueError(f"{yaml_path}: sim_budget.{key} must be positive, got {value}")
        out[key] = int(value)
    return out


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


def _parse_model_patches(raw: dict[str, Any], yaml_path: Path) -> ModelPatches:
    unknown = [k for k in raw if k not in _ALLOWED_MODEL_PATCH_KEYS]
    if unknown:
        raise ValueError(f"{yaml_path}: model_patches has unknown key(s): {unknown}")

    options = _parse_model_patch_options(raw.get("options") or {}, yaml_path)

    mesh_collisions = raw.get("mesh_collisions")
    if mesh_collisions is not None and mesh_collisions not in _ALLOWED_MESH_COLLISIONS_VALUES:
        raise ValueError(
            f"{yaml_path}: model_patches.mesh_collisions must be one of "
            f"{_ALLOWED_MESH_COLLISIONS_VALUES}, got '{mesh_collisions}'"
        )

    sites = {
        name: _parse_model_patch_site(name, raw_site, yaml_path)
        for name, raw_site in (raw.get("sites") or {}).items()
    }
    geoms = {
        name: _parse_model_patch_geom(name, raw_geom, yaml_path)
        for name, raw_geom in (raw.get("geoms") or {}).items()
    }

    return ModelPatches(options=options, mesh_collisions=mesh_collisions, sites=sites, geoms=geoms)


def _parse_model_patch_options(raw: dict[str, Any], yaml_path: Path) -> ModelPatchOptions:
    unknown = [k for k in raw if k not in _ALLOWED_MODEL_PATCH_OPTION_KEYS]
    if unknown:
        raise ValueError(f"{yaml_path}: model_patches.options has unknown key(s): {unknown}")

    solver = raw.get("solver")
    if solver is not None and solver not in _ALLOWED_SOLVERS:
        raise ValueError(
            f"{yaml_path}: model_patches.options.solver must be one of "
            f"{_ALLOWED_SOLVERS}, got '{solver}'"
        )

    return ModelPatchOptions(
        solver=solver,
        iterations=int(raw["iterations"]) if "iterations" in raw else None,
        timestep=float(raw["timestep"]) if "timestep" in raw else None,
    )


def _parse_model_patch_site(name: str, raw: dict[str, Any], yaml_path: Path) -> ModelPatchSite:
    unknown = [k for k in raw if k not in _ALLOWED_MODEL_PATCH_SITE_KEYS]
    if unknown:
        raise ValueError(
            f"{yaml_path}: model_patches.sites['{name}'] has unknown key(s): {unknown}"
        )
    if "body" not in raw:
        raise ValueError(f"{yaml_path}: model_patches.sites['{name}'] is missing required 'body'")
    if "pos" not in raw:
        raise ValueError(f"{yaml_path}: model_patches.sites['{name}'] is missing required 'pos'")

    pos = tuple(float(x) for x in raw["pos"])
    if len(pos) != 3:
        raise ValueError(f"{yaml_path}: model_patches.sites['{name}'] pos must have 3 elements")
    quat = tuple(float(x) for x in raw.get("quat", _IDENTITY_QUAT))
    if len(quat) != 4:
        raise ValueError(f"{yaml_path}: model_patches.sites['{name}'] quat must have 4 elements")

    return ModelPatchSite(body=raw["body"], pos=pos, quat=quat)


def _parse_model_patch_geom(name: str, raw: dict[str, Any], yaml_path: Path) -> ModelPatchGeom:
    unknown = [k for k in raw if k not in _ALLOWED_MODEL_PATCH_GEOM_KEYS]
    if unknown:
        raise ValueError(
            f"{yaml_path}: model_patches.geoms['{name}'] has unknown key(s): {unknown}"
        )
    missing = [k for k in ("body", "type", "size") if k not in raw]
    if missing:
        raise ValueError(
            f"{yaml_path}: model_patches.geoms['{name}'] is missing required key(s): {missing}"
        )

    geom_type = raw["type"]
    if geom_type not in _ALLOWED_GEOM_TYPES:
        raise ValueError(
            f"{yaml_path}: model_patches.geoms['{name}'] type must be one of "
            f"{_ALLOWED_GEOM_TYPES}, got '{geom_type}'"
        )

    # size length is type-dependent (mujoco semantics); the compiler rejects
    # a bad length with a named error, so it is deliberately not checked here.
    size = tuple(float(x) for x in raw["size"])

    pos = tuple(float(x) for x in raw["pos"]) if "pos" in raw else None
    if pos is not None and len(pos) != 3:
        raise ValueError(f"{yaml_path}: model_patches.geoms['{name}'] pos must have 3 elements")

    fromto = tuple(float(x) for x in raw["fromto"]) if "fromto" in raw else None
    if fromto is not None and len(fromto) != 6:
        raise ValueError(f"{yaml_path}: model_patches.geoms['{name}'] fromto must have 6 elements")

    quat = tuple(float(x) for x in raw.get("quat", _IDENTITY_QUAT))
    if len(quat) != 4:
        raise ValueError(f"{yaml_path}: model_patches.geoms['{name}'] quat must have 4 elements")

    return ModelPatchGeom(
        body=raw["body"], type=geom_type, size=size, pos=pos, fromto=fromto, quat=quat
    )


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
    _check_names_exist(spec.sensors.values(), "sensor", model.sensor)

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
