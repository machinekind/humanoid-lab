"""Actuator presets: a Hydra axis separate from the robot.

A preset (`<robot_dir>/actuators/<name>.yaml`) picks an actuator model and
carries per-joint-group parameters (kp/kd, effort/velocity limits, armature,
friction). resolve() maps those group params onto every actuated joint via
the robot's joint_groups; action_scale() derives the per-joint action scale
for the preset's model.

load_actuator_preset's `overrides` (cfg.actuators.overrides, or a CLI --set
parsed by parse_set_overrides) is deep-merged onto the loaded yaml dict
before validation: a top-level key (model/soft_limit_factor/
action_scale_factor) replaces that preset field, `groups.<group>.<param>`
patches one per-group value in place. This is the ONE choke point overrides
flow through, so every consumer (train, eval, sizing, build/check CLIs)
resolves identical values.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from humanoid_lab.actuators.models import ACTUATOR_MODELS, JointActuatorParams
from humanoid_lab.robot.spec import RobotSpec

_OPTIONAL_PARAM_KEYS = ("kp", "kd", "velocity_limit", "armature", "frictionloss")

# Schema for the raw preset dict, checked AFTER overrides are merged in: a
# typo'd override key (e.g. kp_) must fail loudly here rather than vanish
# silently inside resolve()'s explicit key extraction below.
_TOP_LEVEL_KEYS = frozenset({"model", "soft_limit_factor", "action_scale_factor", "groups"})
_GROUP_PARAM_KEYS = frozenset({"effort_limit", *_OPTIONAL_PARAM_KEYS})


def _deep_merge(base: dict, patch: dict) -> dict:
    """Recursively merge `patch` onto `base`; patch wins, non-dict values replace.

    Neither argument is mutated. A key present in both as a dict merges
    recursively (so a single-group override leaves every other group
    untouched); any other patch value (scalar, list, or a dict overwriting a
    non-dict) replaces the base value outright.
    """
    result = dict(base)
    for key, value in patch.items():
        base_value = result.get(key)
        if isinstance(base_value, dict) and isinstance(value, dict):
            result[key] = _deep_merge(base_value, value)
        else:
            result[key] = value
    return result


def _validate_preset_keys(data: dict, label: str) -> None:
    """Reject any top-level or per-group key outside the known schema."""
    unknown_top = set(data) - _TOP_LEVEL_KEYS
    if unknown_top:
        raise ValueError(
            f"{label}: unknown key(s) {sorted(unknown_top)}; known top-level keys: "
            f"{sorted(_TOP_LEVEL_KEYS)}"
        )
    for group_name, group_params in (data.get("groups") or {}).items():
        unknown = set(group_params) - _GROUP_PARAM_KEYS
        if unknown:
            raise ValueError(
                f"{label}: group '{group_name}' has unknown key(s) {sorted(unknown)}; "
                f"known group keys: {sorted(_GROUP_PARAM_KEYS)}"
            )


@dataclass(frozen=True)
class ActuatorPreset:
    """Parsed contents of an actuators/<name>.yaml preset file."""

    model: str
    soft_limit_factor: float
    action_scale_factor: float
    groups: dict[str, dict[str, float]]


def load_actuator_preset(
    robot_dir: Path, name: str, overrides: dict | None = None
) -> ActuatorPreset:
    """Load `<robot_dir>/actuators/<name>.yaml` into an ActuatorPreset.

    `overrides`, if given, is deep-merged onto the yaml dict before
    validation (see module docstring). None/{} behaves exactly as the
    override-free load.
    """
    robot_dir = Path(robot_dir)
    yaml_path = robot_dir / "actuators" / f"{name}.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"actuator preset not found at {yaml_path}")

    with yaml_path.open() as f:
        raw = yaml.safe_load(f) or {}

    if overrides:
        raw = _deep_merge(raw, overrides)
    _validate_preset_keys(raw, str(yaml_path))

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


def effective_gains(
    actuator_gainprm, actuator_biasprm, joint_names, *, model: str, preset: str
) -> dict:
    """The gain block run.json stamps, read back off a BUILT model.

    Not the preset yaml's numbers. Gains reach the physics through
    load_actuator_preset's `overrides` deep-merge and then through
    actuators/models.py's inject, and an `actuators.overrides` entry never
    appears in the yaml at all -- so a stamp taken from the yaml can record
    numbers the run never used. This reads what mujoco ended up holding.

    For the `pd` model those params ARE the PD gains: PositionPD.inject
    writes gainprm=(kp,0,0) and biasprm=(0,-kp,-kd), so kp is
    `gainprm[:, 0]` and kd is `-biasprm[:, 2]`. For `ideal_torque` they are
    not gains at all (gainprm[0] is 1.0, there is no bias term) and the
    numbers come out as 1.0 and 0.0. They are stamped anyway: `model` is what
    makes them readable, and a run.json whose shape depended on the actuator
    model would need branching at every reader.

    `joint_names` is the canonical actuated-joint order, which is also the
    actuator column order (robot/build.py injects in that order).
    """
    kp = [float(row[0]) for row in actuator_gainprm]
    kd = [-float(row[2]) for row in actuator_biasprm]
    names = list(joint_names)
    if len(kp) != len(names) or len(kd) != len(names):
        raise ValueError(
            f"gain table has {len(kp)} actuators and bias table {len(kd)}, but the "
            f"robot declares {len(names)} actuated joints {names} -- the stamp would "
            "mislabel every column"
        )
    return {"preset": preset, "model": model, "joints": names, "kp": kp, "kd": kd}


def parse_set_overrides(items: list[str]) -> dict:
    """Parse CLI `--set PATH=VALUE` items into a load_actuator_preset overrides dict.

    PATH is one segment (a top-level key, e.g. `soft_limit_factor=0.8` or
    `model=ideal_torque`) or two (`<group>.<param>=<value>`, e.g.
    `knee.kp=80`, which becomes `{"groups": {"knee": {"kp": 80.0}}}`).
    VALUE is parsed as a float when possible, else kept as a string. More
    than two path segments is an error -- the preset schema has no deeper
    nesting.
    """
    result: dict = {}
    for item in items:
        path, sep, value_str = item.partition("=")
        if not sep:
            raise ValueError(f"--set item {item!r} is not PATH=VALUE")
        segments = path.split(".")
        if len(segments) > 2:
            raise ValueError(
                f"--set path {path!r} has more than two segments; use PATH=VALUE or "
                "GROUP.PARAM=VALUE"
            )
        try:
            value: float | str = float(value_str)
        except ValueError:
            value = value_str

        if len(segments) == 1:
            result[segments[0]] = value
        else:
            group, param = segments
            result.setdefault("groups", {}).setdefault(group, {})[param] = value
    return result
