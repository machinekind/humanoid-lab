"""Actuator model strategies, selected by string key via ACTUATOR_MODELS.

Each model knows how to (a) inject a mujoco.MjSpec actuator equivalent to its
physical behavior for one joint, and (b) map a policy action to a ctrl value
and to an action-scale number. Robot/preset code never branches on the model
type directly; it looks the model up in ACTUATOR_MODELS by the string key
carried in the actuator preset yaml.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco


@dataclass(frozen=True)
class JointActuatorParams:
    """Per-joint actuator parameters, resolved from an actuator preset group.

    Only effort_limit is required; the rest are optional because not every
    actuator model uses every field (e.g. ideal-torque has no kp/kd).

    velocity_limit is carried through for later actuator models (DC-motor with
    speed saturation) and for sizing reports, even though a MuJoCo general
    actuator injected via PositionPD/IdealTorque cannot enforce a velocity
    limit in the XML: MuJoCo actuators clamp force/ctrl, not joint speed.
    """

    effort_limit: float
    kp: float | None = None
    kd: float | None = None
    velocity_limit: float | None = None
    armature: float | None = None
    frictionloss: float | None = None


class ActuatorModel:
    """Strategy interface implemented by each actuator model."""

    def inject(
        self,
        spec: mujoco.MjSpec,
        joint_name: str,
        params: JointActuatorParams,
        *,
        soft_limit_factor: float = 0.9,
    ) -> None:
        """Add a general actuator to `spec` driving `joint_name`."""
        raise NotImplementedError

    def ctrl_from_action(self, action, default_pose, action_scale):
        """Map a policy action to a ctrl value. Pure array arithmetic: no
        python branching on the values, so this is safe to call under jax
        jit/vmap."""
        raise NotImplementedError

    def action_scale(self, params: JointActuatorParams, action_scale_factor: float) -> float:
        """The action_scale for one joint under this model."""
        raise NotImplementedError


class PositionPD(ActuatorModel):
    """A MuJoCo position-servo equivalent, built from a general actuator.

    gainprm=(kp, 0, 0), biasprm=(0, -kp, -kd) reproduces mujoco's builtin
    <position> actuator: ctrl is a target joint angle, force = kp*(ctrl - q) -
    kd*qvel. ctrl is deliberately NOT clamped to the joint range
    (ctrllimited=False), matching asimov-mjlab: a setpoint past the kinematic
    limit lets the PD loop keep pulling at a limit instead of fighting the
    policy there, and a range-less (continuous) joint stays injectable.
    soft_limit_factor is accepted for interface uniformity; the soft limits it
    describes belong to the RL layer (obs/reward), not to the actuator.
    """

    def inject(self, spec, joint_name, params, *, soft_limit_factor=0.9):
        del soft_limit_factor
        if params.kp is None or params.kd is None:
            raise ValueError(f"pd actuator model requires kp and kd for joint '{joint_name}'")

        joint = spec.joint(joint_name)
        if joint is None:
            raise ValueError(f"joint '{joint_name}' not found in spec")

        kp, kd, effort = params.kp, params.kd, params.effort_limit

        gainprm = [0.0] * mujoco.mjNGAIN
        gainprm[0] = kp
        biasprm = [0.0] * mujoco.mjNBIAS
        biasprm[1] = -kp
        biasprm[2] = -kd

        spec.add_actuator(
            name=joint_name,
            trntype=mujoco.mjtTrn.mjTRN_JOINT,
            target=joint_name,
            gaintype=mujoco.mjtGain.mjGAIN_FIXED,
            gainprm=gainprm,
            biastype=mujoco.mjtBias.mjBIAS_AFFINE,
            biasprm=biasprm,
            forcerange=[-effort, effort],
            forcelimited=True,
            ctrllimited=False,
        )

    def ctrl_from_action(self, action, default_pose, action_scale):
        return default_pose + action * action_scale

    def action_scale(self, params, action_scale_factor):
        if params.kp is None:
            raise ValueError("pd action_scale requires kp")
        return action_scale_factor * params.effort_limit / params.kp


class IdealTorque(ActuatorModel):
    """A direct, idealized torque actuator: ctrl is commanded torque.

    No position/velocity feedback and no reflected inertia beyond whatever
    armature is set on the joint. Used for motor-sizing runs where the task
    reads off per-joint torque demand before a motor is chosen.
    """

    def inject(self, spec, joint_name, params, *, soft_limit_factor=0.9):
        # soft_limit_factor is a PD/position-actuator concept (it shrinks a
        # ctrlrange expressed in joint-angle units); ideal-torque's ctrlrange
        # is +/- effort_limit regardless, so it is unused here.
        del soft_limit_factor

        joint = spec.joint(joint_name)
        if joint is None:
            raise ValueError(f"joint '{joint_name}' not found in spec")

        effort = params.effort_limit
        gainprm = [0.0] * mujoco.mjNGAIN
        gainprm[0] = 1.0

        spec.add_actuator(
            name=joint_name,
            trntype=mujoco.mjtTrn.mjTRN_JOINT,
            target=joint_name,
            gaintype=mujoco.mjtGain.mjGAIN_FIXED,
            gainprm=gainprm,
            biastype=mujoco.mjtBias.mjBIAS_NONE,
            forcerange=[-effort, effort],
            forcelimited=True,
            ctrlrange=[-effort, effort],
            ctrllimited=True,
        )

    def ctrl_from_action(self, action, default_pose, action_scale):
        # Ideal-torque ctrl is commanded torque directly; there is no default
        # pose to offset from.
        del default_pose
        return action * action_scale

    def action_scale(self, params, action_scale_factor):
        # Unlike pd, ideal-torque's action_scale is just the effort limit
        # itself; action_scale_factor does not apply to this model.
        del action_scale_factor
        return params.effort_limit


class _NotImplementedActuatorModel(ActuatorModel):
    """Placeholder for an actuator model named in PLAN.md but not built yet.

    Registering the key now (rather than leaving it out of ACTUATOR_MODELS)
    lets preset yaml validation give a "not implemented" error instead of an
    "unknown model" error once someone writes `model: dc_motor_speed_saturation`.
    """

    def __init__(self, key: str):
        self._key = key

    def _not_implemented(self):
        raise NotImplementedError(
            f"actuator model '{self._key}' is not implemented yet (see PLAN.md's "
            "src/<pkg>/actuators/ list)"
        )

    def inject(self, *args, **kwargs):
        self._not_implemented()

    def ctrl_from_action(self, *args, **kwargs):
        self._not_implemented()

    def action_scale(self, *args, **kwargs):
        self._not_implemented()


ACTUATOR_MODELS: dict[str, ActuatorModel] = {
    "pd": PositionPD(),
    "ideal_torque": IdealTorque(),
    "dc_motor_speed_saturation": _NotImplementedActuatorModel("dc_motor_speed_saturation"),
    "delayed": _NotImplementedActuatorModel("delayed"),
}
