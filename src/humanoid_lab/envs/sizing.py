"""Motor-sizing task (task=sizing): Joystick, penalty-shifted and
instrumented for PLAN.md's "First experiments" #2 sizing loop
(`robot=asimov_v1 task=sizing actuators=sizing_ideal`).

Deliberately a thin Joystick subclass, not a new task: sizing wants the SAME
walking behavior (tracking, gait, fall handling, obs) so the torque/speed
percentiles this task exists to produce describe a real gait, not some other
policy. The only behavioral difference is reward.scales.torques/energy at
5x joystick's starting values (see default_config below) -- PLAN.md's
sizing loop wants a torque-frugal gait so the demand percentiles the sizing
report reads off reflect what the task NEEDS, not what a sloppily-penalized
policy wastes. Domain randomization is expected off for sizing runs
(PLAN.md: "DR off"), but that is a run-config choice (configs/config.yaml's
domain_rand: false default, gated in train.py), not something this env
enforces in code.

Adds three per-step scalar metrics -- sizing/tau_frac_max,
sizing/omega_frac_max, sizing/mech_power -- read back out of
`runs/<name>/checkpoints/.../` rollouts by sizing/collect.py and reduced
into percentile tables/plots by sizing/report.py. These are metrics
(state.metrics), not reward terms: they never affect `reward`.
"""

from __future__ import annotations

import jax.numpy as jp
from ml_collections import config_dict
from mujoco_playground._src import mjx_env

from humanoid_lab.envs.joystick import Joystick
from humanoid_lab.envs.joystick import default_config as _joystick_default_config
from humanoid_lab.robot.presets import resolve as _resolve_preset
from humanoid_lab.rewards import terms

# PLAN.md "First experiments" #2: "Torque and energy penalties on ...
# generous effort caps" -- 5x joystick's starting reward.scales.torques/
# energy so the sizing loop's percentiles reflect torque/energy NEED, not
# sloppy-policy noise.
_PENALTY_MULT = 5.0


def default_config() -> config_dict.ConfigDict:
    cfg = _joystick_default_config()
    cfg.reward.scales.torques = cfg.reward.scales.torques * _PENALTY_MULT
    cfg.reward.scales.energy = cfg.reward.scales.energy * _PENALTY_MULT
    return cfg


class Sizing(Joystick):
    """Joystick with torque/energy penalties sharpened and per-step
    torque/speed/power telemetry added to `state.metrics`.
    """

    def __init__(self, robot_dir, preset_name, config=None, config_overrides=None, actuator_overrides=None):
        super().__init__(
            robot_dir, preset_name, config or default_config(), config_overrides, actuator_overrides
        )

        # Per-joint velocity_limit, canonical actuated_joints order.
        # HumanoidEnv doesn't carry this (only kp/kd/effort_limit have an
        # XML/actuator-forcerange equivalent to read back after injection --
        # see actuators/models.py's JointActuatorParams docstring: a MuJoCo
        # actuator clamps force/ctrl, never joint speed), so it is resolved
        # here directly from the already-loaded preset + robot spec.
        params_by_joint = _resolve_preset(self._preset, self._robot_spec)
        missing = [
            n
            for n in self._robot_spec.actuated_joints
            if params_by_joint[n].velocity_limit is None
        ]
        if missing:
            raise ValueError(
                f"actuator preset '{preset_name}' has no velocity_limit for joint(s) "
                f"{missing}; task=sizing requires one per actuated joint for "
                "sizing/omega_frac_max"
            )
        self._velocity_limit = jp.array(
            [params_by_joint[n].velocity_limit for n in self._robot_spec.actuated_joints]
        )

    # -- sizing telemetry -----------------------------------------------
    def _sizing_metrics(self, data) -> dict:
        """Per-step scalar sizing metrics, added to state.metrics.

        tau/omega are already in canonical actuated_joints order (data.
        actuator_force follows the actuator injection order, robot/build.py
        injects in robot_spec.actuated_joints order; data.qvel[self._vadr]
        is the same order by construction, see envs/base.py).
        """
        tau = data.actuator_force
        omega = data.qvel[self._vadr]
        tau_frac = jp.abs(tau) / self._torque_cap
        omega_frac = jp.abs(omega) / self._velocity_limit
        return {
            "sizing/tau_frac_max": jp.max(tau_frac),
            "sizing/omega_frac_max": jp.max(omega_frac),
            # sum |tau*omega| over actuated joints -- same math as
            # rewards/terms.py's energy() term, reused rather than
            # reimplemented (rewards/terms.py is off-limits to edit, but
            # nothing stops importing its pure functions).
            "sizing/mech_power": terms.energy(omega, tau),
        }

    # -- reset / step: CRITICAL for scan-carry parity, mirroring Joystick's
    # own reward/* key discipline (see joystick.py's reset()/step()
    # docstrings) -- the sizing/* keys must be present identically after
    # reset() and every step(), or brax's training scan chokes on a
    # changing metrics pytree structure across steps.
    def reset(self, rng) -> mjx_env.State:
        state = super().reset(rng)
        metrics = {**state.metrics, **self._sizing_metrics(state.data)}
        return state.replace(metrics=metrics)

    def step(self, state: mjx_env.State, action) -> mjx_env.State:
        next_state = super().step(state, action)
        metrics = {**next_state.metrics, **self._sizing_metrics(next_state.data)}
        return next_state.replace(metrics=metrics)
