"""Resolved task constants, shared by every backend.

This is a shared module. At module scope it may import only array_api_compat,
numpy, the standard library and other shared modules. Never jax, torch,
mujoco, brax, ml_collections or any env.

`TaskConstants` is the frozen record the shared task math reads in place of
an env: resolved joint addresses, poses, scales, limits, foot and knee ids,
budgets and the actuator model. `xp_of` picks the array namespace for a call.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import array_api_compat
import numpy as np


def xp_of(*arrays):
    """The array namespace of the data arrays in a call.

    None, bool, int and float are skipped. Pass data arrays only, never index
    arrays or numpy masks. With no array left (a scalar call) this returns
    plain numpy, which handles Python scalars; array_api_compat.numpy does not.
    """
    rest = [a for a in arrays if a is not None and not isinstance(a, (bool, int, float))]
    if not rest:
        return np
    return array_api_compat.array_namespace(*rest)


@dataclasses.dataclass(frozen=True, eq=False)
class TaskConstants:
    """The constants one task env resolves at construction.

    Joystick builds it once. Fields hold the same objects the env uses (jp
    arrays on JAX); a backend that needs numpy indices or tensors converts
    them once. Config scalars are Python floats.
    """

    dt: float
    action_size: int
    n_feet: int
    qadr: Any
    vadr: Any
    base_qadr: int
    base_vadr: int
    sensor_adr: dict
    default_pose: Any
    pose_anchor: Any
    reset_qpos: Any
    pose_weight: Any
    pose_l1_weight: Any
    action_scale: Any
    actuator_model: Any
    neutral_ctrl: Any
    ctrl_lo: Any
    ctrl_hi: Any
    soft_lo: Any
    soft_hi: Any
    torque_cap: Any
    foot_site_ids: Any
    foot_site_sole_offset: Any
    foot_root_body: Any
    foot_dof_mask: Any
    foot_geom_ids: Any
    foot_geom_radius: Any
    foot_geom_member: Any
    ankle_pair: Any
    knee_pair: Any
    knee_qidx: Any
    cmd_vmax: Any
    deadband: float
    orientation_tol: float
    heading_obs: bool
    gait_dur_ema_on: bool
    knee_stance_on: bool
    gait_symmetry_on: bool
    gait_symmetry_income_on: bool
    feet_apex_min_on: bool
    naconmax_per_env: Any
    njmax: Any
    act_perm: Any = None
    act_sign: Any = None
    state_perm: Any = None
    state_sign: Any = None
    priv_perm: Any = None
    priv_sign: Any = None
