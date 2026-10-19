"""Observation signals and the joystick observation catalog, shared by every backend.

This is a shared module. At module scope it may import only array_api_compat,
numpy, the standard library and other shared modules. Never jax, torch,
mujoco, brax, ml_collections or any env.

Every function takes arrays with any leading batch shape and reads the last
axis. JAX calls them per world inside its vmap. The Torch runner calls them on
the whole batch. Both must give the same bits, so the op forms below are fixed:

- The quaternion inverse is q * [1, -1, -1, -1].
- A dot product is xp.vecdot. A sum of a product changes bits under vmap.
- A cross product is xp.linalg.cross(axis=-1).
- A contraction is xp.matmul. (jacp * qvel).sum differs by 8e-8.
- The foot contact is a mask and any, not a scatter-max.
- Scalars are Python floats. Arrays made here carry the dtype and device of
  the input they combine with.
- Concatenation is concat(axis=-1).

`readings` reads the signals out of a data object. `catalog` builds the named
observation vectors from the readings and the env's info. `observe` gathers,
adds actor noise and mirrors.
"""

from __future__ import annotations

import array_api_compat

from humanoid_lab.envs import progress, symmetry
from humanoid_lab.envs.constants import xp_of

# The fields of the data object that `readings` reads. mjx.Data has them all.
# The Torch runner passes a namespace of views with these names.
DATA_FIELDS = (
    "qpos",
    "qvel",
    "qacc",
    "sensordata",
    "actuator_force",
    "xpos",
    "site_xpos",
    "geom_xpos",
    "subtree_com",
    "cdof",
)

_QUAT_CONJ = (1.0, -1.0, -1.0, -1.0)


def _dev(x):
    return array_api_compat.device(x)


def quat_inv(q):
    xp = xp_of(q)
    return q * xp.asarray(list(_QUAT_CONJ), dtype=q.dtype, device=_dev(q))


def quat_rotate(q, v):
    """Rotate v by q, op for op as brax.math.rotate."""
    xp = xp_of(q, v)
    s, u = q[..., :1], q[..., 1:]
    r = 2 * (xp.vecdot(u, v, axis=-1)[..., None] * u) + (s * s - xp.vecdot(u, u, axis=-1)[..., None]) * v
    return r + 2 * s * xp.linalg.cross(u, v, axis=-1)


def quat_rotate_inv(q, v):
    return quat_rotate(quat_inv(q), v)


def quat(sensordata, qpos, sensor_adr, base_qadr):
    """Base orientation (w, x, y, z): the quat sensor, else the free joint's qpos."""
    if "quat" in sensor_adr:
        a = sensor_adr["quat"]
        return sensordata[..., a : a + 4]
    return qpos[..., base_qadr + 3 : base_qadr + 7]


def gyro(sensordata, qvel, sensor_adr, base_vadr):
    """Base angular rate, body frame. A free joint's angular velocity is
    already in the body frame, like a body-mounted gyro."""
    if "gyro" in sensor_adr:
        a = sensor_adr["gyro"]
        return sensordata[..., a : a + 3]
    return qvel[..., base_vadr + 3 : base_vadr + 6]


def linvel(sensordata, qvel, q, sensor_adr, base_vadr):
    """Base linear velocity, body frame."""
    if "linvel" in sensor_adr:
        a = sensor_adr["linvel"]
        return sensordata[..., a : a + 3]
    return quat_rotate_inv(q, qvel[..., base_vadr : base_vadr + 3])


def gravity_body(q):
    """World down as seen in the body frame."""
    xp = xp_of(q)
    return quat_rotate_inv(q, xp.asarray([0.0, 0.0, -1.0], dtype=q.dtype, device=_dev(q)))


def base_yaw(q):
    return xp_of(q).atan2(
        2.0 * (q[..., 0] * q[..., 3] + q[..., 1] * q[..., 2]),
        1.0 - 2.0 * (q[..., 2] ** 2 + q[..., 3] ** 2),
    )


def wrap_angle(a):
    xp = xp_of(a)
    return xp.atan2(xp.sin(a), xp.cos(a))


def heading_err(yaw_ref, yaw):
    """Wrapped heading error, clipped to +-1 rad."""
    xp = xp_of(yaw_ref, yaw)
    return xp.clip(wrap_angle(yaw_ref - yaw), -1.0, 1.0)


def foot_clearance(site_xpos, foot_site_ids, sole_offset):
    """Per-foot sole height above the floor: site z minus the site-to-sole offset."""
    return site_xpos[..., foot_site_ids, 2] - sole_offset


def foot_contact(geom_xpos, foot_geom_ids, radius, member):
    """Per-foot contact bool: any geom of the foot (member is (G, F)) within
    0.005 of touching the floor. Assumes capsule geoms."""
    xp = xp_of(geom_xpos)
    z = geom_xpos[..., foot_geom_ids, 2]
    per = z < radius + 0.005
    return xp.any(per[..., :, None] & member, axis=-2)


def foot_linvel(site_xpos, subtree_com, cdof, qvel, foot_site_ids, root_body, dof_mask):
    """Per-foot linear velocity at the foot site, world frame.

    The site Jacobian is built from cdof and masked to the dofs that move the
    foot, then contracted with qvel.
    """
    xp = xp_of(site_xpos, cdof, qvel)
    p = site_xpos[..., foot_site_ids, :]
    off = p - subtree_com[..., root_body, :]
    ang, lin = cdof[..., None, :, :3], cdof[..., None, :, 3:]
    jacp = (lin + xp.linalg.cross(ang, off[..., :, None, :], axis=-1)) * dof_mask[..., :, :, None]
    return xp.matmul(xp.matrix_transpose(jacp), qvel[..., None, :, None])[..., 0]


def foot_pos_base(site_xpos, xpos, foot_site_ids, root_body, q):
    """Per-foot site position relative to the base, in the base frame, shape (..., F, 3).

    The base position is xpos[root_body]. It comes from the same forward
    kinematics stage as site_xpos and the quat sensor. The free joint's qpos
    runs one substep ahead of them, so it is not used here. `root_body` is
    the body id of the free joint. `[..., 0]` of the result is each foot's
    signed offset along the base's forward axis. A foot ahead of the base
    has a positive x.
    """
    rel = site_xpos[..., foot_site_ids, :] - xpos[..., root_body, :][..., None, :]
    return quat_rotate_inv(q[..., None, :], rel)


def body_y_sep(xpos, pair, q):
    """|y| distance between two bodies, base frame."""
    xp = xp_of(xpos, q)
    delta = xpos[..., pair[0], :] - xpos[..., pair[1], :]
    return xp.abs(quat_rotate_inv(q, delta)[..., 1])


def fall(base_height, gravity_z, min_height, max_tilt_gz):
    return (base_height < min_height) | (gravity_z > max_tilt_gz)


def readings(d, consts):
    """The signals the catalog and the rewards read, from any object with DATA_FIELDS."""
    adr = consts.sensor_adr
    b, v = consts.base_qadr, consts.base_vadr
    q = quat(d.sensordata, d.qpos, adr, b)
    out = {
        "quat": q,
        "gyro": gyro(d.sensordata, d.qvel, adr, v),
        "linvel": linvel(d.sensordata, d.qvel, q, adr, v),
        "gravity": gravity_body(q),
        "yaw": base_yaw(q),
        "qpos_act": d.qpos[..., consts.qadr],
        "qvel_act": d.qvel[..., consts.vadr],
        "qacc_act": d.qacc[..., consts.vadr],
        "actuator_force": d.actuator_force,
        "base_height": d.qpos[..., b + 2],
        "height": d.qpos[..., b + 2 : b + 3],
        "foot_clearance": foot_clearance(d.site_xpos, consts.foot_site_ids, consts.foot_site_sole_offset),
        "foot_linvel": foot_linvel(
            d.site_xpos,
            d.subtree_com,
            d.cdof,
            d.qvel,
            consts.foot_site_ids,
            consts.foot_root_body,
            consts.foot_dof_mask,
        ),
        "contact": foot_contact(
            d.geom_xpos, consts.foot_geom_ids, consts.foot_geom_radius, consts.foot_geom_member
        ),
    }
    if consts.ankle_pair is not None:
        out["feet_sep"] = body_y_sep(d.xpos, consts.ankle_pair, q)
    if consts.knee_pair is not None:
        out["knee_sep"] = body_y_sep(d.xpos, consts.knee_pair, q)
    # Only the landing terms read the feet's base-frame position. Off, the
    # readings and the trace are unchanged.
    if consts.step_length_on or consts.foot_reach_on or consts.knee_extend_on:
        out["foot_pos_base"] = foot_pos_base(d.site_xpos, d.xpos, consts.foot_site_ids, consts.root_body, q)
    return out


def catalog(r, info, consts):
    """name -> observation vector, with the batch shape of the readings.

    Only gyro, gravity, joint_pos, joint_vel and last_action are signals a
    real robot can expose. The rest (linvel, height, actuator_force, contacts)
    are sim-only and belong on the privileged (critic) list.
    """
    xp = xp_of(r["gyro"])
    out = {
        "gyro": r["gyro"],
        "gravity": r["gravity"],
        "joint_pos": r["qpos_act"] - consts.default_pose,
        "joint_vel": r["qvel_act"],
        "last_action": info["last_action"],
        "linvel": r["linvel"],
        "height": r["height"],
        "actuator_force": r["actuator_force"],
        "contacts": xp.astype(r["contact"], xp.float32),
        "command": info["command"],
    }
    if "yaw_ref" in info:
        out["heading_err"] = heading_err(info["yaw_ref"], r["yaw"])[..., None]
    else:
        out["heading_err"] = xp.zeros(r["gyro"].shape[:-1] + (1,), dtype=xp.float32, device=_dev(r["gyro"]))
    if consts.frozen_phase_zeros:
        # gait.frozen_phase_obs "zeros" on a frozen clock: no left/right marker.
        out["phase"] = xp.zeros(
            r["gyro"].shape[:-1] + (2 * consts.n_feet,), dtype=xp.float32, device=_dev(r["gyro"])
        )
    else:
        leg = progress.leg_phases(info["phase"])
        out["phase"] = xp.concat([xp.cos(leg), xp.sin(leg)], axis=-1)
    return out


def gather(cat, names):
    missing = [n for n in names if n not in cat]
    if missing:
        raise KeyError(f"unknown obs component(s) {missing}; available: {sorted(cat)}")
    xp = xp_of(*[cat[n] for n in names])
    return xp.concat([cat[n] for n in names], axis=-1)


def noise_scales(cat, names, obs_noise):
    """Per-entry actor noise scale from obs_noise by component name (no entry = 0)."""
    xp = xp_of(*[cat[n] for n in names])
    return xp.concat(
        [
            xp.full(cat[n].shape, float(obs_noise.get(n, 0.0)), dtype=xp.float32, device=_dev(cat[n]))
            for n in names
        ],
        axis=-1,
    )


def add_noise(clean, noise, scales):
    return clean + noise * scales


def observe(cat, state_names, priv_names, noise=None, obs_noise=None, mirror=None):
    """{'state', 'privileged_state'} from the catalog.

    noise is a uniform(-1, 1) draw of the actor vector's shape and goes on the
    actor vector only. mirror is (flag, state_perm, state_sign, priv_perm,
    priv_sign) and is applied last.
    """
    state = gather(cat, state_names)
    if noise is not None:
        state = add_noise(state, noise, noise_scales(cat, state_names, obs_noise or {}))
    priv = gather(cat, priv_names)
    if mirror is not None:
        flag, state_perm, state_sign, priv_perm, priv_sign = mirror
        state = symmetry.mirror(state, flag, state_perm, state_sign)
        priv = symmetry.mirror(priv, flag, priv_perm, priv_sign)
    return {"state": state, "privileged_state": priv}
