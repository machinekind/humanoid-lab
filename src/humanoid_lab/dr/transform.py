"""Domain randomization draw table and field transform, shared by both backends.

`DR_DRAWS` lists every random draw a config asks for: its per-world shape, its
bounds and the JAX key that makes it. `dr_fields` turns one set of draws into
the per-world model fields. It is array-agnostic and takes draws with any
leading batch shape. Only numpy, the standard library and array_api_compat are
imported here.

The JAX adapter (dr/randomize.py) makes the draws with the key scheme of the
original code, so scaling stays inside jax.random.uniform(minval, maxval). The
Torch runner feeds the same table with its own generator.
"""

from __future__ import annotations

import numpy as np
from array_api_compat import array_namespace

# Defaults for each dr_cfg field. configs/dr/default.yaml mirrors this shape
# exactly (tests/integration/test_randomize.py pins the two together); missing keys fall
# back to these values.
DEFAULT_DR = {
    # The always-on base set (once DR itself is enabled): floor friction,
    # base and link mass, and the single gain/kd scalar used when
    # joint_gains is off. Multiplicative ranges.
    "base": {
        "floor_friction": [0.6, 1.2],
        "base_mass": [0.7, 1.3],
        "link_mass": [0.9, 1.1],
        "gain_fallback": [0.8, 1.2],
    },
    "com_offset": {"enable": False, "xy": 0.02, "z": 0.01},
    "joint_gains": {"enable": False, "gain_pct": 0.2, "kd_pct": 0.2},
    "dof": {
        "enable": False,
        "damping": [0.9, 1.1],
        "armature": [0.9, 1.1],
        "frictionloss": [0.9, 1.1],
    },
    "foot_friction": {"enable": False, "range": [0.8, 1.2]},
    # Weak-skewed: the failure mode this guards is motors too weak to pull
    # the hips back under the body, not a symmetric strength spread.
    "motor_strength": {"enable": False, "range": [0.5, 1.1]},
    # Additive +-kg draw on the base body, on top of base_mass's scale: a
    # payload weighs the same on a light robot and a heavy one.
    "base_mass_add": {"enable": False, "kg": 1.0},
}


def field_cfg(dr_cfg, name):
    return {**DEFAULT_DR[name], **((dr_cfg or {}).get(name) or {})}


def resolved_cfg(dr_cfg):
    return {name: field_cfg(dr_cfg, name) for name in DEFAULT_DR}


def check_dr_enabled(domain_rand, dr_cfg):
    """Raise if domain_rand is off while a dr field has enable=true."""
    if domain_rand:
        return
    enabled = [k for k, v in dr_cfg.items() if isinstance(v, dict) and v.get("enable")]
    if enabled:
        raise ValueError(
            f"domain_rand=false disables the whole dr block, but "
            f"cfg.dr fields {enabled} have enable=true; either set "
            f"domain_rand=true or disable those fields too"
        )


def DR_DRAWS(cfg, dims):
    """name -> (per-world shape, lo, hi, JAX key slot) for the enabled draws.

    cfg is resolved_cfg(dr_cfg). dims holds nbody, nu, nv and nfoot. The slot
    is ("split", i) for the i-th key of split(rng, 5), or ("fold", i, pos, n)
    for fold_in(rng, _DRAW + i), taken as sub-key pos of an n-way split when
    pos is not None. List bounds are per element.
    """
    b = cfg["base"]
    d = {
        "floor_friction": ((), *b["floor_friction"], ("split", 0)),
        "base_scale": ((), *b["base_mass"], ("split", 1)),
        "link_scale": ((dims["nbody"],), *b["link_mass"], ("split", 2)),
    }
    if cfg["base_mass_add"]["enable"]:
        kg = cfg["base_mass_add"]["kg"]
        d["mass_add"] = ((), -kg, kg, ("fold", 6, None, None))
    if cfg["joint_gains"]["enable"]:
        pct, kpct = cfg["joint_gains"]["gain_pct"], cfg["joint_gains"]["kd_pct"]
        d["gain_scale"] = ((dims["nu"],), 1 - pct, 1 + pct, ("fold", 1, 0, 2))
        d["kd_scale"] = ((dims["nu"],), 1 - kpct, 1 + kpct, ("fold", 1, 1, 2))
    else:
        d["gain_scalar"] = ((), *b["gain_fallback"], ("split", 3))
        d["kd_scalar"] = ((), *b["gain_fallback"], ("split", 4))
    if cfg["motor_strength"]["enable"]:
        d["motor_scale"] = ((dims["nu"],), *cfg["motor_strength"]["range"], ("fold", 5, None, None))
    if cfg["com_offset"]["enable"]:
        xy, z = cfg["com_offset"]["xy"], cfg["com_offset"]["z"]
        d["com_offset"] = ((3,), [-xy, -xy, -z], [xy, xy, z], ("fold", 2, None, None))
    if cfg["dof"]["enable"]:
        for pos, key in enumerate(("damping", "armature", "frictionloss")):
            d[f"dof_{key}"] = ((dims["nv"],), *cfg["dof"][key], ("fold", 3, pos, 3))
    if cfg["foot_friction"]["enable"]:
        d["foot_scale"] = ((dims["nfoot"],), *cfg["foot_friction"]["range"], ("fold", 4, None, None))
    return d


def dr_masks(ngeom, nbody, floor_id, root_id, foot_geom_ids):
    """Numpy masks and gather positions that dr_fields reads, built from ids."""
    foot_geom_ids = np.asarray(foot_geom_ids, np.int64)
    floor_mask = np.zeros((ngeom, 3), bool)
    floor_mask[floor_id, 0] = True
    foot_mask = np.zeros((ngeom, 3), bool)
    foot_mask[foot_geom_ids, 0] = True
    geom_foot_pos = np.zeros(ngeom, np.int64)
    geom_foot_pos[foot_geom_ids] = np.arange(len(foot_geom_ids))
    root_mask = np.zeros(nbody, bool)
    root_mask[root_id] = True
    cols = np.arange(10)
    return dict(
        floor_mask=floor_mask,
        foot_mask=foot_mask,
        geom_foot_pos=geom_foot_pos,
        root_mask=root_mask,
        col0=cols == 0,
        col1=cols == 1,
        col2=cols == 2,
    )


def dr_fields(base, masks, draws):
    """Per-world model fields from one set of draws.

    base: unbatched model arrays (geom_friction, body_mass, actuator_gainprm,
    actuator_biasprm, actuator_forcerange, plus body_ipos and dof_* when those
    were drawn). draws: name -> (..., *shape) scaled draws, as DR_DRAWS lists.
    Every output has the draws' leading batch dims.
    """
    xp = array_namespace(*draws.values())
    root = masks["root_mask"]
    gf = xp.where(masks["floor_mask"], draws["floor_friction"][..., None, None], base["geom_friction"])
    bm = base["body_mass"] * draws["link_scale"]
    bm = xp.where(root, base["body_mass"] * draws["base_scale"][..., None], bm)
    if "mass_add" in draws:
        bm = xp.where(root, bm + draws["mass_add"][..., None], bm)
    if "gain_scale" in draws:
        g, k = draws["gain_scale"], draws["kd_scale"]
    else:
        g, k = draws["gain_scalar"][..., None], draws["kd_scalar"][..., None]
    gp = base["actuator_gainprm"]
    bp = base["actuator_biasprm"]
    gp = xp.where(masks["col0"], gp * g[..., None], gp)
    bp = xp.where(masks["col1"], bp * g[..., None], bp)
    bp = xp.where(masks["col2"], bp * k[..., None], bp)
    motor = draws["motor_scale"] if "motor_scale" in draws else (draws["gain_scale"] if "gain_scale" in draws else g)
    fr = base["actuator_forcerange"] * motor[..., None]
    if "foot_scale" in draws:
        mult = xp.take(draws["foot_scale"], masks["geom_foot_pos"], axis=-1)
        gf = xp.where(masks["foot_mask"], gf * mult[..., None], gf)
    out = dict(
        geom_friction=gf,
        body_mass=bm,
        actuator_gainprm=gp,
        actuator_biasprm=bp,
        actuator_forcerange=fr,
    )
    if "com_offset" in draws:
        bi = base["body_ipos"]
        out["body_ipos"] = xp.where(root[:, None], bi + draws["com_offset"][..., None, :], bi)
    if "dof_damping" in draws:
        for f in ("dof_damping", "dof_armature", "dof_frictionloss"):
            out[f] = base[f] * draws[f]
    return out
