"""dr/transform.py on toy arrays: no model, no mujoco."""

import contextlib

import jax.numpy as jnp
import numpy as np
import pytest
from task_toys import BATCHES, MODES, assert_batched_matches_stacked

from humanoid_lab.dr import transform

NGEOM, NBODY, NU, NV, NFOOT = 7, 5, 4, 9, 2
DIMS = {"nbody": NBODY, "nu": NU, "nv": NV, "nfoot": NFOOT}
ALL_ON = {k: {"enable": True} for k, v in transform.DEFAULT_DR.items() if "enable" in v}


def _masks():
    return transform.dr_masks(NGEOM, NBODY, floor_id=0, root_id=1, foot_geom_ids=[3, 5])


def _base(cfg):
    rng = np.random.default_rng(1)
    f = lambda *s: rng.uniform(0.5, 2.0, s).astype(np.float32)
    base = {
        "geom_friction": f(NGEOM, 3),
        "body_mass": f(NBODY),
        "actuator_gainprm": f(NU, 10),
        "actuator_biasprm": f(NU, 10),
        "actuator_forcerange": f(NU, 2),
    }
    if cfg["com_offset"]["enable"]:
        base["body_ipos"] = f(NBODY, 3)
    if cfg["dof"]["enable"]:
        for k in ("dof_damping", "dof_armature", "dof_frictionloss"):
            base[k] = f(NV)
    return base


def _draws(cfg, batch, seed=2):
    rng = np.random.default_rng(seed)
    table = transform.DR_DRAWS(cfg, DIMS)
    return {
        k: rng.uniform(0.5, 1.5, (*batch, *shape)).astype(np.float32)
        for k, (shape, _lo, _hi, _slot) in table.items()
    }


def _cases():
    return {
        "defaults": transform.resolved_cfg(None),
        "all_on": transform.resolved_cfg(ALL_ON),
        "gains_off": transform.resolved_cfg({**ALL_ON, "joint_gains": {"enable": False}}),
    }


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("batch", BATCHES, ids=str)
@pytest.mark.parametrize("case", sorted(_cases()))
def test_dr_fields_batched_equals_stacked(case, batch, mode):
    cfg = _cases()[case]
    base, masks = _base(cfg), _masks()
    draws = _draws(cfg, batch)
    names = sorted(draws)

    def fn(*arrs):
        xp = jnp if mode != "numpy" else np
        b = {k: xp.asarray(v) for k, v in base.items()}
        m = {k: xp.asarray(v) for k, v in masks.items()}
        return transform.dr_fields(b, m, dict(zip(names, arrs)))

    assert_batched_matches_stacked(fn, [draws[n] for n in names], batch, mode)


def test_dr_fields_values_on_numpy():
    cfg = _cases()["all_on"]
    base, masks = _base(cfg), _masks()
    d = _draws(cfg, ())
    out = transform.dr_fields(base, masks, d)
    # floor friction: only [floor, 0] is replaced
    want = base["geom_friction"].copy()
    want[0, 0] = d["floor_friction"]
    want[[3, 5], 0] *= d["foot_scale"]
    np.testing.assert_array_equal(out["geom_friction"], want)
    # base mass: scaled, then plus the additive draw, on the root only
    bm = base["body_mass"] * d["link_scale"]
    bm[1] = base["body_mass"][1] * d["base_scale"] + d["mass_add"]
    np.testing.assert_allclose(out["body_mass"], bm, rtol=1e-6)
    # motor_scale drives forcerange when drawn
    np.testing.assert_array_equal(
        out["actuator_forcerange"], base["actuator_forcerange"] * d["motor_scale"][:, None]
    )
    ipos = base["body_ipos"].copy()
    ipos[1] += d["com_offset"]
    np.testing.assert_allclose(out["body_ipos"], ipos, rtol=1e-6)


def test_forcerange_rides_the_gain_draw_without_motor_strength():
    cfg = transform.resolved_cfg({"joint_gains": {"enable": True}})
    base, masks, d = _base(cfg), _masks(), _draws(cfg, ())
    out = transform.dr_fields(base, masks, d)
    np.testing.assert_array_equal(
        out["actuator_forcerange"], base["actuator_forcerange"] * d["gain_scale"][:, None]
    )
    cfg = transform.resolved_cfg(None)
    base, d = _base(cfg), _draws(cfg, ())
    out = transform.dr_fields(base, masks, d)
    np.testing.assert_array_equal(
        out["actuator_forcerange"], base["actuator_forcerange"] * d["gain_scalar"]
    )
    np.testing.assert_array_equal(out["actuator_gainprm"][:, 0], base["actuator_gainprm"][:, 0] * d["gain_scalar"])
    np.testing.assert_array_equal(out["actuator_gainprm"][:, 1:], base["actuator_gainprm"][:, 1:])


def test_optional_fields_appear_only_when_drawn():
    five = {"geom_friction", "body_mass", "actuator_gainprm", "actuator_biasprm", "actuator_forcerange"}
    cfg = transform.resolved_cfg(None)
    assert set(transform.dr_fields(_base(cfg), _masks(), _draws(cfg, ()))) == five
    cfg = transform.resolved_cfg({"com_offset": {"enable": True}})
    out = transform.dr_fields(_base(cfg), _masks(), _draws(cfg, ()))
    assert set(out) == five | {"body_ipos"}
    cfg = transform.resolved_cfg({"dof": {"enable": True}})
    out = transform.dr_fields(_base(cfg), _masks(), _draws(cfg, ()))
    assert set(out) == five | {"dof_damping", "dof_armature", "dof_frictionloss"}


def test_dr_fields_maps_unsorted_foot_geoms_to_their_draws():
    # Foot geom 6 is foot 0 and geom 1 is foot 1. The floor and the root are
    # at ids other than the toy defaults.
    cfg = transform.resolved_cfg({"foot_friction": {"enable": True}})
    masks = transform.dr_masks(NGEOM, NBODY, floor_id=2, root_id=4, foot_geom_ids=[6, 1])
    base, d = _base(cfg), _draws(cfg, ())
    d["foot_scale"] = np.array([2.0, 3.0], np.float32)
    out = transform.dr_fields(base, masks, d)
    want = base["geom_friction"].copy()
    want[2, 0] = d["floor_friction"]
    want[6, 0] *= 2.0
    want[1, 0] *= 3.0
    np.testing.assert_array_equal(out["geom_friction"], want)
    bm = base["body_mass"] * d["link_scale"]
    bm[4] = base["body_mass"][4] * d["base_scale"]
    np.testing.assert_allclose(out["body_mass"], bm, rtol=1e-6)


def test_field_cfg_fills_defaults():
    assert transform.field_cfg(None, "foot_friction") == transform.DEFAULT_DR["foot_friction"]
    # The keys a partial config leaves out come from DEFAULT_DR.
    got = transform.field_cfg({"dof": {"enable": True, "damping": [0.5, 1.5]}}, "dof")
    assert got == {**transform.DEFAULT_DR["dof"], "enable": True, "damping": [0.5, 1.5]}
    assert set(transform.resolved_cfg(None)) == set(transform.DEFAULT_DR)


@pytest.mark.parametrize(
    "domain_rand,cfg,refused",
    [
        (True, {"dof": {"enable": True}}, False),
        (False, {"base": {"floor_friction": [1, 1]}, "dof": {"enable": False}}, False),
        (False, {"dof": {"enable": True}, "foot_friction": {"enable": False}, "motor_strength": {"enable": True}}, True),
    ],
    ids=["dr_on", "nothing_enabled", "fields_enabled"],
)
def test_check_dr_enabled_refuses_enabled_fields_with_dr_off(domain_rand, cfg, refused):
    with pytest.raises(ValueError) if refused else contextlib.nullcontext():
        transform.check_dr_enabled(domain_rand, cfg)
