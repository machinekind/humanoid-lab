"""dr/transform.py on toy arrays: no model, no mujoco."""

import jax.numpy as jnp
import numpy as np
import pytest
from task_toys import BATCHES, MODES, assert_batched_matches_stacked

from humanoid_lab.dr import transform

NGEOM, NBODY, NU, NV, NFOOT = 7, 5, 4, 9, 2
DIMS = dict(nbody=NBODY, nu=NU, nv=NV, nfoot=NFOOT)
ALL_ON = {k: {"enable": True} for k, v in transform.DEFAULT_DR.items() if "enable" in v}


def _masks():
    return transform.dr_masks(NGEOM, NBODY, floor_id=0, root_id=1, foot_geom_ids=[3, 5])


def _base(cfg):
    rng = np.random.default_rng(1)
    f = lambda *s: rng.uniform(0.5, 2.0, s).astype(np.float32)
    base = dict(
        geom_friction=f(NGEOM, 3),
        body_mass=f(NBODY),
        actuator_gainprm=f(NU, 10),
        actuator_biasprm=f(NU, 10),
        actuator_forcerange=f(NU, 2),
    )
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


def test_draw_table_defaults_use_the_five_base_keys():
    t = transform.DR_DRAWS(transform.resolved_cfg(None), DIMS)
    assert {k: v[3] for k, v in t.items()} == {
        "floor_friction": ("split", 0),
        "base_scale": ("split", 1),
        "link_scale": ("split", 2),
        "gain_scalar": ("split", 3),
        "kd_scalar": ("split", 4),
    }
    assert t["link_scale"][0] == (NBODY,)
    assert t["gain_scalar"][0] == ()


def test_draw_table_fold_indices_and_shapes():
    t = transform.DR_DRAWS(transform.resolved_cfg(ALL_ON), DIMS)
    assert "gain_scalar" not in t and "kd_scalar" not in t
    slots = {k: v[3] for k, v in t.items()}
    assert slots["gain_scale"] == ("fold", 1, 0, 2)
    assert slots["kd_scale"] == ("fold", 1, 1, 2)
    assert slots["com_offset"] == ("fold", 2, None, None)
    assert [slots[f"dof_{k}"] for k in ("damping", "armature", "frictionloss")] == [
        ("fold", 3, 0, 3),
        ("fold", 3, 1, 3),
        ("fold", 3, 2, 3),
    ]
    assert slots["foot_scale"] == ("fold", 4, None, None)
    assert slots["motor_scale"] == ("fold", 5, None, None)
    assert slots["mass_add"] == ("fold", 6, None, None)
    folds = {s[1] for s in slots.values() if s[0] == "fold"}
    assert folds == {1, 2, 3, 4, 5, 6}
    shapes = {k: v[0] for k, v in t.items()}
    assert shapes["gain_scale"] == (NU,) and shapes["com_offset"] == (3,)
    assert shapes["dof_damping"] == (NV,) and shapes["foot_scale"] == (NFOOT,)
    assert t["com_offset"][1:3] == ([-0.02, -0.02, -0.01], [0.02, 0.02, 0.01])
    assert t["mass_add"][1:3] == (-1.0, 1.0)
    assert t["gain_scale"][1:3] == (1 - 0.2, 1 + 0.2)


@pytest.mark.parametrize("field,names", [
    ("com_offset", {"com_offset"}),
    ("joint_gains", {"gain_scale", "kd_scale"}),
    ("dof", {"dof_damping", "dof_armature", "dof_frictionloss"}),
    ("foot_friction", {"foot_scale"}),
    ("motor_strength", {"motor_scale"}),
    ("base_mass_add", {"mass_add"}),
])
def test_draw_table_gates_each_field(field, names):
    base_names = set(transform.DR_DRAWS(transform.resolved_cfg(None), DIMS))
    t = transform.DR_DRAWS(transform.resolved_cfg({field: {"enable": True}}), DIMS)
    extra = set(t) - base_names
    if field == "joint_gains":
        assert extra == names and base_names - set(t) == {"gain_scalar", "kd_scalar"}
    else:
        assert extra == names


def test_dr_masks_positions():
    m = transform.dr_masks(NGEOM, NBODY, floor_id=2, root_id=4, foot_geom_ids=[6, 1])
    assert m["floor_mask"].shape == (NGEOM, 3)
    assert np.argwhere(m["floor_mask"]).tolist() == [[2, 0]]
    assert np.argwhere(m["foot_mask"]).tolist() == [[1, 0], [6, 0]]
    assert m["geom_foot_pos"][6] == 0 and m["geom_foot_pos"][1] == 1
    assert np.flatnonzero(m["root_mask"]).tolist() == [4]
    assert m["root_mask"].shape == (NBODY,)
    for i, k in enumerate(("col0", "col1", "col2")):
        assert m[k].shape == (10,) and np.flatnonzero(m[k]).tolist() == [i]


def test_field_cfg_fills_defaults():
    assert transform.field_cfg(None, "foot_friction") == {"enable": False, "range": [0.8, 1.2]}
    assert transform.field_cfg({"foot_friction": {"enable": True}}, "foot_friction")["range"] == [0.8, 1.2]
    assert set(transform.resolved_cfg(None)) == set(transform.DEFAULT_DR)


def test_check_dr_enabled_passes_when_domain_rand_is_on():
    transform.check_dr_enabled(True, {"dof": {"enable": True}})


def test_check_dr_enabled_passes_when_nothing_is_enabled():
    transform.check_dr_enabled(False, {"base": {"floor_friction": [1, 1]}, "dof": {"enable": False}})


def test_check_dr_enabled_names_the_enabled_fields():
    cfg = {"dof": {"enable": True}, "foot_friction": {"enable": False}, "motor_strength": {"enable": True}}
    with pytest.raises(ValueError, match=r"domain_rand=false disables the whole dr block") as e:
        transform.check_dr_enabled(False, cfg)
    assert "['dof', 'motor_strength']" in str(e.value)
