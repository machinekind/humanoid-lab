"""Domain randomization, brax randomization_fn convention.

A vmapped function builds per-environment copies of the physics fields,
then tree_replace produces a batched model plus an in_axes template. The
draw table and the field math live in dr/transform.py, shared with the Torch
runner. This module is the JAX draw adapter: it makes each draw with the key
scheme below, then calls transform.dr_fields inside the vmap.
Nothing is hardcoded to one robot: the floor geom is found by searching the
model for
the plane type (error if none/multiple, overridable by name), the base body
is whichever body holds the model's one free joint, and the foot geoms come
from the RobotSpec instead of a robot-specific name list.

The five original fields (floor friction, base and link mass, one gain/kd
scale) draw from `r1..r5 = jax.random.split(rng, 5)`. Every other field
draws from a folded-in sub-key so it leaves r1..r5 unchanged. With every
extra field disabled, the output matches that original 5-field code.

Fold-in index taxonomy (jax.random.fold_in(rng, _DRAW + i)): 1 joint_gains,
2 com_offset, 3 dof, 4 foot_friction, 5 motor_strength, 6 base_mass_add.
0 is reserved and 7 up are free for future fields. motor_strength decouples forcerange from
joint_gains' gain_scale: disabled, forcerange keeps riding gain_scale
bitwise (a weak-motor world also reads as soft); enabled, it draws its own
per-actuator sample, so weak and soft are no longer the same draw.

_DRAW is why the taxonomy is safe. `fold_in(key, i)` is bit-identical to
`split(key, n)[i]` for every i < n, so a field folding in its raw table
index keys off one of `rand()`'s own five base keys: index 2 IS r3, the
link-mass key, and index 4 IS r5, the kd key. Those two fields sampled
straight off the folded key, so com_offset was an affine image of the first
three link-mass scales and the first foot's friction scale was an affine
image of the kd scale -- correlation 1.0 in both, measured. Two DR axes
were one axis wearing two names. Offsetting the domain puts every field's
key out of reach of any split of `rng`, however wide that split later
grows. envs/joystick.py::_sample_command carries the same offset for the
same reason. tests/integration/test_randomize.py pins both halves.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

from humanoid_lab.dr import transform

# Base of the optional fields' fold_in domain, disjoint from any split of the
# same key (see the module docstring). Same constant, same reason, as
# envs/joystick.py's pure command draws.
_DRAW = 0x100

# configs/dr/default.yaml mirrors transform.DEFAULT_DR exactly
# (tests/integration/test_randomize.py pins the two together).
_DEFAULT_DR = transform.DEFAULT_DR
_field_cfg = transform.field_cfg


def jax_dr_draws(rng, table):
    """One world's draws for a transform.DR_DRAWS table, with the key scheme above.

    Call it per world (under vmap). The range scaling stays inside
    jax.random.uniform(minval, maxval), so the values match the original code
    bit for bit. Each enabled fold-in field calls jax.random.fold_in once.
    """
    base_keys = jax.random.split(rng, 5)
    folded = {}

    def key_of(slot):
        if slot[0] == "split":
            return base_keys[slot[1]]
        _, idx, pos, n = slot
        if idx not in folded:
            k = jax.random.fold_in(rng, _DRAW + idx)
            folded[idx] = k if pos is None else jax.random.split(k, n)
        return folded[idx] if pos is None else folded[idx][pos]

    draws = {}
    for name, (shape, lo, hi, slot) in table.items():
        lo = jnp.array(lo) if isinstance(lo, list) else lo
        hi = jnp.array(hi) if isinstance(hi, list) else hi
        draws[name] = jax.random.uniform(key_of(slot), shape, minval=lo, maxval=hi)
    return draws


def _find_floor_geom_id(mj_model: mujoco.MjModel, floor_geom_name: str | None) -> int:
    if floor_geom_name is not None:
        return mj_model.geom(floor_geom_name).id
    plane_ids = [i for i in range(mj_model.ngeom) if mj_model.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE]
    if len(plane_ids) != 1:
        raise ValueError(
            f"expected exactly one plane geom to identify the floor, found {len(plane_ids)}; "
            "pass floor_geom_name= to disambiguate"
        )
    return plane_ids[0]


def _find_base_body_id(mj_model: mujoco.MjModel) -> int:
    free_joint_ids = [i for i in range(mj_model.njnt) if mj_model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE]
    if len(free_joint_ids) != 1:
        raise ValueError(f"expected exactly one free joint to identify the base body, found {len(free_joint_ids)}")
    return int(mj_model.jnt_bodyid[free_joint_ids[0]])


def make_domain_randomize(mj_model, robot_spec, dr_cfg=None, floor_geom_name=None):
    """Returns a domain_randomize(model, rng) closure.

    The closure needs geom and body ids, resolved here once outside jit from
    `mj_model` (names) and `robot_spec` (foot_geoms). `dr_cfg=None` (or all
    fields `enable: False`) reproduces the original 5-field DR (floor
    friction, base/link mass, single gain/kd scale) bitwise.
    """
    cfg = transform.resolved_cfg(dr_cfg)
    foot_cfg = cfg["foot_friction"]

    floor_id = _find_floor_geom_id(mj_model, floor_geom_name)
    root_id = _find_base_body_id(mj_model)
    foot_ids = jnp.array([mj_model.geom(name).id for name in robot_spec.foot_geoms])
    dims = dict(nbody=mj_model.nbody, nu=mj_model.nu, nv=mj_model.nv, nfoot=len(robot_spec.foot_geoms))
    table = transform.DR_DRAWS(cfg, dims)
    masks = {
        k: jnp.asarray(v)
        for k, v in transform.dr_masks(
            mj_model.ngeom, mj_model.nbody, floor_id, root_id, np.asarray(foot_ids)
        ).items()
    }
    drawn = (["body_ipos"] if "com_offset" in table else []) + (
        ["dof_damping", "dof_armature", "dof_frictionloss"] if "dof_damping" in table else []
    )

    def domain_randomize(model, rng):
        if foot_cfg["enable"]:
            # Equal-priority contacts take the element-wise max of the two
            # geoms' friction, so a foot draw below the floor's draw would
            # never reach the contact. Priority 1 on the feet makes the
            # foot's friction win outright. geom_priority is a static numpy
            # field in mjx (resolved at collision-pair setup, not under
            # jit), so it stays unbatched and is set with numpy indexing --
            # identical across envs, so it belongs on the base model.
            #
            # STATIC FIELDS MUST BE SET BEFORE in_axes IS DERIVED: an
            # np.ndarray-typed field like geom_priority is aux_data (part of
            # the pytree treedef, not a leaf) on mjx.Model. Mutating it AFTER
            # `in_axes` is built from `model` -- or after `model` has already
            # been batched -- changes the treedef out from under `in_axes`,
            # so `jax.vmap(fn, in_axes=[in_axes, 0])` dies with a
            # pytree-prefix mismatch the moment this branch runs. Applying it
            # here, first, keeps `model`'s treedef consistent for every step
            # below.
            priority = model.geom_priority.copy()
            priority[foot_ids] = 1
            model = model.tree_replace({"geom_priority": priority})

        base = {
            f: getattr(model, f)
            for f in (
                "geom_friction",
                "body_mass",
                "actuator_gainprm",
                "actuator_biasprm",
                "actuator_forcerange",
                *drawn,
            )
        }

        @jax.vmap
        def rand(rng):
            return transform.dr_fields(base, masks, jax_dr_draws(rng, table))

        out = rand(rng)

        in_axes = jax.tree_util.tree_map(lambda x: None, model)
        in_axes = in_axes.tree_replace({k: 0 for k in out})
        model = model.tree_replace(out)

        return model, in_axes

    return domain_randomize
