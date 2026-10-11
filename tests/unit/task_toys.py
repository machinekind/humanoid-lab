"""Toy inputs for the shared task-math unit tests.

A helper module, not a test file. tests/unit is on sys.path for its tests, so
they import it as `task_toys`. Nothing here builds a model or an env.
"""

import dataclasses

import numpy as np

from humanoid_lab.envs.constants import TaskConstants

N_FEET = 2
NU = 4
NV = 10
NBODY = 6
GEOMS_PER_FOOT = 3

# Batch shapes and array modes of the batched-equals-stacked unit tests.
# jit compiles each shared function, which is too slow for the edit loop.
# tests/integration/test_shared_math_jit.py runs the same tests under jit.
BATCHES = [(), (3,), (2, 3)]
MODES = ["numpy", "jax"]


class StubActuatorModel:
    """ctrl_from_action(a, d, s) = d + a * s, the PD-style shape."""

    @staticmethod
    def ctrl_from_action(action, default_pose, action_scale):
        return default_pose + action * action_scale


def _backend(xp_name):
    if xp_name == "numpy":
        return np
    if xp_name == "jax":
        import jax.numpy as jnp

        return jnp
    raise ValueError(f"unknown xp_name {xp_name!r}")


def toy_consts(xp_name="numpy") -> TaskConstants:
    """A small TaskConstants: 2 feet, nu=4, nv=10, nbody=6, 8 geoms (3 per
    foot plus 2 unowned), all three sensors present. Float arrays are numpy
    or jax.numpy; index arrays and the foot tables stay numpy."""
    xp = _backend(xp_name)

    def f(values):
        return xp.asarray(np.asarray(values, np.float32))

    rng = np.random.default_rng(0)
    geom_foot = np.array([0, 0, 0, 1, 1, 1, -1, -1])
    member = np.zeros((len(geom_foot), N_FEET), bool)
    for g, foot in enumerate(geom_foot):
        if foot >= 0:
            member[g, foot] = True
    dof_mask = np.zeros((N_FEET, NV), np.float32)
    dof_mask[0, :5] = 1.0
    dof_mask[1, [0, 1, 5, 6, 7]] = 1.0
    default_pose = f(rng.uniform(-0.5, 0.5, NU))
    action_scale = f(np.full(NU, 0.5))
    model = StubActuatorModel()
    return TaskConstants(
        dt=0.02,
        action_size=NU,
        n_feet=N_FEET,
        qadr=xp.asarray(np.arange(7, 7 + NU)),
        vadr=xp.asarray(np.arange(6, 6 + NU)),
        base_qadr=0,
        base_vadr=0,
        sensor_adr={"gyro": 0, "quat": 3, "linvel": 7},
        default_pose=default_pose,
        pose_anchor=default_pose,
        reset_qpos=f(rng.uniform(-0.5, 0.5, 7 + NU)),
        pose_weight=f(np.ones(NU)),
        pose_l1_weight=f(np.full(NU, 0.5)),
        action_scale=action_scale,
        actuator_model=model,
        neutral_ctrl=model.ctrl_from_action(xp.zeros(NU, dtype=xp.float32), default_pose, action_scale),
        ctrl_lo=f(np.full(NU, -2.0)),
        ctrl_hi=f(np.full(NU, 2.0)),
        soft_lo=f(np.full(NU, -1.0)),
        soft_hi=f(np.full(NU, 1.0)),
        torque_cap=f(np.full(NU, 30.0)),
        foot_site_ids=np.array([2, 4]),
        foot_site_sole_offset=f([0.02, 0.02]),
        foot_root_body=np.zeros(N_FEET, np.int32),
        foot_dof_mask=f(dof_mask),
        foot_geom_ids=np.array([0, 1, 2, 3, 4, 5]),
        foot_geom_radius=f(np.full(6, 0.01)),
        foot_geom_member=member[:6],
        ankle_pair=None,
        knee_pair=None,
        knee_qidx=None,
        cmd_vmax=xp.asarray(np.float32(1.0)),
        deadband=0.05,
        orientation_tol=0.0,
        heading_obs=False,
        gait_dur_ema_on=False,
        knee_stance_on=False,
        gait_symmetry_on=False,
        gait_symmetry_income_on=False,
        feet_apex_min_on=False,
        naconmax_per_env=None,
        njmax=None,
        root_body=0,
    )


def toy_replace(consts, **changes) -> TaskConstants:
    return dataclasses.replace(consts, **changes)


def _leaves(out):
    if isinstance(out, dict):
        return [x for k in sorted(out) for x in _leaves(out[k])]
    if isinstance(out, (tuple, list)):
        return [x for v in out for x in _leaves(v)]
    return [out]


def _like(out, leaves):
    """Rebuild the structure of `out` from an iterator of leaves."""
    if isinstance(out, dict):
        return {k: _like(out[k], leaves) for k in sorted(out)}
    if isinstance(out, (tuple, list)):
        return [_like(v, leaves) for v in out]
    return next(leaves)


def assert_batched_matches_stacked(fn, args, batch_shape, mode="numpy", atol=0.0):
    """fn on the full batch must equal fn per item, stacked.

    args are numpy arrays or non-array values. An array whose leading dims
    equal batch_shape is batched; anything else passes through unchanged.
    mode is 'numpy', 'jax' (eager) or 'jit'. Numpy must match exactly. On jax,
    eager or jit, a batched exp and a single-item exp can differ in the last
    bit, so the comparison uses rtol=1e-6 plus atol (default 0; a contraction that
    sums to near zero needs a small one). Under jit the non-array args are
    closed over and the arrays are traced. One jitted function serves the full
    call and every item, so it compiles once per input shape.
    """
    batch_shape = tuple(batch_shape)
    nb = len(batch_shape)
    size = int(np.prod(batch_shape, dtype=int))

    def is_array(a):
        return isinstance(a, (np.ndarray, np.generic))

    def is_batched(a):
        return is_array(a) and np.shape(a)[:nb] == batch_shape

    if mode == "numpy":
        conv = lambda a: a
        call = fn
    else:
        import jax
        import jax.numpy as jnp

        conv = lambda a: jnp.asarray(a) if is_array(a) else a
        call = fn
        if mode == "jit":
            idx = [i for i, a in enumerate(args) if is_array(a)]

            def traced(*arrs):
                full = list(args)
                for i, v in zip(idx, arrs):
                    full[i] = v
                return fn(*full)

            jitted = jax.jit(traced)
            call = lambda *a: jitted(*[a[i] for i in idx])

    full = call(*[conv(a) for a in args])
    flat = [np.reshape(a, (size,) + np.shape(a)[nb:]) if is_batched(a) else a for a in args]
    items = [
        call(*[conv(a[i]) if is_batched(orig) else conv(a) for a, orig in zip(flat, args)])
        for i in range(size)
    ]
    full_leaves = _leaves(full)
    for k, leaf in enumerate(full_leaves):
        got = np.asarray(leaf)
        assert got.shape[:nb] == batch_shape, (got.shape, batch_shape)
        want = np.stack([np.asarray(_leaves(it)[k]) for it in items]).reshape(got.shape)
        if mode != "numpy":
            np.testing.assert_allclose(got, want, rtol=1e-6, atol=atol)
        else:
            np.testing.assert_array_equal(got, want)
