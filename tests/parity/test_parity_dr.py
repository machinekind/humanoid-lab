"""Identical DR draws give identical per-world model fields on both backends.

JAX applies the shared transform through randomize.make_domain_randomize.
Torch applies it through WarpJoystick's DR block and WarpPhysics.set_dr. The
Torch draws come from replay.ReplayDraws, which calls jax_dr_draws on the
same 16 keys. Both sides get the same merged dr dict.

The transform is elementwise multiplies, adds and selects. Those are exact
IEEE float32 on both backends, so every batched field must be bit-equal.

The cases copy the dr configs of tests/integration/reference_rollouts.py.
"""

from __future__ import annotations

import dataclasses

import jax
import numpy as np
import parity_report
import pytest

torch = pytest.importorskip("torch")

import replay
import warp as wp

from humanoid_lab.dr import randomize, transform
from humanoid_lab.torch_warp import physics
from humanoid_lab.torch_warp.env import WarpJoystick

N = 16
L = 50  # episode length
SEED = 404

_ALL_ON = {
    k: {"enable": True}
    for k in ("com_offset", "joint_gains", "dof", "foot_friction", "motor_strength", "base_mass_add")
}
DR_CASES = {
    "defaults": None,
    "all_on": _ALL_ON,
    "gains_off": {**_ALL_ON, "joint_gains": {"enable": False}},
    "odd_ranges": {
        **_ALL_ON,
        "base": {
            "floor_friction": [0.3, 1.7],
            "base_mass": [0.55, 1.45],
            "link_mass": [0.85, 1.15],
            "gain_fallback": [0.75, 1.33],
        },
        "motor_strength": {"enable": True, "range": [0.37, 1.13]},
        "base_mass_add": {"enable": True, "kg": 2.7},
    },
}
CASES = [
    *[("roboto_origin", dr) for dr in DR_CASES],
    ("asimov_v1", "all_on"),
]


@dataclasses.dataclass
class Built:
    name: str
    hydra: dict
    tenv: WarpJoystick
    jenv: object
    model_v: object  # the batched mjx.Model
    batched: list  # the field names JAX batches


def _batched(model_v, in_axes) -> list:
    """The fields JAX batches, with _rollout_dr_draws' filter."""
    return sorted(
        f.name
        for f in dataclasses.fields(in_axes)
        if isinstance(getattr(in_axes, f.name, None), int)
        and getattr(in_axes, f.name) == 0
        # Size fields such as ncam also read 0 here. A drawn field is batched.
        and np.ndim(getattr(model_v, f.name)) > 0
        and np.shape(getattr(model_v, f.name))[0] == N
    )


@pytest.fixture(scope="module", params=CASES, ids=[f"{r}-{d}" for r, d in CASES])
def built(request, parity_device) -> Built:
    robot, dr = request.param
    device = replay.torch_device(parity_device)
    hydra = replay.hydra_dict([f"robot={robot}", "actuators=deploy_pd"], dr=DR_CASES[dr] or {}, domain_rand=True)
    keys = jax.random.split(jax.random.PRNGKey(SEED), N)
    tenv = WarpJoystick(hydra, N, L, device=device, draws=replay.ReplayDraws(device, dr_keys=keys))
    jenv = replay.jax_env(hydra)
    model_v, in_axes = randomize.make_domain_randomize(jenv.mj_model, jenv.robot_spec, hydra["dr"])(
        jenv.mjx_model, keys
    )
    return Built(f"{robot}-{dr}", hydra, tenv, jenv, model_v, _batched(model_v, in_axes))


def _torch_field(b, name) -> np.ndarray:
    return wp.to_torch(getattr(b.tenv.physics.m, name)).cpu().numpy()


def _first_difference(b, name, expected, actual) -> str:
    """The first differing index, with both bases, the world's draws and both results."""
    idx = tuple(int(i) for i in np.argwhere(expected != actual)[0])
    world, elem = idx[0], idx[1:]
    jax_base = np.asarray(getattr(b.jenv.mjx_model, name))[elem]
    torch_base = replay.to_numpy(b.tenv.physics.base_fields([name])[name])[elem]
    draws = {k: v[world] for k, v in replay.to_numpy(b.tenv.draws.dr(N, _table(b))).items()}
    return (
        f"{b.name} {name}: first difference at {idx}. JAX base {jax_base!r}, Torch base {torch_base!r}. "
        f"JAX {expected[idx]!r}, Torch {actual[idx]!r}. Draws of world {world}: {draws}"
    )


def _table(b):
    m = b.jenv.mj_model
    dims = {"nbody": m.nbody, "nu": m.nu, "nv": m.nv, "nfoot": len(b.jenv.robot_spec.foot_geoms)}
    return transform.DR_DRAWS(transform.resolved_cfg(b.hydra["dr"]), dims)


def test_identical_draws_give_identical_model_fields(built):
    b = built
    assert b.batched, f"{b.name}: JAX batched no field"
    for name in b.batched:
        expected = np.asarray(getattr(b.model_v, name))
        actual = _torch_field(b, name)
        assert actual.shape == expected.shape, f"{b.name} {name}: shape {actual.shape}, JAX {expected.shape}"
        assert actual.dtype == np.float32, f"{b.name} {name}: Torch dtype {actual.dtype}"
        assert expected.dtype == np.float32, f"{b.name} {name}: JAX dtype {expected.dtype}"
        parity_report.record(f"dr[{b.name}]", name, expected, actual, tol="exact")
        if not np.array_equal(expected, actual):
            pytest.fail(_first_difference(b, name, expected, actual))
    # Torch must leave every field JAX keeps unbatched at its base value in every world.
    for name in sorted(set(physics.DR_DTYPES) - set(b.batched)):
        base = np.asarray(getattr(b.jenv.mjx_model, name))
        actual = _torch_field(b, name)
        assert actual.shape[1:] == base.shape, f"{b.name} {name}: shape {actual.shape}, base {base.shape}"
        assert (actual == base).all(), f"{b.name} {name}: JAX leaves it at its base, Torch changes it"


def test_foot_priority_matches(built):
    b = built
    expected = np.asarray(b.model_v.geom_priority)
    actual = np.asarray(b.tenv.physics.mj_model.geom_priority)
    np.testing.assert_array_equal(actual, expected)
    feet = [b.jenv.mj_model.geom(g).id for g in b.jenv.robot_spec.foot_geoms]
    foot_on = bool((b.hydra["dr"].get("foot_friction") or {}).get("enable"))
    want = np.ones(len(feet)) if foot_on else b.jenv.mj_model.geom_priority[feet]
    np.testing.assert_array_equal(actual[feet], want, err_msg=f"{b.name}: foot priority, foot_friction {foot_on}")
    if not foot_on:
        assert np.all(actual[feet] != 1), f"{b.name}: feet read priority 1 with foot_friction off"
    # The priority goes on a model copy. The JAX model inside the Torch env keeps its own.
    np.testing.assert_array_equal(
        b.tenv.jenv.mj_model.geom_priority, b.jenv.mj_model.geom_priority, err_msg=f"{b.name}: the JAX model changed"
    )
