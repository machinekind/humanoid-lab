"""The shared observation signals across array backends and batch shapes.

The batched functions run on numpy and eager jax at batch shapes (), (3,)
and (2, 3). The batched call must equal the per-item calls stacked.
Elementwise functions have value tests only. The toy constants and data
come from task_toys.
tests/integration/test_shared_math_jit.py runs the same checks under jit.
"""

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest
from task_toys import (
    BATCHES,
    MODES,
    NBODY,
    NU,
    NV,
    assert_batched_matches_stacked,
    toy_consts,
    toy_replace,
)

from humanoid_lab.envs import signals

NQ = 7 + NU
NSITE, NGEOM = 5, 6
SENSOR_ADR = {"gyro": 0, "quat": 3, "linvel": 7}
SHAPES = {
    "qpos": (NQ,),
    "qvel": (NV,),
    "qacc": (NV,),
    "sensordata": (10,),
    "actuator_force": (NU,),
    "xpos": (NBODY, 3),
    "site_xpos": (NSITE, 3),
    "geom_xpos": (NGEOM, 3),
    "subtree_com": (NBODY, 3),
    "cdof": (NV, 6),
}


def _unit_quat(rng, batch):
    q = rng.standard_normal(batch + (4,))
    return (q / np.linalg.norm(q, axis=-1, keepdims=True)).astype(np.float32)


def _data(batch, seed=0):
    rng = np.random.default_rng(seed)
    d = {k: rng.uniform(-1.0, 1.0, batch + s).astype(np.float32) for k, s in SHAPES.items()}
    d["geom_xpos"][..., 2] = rng.uniform(0.0, 0.03, batch + (NGEOM,)).astype(np.float32)
    d["sensordata"][..., 3:7] = _unit_quat(rng, batch)
    d["qpos"][..., 3:7] = _unit_quat(rng, batch)
    return d


def _info(batch, seed=1, yaw_ref=True):
    rng = np.random.default_rng(seed)
    f = lambda *s: rng.uniform(-1.0, 1.0, batch + s).astype(np.float32)
    info = {"last_action": f(NU), "command": f(3), "phase": f() * 3.0}
    if yaw_ref:
        info["yaw_ref"] = f() * 6.0
    return info


def _ns(*arrays):
    return SimpleNamespace(**dict(zip(signals.DATA_FIELDS, arrays)))


def _consts(mode):
    return toy_consts("numpy" if mode == "numpy" else "jax")


def _check(fn, args, batch, mode):
    # atol covers contractions that sum to near zero (foot_linvel); numpy stays exact.
    assert_batched_matches_stacked(fn, args, batch, mode, atol=1e-6)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("batch", BATCHES, ids=str)
class TestBatchedEqualsStacked:
    def test_sensor_readers(self, batch, mode):
        d = _data(batch)
        base = (d["sensordata"], d["qpos"], d["qvel"])
        for adr in (SENSOR_ADR, {}):
            _check(lambda s, p, v, adr=adr: signals.quat(s, p, adr, 0), base, batch, mode)
            _check(lambda s, p, v, adr=adr: signals.gyro(s, v, adr, 0), base, batch, mode)
            _check(
                lambda s, p, v, adr=adr: signals.linvel(s, v, signals.quat(s, p, adr, 0), adr, 0), base, batch, mode
            )

    def test_foot_pos_base(self, batch, mode):
        c, d = _consts(mode), _data(batch)
        q = d["sensordata"][..., 3:7]

        def fn(s, x, qq):
            return signals.foot_pos_base(s, x, c.foot_site_ids, c.root_body, qq)

        assert np.shape(fn(d["site_xpos"], d["xpos"], q)) == batch + (c.n_feet, 3)
        _check(fn, (d["site_xpos"], d["xpos"], q), batch, mode)

    @pytest.mark.parametrize("pairs", [False, True])
    def test_readings(self, batch, mode, pairs):
        c = _consts(mode)
        if pairs:
            c = toy_replace(c, ankle_pair=(1, 2), knee_pair=(3, 4))
        d = _data(batch)
        _check(lambda *a: signals.readings(_ns(*a), c), [d[k] for k in signals.DATA_FIELDS], batch, mode)

    @pytest.mark.parametrize("yaw_ref", [True, False])
    def test_catalog(self, batch, mode, yaw_ref):
        c = _consts(mode)
        d, info = _data(batch), _info(batch, yaw_ref=yaw_ref)
        keys = sorted(info)

        def fn(*a):
            n = len(signals.DATA_FIELDS)
            r = signals.readings(_ns(*a[:n]), c)
            return signals.catalog(r, dict(zip(keys, a[n:])), c)

        _check(fn, [d[k] for k in signals.DATA_FIELDS] + [info[k] for k in keys], batch, mode)


@pytest.mark.parametrize("mode", MODES)
def test_the_frozen_phase_is_zeros_with_frozen_phase_zeros(mode):
    c = _consts(mode)
    # Two batch axes check that the zeros take the whole batch shape.
    d, info = _data((2, 3)), _info((2, 3))
    xp = jnp if mode == "jax" else np
    a = lambda x: xp.asarray(x)
    r = signals.readings(_ns(*[a(d[k]) for k in signals.DATA_FIELDS]), c)
    info = {k: a(v) for k, v in info.items()}
    zeros = signals.catalog(r, info, toy_replace(c, frozen_phase_zeros=True))["phase"]
    assert zeros.dtype == xp.float32
    assert np.array_equal(np.asarray(zeros), np.zeros((2, 3, 2 * c.n_feet), np.float32))
    cos_sin = np.asarray(signals.catalog(r, info, c)["phase"])
    leg = np.asarray(info["phase"])[..., None] + np.array([0.0, np.pi], np.float32)
    leg = np.fmod(leg + np.pi, 2 * np.pi) - np.pi
    np.testing.assert_allclose(cos_sin, np.concatenate([np.cos(leg), np.sin(leg)], -1), atol=1e-6)
    assert np.abs(cos_sin).max() > 0.1


def test_a_quarter_turn_about_z():
    h = np.sqrt(0.5, dtype=np.float32)
    q = np.array([h, 0.0, 0.0, h], np.float32)
    assert float(signals.base_yaw(q)) == pytest.approx(np.pi / 2, abs=1e-6)
    np.testing.assert_allclose(signals.quat_rotate(q, np.array([1.0, 0.0, 0.0], np.float32)), [0, 1, 0], atol=1e-6)
    np.testing.assert_allclose(signals.quat_rotate_inv(q, np.array([0.0, 1.0, 0.0], np.float32)), [1, 0, 0], atol=1e-6)
    # a yaw turn leaves gravity alone
    np.testing.assert_allclose(signals.gravity_body(q), [0, 0, -1], atol=1e-6)


def test_foot_pos_base_reads_the_site_in_the_base_frame():
    # base yawed 90 degrees: a site one metre along world +y sits straight ahead
    h = np.sqrt(0.5, dtype=np.float32)
    q = np.array([h, 0.0, 0.0, h], np.float32)
    xpos = np.zeros((3, 3), np.float32)
    xpos[1] = [0.5, -0.2, 0.8]  # the root body
    site_xpos = np.zeros((4, 3), np.float32)
    site_xpos[2] = xpos[1] + [0.0, 1.0, 0.0]
    site_xpos[3] = xpos[1] + [-1.0, 0.0, 0.1]
    got = signals.foot_pos_base(site_xpos, xpos, np.array([2, 3]), 1, q)
    assert got.shape == (2, 3)
    np.testing.assert_allclose(got, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.1]], atol=1e-6)


def test_a_tilt_moves_gravity_into_the_body_axes():
    a = np.float32(0.3)
    q = np.array([np.cos(a / 2), np.sin(a / 2), 0.0, 0.0], np.float32)  # roll by 0.3 about x
    g = signals.gravity_body(q)
    assert g[0] == pytest.approx(0.0, abs=1e-6)
    # The body frame sees the world turned back by -0.3, so gravity leans toward body -y.
    assert g[1] == pytest.approx(-np.sin(0.3), abs=1e-6)
    assert g[2] == pytest.approx(-np.cos(0.3), abs=1e-6)


def test_heading_err_wraps_across_pi_and_clips():
    ref = np.array([3.0, -3.0, 0.2, 1.0, -2.0], np.float32)
    yaw = np.array([-3.0, 3.0, 0.1, 0.0, 0.0], np.float32)
    got = signals.heading_err(ref, yaw)
    # +-6 wraps to the short way round (0.283 rad), then every value clips to +-1
    np.testing.assert_allclose(got, [6 - 2 * np.pi, -6 + 2 * np.pi, 0.1, 1.0, -1.0], atol=1e-5)
    assert np.all(np.abs(got) <= 1.0)


def test_wrap_angle_is_not_clipped():
    assert float(signals.wrap_angle(np.float32(3.0))) == pytest.approx(3.0, abs=1e-6)
    assert float(signals.wrap_angle(np.float32(2 * np.pi + 0.5))) == pytest.approx(0.5, abs=1e-5)


def test_foot_contact_is_true_when_any_geom_of_the_foot_is_low():
    c = toy_consts("numpy")
    z = np.full(NGEOM, 0.5, np.float32)
    geoms = np.zeros((NGEOM, 3), np.float32)
    geoms[:, 2] = z
    geoms[1, 2] = 0.0  # one geom of foot 0
    got = signals.foot_contact(geoms, c.foot_geom_ids, c.foot_geom_radius, c.foot_geom_member)
    assert got.tolist() == [True, False]


def test_the_catalog_keys_and_shapes():
    c, d = toy_consts("numpy"), _ns(*[_data((3,))[k] for k in signals.DATA_FIELDS])
    cat = signals.catalog(signals.readings(d, c), _info((3,)), c)
    sizes = {
        "gyro": 3, "gravity": 3, "joint_pos": NU, "joint_vel": NU, "last_action": NU, "linvel": 3,
        "height": 1, "actuator_force": NU, "contacts": 2, "command": 3, "heading_err": 1, "phase": 4,
    }
    assert {k: v.shape for k, v in cat.items()} == {k: (3, n) for k, n in sizes.items()}
    assert cat["contacts"].dtype == np.float32


def test_heading_err_is_zero_without_yaw_ref():
    c, d = toy_consts("numpy"), _ns(*[_data((3,))[k] for k in signals.DATA_FIELDS])
    cat = signals.catalog(signals.readings(d, c), _info((3,), yaw_ref=False), c)
    assert np.array_equal(cat["heading_err"], np.zeros((3, 1), np.float32))


def test_gather_concatenates_in_order_and_names_a_missing_component():
    cat = {"a": np.ones(2, np.float32), "b": np.zeros(3, np.float32)}
    assert signals.gather(cat, ["b", "a"]).tolist() == [0, 0, 0, 1, 1]
    with pytest.raises(KeyError, match="nope"):
        signals.gather(cat, ["a", "nope"])


def test_noise_goes_on_the_actor_vector_only():
    cat = {"a": np.ones(2, np.float32), "b": np.ones(3, np.float32)}
    noise = np.full(5, 0.5, np.float32)
    out = signals.observe(cat, ["a", "b"], ["b", "a"], noise=noise, obs_noise={"b": 0.2})
    np.testing.assert_allclose(out["state"], [1, 1, 1.1, 1.1, 1.1])
    assert out["privileged_state"].tolist() == [1, 1, 1, 1, 1]
    scales = signals.noise_scales(cat, ["a", "b"], {"b": 0.2})
    np.testing.assert_allclose(scales, [0, 0, 0.2, 0.2, 0.2])
    assert scales.dtype == np.float32


@pytest.mark.parametrize("use_jax", [False, True])
def test_observe_mirrors_last_where_the_flag_is_set(use_jax):
    xp = jnp if use_jax else np
    cat = {"a": xp.asarray([1.0, 2.0], dtype=xp.float32), "b": xp.asarray([3.0], dtype=xp.float32)}
    perm, sign = np.array([1, 0]), np.array([1.0, -1.0], np.float32)
    pperm, psign = np.array([0]), np.array([-1.0], np.float32)
    for flag, state, priv in ((False, [1, 2], [3]), (True, [2, -1], [-3])):
        mirror = (xp.asarray(flag), perm, sign, pperm, psign)
        out = signals.observe(cat, ["a"], ["b"], mirror=mirror)
        assert np.asarray(out["state"]).tolist() == state
        assert np.asarray(out["privileged_state"]).tolist() == priv


def test_observe_adds_noise_before_it_mirrors():
    # Mirroring first would give [2.1, -1].
    cat = {"a": np.array([1.0, 2.0], np.float32)}
    noise = np.array([0.5, 0.0], np.float32)
    perm, sign = np.array([1, 0]), np.array([1.0, -1.0], np.float32)
    mirror = (np.asarray(True), perm, sign, perm, sign)
    out = signals.observe(cat, ["a"], ["a"], noise=noise, obs_noise={"a": 0.2}, mirror=mirror)
    np.testing.assert_allclose(out["state"], [2.0, -1.1], rtol=1e-6)
    assert out["privileged_state"].tolist() == [2, -1]
