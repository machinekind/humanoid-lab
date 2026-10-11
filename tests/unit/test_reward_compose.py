"""The shared reward terms and their composition across array backends and
batch shapes.

Every term, compute_rewards and total_reward run on numpy and eager jax at
batch shapes (), (3,) and (2, 3). The batched call must equal the per-item
calls stacked. tests/integration/test_shared_math_jit.py runs the same checks
under jit.
"""

import dataclasses

import jax.numpy as jnp
import numpy as np
import pytest
from task_toys import BATCHES, MODES, assert_batched_matches_stacked, toy_consts, toy_replace

from humanoid_lab.envs import joystick
from humanoid_lab.envs.constants import TaskConstants, xp_of
from humanoid_lab.rewards import compose, terms

NU = 6
F = 2


def _arrays(batch, seed=0):
    rng = np.random.default_rng(seed)

    def f(*shape, lo=-1.0, hi=1.0):
        return rng.uniform(lo, hi, batch + shape).astype(np.float32)

    return {
        "scalar": f(),
        "pos": f(lo=0.0, hi=2.0),
        "xy": f(2),
        "joints": f(NU),
        "feet": f(F),
        "feet_pos": f(F, lo=0.0, hi=1.0),
        "feet_xy": f(F, 2),
        "contact_f": (rng.uniform(size=batch + (F,)) > 0.5).astype(np.float32),
        "contact_b": rng.uniform(size=batch + (F,)) > 0.5,
        "fall": rng.uniform(size=batch) > 0.5,
        "dur": f(F, lo=0.0, hi=0.6) * (rng.uniform(size=batch + (F,)) > 0.2),
    }


def _cases(a):
    j, ft = a["joints"], a["feet"]
    return {
        "tracking_err_lin": (terms.tracking_err_lin, (a["xy"], a["xy"][..., ::-1].copy())),
        "tracking_err_ang": (terms.tracking_err_ang, (a["scalar"], a["pos"])),
        "tracking_kernel": (terms.tracking_kernel, (a["pos"], 0.25)),
        "tracking_rel_sigma": (terms.tracking_rel_sigma, (a["scalar"], 0.25, 0.1)),
        "tracking_far_blend": (terms.tracking_far_blend, (a["pos"], a["pos"], 0.3, 2.0)),
        "lin_vel_z": (terms.lin_vel_z, (a["scalar"],)),
        "ang_vel_xy": (terms.ang_vel_xy, (a["xy"],)),
        "orientation": (terms.orientation, (a["xy"],)),
        "torques": (terms.torques, (j,)),
        "torque_rate": (terms.torque_rate, (j, j[..., ::-1].copy())),
        "action_rate": (terms.action_rate, (j, j[..., ::-1].copy())),
        "action_accel": (terms.action_accel, (j, j[..., ::-1].copy(), j * 0.5)),
        "energy": (terms.energy, (j, j * 2.0)),
        "pose": (terms.pose, (j, j * 0.5, np.ones(NU, np.float32))),
        "feet_air_time": (terms.feet_air_time, (a["dur"], a["contact_f"], 0.0, 0.1)),
        "feet_air_time_cap": (terms.feet_air_time, (a["dur"], a["contact_f"], 0.3, 0.1)),
        "feet_slip": (terms.feet_slip, (a["feet_xy"], a["contact_f"])),
        "feet_apex": (terms.feet_apex, (a["feet_pos"], a["contact_f"], 0.05)),
        "feet_apex_min": (terms.feet_apex_min, (a["feet_pos"], a["feet_pos"] * 0.7, a["contact_f"], 0.05)),
        "feet_landing": (terms.feet_landing, (ft, a["feet_pos"], 0.3)),
        "feet_phase": (terms.feet_phase, (a["feet_pos"], a["feet_pos"] * 0.5, 0.1)),
        "stand_still": (terms.stand_still, (j, j * 0.5, j * 2.0, 0.2)),
        "termination": (terms.termination, (a["fall"],)),
        "torque_limit": (terms.torque_limit, (j * 40.0, np.full(NU, 30.0, np.float32), 0.8)),
        "pose_l1": (terms.pose_l1, (j, j * 0.5, np.ones(NU, np.float32))),
        "joint_pos_limits": (
            terms.joint_pos_limits,
            (j * 2.0, np.full(NU, -1.0, np.float32), np.full(NU, 1.0, np.float32)),
        ),
        "joint_vel": (terms.joint_vel, (j,)),
        "joint_acc": (terms.joint_acc, (j,)),
        "upward": (terms.upward, (a["scalar"],)),
        "distance_band": (terms.distance_band, (a["pos"], 0.2, 0.5)),
        "feet_contact_without_cmd": (terms.feet_contact_without_cmd, (a["contact_b"], a["scalar"])),
        "feet_air_time_biped": (
            terms.feet_air_time_biped,
            (a["dur"], a["dur"] * 0.5, a["contact_b"], 0.4),
        ),
        "knee_stance": (terms.knee_stance, (ft, a["contact_f"], 0.3)),
        "gait_symmetry": (terms.gait_symmetry, (a["dur"], a["dur"][..., ::-1].copy(), 0.1, 1.0)),
        "gait_symmetry_income": (
            terms.gait_symmetry_income,
            (a["dur"], a["dur"][..., ::-1].copy(), a["contact_b"], 0.1, 1.0),
        ),
    }


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("batch", BATCHES, ids=str)
@pytest.mark.parametrize("name", sorted(_cases(_arrays((2,)))))
def test_term_batched_equals_stacked(name, batch, mode):
    fn, args = _cases(_arrays(batch))[name]
    assert_batched_matches_stacked(fn, args, batch, mode)


def test_term_covers_every_public_function():
    covered = {n.removesuffix("_cap") for n in _cases(_arrays((2,)))}
    public = {n for n in dir(terms) if not n.startswith("_") and callable(getattr(terms, n))}
    assert public - {"xp_of"} <= covered


def test_xp_of_is_numpy_for_scalars_and_python_numbers():
    assert xp_of(0.4, 0.0) is np
    assert xp_of(None, True, 3, 0.5) is np
    assert xp_of() is np


def test_xp_of_follows_the_data_arrays():
    assert xp_of(1.0, jnp.ones(2)) is jnp or xp_of(1.0, jnp.ones(2)).__name__.startswith("jax")
    assert xp_of(np.ones(2)).__name__.startswith("array_api_compat.numpy")


def test_scalar_calls_still_work():
    assert terms.tracking_err_ang(0.4, 0.0) == pytest.approx(0.16)


def test_task_constants_is_frozen():
    consts = toy_consts("numpy")
    assert isinstance(consts, TaskConstants)
    with pytest.raises(dataclasses.FrozenInstanceError):
        consts.dt = 1.0
    assert consts.act_perm is None and consts.priv_sign is None


def test_toy_consts_actuator_stub():
    c = toy_consts("jax")
    out = c.actuator_model.ctrl_from_action(jnp.ones(4), c.default_pose, c.action_scale)
    np.testing.assert_allclose(np.asarray(out), np.asarray(c.default_pose) + 0.5)


# -- composition --------------------------------------------------------------

BASE_KEYS = [
    "tracking_lin_vel", "tracking_ang_vel", "lin_vel_z", "ang_vel_xy", "orientation", "torques",
    "torque_rate", "action_rate", "action_accel", "energy", "pose", "feet_air_time", "feet_slip",
    "feet_phase", "stand_still", "termination", "torque_limit", "feet_apex", "feet_landing",
    "pose_l1", "joint_pos_limits", "joint_vel", "joint_acc", "upward", "feet_distance",
    "knee_distance", "feet_contact_without_cmd", "feet_air_time_biped",
]  # fmt: skip
STYLE_KEYS = {
    "knee_stance_on": "knee_stance",
    "gait_symmetry_on": "gait_symmetry",
    "gait_symmetry_income_on": "gait_symmetry_income",
    "feet_apex_min_on": "feet_apex_min",
}
R_KEYS = ["gyro", "linvel", "gravity", "qpos_act", "qvel_act", "qacc_act", "actuator_force",
          "foot_clearance", "foot_linvel", "base_height"]  # fmt: skip
INFO_KEYS = ["command", "phase", "last_torque", "last_action", "last_last_action", "feet_air_time",
             "feet_contact_time", "last_contact", "swing_apex", "last_apex", "air_dur_ema",
             "stance_dur_ema"]  # fmt: skip


def _compose_inputs(batch, seed=1):
    """Toy readings, info, action and contact masks as a flat name -> array dict."""
    rng = np.random.default_rng(seed)
    nu, nf = 4, 2

    def f(*shape, lo=-1.0, hi=1.0):
        return rng.uniform(lo, hi, batch + shape).astype(np.float32)

    gravity = f(3, lo=-0.3, hi=0.3)
    gravity[..., 2] = -0.9
    command = f(3, lo=-1.0, hi=1.0)
    command[(slice(None),) * len(batch) + (slice(0, 1),)] *= 0.0  # a stand-still draw in the batch
    return {
        "gyro": f(3), "linvel": f(3), "gravity": gravity, "qpos_act": f(nu), "qvel_act": f(nu),
        "qacc_act": f(nu), "actuator_force": f(nu, lo=-40.0, hi=40.0),
        "foot_clearance": f(nf, lo=0.0, hi=0.1), "foot_linvel": f(nf, 3),
        "base_height": f(lo=0.2, hi=0.6),
        "command": command, "phase": f(lo=0.0, hi=6.0), "last_torque": f(nu),
        "last_action": f(nu), "last_last_action": f(nu), "feet_air_time": f(nf, lo=0.0, hi=0.5),
        "feet_contact_time": f(nf, lo=0.0, hi=0.5), "last_contact": rng.uniform(size=batch + (nf,)) > 0.5,
        "swing_apex": f(nf, lo=0.0, hi=0.1), "last_apex": f(nf, lo=0.0, hi=0.1),
        "air_dur_ema": f(nf, lo=0.1, hi=0.5), "stance_dur_ema": f(nf, lo=0.1, hi=0.5),
        "action": f(nu), "first_contact": rng.uniform(size=batch + (nf,)) > 0.5,
        "contact": rng.uniform(size=batch + (nf,)) > 0.5,
    }  # fmt: skip


def _style_consts(xp_name, **flags):
    return toy_replace(toy_consts(xp_name), knee_qidx=np.array([0, 1]), **flags)


def _run(consts, cfg, vals):
    r = {k: vals[k] for k in R_KEYS}
    info = {k: vals[k] for k in INFO_KEYS}
    return compose.compute_rewards(r, info, vals["action"], vals["first_contact"], vals["contact"], consts, cfg)


def _stacked_fn(consts, cfg, names):
    def fn(*arrays):
        return _run(consts, cfg, dict(zip(names, arrays)))

    return fn


ALL_STYLE = dict.fromkeys(STYLE_KEYS, True)


def _cfg(**reward):
    cfg = joystick.default_config()
    for k, v in reward.items():
        cfg.reward[k] = v
    return cfg


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("batch", BATCHES, ids=str)
@pytest.mark.parametrize(
    "reward",
    [{}, dict(tracking_relative=True, tracking_far_weight=0.3, tracking_product=True, shaping_tracking_gate=True)],
    ids=["default", "all_switches"],
)
def test_compute_rewards_batched_equals_stacked(reward, batch, mode):
    vals = _compose_inputs(batch)
    names = sorted(vals)
    consts = _style_consts("numpy" if mode == "numpy" else "jax", orientation_tol=0.01, **ALL_STYLE)
    fn = _stacked_fn(consts, _cfg(**reward), names)
    assert_batched_matches_stacked(fn, [vals[n] for n in names], batch, mode, atol=1e-5)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("batch", BATCHES, ids=str)
def test_total_reward_batched_equals_stacked(batch, mode):
    vals = _compose_inputs(batch)
    rewards, _ = _run(_style_consts("numpy", **ALL_STYLE), _cfg(), vals)
    keys = sorted(rewards)
    scales = joystick.default_config().reward.scales

    def fn(*arrays):
        return compose.total_reward(dict(zip(keys, arrays)), scales, 0.02)

    out = assert_batched_matches_stacked(fn, [np.asarray(rewards[k]) for k in keys], batch, mode, atol=1e-6)
    assert out is None


def test_key_order_is_the_golden_order():
    rewards, fall = _run(toy_consts("numpy"), _cfg(), _compose_inputs((3,)))
    assert list(rewards) == BASE_KEYS
    assert fall.shape == (3,)
    assert set(rewards) - {"termination"} <= set(joystick.default_config().reward.scales)


def test_style_keys_appear_only_when_their_gate_is_on():
    vals = _compose_inputs(())
    for flag, key in STYLE_KEYS.items():
        off, _ = _run(toy_consts("numpy"), _cfg(), vals)
        on, _ = _run(_style_consts("numpy", **{flag: True}), _cfg(), vals)
        assert key not in off
        assert list(on) == BASE_KEYS + [key]
    both, _ = _run(_style_consts("numpy", **ALL_STYLE), _cfg(), vals)
    assert list(both) == BASE_KEYS + list(STYLE_KEYS.values())


def test_zero_scale_terms_are_still_computed():
    cfg = _cfg()
    cfg.reward.scales.pose = 0.0
    rewards, _ = _run(toy_consts("numpy"), cfg, _compose_inputs((3,)))
    assert "pose" in rewards


def test_missing_body_pairs_give_zeros_of_the_batch_shape():
    for batch in BATCHES:
        rewards, _ = _run(toy_consts("numpy"), _cfg(), _compose_inputs(batch))
        for key in ("feet_distance", "knee_distance"):
            assert rewards[key].shape == batch
            assert rewards[key].dtype == np.float32
            assert not rewards[key].any()


def test_body_pairs_use_the_readings():
    vals = _compose_inputs((3,))
    r = {k: vals[k] for k in R_KEYS} | {"feet_sep": np.full(3, 0.5, np.float32), "knee_sep": np.full(3, 0.1, np.float32)}
    info = {k: vals[k] for k in INFO_KEYS}
    cfg = _cfg()
    rewards, _ = compose.compute_rewards(r, info, vals["action"], vals["first_contact"], vals["contact"], toy_consts("numpy"), cfg)
    np.testing.assert_array_equal(rewards["feet_distance"], terms.distance_band(r["feet_sep"], *cfg.reward.feet_distance_range))
    np.testing.assert_array_equal(rewards["knee_distance"], terms.distance_band(r["knee_sep"], *cfg.reward.knee_distance_range))


def test_flipping_tracking_product_changes_values_and_not_the_order():
    vals = _compose_inputs((3,))
    off, _ = _run(toy_consts("numpy"), _cfg(tracking_product=False), vals)
    on, _ = _run(toy_consts("numpy"), _cfg(tracking_product=True), vals)
    assert list(on) == list(off)
    np.testing.assert_array_equal(on["tracking_lin_vel"], off["tracking_lin_vel"] * off["tracking_ang_vel"])
    assert not np.array_equal(on["tracking_lin_vel"], off["tracking_lin_vel"])


def test_the_config_is_read_at_call_time():
    vals = _compose_inputs((3,))
    cfg = _cfg()
    first, _ = _run(toy_consts("numpy"), cfg, vals)
    cfg.reward.shaping_tracking_gate = True
    second, _ = _run(toy_consts("numpy"), cfg, vals)
    assert not np.array_equal(first["feet_apex"], second["feet_apex"])


def test_orientation_tolerance_is_a_floor_at_zero():
    vals = _compose_inputs((3,))
    plain, _ = _run(toy_consts("numpy"), _cfg(), vals)
    cone, _ = _run(toy_replace(toy_consts("numpy"), orientation_tol=0.02), _cfg(), vals)
    np.testing.assert_array_equal(cone["orientation"], np.maximum(plain["orientation"] - 0.02, 0.0))


def test_the_deadband_masks_moving_and_stand_still_terms():
    vals = _compose_inputs((3,))
    vals["command"] = np.zeros_like(vals["command"])
    rewards, _ = _run(toy_consts("numpy"), _cfg(), vals)
    for key in ("feet_slip", "feet_phase", "feet_landing", "feet_air_time"):
        assert not rewards[key].any()


def test_total_reward_is_the_scaled_sum_and_clips_at_100():
    scales = {"a": 2.0, "b": -1.0}
    rewards = {"a": np.array([1.0, 1e6, -1e6], np.float32), "b": np.array([0.5, 0.0, 0.0], np.float32)}
    out = compose.total_reward(rewards, scales, 0.1)
    np.testing.assert_allclose(out, [(2.0 - 0.5) * 0.1, 100.0, -100.0], rtol=1e-6)
    out_jax = compose.total_reward({k: jnp.asarray(v) for k, v in rewards.items()}, scales, 0.1)
    np.testing.assert_allclose(np.asarray(out_jax), out, rtol=1e-6)
