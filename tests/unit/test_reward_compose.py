"""The shared reward terms and their composition across array backends and
batch shapes.

Every term and compute_rewards run on numpy and eager jax at batch shapes
(), (3,) and (2, 3). The batched call must equal the per-item calls stacked.
tests/integration/test_shared_math_jit.py runs the same checks under jit.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from task_toys import (
    BATCHES,
    MODES,
    assert_batched_matches_stacked,
    toy_consts,
    toy_replace,
)

from humanoid_lab.envs import joystick, progress
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
        "target": f(lo=-0.1, hi=0.4),
        "feet_x": f(F, lo=-0.3, hi=0.4),
        "frac": f(F, lo=0.0, hi=1.0),
        "contact_c": rng.uniform(size=batch + (F,)) > 0.5,
        "knee": f(F, lo=0.0, hi=1.2),
        "gx": f(lo=-0.3, hi=0.3),
        "vx": f(lo=-0.5, hi=1.2),
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
        "step_length": (terms.step_length, (a["feet_x"], a["contact_b"], a["target"], 1.5)),
        "step_length_pair": (
            terms.step_length,
            (a["feet_x"], a["contact_b"], a["target"], 1.0, True, a["frac"]),
        ),
        "foot_reach": (terms.foot_reach, (a["feet_x"], a["contact_b"], a["target"], -0.05, 2.0)),
        "foot_reach_pair": (
            terms.foot_reach,
            (a["feet_x"], a["contact_b"], a["target"], 0.0, 2.0, True, a["frac"]),
        ),
        "knee_extend": (
            terms.knee_extend,
            (a["knee"], a["feet_x"], a["contact_c"], a["contact_b"], a["dur"], 0.2, 1.0, 0.0, 0.2, 0.5),
        ),
        "knee_extend_pair": (
            terms.knee_extend,
            (a["knee"], a["feet_x"], a["contact_c"], a["contact_b"], a["dur"], 0.2, 1.0, 0.05, 0.2, 0.5)
            + (True, a["frac"]),
        ),
        "forward_lean": (terms.forward_lean, (a["gx"], a["vx"], 8.0, 1.0, 0.05)),
    }


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("batch", BATCHES, ids=str)
@pytest.mark.parametrize("name", sorted(_cases(_arrays((2,)))))
def test_term_batched_equals_stacked(name, batch, mode):
    fn, args = _cases(_arrays(batch))[name]
    assert_batched_matches_stacked(fn, args, batch, mode)
    if mode == "numpy" and all(np.ndim(a) == 0 and np.asarray(a).dtype.kind == "f" for a in args):
        # A call on float scalars alone also takes Python floats.
        np.testing.assert_allclose(fn(*(float(a) for a in args)), fn(*args), rtol=1e-6)


# -- landing and lean terms -----------------------------------------------------


def _np(*xs):
    return [np.asarray(x, np.float32) for x in xs]


def test_step_length_floors_the_target():
    foot_x, target = _np([0.03, 0.0], 0.01)
    fc = np.array([True, False])
    pay, frac = terms.step_length(foot_x, fc, target)
    pay_floor, _ = terms.step_length(foot_x, fc, np.float32(terms.STEP_LENGTH_MIN_TARGET))
    assert float(pay) == pytest.approx(0.03 / 0.05) == pytest.approx(float(pay_floor))
    np.testing.assert_allclose(frac, [0.6, 0.0], rtol=1e-6)


def test_step_length_clips_and_raises_to_power():
    foot_x, target = _np([0.3, -0.1], 0.2)
    fc = np.array([True, True])
    pay, frac = terms.step_length(foot_x, fc, target)
    np.testing.assert_array_equal(frac, [1.0, 0.0])
    assert float(pay) == 1.0
    foot_x = np.asarray([0.1, 0.0], np.float32)
    _, frac2 = terms.step_length(foot_x, fc, target, power=2.0)
    np.testing.assert_allclose(frac2, [0.25, 0.0], rtol=1e-6)


def test_step_length_pair_min_reads_the_other_foot():
    foot_x, target, last = _np([0.2, 0.0], 0.2, [0.9, 0.3])
    fc = np.array([True, False])
    pay, frac = terms.step_length(foot_x, fc, target, pair_min=True, last_frac=last)
    # foot 0 lands with frac 1; the other foot's last landing earned 0.3
    assert float(pay) == pytest.approx(0.3)
    np.testing.assert_array_equal(frac, [1.0, 0.0])  # recorded before the pair-min
    pay_off, _ = terms.step_length(foot_x, fc, target)
    assert float(pay_off) == 1.0


@pytest.mark.parametrize("name", ["step_length", "foot_reach", "knee_extend"])
def test_landing_terms_need_two_feet(name):
    # knee_extend checks the foot count only under pair_min.
    x, b, target = np.zeros(3, np.float32), np.zeros(3, bool), np.float32(0.2)
    args, kwargs = {
        "step_length": ((x, b, target), {}),
        "foot_reach": ((x, b, target), {}),
        "knee_extend": ((x, x, b, b, x, 0.2, 1.0), {"pair_min": True, "last_frac": x}),
    }[name]
    with pytest.raises(ValueError, match=f"{name} needs two feet"):
        getattr(terms, name)(*args, **kwargs)


def test_foot_reach_floor_and_x_min():
    fc = np.array([True, True])
    foot_x = np.asarray([0.01, 0.0], np.float32)
    pay, frac = terms.foot_reach(foot_x, fc, np.float32(-0.1), power=1.0)
    np.testing.assert_allclose(frac, [0.5, 0.0], rtol=1e-6)  # target floored at 0.02
    assert float(pay) == pytest.approx(0.5)
    foot_x = np.asarray([0.1, -0.1], np.float32)
    _, frac = terms.foot_reach(foot_x, fc, np.float32(0.2), x_min=-0.1, power=1.0)
    np.testing.assert_allclose(frac, [2.0 / 3.0, 0.0], rtol=1e-6)
    _, frac = terms.foot_reach(foot_x, fc, np.float32(0.2), x_min=-0.1)
    np.testing.assert_allclose(frac, [4.0 / 9.0, 0.0], rtol=1e-6)


def _knee(knee, foot_x, contact, first, air, **kw):
    args = (*_np(knee, foot_x), np.array(contact), np.array(first), *_np(air))
    return terms.knee_extend(*args, 0.2, 1.0, **kw)


def test_knee_extend_pays_late_swing():
    # ext = (1.0 - q) / 0.8: foot 0 at 1.0 (bent), foot 1 at 0.6 -> 0.5
    pay, td = _knee([1.0, 0.6], [0.1, 0.1], [False, False], [False, False], [0.0, 0.0])
    assert float(pay) == pytest.approx(0.5)
    np.testing.assert_array_equal(td, [0.0, 0.0])
    # in contact or behind x_split pays nothing
    pay, _ = _knee([0.2, 0.2], [0.1, -0.1], [True, False], [False, False], [0.0, 0.0])
    assert float(pay) == 0.0
    pay, _ = _knee([0.2, 0.2], [0.1, 0.1], [False, False], [False, False], [0.0, 0.0], x_split=0.15)
    assert float(pay) == 0.0


def test_knee_extend_touchdown_needs_a_real_step():
    kw = {"td_mult": 2.0}
    # a real step: in the air >= 0.2 s, landing ahead of the base
    pay, td = _knee([0.2, 0.2], [0.1, 0.1], [True, True], [True, False], [0.3, 0.3], **kw)
    np.testing.assert_allclose(td, [1.0, 0.0])
    assert float(pay) == pytest.approx(2.0)
    # a tap (too short) and a landing behind record 0
    pay, td = _knee([0.2, 0.2], [0.1, -0.1], [True, True], [True, True], [0.1, 0.3], **kw)
    np.testing.assert_array_equal(td, [0.0, 0.0])
    assert float(pay) == 0.0


def test_knee_extend_td_mult_scales_only_the_touchdown():
    args = ([0.6, 0.6], [0.1, 0.1], [True, False], [True, False], [0.3, 0.0])
    late = 0.5  # foot 1 in late swing
    for mult in (0.0, 1.0, 3.0):
        pay, td = _knee(*args, td_mult=mult)
        assert float(pay) == pytest.approx(late + mult * 0.5)
        np.testing.assert_allclose(td, [0.5, 0.0])


def test_knee_extend_records_td_ext_before_the_pair_min():
    last = np.asarray([0.9, 0.1], np.float32)
    pay, td = _knee([0.2, 0.2], [0.1, 0.1], [True, True], [True, False], [0.3, 0.3],
                    td_mult=1.0, pair_min=True, last_frac=last)  # fmt: skip
    np.testing.assert_allclose(td, [1.0, 0.0])
    assert float(pay) == pytest.approx(0.1)


def test_forward_lean_at_zero_degrees_and_target():
    gx = np.asarray([0.1, -0.2, 0.05], np.float32)
    vx = np.asarray([0.5, 1.0, 3.0], np.float32)
    np.testing.assert_allclose(terms.forward_lean(gx, vx, 0.0, 1.0, 0.1), np.exp(-gx**2 / 0.1), rtol=1e-6)
    vx = np.asarray([-0.5, 0.5, 3.0], np.float32)
    target = np.sin(np.radians(10.0) * np.clip(vx / 1.0, 0.0, 1.0))
    out = terms.forward_lean(target.astype(np.float32), vx, 10.0, 1.0, 0.1)
    np.testing.assert_allclose(out, 1.0, rtol=1e-6)
    off = terms.forward_lean(np.zeros(3, np.float32), vx, 10.0, 1.0, 0.1)
    np.testing.assert_allclose(off, np.exp(-target**2 / 0.1), rtol=1e-5)


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
          "foot_clearance", "foot_linvel", "base_height", "foot_pos_base"]  # fmt: skip
INFO_KEYS = ["command", "phase", "last_torque", "last_action", "last_last_action", "feet_air_time",
             "feet_contact_time", "last_contact", "swing_apex", "last_apex", "air_dur_ema",
             "stance_dur_ema", "land_frac"]  # fmt: skip


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
        # Drawn last, so the values above do not move.
        "foot_pos_base": f(nf, 3, lo=-0.2, hi=0.4), "land_frac": f(nf, 3, lo=0.0, hi=1.0),
    }  # fmt: skip


def _style_consts(xp_name, **flags):
    return toy_replace(toy_consts(xp_name), knee_qidx=np.array([0, 1]), **flags)


def _run(consts, cfg, vals):
    """compute_rewards' (rewards, fall, land), or (rewards, fall) when land is
    None: assert_batched_matches_stacked cannot stack None leaves."""
    r = {k: vals[k] for k in R_KEYS}
    info = {k: vals[k] for k in INFO_KEYS}
    rewards, fall, land = compose.compute_rewards(
        r, info, vals["action"], vals["first_contact"], vals["contact"], consts, cfg
    )
    return (rewards, fall) if land is None else (rewards, fall, land)


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
    [{}, {"tracking_relative": True, "tracking_far_weight": 0.3, "tracking_product": True, "shaping_tracking_gate": True}],
    ids=["default", "all_switches"],
)
def test_compute_rewards_batched_equals_stacked(reward, batch, mode):
    vals = _compose_inputs(batch)
    names = sorted(vals)
    consts = _style_consts("numpy" if mode == "numpy" else "jax", orientation_tol=0.01, **ALL_STYLE)
    fn = _stacked_fn(consts, _cfg(**reward), names)
    assert_batched_matches_stacked(fn, [vals[n] for n in names], batch, mode, atol=1e-5)


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


def test_missing_body_pairs_give_zeros_of_the_batch_shape():
    for batch in BATCHES:
        rewards, _ = _run(toy_consts("numpy"), _cfg(), _compose_inputs(batch))
        for key in ("feet_distance", "knee_distance"):
            assert rewards[key].shape == batch
            assert rewards[key].dtype == np.float32
            assert not rewards[key].any()


def test_body_pairs_use_the_readings():
    vals = _compose_inputs((3,))
    feet_sep = np.array([0.3, 0.1, 0.55], np.float32)
    knee_sep = np.array([0.1, 0.25, 0.0], np.float32)
    r = {k: vals[k] for k in R_KEYS} | {"feet_sep": feet_sep, "knee_sep": knee_sep}
    info = {k: vals[k] for k in INFO_KEYS}
    cfg = _cfg()
    cfg.reward.feet_distance_range = (0.2, 0.5)
    cfg.reward.knee_distance_range = (0.05, 0.2)
    rewards, _, _ = compose.compute_rewards(r, info, vals["action"], vals["first_contact"], vals["contact"], toy_consts("numpy"), cfg)
    # 1 inside the band. Outside it, (1 + exp(-100 * excursion)) / 2.
    np.testing.assert_allclose(rewards["feet_distance"], [1.0, (1 + np.exp(-10)) / 2, (1 + np.exp(-5)) / 2], rtol=1e-5)
    np.testing.assert_allclose(rewards["knee_distance"], [1.0, (1 + np.exp(-5)) / 2, (1 + np.exp(-5)) / 2], rtol=1e-5)


def test_flipping_tracking_product_changes_values_and_not_the_order():
    vals = _compose_inputs((3,))
    off, _ = _run(toy_consts("numpy"), _cfg(tracking_product=False), vals)
    on, _ = _run(toy_consts("numpy"), _cfg(tracking_product=True), vals)
    assert list(on) == list(off)
    np.testing.assert_array_equal(on["tracking_lin_vel"], off["tracking_lin_vel"] * off["tracking_ang_vel"])
    assert not np.array_equal(on["tracking_lin_vel"], off["tracking_lin_vel"])


# v_full is above most toy command speeds, so the scale is unclipped on most rows.
LIFT = {"v_full": 2.0, "min_frac": 0.25}


def _lift_cfg(**reward):
    cfg = _cfg(**reward)
    cfg.gait.lift.v_full = LIFT["v_full"]
    cfg.gait.lift.min_frac = LIFT["min_frac"]
    return cfg


def test_lift_off_matches_no_lift_block():
    vals = _compose_inputs((3,))
    consts = _style_consts("numpy", **ALL_STYLE)
    off, _ = _run(consts, _cfg(), vals)
    bare = _cfg()
    del bare.gait["lift"]
    assert "lift" not in bare.gait
    plain, _ = _run(consts, bare, vals)
    assert list(off) == list(plain)
    for k in plain:
        np.testing.assert_array_equal(off[k], plain[k], err_msg=k)


def test_lift_scales_feet_phase_and_feet_apex_only():
    vals = _compose_inputs((4,))
    # A stand-still row, a slow row under the floor, one mid row, one past v_full.
    vals["command"] = np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]], np.float32)
    consts = _style_consts("numpy", **ALL_STYLE)
    cfg = _lift_cfg()
    on, _ = _run(consts, cfg, vals)
    off, _ = _run(consts, _cfg(), vals)
    cmd = vals["command"]
    ls = progress.lift_scale(cmd, LIFT["v_full"], LIFT["min_frac"])
    np.testing.assert_array_equal(ls, [0.25, 0.25, 0.5, 1.0])
    moving = progress.cmd_speed(cmd) > consts.deadband
    target, _ = progress.gait_targets(vals["phase"], cfg.gait.duty, cfg.gait.swing_height)
    np.testing.assert_array_equal(
        on["feet_phase"], terms.feet_phase(vals["foot_clearance"], target * ls[..., None], cfg.reward.phase_sigma) * moving
    )
    np.testing.assert_array_equal(
        on["feet_apex"],
        terms.feet_apex(vals["swing_apex"], vals["first_contact"], cfg.reward.apex_target * ls[..., None]) * moving,
    )
    # feet_apex_min keeps the unscaled target. So does every other term.
    for k in set(on) - {"feet_phase", "feet_apex"}:
        np.testing.assert_array_equal(on[k], off[k], err_msg=k)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("batch", BATCHES, ids=str)
def test_compute_rewards_with_lift_batched_equals_stacked(batch, mode):
    if mode == "jax" and tuple(batch) != (3,):
        pytest.skip("eager jax runs at (3,) only; test_shared_math_jit covers jit at (2, 3)")
    vals = _compose_inputs(batch)
    names = sorted(vals)
    consts = _style_consts("numpy" if mode == "numpy" else "jax", **ALL_STYLE)
    fn = _stacked_fn(consts, _lift_cfg(), names)
    assert_batched_matches_stacked(fn, [vals[n] for n in names], batch, mode, atol=1e-5)


# -- landing and lean terms in compose -------------------------------------------

KEPT_FLAGS = {f"{t}_on": t for t in compose.KEPT_TERMS}
ALL_KEPT = dict.fromkeys(KEPT_FLAGS, True)


def _kept_cfg(pair_min=True, target="clock", scales=compose.KEPT_TERMS):
    """default_config with the given kept terms on and every required parameter set."""
    cfg = _cfg()
    rcfg = cfg.reward
    for t in scales:
        rcfg.scales[t] = 1.0
    rcfg.step_length.update(target=target, target_a=0.05, target_b=0.2, pair_min=pair_min)
    rcfg.foot_reach.update(target_a=0.05, target_b=0.15, pair_min=pair_min)
    rcfg.knee_extend.update(flex_lo=0.1, flex_hi=0.6, td_min_air=0.1, td_mult=0.5, pair_min=pair_min)
    rcfg.forward_lean.update(max_deg=5.0, vx_max=1.0, sigma=0.05)
    return cfg


def _kept_consts(xp_name, cfg, style=False):
    """Gates that follow cfg: a kept term is on when its scale is nonzero."""
    flags = {f"{t}_on": bool(cfg.reward.scales[t]) for t in compose.KEPT_TERMS}
    return _style_consts(
        xp_name, land_frac_on=compose.land_frac_on(cfg.reward), **flags, **(ALL_STYLE if style else {})
    )


def _fwd_rows(vals):
    """Rows 0 and 1 walk forward. Row 2 is below FWD_MIN_VX and row 3 backs up."""
    vals["command"] = np.array([[0.6, 0.1, 0.0], [1.4, 0.0, 0.2], [0.08, 0.0, 0.0], [-0.4, 0.0, 0.0]], np.float32)
    return vals


def test_land_frac_on_reads_config_only():
    assert not compose.land_frac_on(joystick.default_config().reward)
    assert not compose.land_frac_on({})
    assert not compose.land_frac_on({"scales": {"step_length": 1.0}})
    assert not compose.land_frac_on({"scales": {"step_length": 0.0}, "step_length": {"pair_min": True}})
    # forward_lean has no landing slot.
    assert not compose.land_frac_on({"scales": {"forward_lean": 1.0}, "forward_lean": {"pair_min": True}})
    for t in compose.LANDING_SLOTS:
        assert compose.land_frac_on({"scales": {t: 0.5}, t: {"pair_min": True}})
    assert compose.land_frac_on(_kept_cfg().reward)
    assert not compose.land_frac_on(_kept_cfg(pair_min=False).reward)


def test_kept_keys_come_after_the_style_terms_in_kept_order():
    vals = _compose_inputs((3,))
    cfg = _kept_cfg()
    rewards, _, _ = _run(_kept_consts("numpy", cfg, style=True), cfg, vals)
    assert list(rewards) == BASE_KEYS + list(STYLE_KEYS.values()) + list(compose.KEPT_TERMS)
    rewards, _, _ = _run(_kept_consts("numpy", cfg), cfg, vals)
    assert list(rewards) == BASE_KEYS + list(compose.KEPT_TERMS)
    for t in compose.KEPT_TERMS:
        one = _kept_cfg(scales=(t,))
        out = _run(_kept_consts("numpy", one), one, vals)
        assert list(out[0]) == BASE_KEYS + [t]


def test_kept_terms_off_leave_the_rewards_unchanged():
    vals = _compose_inputs((3,))
    consts = _style_consts("numpy", **ALL_STYLE)
    plain = _run(consts, _cfg(), vals)
    # Parameters set but every scale 0: the gates are off.
    off_cfg = _kept_cfg(scales=())
    off = _run(_kept_consts("numpy", off_cfg, style=True), off_cfg, vals)
    assert len(off) == 2  # land is None
    assert list(off[0]) == list(plain[0])
    for k in plain[0]:
        np.testing.assert_array_equal(off[0][k], plain[0][k], err_msg=k)


def _direct(vals, cfg, consts):
    """The landing fractions of the three slot terms from direct terms.* calls."""
    rcfg = cfg.reward
    cmd = vals["command"]
    vx = cmd[..., 0]
    foot_x = vals["foot_pos_base"][..., 0]
    last = vals["land_frac"]
    fc = vals["first_contact"]
    p = rcfg.step_length
    if p.target == "clock":
        target = vx / (2.0 * progress.clock_freq(progress.cmd_speed(cmd), consts.cmd_vmax, cfg.gait.freq))
    else:
        target = p.target_a + p.target_b * np.clip(vx, 0.0, None)
    pm = bool(p.pair_min)
    _, step_frac = terms.step_length(foot_x, fc, target, p.power, pm, last[..., 1] if pm else None)
    p = rcfg.foot_reach
    target = p.target_a + p.target_b * np.clip(vx, 0.0, None)
    _, reach_frac = terms.foot_reach(foot_x, fc, target, p.x_min, p.power, pm, last[..., 0] if pm else None)
    p = rcfg.knee_extend
    _, td_ext = terms.knee_extend(
        vals["qpos_act"][..., consts.knee_qidx], foot_x, vals["contact"], fc, vals["feet_air_time"],
        p.flex_lo, p.flex_hi, p.x_split, p.td_min_air, p.td_mult, pm, last[..., 2] if pm else None,
    )  # fmt: skip
    return {"foot_reach": reach_frac, "step_length": step_frac, "knee_extend": td_ext}


@pytest.mark.parametrize("target", ["clock", "linear"])
@pytest.mark.parametrize("pair_min", [False, True], ids=["plain", "pair_min"])
def test_kept_terms_pay_only_on_forward_rows(pair_min, target):
    vals = _fwd_rows(_compose_inputs((4,)))
    # On every row foot 0 lands far ahead after a long swing with a straight
    # knee, so each landing term earns a nonzero fraction.
    vals["foot_pos_base"][..., 0] = np.array([1.0, 0.0], np.float32)
    vals["first_contact"][:] = np.array([True, False])
    vals["contact"][:] = np.array([True, False])
    vals["feet_air_time"][:] = 0.5
    vals["qpos_act"][..., :2] = 0.0
    cfg = _kept_cfg(pair_min=pair_min, target=target)
    out = _run(_kept_consts("numpy", cfg), cfg, vals)
    for t in compose.KEPT_TERMS:
        assert out[0][t][:2].all(), t
        assert not out[0][t][2:].any(), t  # fwd is off on rows 2 and 3
    if pair_min:
        # The record ignores fwd: row 3 backs up and still records its fractions.
        assert out[2][3].any()


def test_a_disabled_landing_terms_slot_is_zero():
    vals = _fwd_rows(_compose_inputs((4,)))
    cfg = _kept_cfg(scales=("foot_reach", "forward_lean"))
    consts = _kept_consts("numpy", cfg)
    rewards, _, land = _run(consts, cfg, vals)
    assert list(rewards)[-2:] == ["foot_reach", "forward_lean"]
    fracs = _direct(vals, cfg, consts)
    np.testing.assert_allclose(land[..., 0], fracs["foot_reach"], rtol=1e-6)
    zeros = np.zeros((4, F), np.float32)
    np.testing.assert_array_equal(land[..., 1], zeros)
    np.testing.assert_array_equal(land[..., 2], zeros)


def test_pair_min_reads_the_other_foots_slot():
    """Foot 0 lands with every fraction at 1. Each slot of last holds a
    different value for foot 1, so a wrong slot or a missing flip shows."""
    vals = _fwd_rows(_compose_inputs((4,)))
    vals["foot_pos_base"][..., 0] = np.array([1.0, 0.0], np.float32)  # far ahead, foot 1 at the base
    vals["first_contact"][:] = np.array([True, False])
    vals["contact"][:] = np.array([True, False])
    vals["feet_air_time"][:] = 0.5
    vals["qpos_act"][..., :2] = 0.0  # knees straight: extension 1
    last = np.zeros((4, F, 3), np.float32)
    last[..., 1, :] = np.array([0.3, 0.4, 0.2], np.float32)  # foot 1: reach, step, knee
    last[..., 0, :] = 0.9  # foot 0's own record must not count
    vals["land_frac"] = last
    cfg = _kept_cfg()
    cfg.reward.knee_extend.td_mult = 1.0
    rewards, _, _ = _run(_kept_consts("numpy", cfg), cfg, vals)
    np.testing.assert_allclose(rewards["foot_reach"][:2], 0.3, rtol=1e-6)
    np.testing.assert_allclose(rewards["step_length"][:2], 0.4, rtol=1e-6)
    # Foot 1 is in contact, so no late-swing pay: the touchdown bonus alone.
    np.testing.assert_allclose(rewards["knee_extend"][:2], 0.2, rtol=1e-6)


def test_clock_target_follows_the_gait_clock():
    """At a speed past cmd_vmax the clock runs at freq[1], so the target is
    vx / (2 * freq[1])."""
    vals = _compose_inputs((1,))
    vals["command"] = np.array([[1.2, 0.0, 0.0]], np.float32)
    vals["first_contact"][:] = np.array([True, False])
    cfg = _kept_cfg(pair_min=False, scales=("step_length",))
    cfg.gait.freq = (1.0, 2.0)
    want_target = 1.2 / (2.0 * 2.0)
    vals["foot_pos_base"][..., 0] = np.array([0.5 * want_target, 0.0], np.float32)
    rewards, _ = _run(_kept_consts("numpy", cfg), cfg, vals)
    np.testing.assert_allclose(rewards["step_length"], [0.5], rtol=1e-5)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("batch", BATCHES, ids=str)
@pytest.mark.parametrize("pair_min", [False, True], ids=["plain", "pair_min"])
def test_compute_rewards_with_kept_terms_land_batched_equals_stacked(pair_min, batch, mode):
    if mode == "jax" and tuple(batch) != (3,):
        pytest.skip("eager jax runs at (3,) only; test_shared_math_jit covers jit at (2, 3)")
    vals = _compose_inputs(batch)
    names = sorted(vals)
    cfg = _kept_cfg(pair_min=pair_min)
    consts = _kept_consts("numpy" if mode == "numpy" else "jax", cfg, style=True)
    fn = _stacked_fn(consts, cfg, names)
    assert_batched_matches_stacked(fn, [vals[n] for n in names], batch, mode, atol=1e-5)


def test_the_config_is_read_at_call_time():
    """Turning shaping_tracking_gate on in the same cfg object gates the
    positive gait terms by the linear tracking kernel on the next call."""
    vals = _compose_inputs((3,))
    # Every row is in single stance, so the biped term pays.
    vals["contact"][:] = np.array([True, False])
    vals["last_contact"][:] = np.array([True, False])
    cfg = _cfg()
    first, _ = _run(toy_consts("numpy"), cfg, vals)
    cfg.reward.shaping_tracking_gate = True
    second, _ = _run(toy_consts("numpy"), cfg, vals)
    gated = ("feet_air_time", "feet_apex", "feet_air_time_biped")
    assert first["feet_apex"].all() and first["feet_air_time"].all() and first["feet_air_time_biped"].all()
    for k in first:
        want = first[k] * first["tracking_lin_vel"] if k in gated else first[k]
        np.testing.assert_allclose(second[k], want, rtol=1e-6, err_msg=k)


def test_orientation_tolerance_is_a_floor_at_zero():
    vals = _compose_inputs((3,))
    plain, _ = _run(toy_consts("numpy"), _cfg(), vals)
    cone, _ = _run(toy_replace(toy_consts("numpy"), orientation_tol=0.02), _cfg(), vals)
    np.testing.assert_array_equal(cone["orientation"], np.maximum(plain["orientation"] - 0.02, 0.0))


def test_the_deadband_masks_moving_and_stand_still_terms():
    vals = _compose_inputs((3,))
    vals["command"] = np.zeros_like(vals["command"])
    still, _ = _run(toy_consts("numpy"), _cfg(), vals)
    vals["command"] = np.tile(np.array([0.8, 0.0, 0.0], np.float32), (3, 1))
    moving, _ = _run(toy_consts("numpy"), _cfg(), vals)
    for key in ("feet_slip", "feet_phase", "feet_landing", "feet_air_time"):
        assert not still[key].any(), key
        assert moving[key].any(), key
    for key in ("stand_still", "feet_contact_without_cmd"):
        assert still[key].any(), key
        assert not moving[key].any(), key


def test_total_reward_is_the_scaled_sum_and_clips_at_100():
    scales = {"a": 2.0, "b": -1.0}
    rewards = {"a": np.array([1.0, 1e6, -1e6], np.float32), "b": np.array([0.5, 0.0, 0.0], np.float32)}
    out = compose.total_reward(rewards, scales, 0.1)
    np.testing.assert_allclose(out, [(2.0 - 0.5) * 0.1, 100.0, -100.0], rtol=1e-6)
    out_jax = compose.total_reward({k: jnp.asarray(v) for k, v in rewards.items()}, scales, 0.1)
    np.testing.assert_allclose(np.asarray(out_jax), out, rtol=1e-6)
