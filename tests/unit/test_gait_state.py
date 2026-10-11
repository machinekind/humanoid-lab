"""Unit tests for envs/gait_state.py: reseed_flags, clear_done, reset_state, pre_physics and update."""

import jax.numpy as jp
import numpy as np
import pytest
from ml_collections import config_dict
from task_toys import (
    BATCHES,
    MODES,
    assert_batched_matches_stacked,
    toy_consts,
    toy_replace,
)

from humanoid_lab.envs import gait_state, progress
from humanoid_lab.envs.gait_state import GAIT_TRACKER_KEYS, clear_done, reseed_flags
from humanoid_lab.envs.joystick import default_config
from humanoid_lab.rewards import compose

N = 3


def _info(xp):
    f32 = xp.float32
    info = {
        "progress_ema": xp.asarray([0.1, 0.2, 0.3], dtype=f32),
        "steps_since_cmd": xp.asarray([5, 6, 7], dtype=xp.int32),
        "yaw_ref": xp.asarray([1.0, 2.0, 3.0], dtype=f32),
        "command": xp.asarray([[0.5, 0.0, 0.0]] * N, dtype=f32),
    }
    for k in GAIT_TRACKER_KEYS:
        if k == "last_contact":
            info[k] = xp.asarray([[True, True]] * N)
        else:
            info[k] = xp.asarray([[0.3, 0.4]] * N, dtype=f32)
    # a (N, M, K) leaf checks the mask broadcast over two trailing axes
    info["air_dur_ema"] = xp.ones((N, 2, 2), dtype=f32)
    # (N, F, 3): each foot's last landing fractions, LANDING_SLOTS order
    info["land_frac"] = xp.reshape(xp.arange(1, N * 2 * 3 + 1, dtype=f32), (N, 2, 3)) / 10.0
    return info


def _reset_values(xp):
    return {
        "progress_ema": xp.asarray([0.9] * N, dtype=xp.float32),
        "yaw_ref": xp.asarray([-1.0] * N, dtype=xp.float32),
    }


@pytest.fixture(params=["numpy", "jax"])
def xp(request):
    return np if request.param == "numpy" else jp


def _done(xp):
    return xp.asarray([True, False, True])


def _same(a, b):
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def _check_untouched_except(before, after, changed):
    for k in before:
        assert after[k].dtype == before[k].dtype, k
        assert after[k].shape == before[k].shape, k
        if k not in changed:
            _same(after[k], before[k])


def test_progress_clears_done_rows_only(xp):
    before = _info(xp)
    after = clear_done(before, _done(xp), ("progress",), _reset_values(xp))
    _same(after["progress_ema"], np.asarray([0.9, 0.2, 0.9], dtype=np.float32))
    _same(after["steps_since_cmd"], [0, 6, 0])
    _check_untouched_except(before, after, {"progress_ema", "steps_since_cmd"})


def test_gait_zeros_every_tracker_of_done_rows_only(xp):
    before = _info(xp)
    after = clear_done(before, _done(xp), ("gait",), {})
    for k in GAIT_TRACKER_KEYS:
        b, a = np.asarray(before[k]), np.asarray(after[k])
        _same(a[0], np.zeros_like(b[0]))
        _same(a[2], np.zeros_like(b[2]))
        _same(a[1], b[1])
    _check_untouched_except(before, after, set(GAIT_TRACKER_KEYS))


def test_heading_rewrites_yaw_ref_of_done_rows_only(xp):
    before = _info(xp)
    after = clear_done(before, _done(xp), ("heading",), _reset_values(xp))
    _same(after["yaw_ref"], np.asarray([-1.0, 2.0, -1.0], dtype=np.float32))
    _check_untouched_except(before, after, {"yaw_ref"})


def test_land_zeroes_land_frac_of_done_rows_only(xp):
    before = _info(xp)
    after = clear_done(before, _done(xp), ("land",), {})
    b, a = np.asarray(before["land_frac"]), np.asarray(after["land_frac"])
    assert b[0].any() and b[2].any()
    _same(a[0], np.zeros((2, 3), dtype=np.float32))
    _same(a[2], np.zeros((2, 3), dtype=np.float32))
    _same(a[1], b[1])
    _check_untouched_except(before, after, {"land_frac"})


def test_no_flags_changes_nothing_and_does_not_mutate_the_input(xp):
    before = _info(xp)
    keys = set(before)
    after = clear_done(before, _done(xp), (), {})
    assert after is not before
    assert set(before) == keys
    _check_untouched_except(before, after, set())


def test_all_flags_together(xp):
    before = _info(xp)
    after = clear_done(before, _done(xp), gait_state.LAYERS, _reset_values(xp))
    _same(after["yaw_ref"], np.asarray([-1.0, 2.0, -1.0], dtype=np.float32))
    _same(after["steps_since_cmd"], [0, 6, 0])
    _same(np.asarray(after["feet_air_time"])[1], np.asarray(before["feet_air_time"])[1])
    _same(np.asarray(after["feet_air_time"])[0], [0.0, 0.0])
    _same(np.asarray(after["land_frac"])[2], np.zeros((2, 3), dtype=np.float32))
    _same(np.asarray(after["land_frac"])[1], np.asarray(before["land_frac"])[1])


def test_nothing_done_changes_nothing(xp):
    before = _info(xp)
    after = clear_done(before, xp.asarray([False] * N), gait_state.LAYERS, _reset_values(xp))
    _check_untouched_except(before, after, set())


def test_result_leaves_stay_in_the_input_namespace(xp):
    after = clear_done(_info(xp), _done(xp), gait_state.LAYERS, _reset_values(xp))
    want = np.ndarray if xp is np else type(jp.zeros(1))
    for k, v in after.items():
        assert isinstance(v, want), k


# -- reseed_flags ---------------------------------------------------------


def test_default_config_needs_no_layer():
    assert reseed_flags(default_config()) == ()


def test_progress_flag():
    cfg = default_config()
    cfg.no_progress.enable = True
    assert reseed_flags(cfg) == ("progress",)


@pytest.mark.parametrize("key", ["gait_symmetry", "gait_symmetry_income"])
def test_gait_flag(key):
    cfg = default_config()
    cfg.reward.scales[key] = 0.5
    assert reseed_flags(cfg) == ("gait",)


@pytest.mark.parametrize("block", ["state", "privileged"])
def test_heading_flag_from_either_obs_list(block):
    cfg = default_config()
    cfg.obs[block] = tuple(cfg.obs[block]) + ("heading_err",)
    assert reseed_flags(cfg) == ("heading",)


def test_privileged_only_heading_err_still_counts():
    cfg = default_config()
    assert "heading_err" not in cfg.obs.state
    cfg.obs.privileged = ("heading_err",)
    assert reseed_flags(cfg) == ("heading",)


@pytest.mark.parametrize("term", compose.LANDING_SLOTS)
def test_land_flag_needs_a_nonzero_landing_scale_and_pair_min(term):
    cfg = default_config()
    cfg.reward.scales[term] = 0.5
    assert reseed_flags(cfg) == ()
    cfg.reward[term].pair_min = True
    assert reseed_flags(cfg) == ("land",)
    cfg.reward.scales[term] = 0.0
    assert reseed_flags(cfg) == ()


def test_forward_lean_never_asks_for_the_land_layer():
    cfg = default_config()
    cfg.reward.scales.forward_lean = 1.0
    assert reseed_flags(cfg) == ()


def test_flags_come_in_layer_order():
    cfg = default_config()
    cfg.obs.state = tuple(cfg.obs.state) + ("heading_err",)
    cfg.reward.scales.gait_symmetry = -1.0
    cfg.no_progress.enable = True
    cfg.reward.scales.foot_reach = 1.0
    cfg.reward.foot_reach.pair_min = True
    assert reseed_flags(cfg) == ("progress", "gait", "heading", "land")
    assert reseed_flags(cfg) == gait_state.LAYERS


def test_partial_config_is_tolerated():
    import ml_collections

    assert reseed_flags(ml_collections.ConfigDict()) == ()
    assert reseed_flags({}) == ()
    # a reward block without the landing parameter blocks
    assert reseed_flags({"reward": {"scales": {"step_length": 1.0}}}) == ()
    assert reseed_flags({"reward": {"scales": {"step_length": 1.0}, "step_length": {"pair_min": True}}}) == ("land",)


# -- reset_state, reset_offset, pre_physics ------------------------------------

_NU, _NF = 4, 2


def _cfg(**over):
    cfg = {
        "command": {},
        "symmetry": {"enable": False, "mirror_prob": 0.5},
        "no_progress": {"enable": False},
        "push": {
            "enable": False, "interval_steps": 5, "interval_steps_range": None, "vel": 0.4,
            "vel_z": 0.0, "ang_vel_rp": 0.0, "ang_vel_yaw": 0.0,
        },
    }
    for k, v in over.items():
        cfg[k].update(v)
    return config_dict.ConfigDict(cfg)


def _commands(batch):
    rng = np.random.default_rng(3)
    return rng.uniform(-1, 1, batch + (3,)).astype(np.float32)


@pytest.mark.parametrize("batch", BATCHES)
def test_reset_state_base_keys_and_dtypes(batch):
    info = gait_state.reset_state(_commands(batch), toy_consts(), _cfg())
    assert set(info) == {
        "command", "last_action", "last_last_action", "last_torque", "feet_air_time",
        "feet_contact_time", "swing_apex", "last_apex", "last_contact", "phase", "step_count",
        "steps_since_cmd",
    }
    for k in ("last_action", "last_last_action", "last_torque"):
        assert info[k].shape == batch + (_NU,) and info[k].dtype == np.float32
    for k in ("feet_air_time", "feet_contact_time", "swing_apex", "last_apex"):
        assert info[k].shape == batch + (_NF,) and info[k].dtype == np.float32
    assert info["last_contact"].dtype == bool and info["last_contact"].shape == batch + (_NF,)
    assert info["phase"].dtype == np.float32 and info["phase"].shape == batch
    for k in ("step_count", "steps_since_cmd"):
        assert info[k].dtype == np.int32 and info[k].shape == batch
    for k, v in info.items():
        if k != "command":
            assert not np.asarray(v).any(), k


def test_reset_state_jax_dtypes():
    import jax.numpy as jp

    info = gait_state.reset_state(jp.asarray(_commands((3,))), toy_consts("jax"), _cfg())
    assert info["phase"].dtype == jp.float32 and info["step_count"].dtype == jp.int32
    assert info["last_contact"].dtype == jp.bool_


def test_reset_state_gates():
    cmd = _commands((2,))
    consts = toy_replace(toy_consts(), heading_obs=True, gait_dur_ema_on=True)
    yaw = np.array([0.1, 0.2], np.float32)
    info = gait_state.reset_state(cmd, consts, _cfg(), yaw=yaw)
    assert {"yaw_ref", "air_dur_ema", "stance_dur_ema"} <= set(info)
    np.testing.assert_array_equal(info["yaw_ref"], yaw)
    assert info["air_dur_ema"].shape == (2, _NF) and not info["air_dur_ema"].any()
    base = gait_state.reset_state(cmd, toy_consts(), _cfg())
    for k in (
        "yaw_ref", "air_dur_ema", "stance_dur_ema", "push_countdown", "mirror", "progress_ema",
        "cmd_target",
    ):
        assert k not in base
    off = gait_state.reset_state(cmd, toy_consts(), _cfg(command={"ramp": {"lin": 0.0, "yaw": 0.0}}))
    assert "cmd_target" not in off
    _same(off["command"], cmd)
    cd = np.array([4, 6], np.int32)
    info = gait_state.reset_state(cmd, toy_consts(), _cfg(), push_countdown=cd)
    np.testing.assert_array_equal(info["push_countdown"], cd)
    info = gait_state.reset_state(cmd, toy_consts(), _cfg(no_progress={"enable": True}))
    np.testing.assert_allclose(info["progress_ema"], progress.cmd_speed(cmd))


_RAMP = {"ramp": {"lin": 0.5, "yaw": 1.0}}


def test_reset_state_with_the_ramp_serves_zero_and_keeps_the_sample_as_the_target():
    cmd = _commands((3,))
    info = gait_state.reset_state(cmd, toy_consts(), _cfg(command=_RAMP))
    assert list(info)[:2] == ["command", "cmd_target"]
    _same(info["command"], np.zeros_like(cmd))
    _same(info["cmd_target"], cmd)
    info = gait_state.reset_state(
        cmd, toy_consts(), _cfg(command=_RAMP, no_progress={"enable": True})
    )
    # The meter seeds from the served command, zero at reset.
    _same(info["progress_ema"], np.zeros(3, np.float32))


def test_reset_state_mirror_is_u_below_prob():
    u = np.array([0.1, 0.49, 0.5, 0.9], np.float32)
    info = gait_state.reset_state(
        _commands((4,)), toy_consts(), _cfg(symmetry={"enable": True}), mirror_u=u
    )
    np.testing.assert_array_equal(info["mirror"], [True, True, False, False])
    assert info["mirror"].dtype == bool


def test_reset_offset():
    noise = np.array([-1.0, 0.5, 1.0], np.float32)
    np.testing.assert_array_equal(gait_state.reset_offset(noise, 0.0), np.zeros(3))
    np.testing.assert_array_equal(gait_state.reset_offset(noise, 0.2), noise * 0.2)


def _pre_inputs(batch, cfg):
    n = int(np.prod(batch, dtype=int))
    rng = np.random.default_rng(5)
    info = {
        "step_count": (np.arange(n) + 2).reshape(batch).astype(np.int32),
        "mirror": (np.arange(n) % 2 == 0).reshape(batch),
        "push_countdown": (np.arange(n) % 3 - 1).reshape(batch).astype(np.int32),
    }
    draws = {
        "dir": rng.uniform(-1, 1, batch + (2,)).astype(np.float32),
        "next_countdown": (np.arange(n) + 4).reshape(batch).astype(np.int32),
    }
    action = rng.uniform(-6, 6, batch + (_NU,)).astype(np.float32)
    return info, draws, action


def _sym_consts(xp_name):
    perm = np.array([1, 0, 3, 2])
    sign = np.array([1.0, -1.0, 1.0, -1.0], np.float32)
    consts = toy_consts(xp_name)
    return toy_replace(consts, act_perm=perm, act_sign=sign)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("batch", BATCHES)
@pytest.mark.parametrize("ranged", [False, True])
def test_pre_physics_batched_equals_stacked(batch, mode, ranged):
    cfg = _cfg(
        symmetry={"enable": True},
        push={"enable": True, "interval_steps_range": [3, 9] if ranged else None,
              "vel_z": 0.3, "ang_vel_rp": 0.5, "ang_vel_yaw": 0.7},
    )
    info, draws, action = _pre_inputs(batch, cfg)
    draws["vel_z"] = np.full(batch, 0.1, np.float32)
    draws["ang"] = np.full(batch + (3,), 0.2, np.float32)
    consts = _sym_consts("numpy" if mode == "numpy" else "jax")

    def fn(step_count, mirror, countdown, dir_, nxt, vel_z, ang, action):
        i = {"step_count": step_count, "mirror": mirror, "push_countdown": countdown}
        d = {"dir": dir_, "next_countdown": nxt, "vel_z": vel_z, "ang": ang}
        out_info, act, ctrl, kick = gait_state.pre_physics(i, action, consts, cfg, d)
        return {"cd": out_info["push_countdown"], "action": act, "ctrl": ctrl, **kick}

    args = [info["step_count"], info["mirror"], info["push_countdown"], draws["dir"],
            draws["next_countdown"], draws["vel_z"], draws["ang"], action]
    assert_batched_matches_stacked(fn, args, batch, mode)


def test_pre_physics_mirror_clip_and_no_push():
    consts = _sym_consts("numpy")
    cfg = _cfg(symmetry={"enable": True})
    info = {"mirror": np.array([True, False]), "step_count": np.zeros(2, np.int32)}
    action = np.array([[1.0, 2.0, 3.0, 9.0], [1.0, 2.0, 3.0, 9.0]], np.float32)
    out_info, act, ctrl, kick = gait_state.pre_physics(info, action, consts, cfg, None)
    assert kick is None
    np.testing.assert_array_equal(act[0], [2.0, -1.0, 9.0, -3.0])
    np.testing.assert_array_equal(act[1], action[1])
    assert ctrl.max() <= 2.0 and ctrl.min() >= -2.0
    assert ctrl[1, 3] == 2.0
    assert "push_countdown" not in out_info
    # off: the action goes through untouched
    _, act, _, _ = gait_state.pre_physics(info, action, consts, _cfg(), None)
    np.testing.assert_array_equal(act, action)


def test_pre_physics_push_countdown():
    consts = toy_consts()
    cfg = _cfg(push={"enable": True, "interval_steps_range": [3, 9]})
    info = {"step_count": np.zeros(3, np.int32), "push_countdown": np.array([2, 0, -1], np.int32)}
    d = {"dir": np.ones((3, 2), np.float32), "next_countdown": np.array([8, 8, 8], np.int32)}
    out_info, _, _, kick = gait_state.pre_physics(info, np.zeros((3, _NU), np.float32), consts, cfg, d)
    np.testing.assert_array_equal(out_info["push_countdown"], [1, 8, 8])
    assert info["push_countdown"][0] == 2  # the input dict is not changed
    assert not kick["planar"][0].any() and kick["planar"][1].any()
    # fixed interval: no countdown key appears
    cfg = _cfg(push={"enable": True})
    out_info, _, _, kick = gait_state.pre_physics(
        {"step_count": np.array([4, 3], np.int32)}, np.zeros((2, _NU), np.float32), consts, cfg,
        {"dir": np.ones((2, 2), np.float32)},
    )
    assert "push_countdown" not in out_info
    assert kick["planar"][0].any() and not kick["planar"][1].any()


# -- update --------------------------------------------------------------------

_RESAMPLE = 5
_GRACE_SEC = 0.05  # 2.5 control steps at the toy dt of 0.02
_ON_FLAGS = {
    "heading_obs": True,
    "gait_dur_ema_on": True,
    "knee_qidx": np.array([0, 1]),
    "knee_stance_on": True,
    "gait_symmetry_on": True,
    "gait_symmetry_income_on": True,
    "feet_apex_min_on": True,
}


def _ucfg(on=False):
    """default_config with a short resample and grace window. on turns the
    no-progress cut on with a large p_max, and adds a pure draw."""
    cfg = default_config()
    cfg.command.resample_steps = _RESAMPLE
    cfg.no_progress.grace_sec = _GRACE_SEC
    if on:
        cfg.no_progress.enable = True
        cfg.no_progress.p_max = 0.5
        cfg.command.pure_slow_prob = 0.3
    return cfg


def _uconsts(xp_name="numpy", on=False):
    consts = toy_consts(xp_name)
    return toy_replace(consts, **_ON_FLAGS) if on else consts


def _update_inputs(batch, on=False, seed=7):
    """A flat name -> array dict: 'r.*' readings, 'i.*' info, 'd.*' draws, and 'action'.

    steps_since_cmd is a mix of worlds in the grace window, armed worlds and
    worlds due for a resample.
    """
    rng = np.random.default_rng(seed)
    n = int(np.prod(batch, dtype=int))

    def f(*shape, lo=-1.0, hi=1.0):
        return rng.uniform(lo, hi, batch + shape).astype(np.float32)

    def b(*shape):
        return rng.uniform(size=batch + shape) > 0.5

    gravity = f(3, lo=-0.3, hi=0.3)
    gravity[..., 2] = -0.9
    vals = {
        "r.gyro": f(3), "r.linvel": f(3), "r.gravity": gravity, "r.qpos_act": f(_NU),
        "r.qvel_act": f(_NU), "r.qacc_act": f(_NU), "r.actuator_force": f(_NU, lo=-40.0, hi=40.0),
        "r.foot_clearance": f(_NF, lo=0.0, hi=0.1), "r.foot_linvel": f(_NF, 3),
        "r.base_height": f(lo=0.3, hi=0.8), "r.contact": b(_NF), "r.yaw": f(lo=-3.0, hi=3.0),
        "i.command": f(3), "i.phase": f(lo=-3.0, hi=3.0), "i.last_torque": f(_NU),
        "i.last_action": f(_NU), "i.last_last_action": f(_NU),
        "i.feet_air_time": f(_NF, lo=0.0, hi=0.5), "i.feet_contact_time": f(_NF, lo=0.0, hi=0.5),
        "i.last_contact": b(_NF), "i.swing_apex": f(_NF, lo=0.0, hi=0.1),
        "i.last_apex": f(_NF, lo=0.0, hi=0.1),
        "i.step_count": (np.arange(n) + 10).reshape(batch).astype(np.int32),
        "i.steps_since_cmd": ((np.arange(n) * 2 + 4) % 7).reshape(batch).astype(np.int32),
        "action": f(_NU),
        "d.vx": f(lo=-0.8, hi=0.8), "d.vy": f(lo=-0.6, hi=0.6), "d.wz": f(lo=-0.6, hi=0.6),
        "d.zero_on": b(),
    }  # fmt: skip
    if on:
        vals.update({
            "i.yaw_ref": f(lo=-3.0, hi=3.0), "i.air_dur_ema": f(_NF, lo=0.1, hi=0.5),
            "i.stance_dur_ema": f(_NF, lo=0.1, hi=0.5), "i.progress_ema": f(lo=-0.2, hi=0.8),
            "d.pure_slow_on": b(), "d.pure_slow_vx": f(lo=0.1, hi=0.35), "d.u_term": f(lo=0.0, hi=1.0),
        })  # fmt: skip
    return vals


def _split(vals):
    """(info, r, action, draws) from a flat _update_inputs dict."""
    sub = {p: {k[2:]: v for k, v in vals.items() if k.startswith(p + ".")} for p in "rid"}
    u_term = sub["d"].pop("u_term", None)
    return sub["i"], sub["r"], vals["action"], {"command": sub["d"], "u_term": u_term}


def _update(vals, consts, cfg):
    info, r, action, draws = _split(vals)
    return gait_state.update(info, r, action, consts, cfg, draws)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("batch", BATCHES, ids=str)
@pytest.mark.parametrize("on", [False, True], ids=["off", "on"])
def test_update_batched_equals_stacked(on, batch, mode):
    vals = _update_inputs(batch, on)
    names = sorted(vals)
    consts = _uconsts("numpy" if mode == "numpy" else "jax", on)
    cfg = _ucfg(on)

    def fn(*arrays):
        return _update(dict(zip(names, arrays)), consts, cfg)

    assert_batched_matches_stacked(fn, [vals[k] for k in names], batch, mode, atol=1e-5)


@pytest.mark.parametrize("on", [False, True], ids=["off", "on"])
def test_update_keeps_the_info_keys_shapes_and_dtypes(on):
    """reset_state's info goes through update with the same keys, shapes and
    dtypes. With every feature off no key is added and aux holds 'fall' only."""
    consts, cfg = _uconsts("numpy", on), _ucfg(on)
    vals = _update_inputs((3,), on)
    _, r, action, draws = _split(vals)
    info = gait_state.reset_state(
        vals["i.command"], consts, cfg, yaw=vals["r.yaw"] if on else None
    )
    out, rewards, done, aux = gait_state.update(info, r, action, consts, cfg, draws)
    assert list(out) == list(info)
    for k in info:
        assert out[k].shape == info[k].shape and out[k].dtype == info[k].dtype, k
    assert done.dtype == bool and done.shape == (3,)
    assert set(aux) == ({"fall", "cut", "progress_ratio"} if on else {"fall"})
    want = compose.compute_rewards(r, info, action, np.zeros((3, _NF), bool), r["contact"], consts, cfg)[0]
    assert list(rewards) == list(want)


@pytest.mark.parametrize("xp_name", ["numpy", "jax"])
def test_update_does_not_mutate_its_inputs(xp_name):
    consts, cfg = _uconsts(xp_name, True), _ucfg(True)
    vals = _update_inputs((3,), True)
    if xp_name == "jax":
        vals = {k: jp.asarray(v) for k, v in vals.items()}
    info, r, action, draws = _split(vals)
    before = {k: np.array(v) for k, v in vals.items()}
    keys = (list(info), list(r), list(draws["command"]))
    out, _, _, _ = gait_state.update(info, r, action, consts, cfg, draws)
    assert out is not info
    assert (list(info), list(r), list(draws["command"])) == keys
    for k, v in vals.items():
        _same(v, before[k])


def test_yaw_ref_follows_the_yaw_only_above_the_turn_threshold():
    t = progress.TURN_WZ_THRESHOLD
    vals = _update_inputs((4,), True)
    vals["i.command"] = np.array(
        [[0.5, 0.0, 0.0], [0.5, 0.0, t - 0.01], [0.0, 0.0, t + 0.01], [0.0, 0.0, -0.4]], np.float32
    )
    vals["i.yaw_ref"] = np.full(4, 1.0, np.float32)
    vals["r.yaw"] = np.full(4, 0.3, np.float32)
    out, _, _, _ = _update(vals, _uconsts("numpy", True), _ucfg(True))
    np.testing.assert_array_equal(out["yaw_ref"], np.array([1.0, 1.0, 0.3, 0.3], np.float32))


def test_a_landing_pays_the_completed_swings_air_time():
    """Foot 0 was airborne for 0.3 s and lands this step. The reward reads
    the air time from before the update, and only then is it reset."""
    vals = _update_inputs((1,), False)
    vals["i.command"] = np.array([[0.5, 0.0, 0.0]], np.float32)
    vals["i.feet_air_time"] = np.array([[0.3, 0.0]], np.float32)
    vals["i.last_contact"] = np.array([[False, True]])
    vals["r.contact"] = np.array([[True, True]])
    out, rewards, _, _ = _update(vals, _uconsts(), _ucfg())
    # (0.3 - min_air_time 0.1) for the landing foot. A reset air time would pay -0.1.
    np.testing.assert_allclose(rewards["feet_air_time"], [0.2], rtol=1e-6)
    _same(out["feet_air_time"], np.zeros((1, _NF), np.float32))


def test_last_apex_records_the_peak_before_swing_apex_resets():
    """Foot 0 lands with a 0.07 m peak. Foot 1 is still airborne and climbs
    from 0.05 to 0.09."""
    vals = _update_inputs((1,), False)
    vals["i.feet_air_time"] = np.array([[0.3, 0.2]], np.float32)
    vals["i.last_contact"] = np.array([[False, False]])
    vals["r.contact"] = np.array([[True, False]])
    vals["i.swing_apex"] = np.array([[0.07, 0.05]], np.float32)
    vals["i.last_apex"] = np.array([[0.01, 0.02]], np.float32)
    vals["r.foot_clearance"] = np.array([[0.0, 0.09]], np.float32)
    out, _, _, _ = _update(vals, _uconsts(), _ucfg())
    _same(out["last_apex"], np.array([[0.07, 0.02]], np.float32))
    _same(out["swing_apex"], np.array([[0.0, 0.09]], np.float32))


def test_a_due_resample_restarts_the_counter_and_reseeds_progress_ema():
    """World 0 reaches resample_steps on this step and world 1 does not. The
    cut meter scores the command that drove the step, before the resample."""
    vals = _update_inputs((2,), True)
    old = np.array([[0.6, 0.0, 0.0], [0.4, 0.0, 0.0]], np.float32)
    vals["i.command"] = old
    vals["i.steps_since_cmd"] = np.array([_RESAMPLE - 1, 0], np.int32)
    cfg, consts = _ucfg(True), _uconsts("numpy", True)
    info, r, action, draws = _split(vals)
    out, _, _, aux = gait_state.update(info, r, action, consts, cfg, draws)
    new = progress.command_from_draws(draws["command"], cfg.command)
    _same(out["command"][0], new[0])
    _same(out["command"][1], old[1])
    _same(out["steps_since_cmd"], [0, 1])
    alpha = consts.dt / cfg.no_progress.ema_sec
    served = progress.served(r["linvel"][..., :2], r["gyro"][..., 2], old)
    ema = (1.0 - alpha) * info["progress_ema"] + alpha * served
    _same(out["progress_ema"][0], progress.cmd_speed(new)[0])
    _same(out["progress_ema"][1], ema[1])
    np.testing.assert_allclose(aux["progress_ratio"], ema / progress.cmd_speed(old), rtol=1e-6)


def test_the_cut_is_u_below_the_hazard_when_armed():
    """Zero served speed and a zero meter give a ratio of 0, so the hazard is
    p_max (0.5). The cut is u < p_max only when the command is above the
    deadband and the grace window has passed."""
    cfg, consts = _ucfg(True), _uconsts("numpy", True)
    vals = _update_inputs((5,), True)
    walk, stand = [0.5, 0.0, 0.0], [0.03, 0.0, 0.0]
    vals["i.command"] = np.array([walk, walk, walk, stand, walk], np.float32)
    # After the increment: 4, 4, 4, 4 steps (past the 2.5 step grace) and 1 step (inside it).
    vals["i.steps_since_cmd"] = np.array([3, 3, 3, 3, 0], np.int32)
    vals["d.u_term"] = np.array([0.3, 0.7, 0.49, 0.0, 0.0], np.float32)
    vals["i.progress_ema"] = np.zeros(5, np.float32)
    vals["r.linvel"] = np.zeros((5, 3), np.float32)
    vals["r.gyro"] = np.zeros((5, 3), np.float32)
    vals["r.base_height"] = np.full(5, 0.6, np.float32)
    vals["r.gravity"] = np.tile(np.array([0.0, 0.0, -1.0], np.float32), (5, 1))
    _, _, done, aux = _update(vals, consts, cfg)
    _same(aux["fall"], [False] * 5)
    _same(aux["cut"], [True, False, True, False, False])
    _same(done, aux["cut"])
    _same(aux["progress_ratio"], np.zeros(5, np.float32))


def test_the_cut_follows_the_deadband_constant():
    """The same stand-ish command arms at the default deadband and not at a
    wider one, which is how command.deadband reaches the cut."""
    cfg = _ucfg(True)
    vals = _update_inputs((1,), True)
    vals["i.command"] = np.array([[0.1, 0.0, 0.0]], np.float32)
    vals["i.steps_since_cmd"] = np.array([3], np.int32)
    vals["d.u_term"] = np.array([0.0], np.float32)
    vals["i.progress_ema"] = np.zeros(1, np.float32)
    vals["r.linvel"] = np.zeros((1, 3), np.float32)
    _, _, _, aux = _update(vals, _uconsts("numpy", True), cfg)
    _same(aux["cut"], [True])
    wide = toy_replace(_uconsts("numpy", True), deadband=0.15)
    _, _, _, aux = _update(vals, wide, cfg)
    _same(aux["cut"], [False])


@pytest.mark.parametrize("deadband", [0.05, 0.15])
def test_the_phase_is_frozen_below_the_deadband(deadband):
    consts = toy_replace(_uconsts(), deadband=deadband)
    cfg = _ucfg()
    vals = _update_inputs((3,), False)
    speeds = [deadband - 0.01, deadband + 0.05, 0.6]
    vals["i.command"] = np.array([[s, 0.0, 0.0] for s in speeds], np.float32)
    vals["i.phase"] = np.array([0.5, 0.5, 0.5], np.float32)
    out, _, _, _ = _update(vals, consts, cfg)
    assert out["phase"][0] == np.float32(0.5)
    step = progress.phase_dt(vals["i.command"], consts.cmd_vmax, cfg.gait.freq, consts.dt, deadband)
    assert step[0] == 0.0 and (step[1:] > 0).all()
    np.testing.assert_allclose(out["phase"][1:], 0.5 + step[1:], rtol=1e-6)


def test_the_phase_wraps_into_minus_pi_to_pi():
    vals = _update_inputs((1,), False)
    vals["i.command"] = np.array([[0.8, 0.0, 0.0]], np.float32)
    vals["i.phase"] = np.array([np.pi - 0.01], np.float32)
    out, _, _, _ = _update(vals, _uconsts(), _ucfg())
    assert -np.pi <= out["phase"][0] < -np.pi + 0.5


def test_the_duration_emas_fold_the_completed_modes():
    """Foot 0 lands after a 0.3 s swing and foot 1 lifts off after a 0.4 s
    stance. Each EMA takes its completed duration with weight alpha."""
    consts, cfg = _uconsts("numpy", True), _ucfg(True)
    alpha = cfg.reward.gait_symmetry_alpha
    vals = _update_inputs((1,), True)
    vals["i.feet_air_time"] = np.array([[0.3, 0.0]], np.float32)
    vals["i.feet_contact_time"] = np.array([[0.0, 0.4]], np.float32)
    vals["i.last_contact"] = np.array([[False, False]])
    vals["r.contact"] = np.array([[True, False]])
    vals["i.air_dur_ema"] = np.array([[0.2, 0.2]], np.float32)
    vals["i.stance_dur_ema"] = np.array([[0.3, 0.3]], np.float32)
    out, _, _, _ = _update(vals, consts, cfg)
    np.testing.assert_allclose(out["air_dur_ema"], [[(1 - alpha) * 0.2 + alpha * 0.3, 0.2]], rtol=1e-6)
    np.testing.assert_allclose(out["stance_dur_ema"], [[0.3, (1 - alpha) * 0.3 + alpha * 0.4]], rtol=1e-6)


def test_the_timers_and_histories_advance():
    vals = _update_inputs((1,), False)
    vals["i.feet_air_time"] = np.array([[0.0, 0.1]], np.float32)
    vals["i.feet_contact_time"] = np.array([[0.2, 0.0]], np.float32)
    vals["i.last_contact"] = np.array([[True, False]])
    vals["r.contact"] = np.array([[True, False]])
    out, _, _, _ = _update(vals, _uconsts(), _ucfg())
    np.testing.assert_allclose(out["feet_air_time"], [[0.0, 0.12]], rtol=1e-6)
    np.testing.assert_allclose(out["feet_contact_time"], [[0.22, 0.0]], rtol=1e-6)
    _same(out["last_contact"], vals["r.contact"])
    _same(out["last_last_action"], vals["i.last_action"])
    _same(out["last_action"], vals["action"])
    _same(out["last_torque"], vals["r.actuator_force"])
    _same(out["step_count"], vals["i.step_count"] + 1)


def _ramp_ucfg(on=False):
    cfg = _ucfg(on)
    cfg.command.ramp.lin = 0.5
    cfg.command.ramp.yaw = 1.0
    return cfg


def test_update_with_the_ramp_resamples_into_cmd_target_and_slews_every_row():
    """World 0 is due for a resample and world 1 is not. The target changes
    on world 0 only. The command moves toward the target on both."""
    vals = _update_inputs((2,), False)
    vals["i.steps_since_cmd"] = np.array([_RESAMPLE - 1, 0], np.int32)
    vals["i.command"] = np.array([[0.0, 0.0, 0.0], [0.3, -0.2, 0.1]], np.float32)
    old_target = np.array([[0.5, 0.5, 0.5], [-0.6, 0.2, 0.1001]], np.float32)
    vals["i.cmd_target"] = old_target
    cfg, consts = _ramp_ucfg(), _uconsts()
    info, r, action, draws = _split(vals)
    out, _, _, _ = gait_state.update(info, r, action, consts, cfg, draws)
    new = progress.command_from_draws(draws["command"], cfg.command)
    _same(out["cmd_target"][0], new[0])
    _same(out["cmd_target"][1], old_target[1])
    lim = np.array([0.5, 0.5, 1.0], np.float32) * consts.dt
    want = vals["i.command"] + np.clip(out["cmd_target"] - vals["i.command"], -lim, lim)
    np.testing.assert_allclose(out["command"], want, rtol=1e-6, atol=1e-7)
    # World 1's wz is within one step of its target and lands on it.
    np.testing.assert_allclose(out["command"][1, 2], old_target[1, 2], rtol=1e-6)
    _same(out["steps_since_cmd"], [0, 1])


def test_the_ramp_reseeds_progress_ema_from_the_slewed_command():
    vals = _update_inputs((2,), True)
    vals["i.steps_since_cmd"] = np.array([_RESAMPLE - 1, 0], np.int32)
    vals["i.cmd_target"] = vals["i.command"] * 0.5
    cfg, consts = _ramp_ucfg(True), _uconsts("numpy", True)
    out, _, _, _ = _update(vals, consts, cfg)
    _same(out["progress_ema"][0], progress.cmd_speed(out["command"])[0])


def test_update_with_the_ramp_keeps_the_reset_keys():
    consts, cfg = _uconsts(), _ramp_ucfg()
    vals = _update_inputs((3,), False)
    _, r, action, draws = _split(vals)
    info = gait_state.reset_state(vals["i.command"], consts, cfg)
    out, _, _, _ = gait_state.update(info, r, action, consts, cfg, draws)
    assert list(out) == list(info)
    assert "cmd_target" in out
    for k in info:
        assert out[k].shape == info[k].shape and out[k].dtype == info[k].dtype, k


# -- land_frac -------------------------------------------------------------------

_LAND_FLAGS = {
    "knee_qidx": np.array([0, 1]),
    "step_length_on": True,
    "foot_reach_on": True,
    "knee_extend_on": True,
    "forward_lean_on": True,
    "land_frac_on": True,
}


def _land_ucfg():
    """_ucfg with the four kept terms on and pair_min on the three landing terms."""
    cfg = _ucfg()
    rcfg = cfg.reward
    for t in compose.KEPT_TERMS:
        rcfg.scales[t] = 1.0
    rcfg.step_length.pair_min = True
    rcfg.foot_reach.update(target_a=0.05, target_b=0.15, pair_min=True)
    rcfg.knee_extend.update(flex_lo=0.1, flex_hi=0.6, td_min_air=0.1, td_mult=0.5, pair_min=True)
    rcfg.forward_lean.update(max_deg=5.0, vx_max=1.0, sigma=0.05)
    assert compose.land_frac_on(rcfg)
    return cfg


def _land_uconsts(xp_name="numpy"):
    return toy_replace(toy_consts(xp_name), **_LAND_FLAGS)


def _land_inputs(batch, seed=7):
    """_update_inputs plus the feet's base-frame position and a previous land_frac."""
    vals = _update_inputs(batch, False, seed)
    rng = np.random.default_rng(seed + 1)
    vals["r.foot_pos_base"] = rng.uniform(-0.2, 0.4, batch + (_NF, 3)).astype(np.float32)
    vals["i.land_frac"] = rng.uniform(0.0, 1.0, batch + (_NF, 3)).astype(np.float32)
    vals["i.command"][..., 0] = np.abs(vals["i.command"][..., 0])  # mostly forward
    return vals


def test_reset_state_seeds_land_frac_only_when_on():
    cmd = _commands((3,))
    info = gait_state.reset_state(cmd, _land_uconsts(), _land_ucfg())
    assert info["land_frac"].shape == (3, _NF, 3) and info["land_frac"].dtype == np.float32
    assert not info["land_frac"].any()
    info = gait_state.reset_state(jp.asarray(cmd), _land_uconsts("jax"), _land_ucfg())
    assert info["land_frac"].dtype == jp.float32
    # The terms on without pair_min: no land_frac.
    no_pair = toy_replace(_land_uconsts(), land_frac_on=False)
    assert "land_frac" not in gait_state.reset_state(cmd, no_pair, _ucfg())
    assert "land_frac" not in gait_state.reset_state(cmd, toy_consts(), _ucfg())


def test_update_records_land_only_on_first_contact_feet():
    vals = _land_inputs((4,))
    vals["i.feet_air_time"] = np.array([[0.3, 0.0], [0.3, 0.3], [0.0, 0.0], [0.0, 0.3]], np.float32)
    vals["r.contact"] = np.array([[True, True], [False, True], [True, True], [True, False]])
    vals["i.last_contact"] = np.zeros((4, _NF), bool)
    first_contact = np.array([[True, False], [False, True], [False, False], [False, False]])
    consts, cfg = _land_uconsts(), _land_ucfg()
    info, r, action, draws = _split(vals)
    _, _, land = compose.compute_rewards(r, info, action, first_contact, r["contact"], consts, cfg)
    out, rewards, _, _ = gait_state.update(info, r, action, consts, cfg, draws)
    want = np.where(first_contact[..., None], land, vals["i.land_frac"])
    _same(out["land_frac"], want)
    assert not np.array_equal(want, vals["i.land_frac"])
    assert list(rewards)[-4:] == list(compose.KEPT_TERMS)


def test_update_with_land_keeps_the_reset_keys():
    consts, cfg = _land_uconsts(), _land_ucfg()
    vals = _land_inputs((3,))
    _, r, action, draws = _split(vals)
    info = gait_state.reset_state(vals["i.command"], consts, cfg)
    out, _, _, _ = gait_state.update(info, r, action, consts, cfg, draws)
    assert list(out) == list(info)
    for k in info:
        assert out[k].shape == info[k].shape and out[k].dtype == info[k].dtype, k


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("batch", BATCHES, ids=str)
def test_update_land_batched_equals_stacked(batch, mode):
    if mode == "jax" and tuple(batch) != (3,):
        pytest.skip("eager jax runs at (3,) only; test_shared_math_jit covers jit at (2, 3)")
    vals = _land_inputs(batch)
    names = sorted(vals)
    consts = _land_uconsts("numpy" if mode == "numpy" else "jax")
    cfg = _land_ucfg()

    def fn(*arrays):
        return _update(dict(zip(names, arrays)), consts, cfg)

    assert_batched_matches_stacked(fn, [vals[k] for k in names], batch, mode, atol=1e-5)
