"""The command, gait clock and command-draw math across array backends.

Every function runs on numpy and eager jax at batch shapes (), (3,) and
(2, 3). The batched call must equal the per-item calls stacked.
tests/integration/test_shared_math_jit.py runs the same checks under jit.
"""

import itertools
import math

import jax.numpy as jnp
import numpy as np
import pytest
from task_toys import BATCHES, MODES, assert_batched_matches_stacked

from humanoid_lab.envs import progress

FREQ = (1.0, 2.5)
DT = 0.02
VMAX = np.float32(1.0)
DEADBAND = 0.15


def _commands(batch, seed=0):
    rng = np.random.default_rng(seed)
    cmd = rng.uniform(-1.0, 1.0, batch + (3,)).astype(np.float32)
    return cmd


def _phases(batch, seed=1):
    """The clock only advances, so phases are not negative (fmod keeps sign)."""
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 40.0, batch).astype(np.float32)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("batch", BATCHES)
def test_cmd_speed(batch, mode):
    assert_batched_matches_stacked(progress.cmd_speed, [_commands(batch)], batch, mode)


def test_cmd_speed_values():
    out = progress.cmd_speed(np.array([[0.3, 0.4, 0.0], [0.0, 0.0, -1.0]], np.float32))
    np.testing.assert_allclose(out, [0.5, progress.YAW_SPEED_WEIGHT], rtol=1e-6)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("batch", BATCHES, ids=str)
def test_fwd_mask_and_lift_scale(batch, mode):
    cmd = _commands(batch)
    assert_batched_matches_stacked(lambda c: progress.fwd_mask(c, DEADBAND), (cmd,), batch, mode)
    assert_batched_matches_stacked(lambda c: progress.lift_scale(c, 0.5, 0.25), (cmd,), batch, mode)


@pytest.mark.parametrize("deadband, threshold", [(0.05, 0.1), (0.15, 0.15)])
def test_fwd_mask_threshold_is_the_larger_of_the_floor_and_the_deadband(deadband, threshold):
    vx = np.array([threshold - 1e-3, threshold, threshold + 1e-3, -1.0], np.float32)
    cmd = np.stack([vx, np.zeros(4, np.float32), np.zeros(4, np.float32)], axis=-1)
    np.testing.assert_array_equal(progress.fwd_mask(cmd, deadband), [False, False, True, False])
    assert progress.FWD_MIN_VX == 0.1


def test_lift_scale_clips_into_min_frac_and_one():
    cmd = np.array([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0], [0.0, 0.3, 0.0], [2.0, 0.0, 0.0]], np.float32)
    np.testing.assert_allclose(progress.lift_scale(cmd, 0.5, 0.25), [0.25, 0.5, 0.6, 1.0], rtol=1e-6)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("batch", BATCHES)
def test_clock_freq(batch, mode):
    speed = np.abs(_commands(batch)[..., 0]) * 2.0
    assert_batched_matches_stacked(
        lambda s: progress.clock_freq(s, VMAX, FREQ), [speed], batch, mode
    )


def test_clock_freq_clamps_to_the_range():
    speed = np.array([0.0, 0.5, 1.0, 5.0], np.float32)
    np.testing.assert_allclose(progress.clock_freq(speed, VMAX, FREQ), [1.0, 1.75, 2.5, 2.5])


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("batch", BATCHES)
def test_phase_dt(batch, mode):
    assert_batched_matches_stacked(
        lambda c: progress.phase_dt(c, VMAX, FREQ, DT, DEADBAND), [_commands(batch)], batch, mode
    )


@pytest.mark.parametrize("xp", [np, jnp])
def test_phase_dt_is_frozen_at_the_deadband_and_moves_above_it(xp):
    below = xp.asarray(np.array([0.1, 0.0, 0.0], np.float32))
    above = xp.asarray(np.array([0.2, 0.0, 0.0], np.float32))
    at = xp.asarray(np.array([DEADBAND, 0.0, 0.0], np.float32))
    assert float(progress.phase_dt(below, VMAX, FREQ, DT, DEADBAND)) == 0.0
    assert float(progress.phase_dt(at, VMAX, FREQ, DT, DEADBAND)) == 0.0
    assert float(progress.phase_dt(above, VMAX, FREQ, DT, DEADBAND)) > 0.0
    # A turn counts toward speed: wz 0.6 * 0.3 = 0.18 > 0.15.
    turn = xp.asarray(np.array([0.0, 0.0, 0.6], np.float32))
    assert float(progress.phase_dt(turn, VMAX, FREQ, DT, DEADBAND)) > 0.0


def test_phase_dt_matches_the_formula_in_float32():
    cmd = np.array([0.5, 0.0, 0.0], np.float32)
    got = progress.phase_dt(cmd, VMAX, FREQ, DT, DEADBAND)
    frac = np.clip(np.float32(0.5) / VMAX, 0.0, 1.0)
    want = 2 * math.pi * DT * (FREQ[0] + (FREQ[1] - FREQ[0]) * frac)
    np.testing.assert_allclose(got, want, rtol=1e-6)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("batch", BATCHES)
def test_leg_phases(batch, mode):
    assert_batched_matches_stacked(progress.leg_phases, [_phases(batch)], batch, mode)


def test_leg_phases_are_antiphase_and_wrapped():
    out = progress.leg_phases(_phases((50,)))
    assert out.shape == (50, 2)
    assert out.min() >= -math.pi - 1e-6 and out.max() <= math.pi + 1e-6
    diff = np.abs(np.mod(out[:, 1] - out[:, 0] + math.pi, 2 * math.pi) - math.pi)
    np.testing.assert_allclose(diff, math.pi, atol=1e-4)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("batch", BATCHES)
def test_gait_targets(batch, mode):
    assert_batched_matches_stacked(
        lambda p: progress.gait_targets(p, 0.6, 0.08), [_phases(batch)], batch, mode
    )


def test_gait_targets_swing_is_the_complement_of_stance():
    rz, stance = progress.gait_targets(_phases((40,)), 0.6, 0.08)
    assert rz.shape == stance.shape == (40, 2)
    assert np.all(rz[stance] == 0.0)
    assert np.all(rz >= 0.0) and np.all(rz <= 0.08 + 1e-6)


# -- command from draws ----------------------------------------------------

CFG_ALL = {
    "pure_wz_prob": 0.2,
    "pure_vy_prob": 0.2,
    "pure_slow_prob": 0.2,
    "pure_fast_prob": 0.2,
    "pure_back_prob": 0.2,
    "straight_prob": 0.2,
}


class _Cfg(dict):
    """dict with the .get the real config has."""


def _draws(batch, on=(), zero=False, seed=3):
    rng = np.random.default_rng(seed)
    d = {
        "vx": rng.uniform(-1, 1, batch).astype(np.float32) + 2.0,
        "vy": rng.uniform(-1, 1, batch).astype(np.float32) + 3.0,
        "wz": rng.uniform(-1, 1, batch).astype(np.float32) + 4.0,
        "zero_on": np.full(batch, zero),
    }
    for name, _i, _p, rng_key in progress.COMMAND_DRAWS:
        d[f"{name}_on"] = np.full(batch, name in on)
        if rng_key:
            d[f"{name}_vx"] = rng.uniform(-1, 1, batch).astype(np.float32) + 5.0 + len(name)
    return d


def _stack(d):
    keys = sorted(d)
    return keys, [d[k] for k in keys]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("batch", BATCHES)
def test_command_from_draws_batched_matches_stacked(batch, mode):
    d = _draws(batch)
    rng = np.random.default_rng(9)
    for name in d:
        if name.endswith("_on"):
            d[name] = rng.random(batch) < 0.5
    cfg = _Cfg(CFG_ALL)
    keys, vals = _stack(d)
    assert_batched_matches_stacked(
        lambda *v: progress.command_from_draws(dict(zip(keys, v)), cfg), vals, batch, mode
    )


def test_stock_draws_pass_through():
    d = _draws((4,))
    out = progress.command_from_draws(d, _Cfg())
    np.testing.assert_array_equal(out, np.stack([d["vx"], d["vy"], d["wz"]], axis=-1))


def test_each_pure_draw_gives_a_clean_single_axis_command():
    cfg = _Cfg(CFG_ALL)
    d = _draws((4,), on=("pure_wz",))
    out = progress.command_from_draws(d, cfg)
    np.testing.assert_array_equal(out[:, :2], 0.0)
    np.testing.assert_array_equal(out[:, 2], d["wz"])
    d = _draws((4,), on=("pure_vy",))
    out = progress.command_from_draws(d, cfg)
    np.testing.assert_array_equal(out[:, [0, 2]], 0.0)
    np.testing.assert_array_equal(out[:, 1], d["vy"])
    for name in ("pure_slow", "pure_fast", "pure_back"):
        d = _draws((4,), on=(name,))
        out = progress.command_from_draws(d, cfg)
        np.testing.assert_array_equal(out[:, 0], d[f"{name}_vx"])
        np.testing.assert_array_equal(out[:, 1:], 0.0)


def test_zeros_are_positive_zero():
    cfg = _Cfg(CFG_ALL)
    for on in (("pure_wz",), ("pure_vy",), ("pure_slow",)):
        out = progress.command_from_draws(_draws((4,), on=on), cfg)
        zeros = out[out == 0.0]
        assert zeros.size and not np.any(np.signbit(zeros))
    out = progress.command_from_draws(_draws((4,), zero=True), _Cfg())
    assert not np.any(np.signbit(out))


def test_a_later_draw_overwrites_an_earlier_one():
    cfg = _Cfg(CFG_ALL)
    names = [n for n, *_ in progress.COMMAND_DRAWS]
    d = _draws((4,), on=names)
    out = progress.command_from_draws(d, cfg)
    np.testing.assert_array_equal(out[:, 0], d["pure_back_vx"])
    np.testing.assert_array_equal(out[:, 1:], 0.0)
    d = _draws((4,), on=names[:3])
    out = progress.command_from_draws(d, cfg)
    np.testing.assert_array_equal(out[:, 0], d["pure_slow_vx"])


def test_straight_keeps_vx_and_zeroes_vy_and_wz():
    d = _draws((4,), on=("straight",))
    d["vy"] = -d["vy"]
    d["wz"] = -d["wz"]
    out = progress.command_from_draws(d, _Cfg(CFG_ALL))
    np.testing.assert_array_equal(out[:, 0], d["vx"])
    np.testing.assert_array_equal(out[:, 1:], 0.0)
    assert not np.any(np.signbit(out[:, 1:]))


def test_straight_is_order_independent(monkeypatch):
    cfg = _Cfg(CFG_ALL)
    others = ("pure_wz", "pure_vy", "pure_slow", "pure_fast", "pure_back")
    default = progress.COMMAND_DRAWS
    straight = tuple(e for e in default if e[0] == "straight")
    moved = straight + tuple(e for e in default if e[0] != "straight")
    assert len(straight) == 1 and moved != default
    for n in range(len(others) + 1):
        for subset in itertools.combinations(others, n):
            d = _draws((4,), on=subset + ("straight",))
            d["vy"] = -d["vy"]
            want = progress.command_from_draws(d, cfg)
            monkeypatch.setattr(progress, "COMMAND_DRAWS", moved)
            got = progress.command_from_draws(d, cfg)
            monkeypatch.setattr(progress, "COMMAND_DRAWS", default)
            np.testing.assert_array_equal(got, want)
            assert np.array_equal(np.signbit(got), np.signbit(want)), subset


def test_standing_still_overrides_every_draw():
    cfg = _Cfg(CFG_ALL)
    names = [n for n, *_ in progress.COMMAND_DRAWS]
    out = progress.command_from_draws(_draws((4,), on=names, zero=True), cfg)
    np.testing.assert_array_equal(out, np.zeros((4, 3), np.float32))


def test_absent_probability_keys_equal_probability_zero():
    d = _draws((6,), on=("pure_wz", "pure_slow"))
    absent = progress.command_from_draws(d, _Cfg())
    zeros = progress.command_from_draws(d, _Cfg({k: 0.0 for k in CFG_ALL}))
    np.testing.assert_array_equal(absent, zeros)
    np.testing.assert_array_equal(absent[:, 0], d["vx"])


def test_a_disabled_draw_needs_no_draws():
    d = _draws((3,))
    for k in [k for k in d if k.startswith("pure_")]:
        del d[k]
    out = progress.command_from_draws(d, _Cfg({"pure_vy_prob": 0.0}))
    assert out.shape == (3, 3)


def test_command_dtype_follows_the_draws():
    out = progress.command_from_draws(_draws((2,)), _Cfg())
    assert out.dtype == np.float32


# The command ramp.

LIN, YAW = 0.5, 1.0


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("batch", BATCHES, ids=str)
def test_ramp_command_batched_equals_stacked(batch, mode):
    assert_batched_matches_stacked(
        lambda c, t: progress.ramp_command(c, t, LIN, YAW, DT),
        [_commands(batch, seed=4), _commands(batch, seed=5)],
        batch,
        mode,
    )


def test_a_ramp_step_never_exceeds_the_rate_times_dt():
    cmd, target = _commands((64,), seed=6), _commands((64,), seed=7)
    step = progress.ramp_command(cmd, target, LIN, YAW, DT) - cmd
    tol = 1e-6
    assert np.all(np.abs(step[:, :2]) <= LIN * DT + tol)
    assert np.all(np.abs(step[:, 2]) <= YAW * DT + tol)
    # A far target moves by the full limit, with the sign of the difference.
    far = np.abs(target - cmd) > np.array([LIN, LIN, YAW]) * DT
    limit = np.broadcast_to(np.array([LIN, LIN, YAW], np.float32) * DT, cmd.shape)
    np.testing.assert_allclose(
        step[far], (np.sign(target - cmd) * limit)[far], rtol=1e-5, atol=1e-6
    )


def test_the_ramp_reaches_the_target_and_stays_there():
    cmd = np.zeros(3, np.float32)
    target = np.array([0.4, -0.3, 0.6], np.float32)
    # 0.4 m/s at 0.5 m/s^2 takes 0.8 s, 40 steps. 0.6 rad/s at 1 rad/s^2 takes 30.
    for _ in range(45):
        cmd = progress.ramp_command(cmd, target, LIN, YAW, DT)
    np.testing.assert_allclose(cmd, target, atol=1e-6)
    for _ in range(10):
        cmd = progress.ramp_command(cmd, target, LIN, YAW, DT)
        np.testing.assert_allclose(cmd, target, atol=1e-6)


def test_the_lin_and_yaw_limits_act_independently():
    cmd = np.zeros(3, np.float32)
    target = np.array([1.0, 1.0, 1.0], np.float32)
    fast_lin = progress.ramp_command(cmd, target, 10.0, YAW, DT)
    np.testing.assert_allclose(fast_lin, [0.2, 0.2, YAW * DT], rtol=1e-6)
    fast_yaw = progress.ramp_command(cmd, target, LIN, 10.0, DT)
    np.testing.assert_allclose(fast_yaw, [LIN * DT, LIN * DT, 0.2], rtol=1e-6)


def test_ramp_on_reads_the_config():
    assert not progress.ramp_on(None)
    assert not progress.ramp_on({})
    assert not progress.ramp_on({"vx": (-1.0, 1.0)})
    assert not progress.ramp_on({"ramp": {"lin": 0.0, "yaw": 0.0}})
    assert not progress.ramp_on({"ramp": {}})
    assert progress.ramp_on({"ramp": {"lin": 0.5, "yaw": 1.0}})
    assert progress.ramp_on({"ramp": {"lin": 0.5}})
