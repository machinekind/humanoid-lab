"""The push schedule and kick across array backends.

Every function runs on numpy and eager jax at batch shapes (), (3,) and
(2, 3). The batched call must equal the per-item calls stacked.
tests/integration/test_shared_math_jit.py runs the same checks under jit.
"""

import numpy as np
import pytest
from ml_collections import config_dict
from task_toys import BATCHES, MODES, assert_batched_matches_stacked

from humanoid_lab.envs import push


def _cfg(**kw):
    base = dict(
        enable=True, interval_steps=5, interval_steps_range=None, vel=0.4, vel_z=0.0,
        ang_vel_rp=0.0, ang_vel_yaw=0.0,
    )
    base.update(kw)
    return config_dict.ConfigDict(base)


def _draws(batch, seed=0):
    rng = np.random.default_rng(seed)
    return {
        "dir": rng.uniform(-1, 1, batch + (2,)).astype(np.float32),
        "vel_z": rng.uniform(-1, 1, batch).astype(np.float32),
        "ang": rng.uniform(-1, 1, batch + (3,)).astype(np.float32),
    }


def _push_now(batch):
    n = int(np.prod(batch, dtype=int))
    return (np.arange(n) % 2 == 0).reshape(batch)


FULL = dict(vel_z=0.3, ang_vel_rp=0.5, ang_vel_yaw=0.7)


def test_draw_table_and_names():
    assert [d[0] for d in push.PUSH_DRAWS] == ["dir", "next_countdown", "vel_z", "ang"]
    assert [d[1] for d in push.PUSH_DRAWS] == [None, 1, 2, 3]
    assert push.push_draw_names(_cfg()) == ("dir",)
    assert push.push_draw_names(_cfg(interval_steps_range=[3, 9], **FULL)) == (
        "dir", "next_countdown", "vel_z", "ang",
    )
    assert push.push_draw_names(_cfg(ang_vel_yaw=0.2)) == ("dir", "ang")


def test_ranges():
    r = push.push_ranges(_cfg(interval_steps_range=[3, 9], **FULL))
    assert r["next_countdown"] == (3, 9)
    assert r["vel_z"] == (-0.3, 0.3)
    assert r["ang"] == ((-0.5, -0.5, -0.7), (0.5, 0.5, 0.7))
    assert set(push.push_ranges(_cfg())) == {"dir"}


def test_a_zero_int_angular_knob_keeps_a_negative_zero_low_bound():
    lo, hi = push.push_ranges(_cfg(ang_vel_rp=0, ang_vel_yaw=0.7))["ang"]
    assert np.signbit(lo[:2]).all() and not np.signbit(hi[:2]).any()


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("batch", BATCHES)
def test_schedule_fixed_interval(batch, mode):
    cfg = _cfg()
    steps = (np.arange(int(np.prod(batch, dtype=int))) + 2).reshape(batch).astype(np.int32)

    def fn(step_count):
        now, cd = push.push_schedule(cfg, step_count)
        assert cd is None
        return {"now": now}

    assert_batched_matches_stacked(fn, [steps], batch, mode)


def test_schedule_fixed_interval_values():
    now, cd = push.push_schedule(_cfg(), np.array([3, 4, 5, 9], np.int32))
    assert cd is None
    np.testing.assert_array_equal(now, [False, True, False, True])


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("batch", BATCHES)
def test_schedule_ranged(batch, mode):
    cfg = _cfg(interval_steps_range=[3, 9])
    n = int(np.prod(batch, dtype=int))
    countdown = (np.arange(n) % 3 - 1).reshape(batch).astype(np.int32)
    nxt = (np.arange(n) + 4).reshape(batch).astype(np.int32)

    def fn(step_count, countdown, nxt):
        now, cd = push.push_schedule(cfg, step_count, countdown, nxt)
        return {"now": now, "cd": cd}

    assert_batched_matches_stacked(fn, [np.zeros(batch, np.int32), countdown, nxt], batch, mode)


def test_schedule_ranged_values():
    cd = np.array([2, 1, 0, -1], np.int32)
    nxt = np.array([7, 7, 7, 7], np.int32)
    now, new = push.push_schedule(_cfg(interval_steps_range=[3, 9]), np.zeros(4, np.int32), cd, nxt)
    np.testing.assert_array_equal(now, [False, False, True, True])
    np.testing.assert_array_equal(new, [1, 0, 7, 7])
    assert new.dtype == np.int32


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("batch", BATCHES)
@pytest.mark.parametrize("knobs", [{}, FULL, dict(vel_z=0.3), dict(ang_vel_yaw=0.7)])
def test_kick(batch, mode, knobs):
    cfg = _cfg(**knobs)
    d = _draws(batch)

    def fn(now, dir_, vel_z, ang):
        kick = push.push_kick(cfg, now, {"dir": dir_, "vel_z": vel_z, "ang": ang})
        return {k: v for k, v in kick.items() if v is not None}

    assert_batched_matches_stacked(fn, [_push_now(batch), d["dir"], d["vel_z"], d["ang"]], batch, mode)


def test_kick_gating_returns_none():
    d = _draws((2,))
    now = np.array([True, False])
    kick = push.push_kick(_cfg(), now, {"dir": d["dir"]})
    assert kick["z"] is None and kick["ang"] is None
    kick = push.push_kick(_cfg(**FULL), now, d)
    assert kick["z"] is not None and kick["ang"] is not None


def test_kick_masks_and_normalizes():
    d = _draws((4,))
    now = np.array([True, False, True, False])
    kick = push.push_kick(_cfg(**FULL), now, d)
    np.testing.assert_allclose(np.linalg.norm(kick["planar"][now], axis=-1), 0.4, rtol=1e-5)
    assert not kick["planar"][~now].any()
    assert not kick["z"][~now].any() and not kick["ang"][~now].any()
    np.testing.assert_array_equal(kick["z"][now], d["vel_z"][now])
    np.testing.assert_array_equal(kick["ang"][now], d["ang"][now])
