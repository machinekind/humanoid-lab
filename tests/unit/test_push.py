"""The push schedule and kick across array backends.

The batched functions run on numpy and eager jax at batch shapes (), (3,)
and (2, 3). The batched call must equal the per-item calls stacked.
Elementwise functions have value tests only.
tests/integration/test_shared_math_jit.py runs the same checks under jit.
"""

import numpy as np
import pytest
from ml_collections import config_dict
from task_toys import BATCHES, MODES, assert_batched_matches_stacked

from humanoid_lab.envs import push


def _cfg(**kw):
    base = {
        "enable": True, "interval_steps": 5, "interval_steps_range": None, "vel": 0.4, "vel_z": 0.0,
        "ang_vel_rp": 0.0, "ang_vel_yaw": 0.0,
    }
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


FULL = {"vel_z": 0.3, "ang_vel_rp": 0.5, "ang_vel_yaw": 0.7}


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


def test_schedule_fixed_interval_values():
    now, cd = push.push_schedule(_cfg(), np.array([3, 4, 5, 9], np.int32))
    assert cd is None
    np.testing.assert_array_equal(now, [False, True, False, True])


def test_schedule_ranged_values():
    cd = np.array([2, 1, 0, -1], np.int32)
    nxt = np.array([7, 7, 7, 7], np.int32)
    now, new = push.push_schedule(_cfg(interval_steps_range=[3, 9]), np.zeros(4, np.int32), cd, nxt)
    np.testing.assert_array_equal(now, [False, False, True, True])
    np.testing.assert_array_equal(new, [1, 0, 7, 7])
    assert new.dtype == np.int32


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("batch", BATCHES)
@pytest.mark.parametrize("knobs", [{}, FULL, {"vel_z": 0.3}, {"ang_vel_yaw": 0.7}])
def test_kick(batch, mode, knobs):
    cfg = _cfg(**knobs)
    d = _draws(batch)

    def fn(now, dir_, vel_z, ang):
        kick = push.push_kick(cfg, now, {"dir": dir_, "vel_z": vel_z, "ang": ang})
        return {k: v for k, v in kick.items() if v is not None}

    assert_batched_matches_stacked(fn, [_push_now(batch), d["dir"], d["vel_z"], d["ang"]], batch, mode)


def test_kick_masks_and_normalizes():
    d = _draws((4,))
    now = np.array([True, False, True, False])
    kick = push.push_kick(_cfg(**FULL), now, d)
    np.testing.assert_allclose(np.linalg.norm(kick["planar"][now], axis=-1), 0.4, rtol=1e-5)
    assert not kick["planar"][~now].any()
    assert not kick["z"][~now].any() and not kick["ang"][~now].any()
    np.testing.assert_array_equal(kick["z"][now], d["vel_z"][now])
    np.testing.assert_array_equal(kick["ang"][now], d["ang"][now])
    # With the vertical and angular knobs at zero, only the planar kick is applied.
    off = push.push_kick(_cfg(), now, {"dir": d["dir"]})
    np.testing.assert_array_equal(off["planar"], kick["planar"])
    for k in ("z", "ang"):
        assert off[k] is None or not np.any(off[k]), k
