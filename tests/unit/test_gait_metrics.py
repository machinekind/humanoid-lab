"""gait_metrics: swing apex and touchdown softness, model-free.

Three properties this file pins. gait_metrics takes an explicit
settle_steps, so the reset transient is excluded at the call site and the
exclusion is testable here. A record with no scorable swing returns
`swings: 0` and null
medians rather than an empty dict, so battery.json's shape never depends on
what the policy did. And a swing already airborne at the first measured
sample is dropped like an end-truncated one: our record is trimmed mid-flight
by the settle window, so its apex may have happened before step 0.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from humanoid_lab.eval.gait import gait_metrics

_G = 9.81


def _rec_from_swing(apex, td_speed, steps=10, feet=2):
    """One triangular swing per foot: rise to `apex`, descend arriving at
    the ground with `td_speed` on the last airborne step."""
    up = np.linspace(0.0, apex, steps)
    down = np.linspace(apex, 0.001, steps)
    z = np.concatenate([[0.0], up, down, [0.0, 0.0]])
    vz = np.gradient(z)
    last_airborne = np.nonzero(z > 0.005)[0][-1]
    vz[last_airborne] = -td_speed  # pre-contact downward speed
    return {
        "foot_clear": np.tile(z[:, None], (1, feet)),
        "foot_vz": np.tile(vz[:, None], (1, feet)),
    }


def _empty(n=20, feet=2):
    return {"foot_clear": np.zeros((n, feet)), "foot_vz": np.zeros((n, feet))}


# -- swing detection ---------------------------------------------------------


def test_apex_measured_per_swing():
    g = gait_metrics(_rec_from_swing(apex=0.05, td_speed=0.3), settle_steps=0)

    assert g["swing_apex_med_m"] == pytest.approx(0.05, abs=1e-6)
    assert g["swings"] == 2  # one per foot


def test_works_for_any_foot_count():
    for feet in (1, 2, 4, 6):
        g = gait_metrics(_rec_from_swing(0.05, 0.3, feet=feet), settle_steps=0)
        assert g["swings"] == feet


def test_blips_and_ground_noise_ignored():
    rec = _empty()
    rec["foot_clear"][5, 0] = 0.02  # 1-step blip: shorter than 2 steps
    rec["foot_clear"][10:12, 1] = 0.003  # below the 5 mm airborne band

    g = gait_metrics(rec, settle_steps=0)

    assert g["swings"] == 0


def test_truncated_final_swing_not_counted():
    """A swing still airborne when the record ends has no touchdown to
    measure, so it is not a swing this function saw."""
    rec = _empty(n=10)
    rec["foot_clear"][6:, :] = 0.04

    assert gait_metrics(rec, settle_steps=0)["swings"] == 0


def test_swing_already_airborne_at_the_first_sample_not_counted():
    """Its apex may have happened before the window opened, so the number
    would be a floor, not a peak."""
    rec = _empty(n=20)
    rec["foot_clear"][:6, 0] = 0.04  # airborne from step 0, lands at 6

    assert gait_metrics(rec, settle_steps=0)["swings"] == 0


def test_two_step_swings_are_the_shortest_counted():
    rec = _empty(n=20)
    rec["foot_clear"][5:7, 0] = 0.03  # exactly 2 steps, lands at 7

    g = gait_metrics(rec, settle_steps=0)

    assert g["swings"] == 1
    assert g["swing_apex_med_m"] == pytest.approx(0.03)


# -- touchdown softness ------------------------------------------------------


def test_touchdown_softness_uses_freefall_reference():
    apex, td = 0.05, 0.3
    g = gait_metrics(_rec_from_swing(apex, td), settle_steps=0)
    v_ff = math.sqrt(2 * _G * apex)

    assert g["touchdown_v_med"] == pytest.approx(td, abs=1e-6)
    assert g["touchdown_softness_med"] == pytest.approx(td / v_ff, abs=1e-3)


def test_a_foot_that_falls_like_a_brick_scores_one():
    apex = 0.05
    v_ff = math.sqrt(2 * _G * apex)

    g = gait_metrics(_rec_from_swing(apex, v_ff), settle_steps=0)

    assert g["touchdown_softness_med"] == pytest.approx(1.0, abs=1e-3)


def test_a_rising_foot_at_touchdown_scores_zero_not_negative():
    """Touchdown speed is the DOWNWARD component; a foot still rising on its
    last airborne step arrived at no downward speed at all."""
    rec = _empty(n=20)
    rec["foot_clear"][5:9, 0] = 0.03
    rec["foot_vz"][8, 0] = +0.4  # rising

    g = gait_metrics(rec, settle_steps=0)

    assert g["touchdown_v_med"] == pytest.approx(0.0)
    assert g["touchdown_softness_med"] == pytest.approx(0.0)


# -- percentiles and sample counts -------------------------------------------


def test_p90_reads_the_tall_swings():
    rec = _empty(n=200, feet=1)
    for i, apex in enumerate([0.02] * 9 + [0.09]):
        rec["foot_clear"][10 * i + 2 : 10 * i + 6, 0] = apex

    g = gait_metrics(rec, settle_steps=0)

    assert g["swings"] == 10
    assert g["swing_apex_med_m"] == pytest.approx(0.02)
    assert g["swing_apex_p90_m"] > 0.02


def test_no_swing_reports_zero_swings_and_null_medians():
    """Never a 0.0 that dilutes: a policy that never lifted a foot has no
    apex, and reporting one as zero would average into a keeper comparison
    as though it had been measured."""
    g = gait_metrics(_empty(), settle_steps=0)

    assert g["swings"] == 0
    assert g["swing_apex_med_m"] is None
    assert g["swing_apex_p90_m"] is None
    assert g["touchdown_v_med"] is None
    assert g["touchdown_softness_med"] is None


def test_the_field_set_is_the_same_whether_or_not_anything_swung():
    assert set(gait_metrics(_empty(), settle_steps=0)) == set(
        gait_metrics(_rec_from_swing(0.05, 0.3), settle_steps=0)
    )


# -- the settle window -------------------------------------------------------


def test_swings_inside_the_settle_window_are_excluded():
    rec = _empty(n=60, feet=1)
    rec["foot_clear"][5:9, 0] = 0.09  # reset transient: a big early lift
    rec["foot_clear"][30:34, 0] = 0.02  # the real gait

    g = gait_metrics(rec, settle_steps=20)

    assert g["swings"] == 1
    assert g["swing_apex_med_m"] == pytest.approx(0.02)


def test_a_swing_straddling_the_settle_boundary_is_dropped():
    """Trimming the record can cut a swing mid-flight; its apex is then a
    floor, not a peak, so it is not counted."""
    rec = _empty(n=60, feet=1)
    rec["foot_clear"][16:24, 0] = 0.05  # opens before the window, lands after

    assert gait_metrics(rec, settle_steps=20)["swings"] == 0


def test_a_record_shorter_than_the_settle_window_measures_nothing():
    g = gait_metrics(_empty(n=10), settle_steps=50)

    assert g["swings"] == 0
    assert g["swing_apex_med_m"] is None
