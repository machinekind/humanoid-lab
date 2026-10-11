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

from humanoid_lab.eval.gait import (
    foot_sides,
    gait_metrics,
    gait_shape,
    landing_metrics,
    metric_window,
    stability_metrics,
    support_fractions,
    travel_metrics,
)

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


# -- foot_sides --------------------------------------------------------------


def test_foot_sides_reads_left_and_right_tokens():
    assert foot_sides(("left_foot", "right_foot")) == ("left", "right")
    assert foot_sides(("right_foot", "left_foot")) == ("right", "left")


def test_foot_sides_falls_back_to_indices():
    assert foot_sides(("foot_a", "foot_b")) == ("foot0", "foot1")
    assert foot_sides(("left_a", "left_b")) == ("foot0", "foot1")


# -- gait shape --------------------------------------------------------------
#
# Row i of a record is the state after step i under cmd[i]. The builder
# below writes the base pose, the command and the contacts directly.

_DT = 0.02
_SETTLE = 50
_N = 100


def _shape_rec(n=_N, *, yaw=0.0, roll=0.0, pos=None, vx=0.0, wz=0.0, contact=2):
    """A record with the base pose, cmd, contact and foot positions of `n` rows.

    yaw and roll are scalars or per-row arrays in radians. The quaternion is
    the yaw rotation times the roll rotation. pos is (n, 3) and defaults to a
    stand at 0.7 m. vx and wz fill the command. contact is the number of the
    two feet down on every row. The feet sit on the floor under the base,
    0.1 m to each side in the world frame.
    """
    y = np.broadcast_to(np.asarray(yaw, dtype=np.float64), (n,))
    r = np.broadcast_to(np.asarray(roll, dtype=np.float64), (n,))
    cy, sy = np.cos(y / 2), np.sin(y / 2)
    cr, sr = np.cos(r / 2), np.sin(r / 2)
    quat = np.stack([cy * cr, cy * sr, sy * sr, sy * cr], axis=-1)
    if pos is None:
        pos = np.zeros((n, 3))
        pos[:, 2] = 0.7
    cmd = np.zeros((n, 3))
    cmd[:, 0] = vx
    cmd[:, 2] = wz
    feet = np.zeros((n, 2), dtype=bool)
    feet[:, :contact] = True
    pos = np.asarray(pos, dtype=np.float64)
    foot_pos = np.repeat(pos[:, None, :], 2, axis=1)
    foot_pos[:, 0, 1] += 0.1
    foot_pos[:, 1, 1] -= 0.1
    foot_pos[:, :, 2] = 0.0
    return {
        "base_quat": quat.astype(np.float32),
        "base_pos": pos.astype(np.float32),
        "cmd": cmd.astype(np.float32),
        "contact": feet,
        "foot_pos": foot_pos.astype(np.float32),
    }


def _walk_pos(n, heading, speed):
    """Positions of a base that moves speed * _DT per row along heading."""
    t = np.arange(n, dtype=np.float64)
    pos = np.zeros((n, 3))
    pos[:, 0] = np.cos(heading) * speed * _DT * t
    pos[:, 1] = np.sin(heading) * speed * _DT * t
    pos[:, 2] = 0.7
    return pos


def _travel(rec, fell_at=None):
    return travel_metrics(rec, dt=_DT, settle_steps=_SETTLE, fell_at=fell_at)


def test_metric_window_stops_before_the_fall_row():
    assert metric_window(100, 50, None) == slice(50, 100)
    assert metric_window(100, 50, 80) == slice(50, 80)
    w = metric_window(100, 50, 30)
    assert len(range(100)[w]) == 0


def test_speed_frac_is_one_when_the_walk_delivers_the_command():
    rec = _shape_rec(yaw=0.7, pos=_walk_pos(_N, 0.7, 0.5), vx=0.5)

    t = _travel(rec)

    assert t["speed_frac"] == pytest.approx(1.0)
    assert t["lateral_drift_m"] == pytest.approx(0.0, abs=1e-4)
    assert t["heading_drift_deg"] == pytest.approx(0.0, abs=1e-2)


def test_speed_frac_reads_the_shortfall():
    rec = _shape_rec(yaw=0.7, pos=_walk_pos(_N, 0.7, 0.25), vx=0.5)

    assert _travel(rec)["speed_frac"] == pytest.approx(0.5)


def test_speed_frac_is_none_below_ten_centimetres_of_commanded_travel():
    # 0.05 m/s over the 49 commanded steps of the window is 0.049 m.
    rec = _shape_rec(vx=0.05)

    t = _travel(rec)

    assert t["speed_frac"] is None
    assert t["heading_drift_deg"] == 0.0
    assert t["lateral_drift_m"] == 0.0


def test_lateral_drift_is_measured_across_the_start_heading():
    # Facing world +y, so the robot's left is world -x.
    pos = _walk_pos(_N, math.pi / 2, 0.5)
    pos[:, 0] -= np.clip(np.arange(_N) - _SETTLE, 0, None) / (_N - 1 - _SETTLE) * 0.1
    rec = _shape_rec(yaw=math.pi / 2, pos=pos, vx=0.5)

    assert _travel(rec)["lateral_drift_m"] == pytest.approx(0.1, abs=1e-4)


def test_heading_drift_nets_out_the_commanded_turn():
    t = np.arange(_N, dtype=np.float64)
    followed = 0.5 * _DT * t
    assert _travel(_shape_rec(yaw=followed, wz=0.5))["heading_drift_deg"] == pytest.approx(
        0.0, abs=1e-2
    )

    lag = math.radians(10.0) * np.clip(t - _SETTLE, 0, None) / (_N - 1 - _SETTLE)
    lagging = _shape_rec(yaw=followed - lag, wz=0.5)
    assert _travel(lagging)["heading_drift_deg"] == pytest.approx(-10.0, abs=1e-2)


def test_heading_drift_wraps_across_180_degrees():
    # 179 deg at the first scored row, 181 deg (read back as -179) at the last.
    t = np.clip(np.arange(_N) - _SETTLE, 0, None) / (_N - 1 - _SETTLE)
    rec = _shape_rec(yaw=np.radians(179.0 + 2.0 * t))

    assert _travel(rec)["heading_drift_deg"] == pytest.approx(2.0, abs=1e-2)


def test_heading_drift_keeps_the_full_miss_on_a_spin():
    # The battery's spin rows command about 360 deg over the scored window.
    n, wz = 300, 1.26
    commanded = np.degrees(wz * _DT * (n - 1 - _SETTLE))
    still = _travel(_shape_rec(n, wz=wz))
    assert still["heading_drift_deg"] == pytest.approx(-commanded, abs=1e-2)

    # Turning at 40% of the command: the miss is 60% of it, past -180.
    yaw = 0.4 * wz * _DT * np.arange(n)
    partial = _travel(_shape_rec(n, yaw=yaw, wz=wz))
    assert partial["heading_drift_deg"] == pytest.approx(-0.6 * commanded, abs=1e-2)
    assert partial["heading_drift_deg"] < -180.0


def test_heading_drift_is_zero_when_the_robot_follows_a_full_spin():
    # The yaw crosses +-180 deg while the turn is commanded.
    n, wz = 300, 1.26
    yaw = wz * _DT * np.arange(n)
    rec = _shape_rec(n, yaw=yaw, wz=wz)
    assert _travel(rec)["heading_drift_deg"] == pytest.approx(0.0, abs=0.05)


def test_travel_metrics_stop_at_the_fall():
    fell_at = 80
    pos = _walk_pos(_N, 0.0, 0.5)
    pos[fell_at:, :2] += 5.0
    yaw = np.zeros(_N)
    yaw[fell_at:] = 1.5
    rec = _shape_rec(yaw=yaw, pos=pos, vx=0.5)
    cut = {k: v[:fell_at] for k, v in rec.items()}

    assert _travel(rec, fell_at) == _travel(cut)
    assert _travel(rec, fell_at)["speed_frac"] == pytest.approx(1.0)
    assert _travel(rec) != _travel(cut)


def test_travel_metrics_are_none_without_two_scored_rows():
    keys = {"speed_frac", "heading_drift_deg", "lateral_drift_m"}
    for t in (_travel(_shape_rec(n=_SETTLE + 1)), _travel(_shape_rec(), fell_at=_SETTLE + 1)):
        assert set(t) == keys
        assert all(v is None for v in t.values())


def test_support_fractions_count_feet_down_per_row():
    rec = _shape_rec()
    rec["contact"][_SETTLE + 20 : _SETTLE + 40, 1] = False
    rec["contact"][_SETTLE + 40 :] = False

    f = support_fractions(rec, settle_steps=_SETTLE, fell_at=None)

    assert f == {"support_frac_0": 0.2, "support_frac_1": 0.4, "support_frac_2": 0.4}


def test_support_fractions_skip_the_settle_window_and_the_fall_row():
    rec = _shape_rec()
    rec["contact"][:_SETTLE] = False
    rec["contact"][-1] = False

    f = support_fractions(rec, settle_steps=_SETTLE, fell_at=_N - 1)

    assert f == {"support_frac_0": 0.0, "support_frac_1": 0.0, "support_frac_2": 1.0}
    empty = support_fractions(rec, settle_steps=_SETTLE, fell_at=_SETTLE)
    assert all(v is None for v in empty.values())


def test_roll_std_reads_the_roll_angle_in_degrees():
    roll = np.where(np.arange(_N) % 2 == 0, 0.1, -0.1)

    s = stability_metrics(_shape_rec(roll=roll), settle_steps=_SETTLE, fell_at=None)

    assert s["roll_std_deg"] == pytest.approx(5.73)


def test_a_steady_roll_has_zero_std():
    s = stability_metrics(_shape_rec(roll=0.2), settle_steps=_SETTLE, fell_at=None)

    assert s["roll_std_deg"] == pytest.approx(0.0)


def test_pelvis_sway_removes_the_linear_drift():
    t = np.clip(np.arange(_N) - _SETTLE, 0, None).astype(np.float64)
    drift = _walk_pos(_N, 0.0, 0.5)
    drift[:, 1] = 0.001 * t
    # A cosine phase: a sine that starts at zero leans with the row index, so
    # the line fit takes part of it and the std reads 1.30 cm.
    wave = 0.02 * np.cos(2 * np.pi * t / 25)
    sway = drift.copy()
    sway[:, 1] += wave
    pure = _walk_pos(_N, 0.0, 0.5)
    pure[:, 1] = wave

    def _sway(pos):
        s = stability_metrics(_shape_rec(pos=pos, vx=0.5), settle_steps=_SETTLE, fell_at=None)
        return s["pelvis_sway_cm"]

    assert _sway(sway) == pytest.approx(1.41, abs=0.05)
    assert _sway(sway) == _sway(pure)
    assert _sway(drift) == pytest.approx(0.0)


def test_gait_shape_keys_are_always_present():
    g = gait_shape(_shape_rec(), dt=_DT, settle_steps=_SETTLE, fell_at=None)

    assert set(g) == {
        "speed_frac",
        "heading_drift_deg",
        "lateral_drift_m",
        "support_frac_0",
        "support_frac_1",
        "support_frac_2",
        "roll_std_deg",
        "pelvis_sway_cm",
        "landing_x_m_foot0",
        "landing_y_m_foot0",
        "landing_x_m_foot1",
        "landing_y_m_foot1",
        "step_length_m",
        "knee_td_rad_foot0",
        "knee_td_rad_foot1",
    }
    assert g["speed_frac"] is None
    assert g["support_frac_2"] == 1.0

    short = gait_shape(_shape_rec(n=10), dt=_DT, settle_steps=_SETTLE, fell_at=None)
    assert set(short) == set(g)
    assert all(v is None for v in short.values())


# -- landings ----------------------------------------------------------------
#
# A touchdown is a contact rising edge. Each foot's first touchdown in the
# scored window is dropped. The helpers below lift a foot for two rows and
# put it down at a chosen row.

_SIDES = ("left", "right")


def _land(rec, f, t, local=None, knee=None):
    """Lift foot f on rows t-2 and t-1 and put it down at row t.

    local is the foot's (x, y) in the base frame at row t. It is rotated into
    the world around the base with the row's yaw. knee sets knee_q[t, f].
    """
    rec["contact"][t - 2 : t, f] = False
    if local is not None:
        q = rec["base_quat"][t].astype(np.float64)
        yaw = 2.0 * math.atan2(q[3], q[0])
        x, y = local
        c, sn = math.cos(yaw), math.sin(yaw)
        world = np.array([c * x - sn * y, sn * x + c * y])
        rec["foot_pos"][t, f, :2] = rec["base_pos"][t, :2] + world
    if knee is not None:
        rec["knee_q"][t, f] = knee


def _landings(rec, fell_at=None, sides=_SIDES):
    return landing_metrics(rec, sides=sides, settle_steps=_SETTLE, fell_at=fell_at)


def test_landing_position_is_read_in_the_base_frame():
    rec = _shape_rec(yaw=math.pi / 2)
    for t in (_SETTLE + 10, _SETTLE + 30):
        _land(rec, 0, t, local=(0.25, 0.1))
    for t in (_SETTLE + 20, _SETTLE + 40):
        _land(rec, 1, t, local=(0.20, -0.1))

    m = _landings(rec)

    assert m["landing_x_m_left"] == pytest.approx(0.25, abs=1e-4)
    assert m["landing_y_m_left"] == pytest.approx(0.1, abs=1e-4)
    assert m["landing_x_m_right"] == pytest.approx(0.2, abs=1e-4)
    assert m["landing_y_m_right"] == pytest.approx(-0.1, abs=1e-4)


def test_each_foot_s_first_landing_is_dropped():
    rec = _shape_rec()
    _land(rec, 0, _SETTLE + 10, local=(0.9, 0.1))
    _land(rec, 0, _SETTLE + 20, local=(0.3, 0.1))
    _land(rec, 0, _SETTLE + 30, local=(0.3, 0.1))

    assert _landings(rec)["landing_x_m_left"] == pytest.approx(0.3, abs=1e-4)


def test_the_edge_at_the_first_scored_row_reads_the_last_settle_row():
    rec = _shape_rec()
    rec["contact"][_SETTLE - 1, 0] = False
    _land(rec, 0, _SETTLE + 20, local=(0.3, 0.1))

    assert _landings(rec)["landing_x_m_left"] == pytest.approx(0.3, abs=1e-4)

    rec["contact"][_SETTLE - 1, 0] = True
    assert _landings(rec)["landing_x_m_left"] is None


def test_step_length_is_landing_foot_x_minus_the_other_foot_x():
    rec = _shape_rec()
    # The other foot stays at x 0. Each scored landing puts its foot 0.3 m ahead.
    _land(rec, 0, _SETTLE + 10, local=(0.3, 0.1))
    _land(rec, 0, _SETTLE + 20, local=(0.3, 0.1))
    _land(rec, 1, _SETTLE + 30, local=(0.2, -0.1))
    _land(rec, 1, _SETTLE + 40, local=(0.2, -0.1))
    rec["foot_pos"][_SETTLE + 40, 0, 0] = 0.1

    # Left scores row 20: 0.3 - 0. Right scores row 40: 0.2 - 0.1.
    assert _landings(rec)["step_length_m"] == pytest.approx(0.2, abs=1e-4)


def test_step_length_is_none_without_two_feet():
    rec = _shape_rec()
    _land(rec, 0, _SETTLE + 10, local=(0.3, 0.1))
    _land(rec, 0, _SETTLE + 20, local=(0.3, 0.1))
    one = {
        "base_quat": rec["base_quat"],
        "base_pos": rec["base_pos"],
        "cmd": rec["cmd"],
        "contact": rec["contact"][:, :1],
        "foot_pos": rec["foot_pos"][:, :1],
    }

    m = _landings(one, sides=("foot0",))

    assert m["step_length_m"] is None
    assert m["landing_x_m_foot0"] == pytest.approx(0.3, abs=1e-4)
    assert _landings(_shape_rec())["step_length_m"] is None


def test_knee_at_touchdown_takes_the_magnitude():
    rec = _shape_rec()
    rec["knee_q"] = np.zeros((_N, 2), dtype=np.float32)
    for t, knee in ((_SETTLE + 10, -0.9), (_SETTLE + 20, -0.4), (_SETTLE + 30, -0.6)):
        _land(rec, 1, t, knee=knee)

    m = _landings(rec)

    assert m["knee_td_rad_right"] == pytest.approx(0.5, abs=1e-3)
    assert m["knee_td_rad_left"] is None


def test_knee_at_touchdown_is_none_without_knee_q():
    rec = _shape_rec()
    _land(rec, 0, _SETTLE + 10)
    _land(rec, 0, _SETTLE + 20)

    m = _landings(rec)

    assert "knee_td_rad_left" in m
    assert "knee_td_rad_right" in m
    assert m["knee_td_rad_left"] is None
    assert m["knee_td_rad_right"] is None
    assert m["landing_x_m_left"] is not None


def test_landings_stop_at_the_fall():
    rec = _shape_rec()
    _land(rec, 0, _SETTLE + 10, local=(0.9, 0.1))
    _land(rec, 0, _SETTLE + 20, local=(0.3, 0.1))
    _land(rec, 0, _SETTLE + 30, local=(0.6, 0.1))

    assert _landings(rec, fell_at=_SETTLE + 30)["landing_x_m_left"] == pytest.approx(0.3, abs=1e-4)
    assert _landings(rec, fell_at=_SETTLE + 20)["landing_x_m_left"] is None
    assert _landings(rec)["landing_x_m_left"] == pytest.approx(0.45, abs=1e-4)


def test_landing_keys_follow_the_side_tokens():
    rec = _shape_rec()

    named = gait_shape(rec, dt=_DT, settle_steps=_SETTLE, fell_at=None, sides=_SIDES)
    plain = gait_shape(rec, dt=_DT, settle_steps=_SETTLE, fell_at=None)

    assert "landing_x_m_left" in named
    assert "knee_td_rad_right" in named
    assert "landing_x_m_foot0" not in named
    assert "landing_x_m_foot0" in plain
    assert "landing_x_m_left" not in plain
