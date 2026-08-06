"""Unit tests for the reward term library (rewards/terms.py): every term is a
pure function of generic arrays, so each is tested directly on synthetic
small arrays without building a model or an env.
"""

import jax.numpy as jp
import pytest

from humanoid_lab.rewards import terms


def _is_finite_scalar(x) -> bool:
    x = jp.asarray(x)
    return x.shape == () and bool(jp.all(jp.isfinite(x)))


def test_tracking_err_lin_is_zero_at_perfect_tracking():
    cmd = jp.array([0.3, -0.2])
    assert terms.tracking_err_lin(cmd, cmd) == pytest.approx(0.0)


def test_tracking_err_lin_sums_both_axes():
    cmd = jp.array([0.3, -0.2])
    off = jp.array([0.0, 0.0])
    assert terms.tracking_err_lin(cmd, off) == pytest.approx(0.09 + 0.04)


def test_tracking_err_ang_is_the_squared_yaw_rate_error():
    assert terms.tracking_err_ang(0.4, 0.0) == pytest.approx(0.16)
    assert terms.tracking_err_ang(0.4, 0.4) == pytest.approx(0.0)


def test_tracking_kernel_is_one_at_zero_error():
    assert terms.tracking_kernel(0.0, sigma=0.25) == pytest.approx(1.0)


def test_tracking_kernel_decays_with_error():
    assert terms.tracking_kernel(0.25, sigma=0.25) == pytest.approx(jp.exp(-1.0))
    assert terms.tracking_kernel(0.25, sigma=0.25) < 1.0


# -- relative kernel width -------------------------------------------------


def test_tracking_rel_sigma_scales_with_the_squared_command():
    # rel_sigma * max(|cmd|, floor)^2, both commands above the floor.
    assert terms.tracking_rel_sigma(0.4, rel_sigma=0.25, floor=0.3) == pytest.approx(0.04)
    assert terms.tracking_rel_sigma(0.8, rel_sigma=0.25, floor=0.3) == pytest.approx(0.16)


def test_relative_kernel_pays_the_same_at_the_same_fraction_of_target():
    """The whole point of the relative kernel: 80% of target scores the same
    at any commanded speed, so a fast command is not a reward cliff."""
    slow = terms.tracking_kernel(
        terms.tracking_err_ang(0.4, 0.8 * 0.4),
        terms.tracking_rel_sigma(0.4, rel_sigma=0.25, floor=0.3),
    )
    fast = terms.tracking_kernel(
        terms.tracking_err_ang(0.8, 0.8 * 0.8),
        terms.tracking_rel_sigma(0.8, rel_sigma=0.25, floor=0.3),
    )
    assert float(slow) == pytest.approx(float(fast), rel=1e-6)
    assert float(slow) == pytest.approx(float(jp.exp(-0.16)), rel=1e-6)


def test_the_absolute_kernel_does_not_pay_the_same_at_the_same_fraction():
    """The cliff the relative kernel removes: at a fixed fraction of target,
    the absolute kernel collapses as the command grows."""
    slow = terms.tracking_kernel(terms.tracking_err_ang(0.4, 0.8 * 0.4), sigma=0.25)
    fast = terms.tracking_kernel(terms.tracking_err_ang(0.8, 0.8 * 0.8), sigma=0.25)
    assert float(fast) < float(slow)


def test_tracking_rel_sigma_floors_a_small_command():
    # Below the floor the width stops shrinking, so small commands share one
    # width instead of sharpening toward a division by zero.
    small = terms.tracking_rel_sigma(0.1, rel_sigma=0.25, floor=0.3)
    smaller = terms.tracking_rel_sigma(0.0, rel_sigma=0.25, floor=0.3)
    assert float(small) == pytest.approx(float(smaller))
    assert float(smaller) == pytest.approx(0.25 * 0.09)


# -- far-field blend -------------------------------------------------------


def test_far_blend_at_weight_zero_returns_the_kernel():
    k = terms.tracking_kernel(0.5, sigma=0.25)
    blended = terms.tracking_far_blend(k, 0.5, weight=0.0, far_sigma=2.5)
    assert float(blended) == float(k)


def test_far_blend_at_weight_one_is_the_far_kernel_alone():
    k = terms.tracking_kernel(0.5, sigma=0.25)
    blended = terms.tracking_far_blend(k, 0.5, weight=1.0, far_sigma=2.5)
    assert float(blended) == pytest.approx(float(jp.exp(-0.5 / 2.5)), rel=1e-6)


def test_far_blend_lifts_a_kernel_that_has_gone_flat():
    """The point of the term: several sigma out the sharp kernel is 0 and has
    no gradient, and the wide one still does."""
    err = 2.25
    k = terms.tracking_kernel(err, sigma=0.25)  # exp(-9), effectively zero
    blended = terms.tracking_far_blend(k, err, weight=0.25, far_sigma=2.5)
    assert float(k) < 1e-3
    assert float(blended) > 0.1


def test_far_blend_stays_within_the_unit_interval():
    for err in (0.0, 0.25, 1.0, 4.0):
        k = terms.tracking_kernel(err, sigma=0.25)
        blended = float(terms.tracking_far_blend(k, err, weight=0.25, far_sigma=2.5))
        assert 0.0 <= blended <= 1.0


def test_far_blend_keeps_the_optimum_at_zero_error():
    """Both kernels peak at zero error, so the blend does too -- the mix-in
    adds gradient at range without moving where the reward is maximal."""
    best = float(terms.tracking_far_blend(terms.tracking_kernel(0.0, 0.25), 0.0, 0.25, 2.5))
    assert best == pytest.approx(1.0)
    for err in (0.01, 0.25, 1.0):
        k = terms.tracking_kernel(err, sigma=0.25)
        assert float(terms.tracking_far_blend(k, err, 0.25, 2.5)) < best


def test_far_blend_alone_pays_a_standing_robot_a_fifth_of_the_maximum():
    """Why tracking_far_weight must never be used without tracking_product or
    tracking_relative: at a yaw rate error of 0.8 rad/s the far term alone
    pays 0.25*exp(-0.64/2.5), about 19% of the maximum angular reward, for
    standing still -- and its gradient is weaker than the penalties a pivot
    attempt incurs, so standing is a stable deadlock."""
    err = 0.8**2
    far_only = float(terms.tracking_far_blend(0.0, err, weight=0.25, far_sigma=2.5))
    assert far_only == pytest.approx(0.19, abs=0.01)


def test_a_zero_command_gives_a_finite_relative_kernel():
    k = terms.tracking_kernel(
        terms.tracking_err_ang(0.0, 0.1),
        terms.tracking_rel_sigma(0.0, rel_sigma=0.25, floor=0.3),
    )
    assert _is_finite_scalar(k)
    assert 0.0 < float(k) < 1.0


def test_action_rate_zero_for_constant_action():
    a = jp.array([0.1, -0.2, 0.3])
    assert terms.action_rate(a, a) == pytest.approx(0.0)


def test_action_rate_nonzero_for_changing_action():
    a = jp.array([0.1, -0.2, 0.3])
    b = jp.array([0.0, 0.0, 0.0])
    assert terms.action_rate(a, b) > 0.0


def test_action_accel_zero_for_constant_action():
    a = jp.array([0.1, -0.2, 0.3])
    assert terms.action_accel(a, a, a) == pytest.approx(0.0)


def test_action_accel_nonzero_when_second_difference_is_nonzero():
    # A constant ramp (a - prev == prev - prev2) gives a zero second
    # difference by construction; use a non-collinear triple instead.
    a = jp.array([0.1, -0.2])
    prev = jp.array([0.0, 0.0])
    prev2 = jp.array([0.05, 0.1])
    assert terms.action_accel(a, prev, prev2) > 0.0


def test_feet_air_time_uses_min_air_time_offset():
    air_time = jp.array([0.3, 0.05])
    first_contact = jp.array([True, True])
    out = terms.feet_air_time(air_time, first_contact, air_time_cap=0.0, min_air_time=0.1)
    assert out == pytest.approx((0.3 - 0.1) + (0.05 - 0.1))


def test_feet_air_time_cap_bounds_reward():
    air_time = jp.array([10.0])
    first_contact = jp.array([True])
    capped = terms.feet_air_time(air_time, first_contact, air_time_cap=0.5, min_air_time=0.1)
    assert capped == pytest.approx(0.5 - 0.1)


# -- feet_air_time_biped ---------------------------------------------------


def test_feet_air_time_biped_pays_zero_in_double_support():
    out = terms.feet_air_time_biped(
        jp.array([0.0, 0.0]), jp.array([0.3, 0.5]), jp.array([True, True]), threshold=0.4
    )
    assert out == pytest.approx(0.0)


def test_feet_air_time_biped_pays_zero_in_flight():
    out = terms.feet_air_time_biped(
        jp.array([0.2, 0.3]), jp.array([0.0, 0.0]), jp.array([False, False]), threshold=0.4
    )
    assert out == pytest.approx(0.0)


def test_feet_air_time_biped_pays_the_smaller_mode_time_in_single_stance():
    # Stance foot 0.3 s into contact, swing foot 0.1 s into its swing: the
    # min prices the shorter dwell, so BOTH times have to grow to earn more.
    out = terms.feet_air_time_biped(
        jp.array([0.0, 0.1]), jp.array([0.3, 0.0]), jp.array([True, False]), threshold=0.4
    )
    assert out == pytest.approx(0.1)


def test_feet_air_time_biped_clamps_at_the_threshold():
    out = terms.feet_air_time_biped(
        jp.array([0.0, 0.9]), jp.array([1.2, 0.0]), jp.array([True, False]), threshold=0.4
    )
    assert out == pytest.approx(0.4)


def test_feet_air_time_biped_pays_from_the_first_instant_of_a_lift():
    """The run-1 defect this term exists to fix: a policy with zero completed
    swings must still see more reward one control step into a lift than it
    sees standing on both feet."""
    dt = 0.02
    lifted = terms.feet_air_time_biped(
        jp.array([0.0, dt]), jp.array([0.5, 0.0]), jp.array([True, False]), threshold=0.4
    )
    standing = terms.feet_air_time_biped(
        jp.array([0.0, 0.0]), jp.array([0.5, 0.5]), jp.array([True, True]), threshold=0.4
    )
    assert float(lifted) > float(standing)


# -- feet_apex / feet_landing ----------------------------------------------


def test_feet_apex_pays_the_fraction_of_the_target_each_swing_reached():
    apex = jp.array([0.025, 0.05])
    first_contact = jp.array([True, True])
    assert terms.feet_apex(apex, first_contact, apex_target=0.05) == pytest.approx(0.5 + 1.0)


def test_feet_apex_clips_at_the_target():
    """A swing higher than asked for pays the same as one that just reaches
    the target: the term prices reaching the apex, not maximizing it."""
    at_target = float(terms.feet_apex(jp.array([0.05]), jp.array([True]), apex_target=0.05))
    over = float(terms.feet_apex(jp.array([0.20]), jp.array([True]), apex_target=0.05))
    assert at_target == pytest.approx(1.0)
    assert over == pytest.approx(1.0)


def test_feet_apex_pays_once_per_swing_at_touchdown():
    """A foot still in the air earns nothing however high it has been. The
    payout lands on the step its swing ends, and on that foot only."""
    apex = jp.array([0.05, 0.05])
    airborne = terms.feet_apex(apex, jp.array([False, False]), apex_target=0.05)
    one_lands = terms.feet_apex(apex, jp.array([False, True]), apex_target=0.05)
    assert airborne == pytest.approx(0.0)
    assert one_lands == pytest.approx(1.0)


def test_feet_landing_is_zero_for_a_foot_moving_up():
    """Only downward speed is priced: a foot pushing off the floor is a
    swing starting, not an impact."""
    rising = terms.feet_landing(jp.array([0.8, 0.0]), jp.array([0.0, 0.0]), glide_height=0.03)
    assert rising == pytest.approx(0.0)


def test_feet_landing_is_zero_at_and_above_the_glide_height():
    """The penalty exists to shape the last few centimetres of a swing; a
    foot still above the band is free to move at any speed."""
    fast_down = jp.array([-1.0])
    assert terms.feet_landing(fast_down, jp.array([0.03]), 0.03) == pytest.approx(0.0)
    assert terms.feet_landing(fast_down, jp.array([0.10]), 0.03) == pytest.approx(0.0)


def test_feet_landing_is_quadratic_in_downward_speed_at_the_floor():
    slow = float(terms.feet_landing(jp.array([-0.4]), jp.array([0.0]), 0.03))
    fast = float(terms.feet_landing(jp.array([-0.8]), jp.array([0.0]), 0.03))
    assert slow == pytest.approx(0.16)
    assert fast == pytest.approx(4.0 * slow)


def test_feet_landing_ramps_linearly_from_the_glide_height_to_the_floor():
    """The proximity gate is 1 at the floor and 0 at glide_height, linear in
    between, so the gradient reads "decelerate as you approach"."""
    at = [
        float(terms.feet_landing(jp.array([-1.0]), jp.array([c]), 0.03))
        for c in (0.0, 0.0075, 0.015, 0.0225, 0.03)
    ]
    assert at == pytest.approx([1.0, 0.75, 0.5, 0.25, 0.0])


def test_feet_landing_sums_over_the_feet():
    both = terms.feet_landing(jp.array([-1.0, -1.0]), jp.array([0.0, 0.0]), 0.03)
    assert both == pytest.approx(2.0)


def test_feet_landing_prices_the_free_fall_touchdown_reference():
    """The physical reference for touchdown softness: a foot that free-falls
    the whole glide band arrives at sqrt(2*9.81*0.03) ~ 0.77 m/s. That speed
    at the floor is the unit this penalty is calibrated in."""
    free_fall = float(jp.sqrt(2 * 9.81 * 0.03))
    assert free_fall == pytest.approx(0.77, abs=0.01)
    at_touchdown = float(terms.feet_landing(jp.array([-free_fall]), jp.array([0.0]), 0.03))
    assert at_touchdown == pytest.approx(free_fall**2, rel=1e-5)


def test_feet_phase_is_one_at_perfect_tracking():
    clearance = jp.array([0.01, 0.0])
    assert terms.feet_phase(clearance, clearance, phase_sigma=0.002) == pytest.approx(1.0)


def test_torque_limit_zero_within_cap():
    force = jp.array([1.0, -2.0])
    cap = jp.array([10.0, 10.0])
    assert terms.torque_limit(force, cap, frac=0.85) == pytest.approx(0.0)


def test_torque_limit_positive_above_cap():
    force = jp.array([9.0, -9.5])
    cap = jp.array([10.0, 10.0])
    assert terms.torque_limit(force, cap, frac=0.85) > 0.0


@pytest.mark.parametrize(
    "fn, args",
    [
        (terms.tracking_err_lin, (jp.array([0.1, 0.2]), jp.array([0.0, 0.3]))),
        (terms.tracking_err_ang, (0.2, -0.1)),
        (terms.tracking_kernel, (0.05, 0.25)),
        (terms.tracking_rel_sigma, (0.5, 0.25, 0.3)),
        (terms.tracking_far_blend, (0.5, 0.05, 0.25, 2.5)),
        (terms.lin_vel_z, (0.3,)),
        (terms.ang_vel_xy, (jp.array([0.1, -0.2]),)),
        (terms.orientation, (jp.array([0.05, -0.02]),)),
        (terms.torques, (jp.array([1.0, -2.0, 3.0]),)),
        (terms.torque_rate, (jp.array([1.0, -2.0]), jp.array([0.5, -1.0]))),
        (terms.action_rate, (jp.array([0.1, 0.2]), jp.array([0.0, 0.1]))),
        (
            terms.action_accel,
            (jp.array([0.1, 0.2]), jp.array([0.0, 0.1]), jp.array([-0.1, 0.0])),
        ),
        (terms.energy, (jp.array([1.0, -1.0]), jp.array([2.0, 3.0]))),
        (terms.pose, (jp.array([0.1, -0.1]), jp.array([0.0, 0.0]), jp.array([1.0, 0.5]))),
        (terms.feet_air_time, (jp.array([0.2, 0.0]), jp.array([True, False]))),
        (
            terms.feet_slip,
            (jp.array([[0.1, 0.0], [0.0, 0.0]]), jp.array([True, False])),
        ),
        (terms.feet_apex, (jp.array([0.04, 0.0]), jp.array([True, False]), 0.05)),
        (
            terms.feet_landing,
            (jp.array([-0.4, 0.1]), jp.array([0.01, 0.05]), 0.03),
        ),
        (terms.feet_phase, (jp.array([0.01, 0.0]), jp.array([0.0, 0.0]), 0.002)),
        (
            terms.stand_still,
            (jp.array([0.1, -0.1]), jp.array([0.0, 0.0]), jp.array([0.01, -0.01])),
        ),
        (terms.termination, (jp.array(False),)),
        (terms.torque_limit, (jp.array([1.0, 2.0]), jp.array([10.0, 10.0]), 0.85)),
        (terms.pose_l1, (jp.array([0.1, -0.1]), jp.array([0.0, 0.0]), jp.array([1.0, 0.5]))),
        (
            terms.joint_pos_limits,
            (jp.array([0.1, -0.4]), jp.array([-0.3, -0.3]), jp.array([0.3, 0.3])),
        ),
        (terms.joint_vel, (jp.array([0.5, -1.0]),)),
        (terms.joint_acc, (jp.array([2.0, -3.0]),)),
        (terms.upward, (jp.array(-0.98),)),
        (terms.distance_band, (jp.array(0.3), 0.16, 0.5)),
        (terms.feet_contact_without_cmd, (jp.array([True, True]), jp.array(-0.98))),
    ],
)
def test_every_term_returns_a_finite_scalar(fn, args):
    assert _is_finite_scalar(fn(*args))


# -- ported robolab terms --------------------------------------------------


def test_pose_l1_is_zero_at_the_default_pose():
    q = jp.array([0.2, -0.3])
    assert terms.pose_l1(q, q, jp.ones(2)) == pytest.approx(0.0)


def test_pose_l1_weights_price_each_joint_separately():
    q = jp.array([0.1, -0.2])
    zero = jp.array([0.0, 0.0])
    assert terms.pose_l1(q, zero, jp.array([1.0, 0.0])) == pytest.approx(0.1)
    assert terms.pose_l1(q, zero, jp.array([1.0, 0.5])) == pytest.approx(0.1 + 0.5 * 0.2)


def test_joint_pos_limits_is_zero_inside_the_soft_band():
    lo, hi = jp.array([-0.3, -0.3]), jp.array([0.3, 0.3])
    assert terms.joint_pos_limits(jp.array([0.29, -0.29]), lo, hi) == pytest.approx(0.0)


def test_joint_pos_limits_charges_the_linear_overshoot_on_both_sides():
    lo, hi = jp.array([-0.3, -0.3]), jp.array([0.3, 0.3])
    assert terms.joint_pos_limits(jp.array([0.4, -0.45]), lo, hi) == pytest.approx(0.1 + 0.15)


def test_joint_vel_and_acc_are_sums_of_squares():
    assert terms.joint_vel(jp.array([0.5, -1.0])) == pytest.approx(1.25)
    assert terms.joint_acc(jp.array([2.0, -3.0])) == pytest.approx(13.0)


def test_upward_is_one_upright_and_falls_with_tilt():
    assert terms.upward(jp.array(-1.0)) == pytest.approx(1.0)
    assert terms.upward(jp.array(0.0)) == pytest.approx(0.0)
    assert terms.upward(jp.array(1.0)) == pytest.approx(-1.0)


def test_distance_band_pays_one_anywhere_inside_the_band():
    assert terms.distance_band(jp.array(0.16), 0.16, 0.5) == pytest.approx(1.0)
    assert terms.distance_band(jp.array(0.33), 0.16, 0.5) == pytest.approx(1.0)
    assert terms.distance_band(jp.array(0.5), 0.16, 0.5) == pytest.approx(1.0)


def test_distance_band_decays_outside_the_band_over_about_a_centimetre():
    # The far side of the band stays at exp(0)=1; the crossed side decays
    # exp(-100*excursion), so 1 cm out pays (1 + e^-1)/2.
    crossed = terms.distance_band(jp.array(0.15), 0.16, 0.5)
    splayed = terms.distance_band(jp.array(0.51), 0.16, 0.5)
    expected = (1.0 + float(jp.exp(-1.0))) / 2.0
    assert crossed == pytest.approx(expected, rel=1e-5)
    assert splayed == pytest.approx(expected, rel=1e-5)
    assert float(terms.distance_band(jp.array(0.05), 0.16, 0.5)) < float(crossed)


def test_feet_contact_without_cmd_needs_every_foot_planted():
    upright = jp.array(-1.0)
    assert terms.feet_contact_without_cmd(jp.array([True, True]), upright) == pytest.approx(1.0)
    assert terms.feet_contact_without_cmd(jp.array([True, False]), upright) == pytest.approx(0.0)


def test_feet_contact_without_cmd_scales_with_uprightness_and_clamps():
    both = jp.array([True, True])
    # clip(-gz, 0, 0.7)/0.7: saturated at 1.0 from gz=-0.7 down, linear
    # toward 0 as the base tips, floored at 0 past horizontal.
    assert terms.feet_contact_without_cmd(both, jp.array(-0.7)) == pytest.approx(1.0)
    assert terms.feet_contact_without_cmd(both, jp.array(-0.35)) == pytest.approx(0.5)
    assert terms.feet_contact_without_cmd(both, jp.array(0.5)) == pytest.approx(0.0)


# -- knee_stance -------------------------------------------------------------


def test_knee_stance_is_free_inside_the_tolerance():
    out = terms.knee_stance(jp.array([0.1, -0.12]), jp.array([True, True]), tol=0.15)
    assert out == pytest.approx(0.0)


def test_knee_stance_charges_only_the_leg_in_contact():
    # Same flexion on both knees; only the stance leg pays, so the swing
    # leg is free to bend as much as the step needs.
    both = terms.knee_stance(jp.array([0.35, 0.35]), jp.array([True, True]), tol=0.15)
    stance_only = terms.knee_stance(jp.array([0.35, 0.35]), jp.array([True, False]), tol=0.15)
    airborne = terms.knee_stance(jp.array([0.35, 0.35]), jp.array([False, False]), tol=0.15)
    assert both == pytest.approx(2 * 0.2**2)
    assert stance_only == pytest.approx(0.2**2)
    assert airborne == pytest.approx(0.0)


def test_knee_stance_is_quadratic_in_the_excess_flexion():
    near = float(terms.knee_stance(jp.array([0.25]), jp.array([True]), tol=0.15))
    far = float(terms.knee_stance(jp.array([0.35]), jp.array([True]), tol=0.15))
    assert near == pytest.approx(0.1**2)
    assert far == pytest.approx(0.2**2)


def test_knee_stance_charges_hyperextension_too():
    # |q| in the excess: a knee locked past straight is as priced as a
    # crouch, so the term cannot be gamed by bending the other way.
    out = terms.knee_stance(jp.array([-0.35]), jp.array([True]), tol=0.15)
    assert out == pytest.approx(0.2**2)


# -- gait_symmetry -----------------------------------------------------------


def test_gait_symmetry_is_zero_for_an_even_gait():
    out = terms.gait_symmetry(jp.array([0.35, 0.35]), jp.array([0.4, 0.4]), floor=0.1)
    assert out == pytest.approx(0.0)


def test_gait_symmetry_charges_the_relative_difference():
    # 20% swing asymmetry around a 0.35 s mean: (0.07/0.35)^2 = 0.04.
    out = terms.gait_symmetry(
        jp.array([0.385, 0.315]), jp.array([0.4, 0.4]), floor=0.1
    )
    assert out == pytest.approx((0.07 / 0.35) ** 2, rel=1e-5)


def test_gait_symmetry_is_cadence_invariant():
    slow = terms.gait_symmetry(jp.array([0.44, 0.36]), jp.array([0.5, 0.5]), floor=0.1)
    fast = terms.gait_symmetry(jp.array([0.22, 0.18]), jp.array([0.25, 0.25]), floor=0.1)
    assert float(slow) == pytest.approx(float(fast), rel=1e-5)


def test_gait_symmetry_stays_disarmed_until_both_feet_have_stepped():
    """The first step of an episode is one-legged by definition: one foot
    has a completed swing on record and the other still reads zero. Charging
    that state would penalize starting to walk at all."""
    first_step = terms.gait_symmetry(
        jp.array([0.09, 0.0]), jp.array([0.0, 0.0]), floor=0.1
    )
    assert first_step == pytest.approx(0.0)


def test_gait_symmetry_arms_per_pair():
    # Swings recorded on both feet, stances not yet: only the swing pair
    # charges.
    out = terms.gait_symmetry(
        jp.array([0.385, 0.315]), jp.array([0.4, 0.0]), floor=0.1
    )
    assert out == pytest.approx((0.07 / 0.35) ** 2, rel=1e-5)
