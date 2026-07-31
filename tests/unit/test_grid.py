"""Pure-function tests for the robustness grid's plant perturbations
(port item 4.4): the torque lag filter, the speed-dependent torque
envelope, the Kt-miscalibration scaling, and the grid cell naming.

Model-free by construction: every function under test takes arrays (or a
duck-typed model holding the four actuator arrays MuJoCo would have) and
returns arrays. The rollout half of the grid -- the explicit-PD substep
loop, which does need a real model -- is
tests/integration/test_grid_env.py.
"""

from __future__ import annotations

import math
import types

import numpy as np
import pytest

from humanoid_lab.eval import grid


# -- lag filter ------------------------------------------------------------


def test_lag_coeff_is_the_first_order_step_response():
    assert grid.lag_coeff(0.005, 0.01) == pytest.approx(1.0 - math.exp(-0.5))
    assert grid.lag_coeff(0.005, 0.005) == pytest.approx(1.0 - math.exp(-1.0))


def test_lag_coeff_goes_to_one_as_the_time_constant_vanishes():
    """The native no-lag limit: a tiny time constant passes the target
    through immediately."""
    assert grid.lag_coeff(0.005, 1e-4) == pytest.approx(1.0, abs=1e-12)
    assert grid.lag_coeff(0.005, 1e-6) == 1.0


def test_lag_coeff_shrinks_as_the_time_constant_grows():
    coeffs = [grid.lag_coeff(0.005, tau) for tau in (0.002, 0.005, 0.02, 0.1)]
    assert coeffs == sorted(coeffs, reverse=True)
    assert all(0.0 < c <= 1.0 for c in coeffs)


def test_lag_coeff_handles_the_zero_and_negative_boundary_without_dividing():
    # lag_tau == 0 is the limit's boundary: a --torque-envelope-only cell
    # runs the same filter as a no-op rather than a separate code branch.
    assert grid.lag_coeff(0.005, 0.0) == 1.0
    assert grid.lag_coeff(0.005, -1.0) == 1.0


def test_lag_update_converges_to_the_target_from_zero():
    dt_sub, lag_tau = 0.005, 0.02
    coeff = grid.lag_coeff(dt_sub, lag_tau)
    target = np.array([10.0, -4.0])
    tau = np.zeros(2)
    for k in range(1, 40):
        tau = np.asarray(grid.lag_update(tau, target, coeff))
        expected = target * (1.0 - math.exp(-k * dt_sub / lag_tau))
        assert tau == pytest.approx(expected, rel=1e-6)


def test_lag_update_at_coeff_one_is_a_passthrough():
    tau = np.asarray(grid.lag_update(np.array([3.0, -7.0]), np.array([1.0, 2.0]), 1.0))
    assert tau == pytest.approx([1.0, 2.0])


# -- torque envelope -------------------------------------------------------


def test_torque_envelope_limit_is_flat_below_omega_b():
    cap = np.array([40.0, 40.0, 40.0])
    qvel = np.array([0.0, 2.0, 5.0])
    limit = np.asarray(grid.torque_envelope_limit(qvel, cap, 5.0, 15.0))
    assert limit == pytest.approx([40.0, 40.0, 40.0])


def test_torque_envelope_limit_ramps_linearly_to_zero_at_omega_0():
    cap = np.array([40.0, 40.0, 40.0])
    # Halfway along the 5 -> 15 ramp is 10 rad/s.
    qvel = np.array([10.0, 15.0, 20.0])
    limit = np.asarray(grid.torque_envelope_limit(qvel, cap, 5.0, 15.0))
    assert limit == pytest.approx([20.0, 0.0, 0.0])


def test_torque_envelope_limit_ignores_the_sign_of_the_speed():
    cap = np.array([40.0, 40.0])
    forward = np.asarray(grid.torque_envelope_limit(np.array([10.0, 12.0]), cap, 5.0, 15.0))
    reverse = np.asarray(grid.torque_envelope_limit(np.array([-10.0, -12.0]), cap, 5.0, 15.0))
    assert forward == pytest.approx(reverse)


def test_torque_envelope_limit_is_per_actuator():
    cap = np.array([40.0, 10.0])
    limit = np.asarray(grid.torque_envelope_limit(np.array([10.0, 10.0]), cap, 5.0, 15.0))
    assert limit == pytest.approx([20.0, 5.0])


def test_apply_torque_envelope_caps_driving_torque():
    cap = np.array([40.0, 40.0])
    # Both same-sign (driving) at 10 rad/s, where the envelope allows 20.
    tau = np.array([35.0, -35.0])
    qvel = np.array([10.0, -10.0])
    out = np.asarray(grid.apply_torque_envelope(tau, qvel, cap, 5.0, 15.0))
    assert out == pytest.approx([20.0, -20.0])


def test_apply_torque_envelope_exempts_braking_torque():
    """Opposite signs is braking (regenerative): it keeps the full static
    cap, because it is not limited by the bus-voltage headroom the ramp
    models."""
    cap = np.array([40.0, 40.0])
    tau = np.array([35.0, -35.0])
    qvel = np.array([-10.0, 10.0])
    out = np.asarray(grid.apply_torque_envelope(tau, qvel, cap, 5.0, 15.0))
    assert out == pytest.approx([35.0, -35.0])


def test_apply_torque_envelope_still_holds_braking_to_the_static_cap():
    cap = np.array([40.0])
    out = np.asarray(grid.apply_torque_envelope(np.array([90.0]), np.array([-10.0]), cap, 5.0, 15.0))
    assert out == pytest.approx([40.0])


def test_apply_torque_envelope_leaves_torque_under_the_limit_alone():
    cap = np.array([40.0, 40.0])
    tau = np.array([5.0, -1.0])
    qvel = np.array([10.0, 10.0])
    out = np.asarray(grid.apply_torque_envelope(tau, qvel, cap, 5.0, 15.0))
    assert out == pytest.approx([5.0, -1.0])


def test_apply_torque_envelope_zeroes_driving_torque_past_omega_0():
    cap = np.array([40.0])
    out = np.asarray(grid.apply_torque_envelope(np.array([30.0]), np.array([20.0]), cap, 5.0, 15.0))
    assert out == pytest.approx([0.0])


def test_zero_qvel_counts_as_driving_and_keeps_the_flat_cap():
    """tau*qvel == 0 lands in the driving branch by the `>= 0` test, and
    the envelope is flat at zero speed, so a stalled joint is unaffected."""
    cap = np.array([40.0])
    out = np.asarray(grid.apply_torque_envelope(np.array([90.0]), np.array([0.0]), cap, 5.0, 15.0))
    assert out == pytest.approx([40.0])


# -- envelope spec parsing --------------------------------------------------


def test_parse_torque_envelope_reads_a_pair():
    assert grid.parse_torque_envelope("5,15") == (5.0, 15.0)
    assert grid.parse_torque_envelope("0,12.5") == (0.0, 12.5)


def test_parse_torque_envelope_passes_none_through():
    assert grid.parse_torque_envelope(None) is None


@pytest.mark.parametrize("spec", ["5", "5,10,15", "a,b", "", "5,"])
def test_parse_torque_envelope_rejects_a_malformed_spec(spec):
    with pytest.raises(ValueError, match="OMEGA_B,OMEGA_0"):
        grid.parse_torque_envelope(spec)


@pytest.mark.parametrize("spec", ["15,5", "10,10", "-1,10"])
def test_parse_torque_envelope_rejects_a_non_positive_ramp(spec):
    with pytest.raises(ValueError, match="0 <= OMEGA_B < OMEGA_0"):
        grid.parse_torque_envelope(spec)


# -- Kt miscalibration ------------------------------------------------------


def _fake_pd_model(nu: int = 2):
    """The four actuator arrays apply_kt_miscalibration touches, shaped and
    filled the way PositionPD.inject leaves a built model: gainprm[:, 0] =
    kp, biasprm[:, 1] = -kp, biasprm[:, 2] = -kd, forcerange = +-effort.
    A plain namespace, not an MjModel: this is a unit test."""
    gainprm = np.zeros((nu, 10))
    gainprm[:, 0] = [50.0, 60.0][:nu]
    biasprm = np.zeros((nu, 10))
    biasprm[:, 1] = [-50.0, -60.0][:nu]
    biasprm[:, 2] = [-2.0, -3.0][:nu]
    forcerange = np.array([[-40.0, 40.0], [-55.0, 55.0]][:nu])
    return types.SimpleNamespace(
        actuator_gainprm=gainprm, actuator_biasprm=biasprm, actuator_forcerange=forcerange
    )


def test_alpha_scales_gains_and_cap_together():
    m = _fake_pd_model()
    grid.apply_kt_miscalibration(m, 2.0)
    assert m.actuator_gainprm[:, 0] == pytest.approx([100.0, 120.0])
    assert m.actuator_biasprm[:, 1] == pytest.approx([-100.0, -120.0])
    assert m.actuator_biasprm[:, 2] == pytest.approx([-4.0, -6.0])
    assert m.actuator_forcerange.ravel() == pytest.approx([-80.0, 80.0, -110.0, 110.0])


def test_alpha_keeps_the_bias_consistent_with_the_new_kp():
    """biasprm[:, 1] must stay -kp after the scale, or the servo law stops
    being kp*(ctrl - qpos) - kd*qvel."""
    m = _fake_pd_model()
    grid.apply_kt_miscalibration(m, 0.7)
    assert m.actuator_biasprm[:, 1] == pytest.approx(-m.actuator_gainprm[:, 0])


def test_alpha_one_is_a_bitwise_no_op():
    m = _fake_pd_model()
    before = (
        m.actuator_gainprm.copy(),
        m.actuator_biasprm.copy(),
        m.actuator_forcerange.copy(),
    )
    grid.apply_kt_miscalibration(m, 1.0)
    assert (m.actuator_gainprm == before[0]).all()
    assert (m.actuator_biasprm == before[1]).all()
    assert (m.actuator_forcerange == before[2]).all()


def test_alpha_leaves_the_unused_gain_columns_alone():
    m = _fake_pd_model()
    m.actuator_gainprm[:, 3] = 9.0
    grid.apply_kt_miscalibration(m, 3.0)
    assert m.actuator_gainprm[:, 3] == pytest.approx([9.0, 9.0])


def test_alpha_rejects_a_non_positive_factor():
    with pytest.raises(ValueError, match="alpha"):
        grid.apply_kt_miscalibration(_fake_pd_model(), 0.0)
    with pytest.raises(ValueError, match="alpha"):
        grid.apply_kt_miscalibration(_fake_pd_model(), -1.0)


# -- the explicit-PD path's guards ------------------------------------------
#
# These fire before the builder touches a model, so a duck-typed env is
# enough and the checks stay in the fast suite. What the loop DOES to the
# physics is tests/integration/test_grid_env.py.


def _fake_env(push=False, no_progress=False, symmetry=False, model="pd"):
    return types.SimpleNamespace(
        _config={
            "push": {"enable": push},
            "no_progress": {"enable": no_progress},
            "symmetry": {"enable": symmetry},
        },
        _preset=types.SimpleNamespace(model=model),
    )


def test_the_explicit_path_refuses_a_cell_that_perturbs_nothing():
    """lag_tau 0 with no envelope means "use the native env unchanged", not
    "use this path with a zero filter state". The baseline cell is the
    native path, and that is the whole honesty requirement."""
    with pytest.raises(ValueError, match="native"):
        grid.make_explicit_pd_rollout_fns(_fake_env(), 0.0)


@pytest.mark.parametrize("flag", ["push", "no_progress", "symmetry"])
def test_the_explicit_path_refuses_an_env_still_running_training_stochastics(flag):
    """The step function mirrors the MEASUREMENT env's step, which has
    pushes, the no-progress cut and the mirror off. An env with one still on
    would have it silently dropped, making the cell a different experiment
    from the battery it is compared against."""
    with pytest.raises(ValueError, match=flag):
        grid.make_explicit_pd_rollout_fns(_fake_env(**{flag: True}), 0.01)


def test_the_explicit_path_refuses_a_non_pd_preset():
    """Under `ideal_torque` the gain/bias params are 1.0 and 0.0, which are
    not gains: the loop would run a servo at kp 1, kd 0."""
    with pytest.raises(ValueError, match="ideal_torque"):
        grid.make_explicit_pd_rollout_fns(_fake_env(model="ideal_torque"), 0.01)


# -- grid cell naming -------------------------------------------------------


@pytest.mark.parametrize(
    "alpha,lag_tau,envelope",
    [
        (1.0, 0.0, None),
        (1.58, 0.005, None),
        (0.8, 0.01, (5.0, 15.0)),
        (1.0, 0.0, (0.0, 12.5)),
    ],
)
def test_cell_name_round_trips_through_the_report_parser(alpha, lag_tau, envelope):
    from humanoid_lab.eval import grid_report

    name = grid.cell_name(alpha, lag_tau, envelope)
    assert name.endswith(".json")
    assert grid_report.parse_cell_name(name) == (
        alpha,
        round(lag_tau * 1000),
        "none" if envelope is None else f"{envelope[0]:g}-{envelope[1]:g}",
    )


def test_cell_name_states_every_axis():
    assert grid.cell_name(1.0, 0.0, None) == "battery_a1_lag0ms_envnone.json"
    assert grid.cell_name(1.58, 0.005, (5.0, 15.0)) == "battery_a1.58_lag5ms_env5-15.json"
