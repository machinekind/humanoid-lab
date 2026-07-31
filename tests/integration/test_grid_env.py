"""The robustness grid through a real env (port item 4.4): the explicit-PD
substep loop, the honesty property that ties it to the native pipeline, and
a mini-grid through the aggregator.

The honesty property (PORT.md's "Eval integrity"): the baseline cell takes
the NATIVE rollout path. There is no shared code path through a tiny lag
value -- `eval/battery.py::run_battery` branches, and the branch is the
divergence point. What must hold instead, and what
`test_explicit_pd_reproduces_the_native_pipeline_at_a_vanishing_lag` proves,
is that the explicit-PD path REPRODUCES the native one as the lag goes to
zero, within a stated tolerance.

Every rollout below drives the same scripted action sequence through both
plants (`_scripted_policy` ignores the observation), so the only difference
between two rollouts is the physics, never the policy's reaction to it.
"""

from __future__ import annotations

import json

import jax
import jax.numpy as jp
import numpy as np
import pytest

from humanoid_lab import paths
from humanoid_lab.eval import grid, grid_report
from humanoid_lab.eval.battery import (
    _measurement_env_overrides,
    rollout,
    scenario_result,
    tracking_error,
)
from humanoid_lab.registry import make_env

ROBOT_DIR = paths.ROBOTS_DIR / "asimov_v1"
PRESET = "sizing_ideal"  # a `pd` model: the explicit-PD path needs real kp/kd

# Rollout length for every comparison below. Long enough to outlive
# eval/battery.py's 50-step settle window with 100 scored steps left, short
# enough that four jitted step functions still compile inside the slow
# suite's budget.
STEPS = 150

# The measured native-vs-explicit agreement at a vanishing lag. See the
# test that uses it: at lag_tau = 1e-4 s the filter coefficient is 1.0 to
# machine precision, so the two paths differ only in float ordering (mujoco
# sums gain*ctrl + bias.(1, qpos, qvel); the explicit loop computes
# kp*(ctrl - qpos) - kd*qvel) and in the substep bookkeeping around it.
# w01-tek measured under 1% on its own track_err_rms; this repo's measured
# number is recorded in docs/configuration.md.
NATIVE_MATCH_REL = 0.01

# A lag long enough to be visibly different from no lag at all. 10 ms is one
# of the rungs w01-tek's stiff_grid.job sweeps (0, 5, 10 ms).
LAG_TAU = 0.010


def _scripted_policy(env):
    """A deterministic action sequence that ignores the observation.

    Both plants therefore see the identical action at every step, so a
    difference between two rollouts is the physics and nothing else. A
    slow joint-space sinusoid (rather than a constant) is what makes the
    servo actually track something: `tracking_error` on a frozen setpoint
    would measure the sag, not the loop.
    """
    nu = env.action_size
    j = jp.arange(nu, dtype=jp.float32)
    step = {"i": 0}

    def inf(_obs, _rng):
        t = step["i"] * env.dt
        step["i"] += 1
        return 0.5 * jp.sin(2.0 * jp.pi * 1.5 * t + j), {}

    def reset_counter():
        step["i"] = 0

    inf.reset_counter = reset_counter
    return inf


def _stand(_i):
    return np.array([0.0, 0.0, 0.0])


@pytest.fixture(scope="module")
def env():
    overrides = _measurement_env_overrides({"hydra_config": {"task": {"env": {}}}})
    return make_env("joystick", ROBOT_DIR, PRESET, overrides)


@pytest.fixture(scope="module")
def inf(env):
    return _scripted_policy(env)


def _roll(env, inf, reset_fn, step_fn):
    inf.reset_counter()
    rec, fell_at, _budget = rollout(
        env, jax.jit(reset_fn), jax.jit(step_fn), inf, _stand, STEPS
    )
    return rec, fell_at


@pytest.fixture(scope="module")
def native(env, inf):
    return _roll(env, inf, env.reset, env.step)


@pytest.fixture(scope="module")
def vanishing_lag(env, inf):
    reset_fn, step_fn = grid.make_explicit_pd_rollout_fns(env, 1e-4)
    return _roll(env, inf, reset_fn, step_fn)


@pytest.fixture(scope="module")
def real_lag(env, inf):
    reset_fn, step_fn = grid.make_explicit_pd_rollout_fns(env, LAG_TAU)
    return _roll(env, inf, reset_fn, step_fn)


@pytest.fixture(scope="module")
def enveloped(env, inf):
    """Envelope only, no lag: the envelope forces the explicit-PD path even
    at lag_tau 0, where the filter is a passthrough."""
    reset_fn, step_fn = grid.make_explicit_pd_rollout_fns(env, 0.0, (2.0, 6.0))
    return _roll(env, inf, reset_fn, step_fn)


# -- the honesty property ----------------------------------------------------


def test_explicit_pd_reproduces_the_native_pipeline_at_a_vanishing_lag(native, vanishing_lag):
    native_rec, native_fell = native
    explicit_rec, explicit_fell = vanishing_lag

    assert native_fell == explicit_fell
    assert len(native_rec["ctrl"]) == len(explicit_rec["ctrl"])

    native_err = tracking_error(native_rec["ctrl"], native_rec["qpos"])["rms"]
    explicit_err = tracking_error(explicit_rec["ctrl"], explicit_rec["qpos"])["rms"]
    assert native_err is not None
    rel = abs(explicit_err - native_err) / native_err
    assert rel < NATIVE_MATCH_REL, (
        f"explicit-PD tracking_err_rms {explicit_err} vs native {native_err} "
        f"is {rel:.4%} apart, over the stated {NATIVE_MATCH_REL:.0%} tolerance"
    )


def test_the_explicit_path_leaves_the_position_setpoint_in_ctrl(native, vanishing_lag):
    """The substep loop writes an applied TORQUE into the torque-mode
    model's ctrl channel. It has to put the position setpoint back, or
    `tracking_error` diffs newton-metres against radians -- the symptom
    w01-tek's own comment records."""
    native_rec, _ = native
    explicit_rec, _ = vanishing_lag
    assert np.asarray(explicit_rec["ctrl"]) == pytest.approx(
        np.asarray(native_rec["ctrl"]), rel=1e-3, abs=1e-4
    )


def test_a_real_lag_actually_changes_the_plant(native, real_lag):
    """The tolerance above would be vacuous if the explicit path were
    insensitive to its own time constant."""
    native_rec, _ = native
    lagged_rec, _ = real_lag
    n = min(len(native_rec["tau"]), len(lagged_rec["tau"]))
    native_tau = np.asarray(native_rec["tau"])[:n]
    lagged_tau = np.asarray(lagged_rec["tau"])[:n]
    assert np.abs(lagged_tau - native_tau).max() > 1.0  # N*m


def test_the_lag_state_zeroes_at_reset(env):
    reset_fn, _ = grid.make_explicit_pd_rollout_fns(env, LAG_TAU)
    state = jax.jit(reset_fn)(jax.random.PRNGKey(0))
    assert np.asarray(state.info["tau_applied"]) == pytest.approx(np.zeros(env.action_size))


def test_the_lag_state_persists_across_control_steps(env, inf):
    reset_fn, step_fn = grid.make_explicit_pd_rollout_fns(env, LAG_TAU)
    reset, step = jax.jit(reset_fn), jax.jit(step_fn)
    state = reset(jax.random.PRNGKey(0))
    state = step(state, jp.zeros(env.action_size))
    first = np.asarray(state.info["tau_applied"])
    state = step(state, jp.zeros(env.action_size))
    second = np.asarray(state.info["tau_applied"])
    assert np.abs(first).max() > 0.0
    assert not np.allclose(first, second)


# -- the torque envelope through real physics --------------------------------


def test_the_envelope_holds_driving_torque_under_the_speed_dependent_cap(env, enveloped):
    """Every recorded (step, joint) sample that was DRIVING must sit inside
    the envelope the joint's own speed allowed."""
    rec, _ = enveloped
    cap = np.asarray(env.mj_model.actuator_forcerange[:, 1])
    tau = np.asarray(rec["tau"])
    qvel = np.asarray(rec["qvel"])
    driving = tau * qvel >= 0.0
    allowed = np.asarray(grid.torque_envelope_limit(qvel, cap, 2.0, 6.0))
    # 1e-4 N*m of slack: the envelope clamps the torque WRITTEN to ctrl, and
    # the recorded number is mujoco's own actuator_force read back out.
    assert (np.abs(tau)[driving] <= allowed[driving] + 1e-4).all()


def test_the_envelope_needs_no_lag_to_take_effect(native, enveloped):
    native_rec, _ = native
    env_rec, _ = enveloped
    n = min(len(native_rec["tau"]), len(env_rec["tau"]))
    assert not np.allclose(
        np.asarray(env_rec["tau"])[:n], np.asarray(native_rec["tau"])[:n]
    )


# -- alpha on a built model --------------------------------------------------


def test_alpha_scales_the_built_model_s_post_injection_gains_and_cap(env):
    """`sizing_ideal`'s numbers reach the model through preset loading and
    the override merge; alpha scales what the model ENDED UP with, so it
    composes with any preset."""
    import copy

    m = copy.deepcopy(env.mj_model)
    kp_before = m.actuator_gainprm[:, 0].copy()
    kd_before = -m.actuator_biasprm[:, 2].copy()
    cap_before = m.actuator_forcerange[:, 1].copy()

    grid.apply_kt_miscalibration(m, 1.58)

    assert m.actuator_gainprm[:, 0] == pytest.approx(1.58 * kp_before)
    assert -m.actuator_biasprm[:, 2] == pytest.approx(1.58 * kd_before)
    assert m.actuator_forcerange[:, 1] == pytest.approx(1.58 * cap_before)
    # The bias tracks the new kp, or the servo law stops holding.
    assert m.actuator_biasprm[:, 1] == pytest.approx(-m.actuator_gainprm[:, 0])
    # The env's own model is untouched: the grid mutates a copy or the eval
    # process's own model, never anything the training path reads.
    assert env.mj_model.actuator_gainprm[:, 0] == pytest.approx(kp_before)


# -- guards ------------------------------------------------------------------


def test_the_explicit_path_refuses_a_cell_that_perturbs_nothing(env):
    """lag_tau 0 with no envelope means "use the native env unchanged", not
    "use this path with a zero filter state" -- see the honesty note."""
    with pytest.raises(ValueError, match="native"):
        grid.make_explicit_pd_rollout_fns(env, 0.0)


def test_the_explicit_path_refuses_an_env_still_running_training_stochastics(env):
    """The step function here mirrors the measurement env's step, which has
    pushes, the no-progress cut and the mirror off. Anything else would be
    silently dropped."""
    overrides = _measurement_env_overrides({"hydra_config": {"task": {"env": {}}}})
    overrides["push"] = {"enable": True}
    pushy = make_env("joystick", ROBOT_DIR, PRESET, overrides)
    with pytest.raises(ValueError, match="push"):
        grid.make_explicit_pd_rollout_fns(pushy, LAG_TAU)


# -- a mini grid through the aggregator --------------------------------------


def test_a_three_cell_grid_scores_through_the_report(tmp_path, env, native, real_lag, enveloped):
    run_dir = tmp_path / "run"
    grid_dir = run_dir / "grid"
    grid_dir.mkdir(parents=True)

    cap = np.asarray(env.mj_model.actuator_forcerange[:, 1])
    cells = {
        (1.0, 0.0, None): native,
        (1.0, LAG_TAU, None): real_lag,
        (1.0, 0.0, (2.0, 6.0)): enveloped,
    }
    for (alpha, lag_tau, envelope), (rec, fell_at) in cells.items():
        result = {
            "run": "mini", "checkpoint": "0",
            "alpha": alpha, "lag_tau": lag_tau,
            "torque_envelope": list(envelope) if envelope else None,
            "stand": scenario_result("stand", rec, fell_at, env.dt, cap, STEPS),
        }
        (grid_dir / grid.cell_name(alpha, lag_tau, envelope)).write_text(json.dumps(result))

    found = grid_report.find_cells(run_dir)
    assert len(found) == 3

    md = grid_report.render_markdown(*grid_report.build_grid([run_dir]))
    assert "0ms" in md and "10ms" in md
    assert "2-6" in md  # the envelope axis is its own row
    assert "MISSING" not in md
