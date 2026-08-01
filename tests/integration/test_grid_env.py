"""The robustness grid through a real env: the explicit-PD
substep loop, the honesty property that ties it to the native pipeline, and
a mini-grid through the aggregator.

The honesty property: the baseline cell takes
the NATIVE rollout path. There is no shared code path through a tiny lag
value -- `eval/battery.py::run_battery` branches, and that branch is the
divergence point. What must hold instead, and what
`test_explicit_pd_reproduces_the_native_pipeline_at_a_vanishing_lag` proves,
is that the explicit-PD path REPRODUCES the native one as the lag goes to
zero, within a stated tolerance.

Every rollout below drives the same scripted action sequence through both
plants (`_scripted_policy` ignores the observation), so the only difference
between two rollouts is the physics, never the policy's reaction to it.

`roboto_origin` with `real_pose_ref`, not `asimov_v1`: there is no trained
checkpoint in this repo, and an untrained policy has to stay upright long
enough for the post-settle window to score anything. Under a neutral or
small scripted action `asimov_v1` is on the floor by step 46 -- inside
`eval/battery.py`'s 50-step settle window, so nothing is measured at all --
while roboto's settled anchor holds it up past step 85. `asimov_v1` also
cannot use `real_pose_ref`: its `home` keyframe is not a standing
equilibrium and `_check_settled` refuses it (see envs/base.py).
"""

from __future__ import annotations

import copy
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

ROBOT_DIR = paths.ROBOTS_DIR / "roboto_origin"
PRESET = "sizing_ideal"  # a `pd` model: the explicit-PD path needs real kp/kd

# Rollout length. Long enough to outlive eval/battery.py's 50-step settle
# window with 30 scored steps left; short of the step-85 fall an untrained
# policy takes here, so no comparison below is a comparison of two different
# falls.
STEPS = 80

# Scripted action amplitude. Large enough that the servo has something to
# track (a frozen setpoint would make `tracking_error` measure the sag, not
# the loop), small enough to stay inside the fall margin above.
AMPLITUDE = 0.3

# The MEASURED native-vs-explicit agreement at a vanishing lag: 1.0e-6
# relative on tracking_err_rms (native 0.0582795420, explicit
# 0.0582795991), measured 2026-07-31 on the CPU backend with the parameters
# in this module. At lag_tau = 1e-4 s the filter coefficient is 1.0 to
# machine precision, so the two paths differ only in float ordering --
# mujoco sums gain*ctrl + bias.(1, qpos, qvel) inside mj_fwdActuation, the
# explicit loop computes kp*(ctrl - qpos) - kd*qvel in JAX -- and in the
# substep bookkeeping around it.
#
# The bar asserted below is 1e-4: a hundredfold margin over the measured
# number for float-ordering drift across platforms.
NATIVE_MATCH_MEASURED = 1.0e-6
NATIVE_MATCH_REL = 1.0e-4

# A lag long enough to be visibly different from no lag at all. 10 ms is the
# top rung of the usual (0, 5, 10 ms) sweep.
LAG_TAU = 0.010

# An envelope that actually bites on this plant. `sizing_ideal` is a
# generous-cap preset (peak |tau| here is 52% of cap) and the joints reach
# about 3.1 rad/s, so wide envelopes clamp nothing at all: the plateau has
# to end well inside the speed range to be a perturbation. See
# docs/configuration.md on choosing OMEGA_B/OMEGA_0 from a real motor curve.
ENVELOPE = (0.5, 2.0)


def _scripted_policy(env):
    """A deterministic action sequence that ignores the observation.

    Both plants therefore see the identical action at every step, so a
    difference between two rollouts is the physics and nothing else.
    """
    j = jp.arange(env.action_size, dtype=jp.float32)
    step = {"i": 0}

    def inf(_obs, _rng):
        t = step["i"] * env.dt
        step["i"] += 1
        return AMPLITUDE * jp.sin(2.0 * jp.pi * 1.5 * t + j), {}

    inf.reset_counter = lambda: step.__setitem__("i", 0)
    return inf


def _stand(_i):
    return np.array([0.0, 0.0, 0.0])


@pytest.fixture(scope="module")
def env():
    overrides = _measurement_env_overrides({"hydra_config": {"task": {"env": {}}}})
    overrides["real_pose_ref"] = True
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
    return _roll(env, inf, *grid.make_explicit_pd_rollout_fns(env, 1e-4))


@pytest.fixture(scope="module")
def real_lag(env, inf):
    return _roll(env, inf, *grid.make_explicit_pd_rollout_fns(env, LAG_TAU))


@pytest.fixture(scope="module")
def enveloped_fns(env):
    """Envelope only, no lag: the envelope forces the explicit-PD path even
    at lag_tau 0, where the filter is a passthrough."""
    return grid.make_explicit_pd_rollout_fns(env, 0.0, ENVELOPE)


@pytest.fixture(scope="module")
def enveloped(env, inf, enveloped_fns):
    return _roll(env, inf, *enveloped_fns)


# -- the honesty property ----------------------------------------------------


def test_explicit_pd_reproduces_the_native_pipeline_at_a_vanishing_lag(native, vanishing_lag):
    native_rec, native_fell = native
    explicit_rec, explicit_fell = vanishing_lag

    assert native_fell == explicit_fell
    assert len(native_rec["ctrl"]) == len(explicit_rec["ctrl"])

    native_err = tracking_error(native_rec["ctrl"], native_rec["qpos"])["rms"]
    explicit_err = tracking_error(explicit_rec["ctrl"], explicit_rec["qpos"])["rms"]
    assert native_err is not None, "nothing outlived the settle window; the rollout is too short"
    rel = abs(explicit_err - native_err) / native_err
    assert rel < NATIVE_MATCH_REL, (
        f"explicit-PD tracking_err_rms {explicit_err} vs native {native_err} is "
        f"{rel:.3e} apart, over the stated {NATIVE_MATCH_REL:.0e} tolerance "
        f"(measured {NATIVE_MATCH_MEASURED:.1e} when this was written)"
    )


def test_the_explicit_path_leaves_the_position_setpoint_in_ctrl(native, vanishing_lag):
    """The substep loop writes an applied TORQUE into the torque-mode
    model's ctrl channel. It has to put the position setpoint back, or
    `tracking_error` diffs newton-metres against radians."""
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
    diff = np.abs(np.asarray(lagged_rec["tau"])[:n] - np.asarray(native_rec["tau"])[:n]).max()
    assert diff > 1.0  # N*m; measured 19.2 at LAG_TAU


def test_the_lag_state_zeroes_at_reset(env):
    reset_fn, _ = grid.make_explicit_pd_rollout_fns(env, LAG_TAU)
    state = jax.jit(reset_fn)(jax.random.PRNGKey(0))
    assert np.asarray(state.info["tau_applied"]) == pytest.approx(np.zeros(env.action_size))


def test_the_lag_state_persists_across_control_steps(env):
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


def test_the_envelope_holds_driving_torque_under_the_speed_dependent_cap(env, enveloped_fns):
    """Checked at the substep boundary, where the clamp is applied.

    The envelope reads the qvel at the START of a substep, and
    `rollout`'s per-control-step arrays hold the state at the END of one --
    a joint that accelerated in between would fail a comparison of the two
    even with the clamp working perfectly. So this drives the plant into
    real joint speeds with the enveloped step function, then runs ONE
    substep of the same loop over that state, where the speed the envelope
    read is exactly the speed the passed-in data carries.
    """
    from mujoco import mjx

    reset, step = jax.jit(enveloped_fns[0]), jax.jit(enveloped_fns[1])
    inf = _scripted_policy(env)
    state = reset(jax.random.PRNGKey(0))
    for _ in range(30):  # get the joints moving; at reset every qvel is 0
        action, _ = inf(state.obs, None)
        state = step(state, action)
    qvel = np.asarray(state.data.qvel)[np.asarray(env._vadr)]
    assert np.abs(qvel).max() > ENVELOPE[0], "the plant never left the envelope's plateau"

    cap = np.asarray(env.mj_model.actuator_forcerange[:, 1])
    _data, tau = grid.explicit_pd_substeps(
        mjx.put_model(grid.torque_mode_model(env.mj_model), impl=env._backend),
        env._qadr, env._vadr,
        jp.array(env.mj_model.actuator_gainprm[:, 0]),
        jp.array(-env.mj_model.actuator_biasprm[:, 2]),
        jp.array(cap), 1.0, state.data, state.data.ctrl,
        jp.zeros(env.action_size), 1, envelope=ENVELOPE,
    )
    tau = np.asarray(tau)
    allowed = np.asarray(grid.torque_envelope_limit(qvel, cap, *ENVELOPE))
    driving = tau * qvel >= 0.0
    assert (np.abs(tau)[driving] <= allowed[driving] + 1e-4).all()
    assert (np.abs(tau) <= cap + 1e-4).all()  # braking keeps the static cap


def test_the_envelope_needs_no_lag_to_take_effect(native, enveloped):
    native_rec, _ = native
    env_rec, _ = enveloped
    n = min(len(native_rec["tau"]), len(env_rec["tau"]))
    diff = np.abs(np.asarray(env_rec["tau"])[:n] - np.asarray(native_rec["tau"])[:n]).max()
    assert diff > 1.0  # N*m; measured 11.7 at ENVELOPE


# -- alpha on a built model --------------------------------------------------


def test_alpha_scales_the_built_model_s_post_injection_gains_and_cap(env):
    """The preset's numbers reach the model through preset loading and the
    `actuators.overrides` merge; alpha scales what the model ENDED UP with,
    so it composes with any preset."""
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
    # The env's own model is untouched: alpha ran on a copy here, and inside
    # run_battery it runs on a model built seconds earlier in the eval
    # process. Nothing training-side ever holds either one.
    assert env.mj_model.actuator_gainprm[:, 0] == pytest.approx(kp_before)


def test_the_envelope_plateau_is_the_alpha_scaled_cap(env):
    """Composition: alpha first, then the envelope reads the cap it left."""
    m = copy.deepcopy(env.mj_model)
    grid.apply_kt_miscalibration(m, 1.58)
    cap = m.actuator_forcerange[:, 1]
    plateau = np.asarray(grid.torque_envelope_limit(np.zeros_like(cap), cap, *ENVELOPE))
    assert plateau == pytest.approx(1.58 * np.asarray(env.mj_model.actuator_forcerange[:, 1]))


# -- a mini grid through the aggregator --------------------------------------


def test_a_three_cell_grid_scores_through_the_report(tmp_path, env, native, real_lag, enveloped):
    run_dir = tmp_path / "mini"
    grid_dir = run_dir / "grid"
    grid_dir.mkdir(parents=True)

    cap = np.asarray(env.mj_model.actuator_forcerange[:, 1])
    cells = {
        (1.0, 0.0, None): native,
        (1.0, LAG_TAU, None): real_lag,
        (1.0, 0.0, ENVELOPE): enveloped,
    }
    for (alpha, lag_tau, envelope), (rec, fell_at) in cells.items():
        result = {
            "run": "mini", "checkpoint": "0",
            "grid": {
                "alpha": alpha, "lag_tau": lag_tau,
                "torque_envelope": list(envelope) if envelope else None,
                "path": "native" if (lag_tau == 0 and envelope is None) else "explicit_pd",
            },
            "stand": scenario_result("stand", rec, fell_at, env.dt, cap, STEPS),
        }
        (grid_dir / grid.cell_name(alpha, lag_tau, envelope)).write_text(json.dumps(result))

    assert len(grid_report.find_cells(run_dir)) == 3

    md = grid_report.render_markdown(grid_report.build_grid([run_dir]))
    rows = [line for line in md.splitlines() if line.startswith("| mini |")]
    assert len(rows) == 2  # alpha 1.0 x {none, the envelope}
    assert "| 0ms | 10ms |" in md

    flat = next(r for r in rows if "| none |" in r)
    assert "MISSING" not in flat  # both lags were run on the flat-cap axis
    assert flat.count("PASS") + flat.count("FAIL") == 2

    ramped = next(r for r in rows if f"| {grid.envelope_tag(ENVELOPE)} |" in r)
    assert "MISSING" in ramped  # lag 10 ms was never run against the envelope

    # The vibration gate ran: the baseline cell is present, so it supplied
    # its own reference.
    assert "- `vibration`" not in md
