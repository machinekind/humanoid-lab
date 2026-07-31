"""Robustness grid: eval-only plant perturbations (port item 4.4).

Three sim2real risks, probed by mutating a run's BUILT, customized model
inside the eval process. No training code changes: `envs/` never imports
this module, and nothing here is reachable from `train.py`.

- `alpha` (Kt miscalibration) scales the effective PD gains and the torque
  cap together, in place on the model the eval process holds.
- `lag_tau` (actuator bandwidth) replaces the model's own instantaneous PD
  actuator with an explicit per-substep PD loop whose torque passes through
  a first-order lag.
- `torque_envelope` (back-EMF droop) caps DRIVING torque by joint speed. It
  can only be evaluated inside the explicit-PD loop, where per-substep qvel
  is in hand, so it forces that path even at `lag_tau` 0.

**The baseline cell takes the native rollout path.** `battery.run_battery`
branches on `lag_tau > 0 or torque_envelope is not None`, and that branch is
where the two pipelines diverge: an unperturbed cell steps `env.step`
itself, exactly as `./run.sh battery` does. There is no shared code path
through a tiny lag value, and w01-tek's `hpc/stiff_grid.job` comment
claiming every cell "runs through the SAME battery.py code path" is false in
that sense -- verified against `wojtek_rl/battery.py:911`, which carries the
same branch.

What IS true, and what `tests/integration/test_grid_env.py` proves, is the
weaker property that makes the grid's numbers comparable: the explicit-PD
path reproduces the native pipeline within a stated tolerance as the lag
goes to zero. See docs/configuration.md's "Robustness grid (eval-only)" for
this repo's measured number.

Ported from w01-tek's `training/wojtek_rl/battery.py:570-882`
(`apply_kt_miscalibration`, `lag_coeff`, `lag_update`,
`torque_envelope_limit`, `apply_torque_envelope`, `parse_torque_envelope`,
`_torque_mode_model`, `_explicit_pd_substeps`, `make_lagged_rollout_fns`),
split into its own module here rather than bolted onto `eval/battery.py`:
the battery's job is the fixed number table, and 250 lines of plant
surgery inside it would bury that.
"""

from __future__ import annotations

import copy

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from mujoco import mjx

# The actuator model the explicit-PD path can read gains off. Every preset
# shipped today resolves to this one (see docs/configuration.md's
# actuator_gains block); an `ideal_torque` preset's gainprm/biasprm are 1.0
# and 0.0, which are not gains at all, so the loop refuses it rather than
# running a servo with kp 1 and kd 0.
_PD_MODEL = "pd"


# -- Kt miscalibration -------------------------------------------------------


def apply_kt_miscalibration(model, alpha: float) -> None:
    """Scale the model's effective PD gains and torque cap by `alpha`, in
    place.

    Models a firmware torque-constant (Kt) error: the commanded kp/kd and
    the torque ceiling all scale with the real Kt together, because a servo
    loop closed in "torque units" that are wrong by a factor delivers a
    proportionally wrong torque at every point of the loop.

    Reads `actuator_gainprm[:, 0]` (kp), `actuator_biasprm[:, 1]` (-kp) and
    `[:, 2]` (-kd), and `actuator_forcerange`. Those are the POST-INJECTION
    values -- what `robot/presets.py::effective_gains` stamps into run.json,
    after preset loading and after the `actuators.overrides` merge -- so
    this composes with any preset instead of assuming a particular one. Both
    bias columns scale so the actuator's affine bias stays
    `-kp*qpos - kd*qvel` under the new kp.

    `alpha == 1.0` writes nothing, so it is a bitwise no-op rather than a
    numerically-equal rewrite.
    """
    if alpha <= 0.0:
        raise ValueError(f"alpha must be positive (it scales a torque constant), got {alpha}")
    if alpha == 1.0:
        return
    model.actuator_gainprm[:, 0] *= alpha
    model.actuator_biasprm[:, 1] *= alpha
    model.actuator_biasprm[:, 2] *= alpha
    model.actuator_forcerange[:, :] *= alpha


# -- the first-order torque lag ----------------------------------------------


def lag_coeff(dt_sub: float, lag_tau: float) -> float:
    """First-order filter step coefficient for one physics substep:
    `tau <- tau + coeff * (target - tau)`.

    As `lag_tau` shrinks, `dt_sub / lag_tau` grows and the coefficient goes
    to 1 -- immediate passthrough, the native no-lag limit. A larger
    `lag_tau` slows the filter.

    A plain Python float in, float out: `lag_tau` is a CLI scalar evaluated
    once per battery run, never a traced value, so `np.exp` is the right
    call here.

    `lag_tau <= 0` is that limit's boundary and is handled explicitly
    (coeff 1.0) rather than run through the formula, which would divide by
    zero. That lets an envelope-only cell (`lag_tau` 0, envelope set) run
    the same filter as a no-op instead of needing a second code branch --
    see `make_explicit_pd_rollout_fns`.
    """
    if lag_tau <= 0:
        return 1.0
    return 1.0 - float(np.exp(-dt_sub / lag_tau))


def lag_update(tau_applied, tau_target, coeff):
    """One first-order-filter step, vectorized over actuators.

    Pulled out of the substep loop so its step response -- from zero, after
    k updates, `tau_target * (1 - exp(-k * dt_sub / lag_tau))` -- is
    unit-testable with no physics.
    """
    return tau_applied + coeff * (tau_target - tau_applied)


# -- the speed-dependent torque envelope -------------------------------------


def torque_envelope_limit(qvel, cap, omega_b: float, omega_0: float):
    """Per-actuator maximum |torque| the DRIVING envelope permits at joint
    speed `qvel` (rad/s, vectorized over actuators).

    `cap` up to `omega_b`, ramping linearly to 0 at `omega_0`, and 0 beyond.
    `cap` is the model's static torque limit -- already alpha-scaled when
    the caller applied `apply_kt_miscalibration` first, which is how the
    envelope's plateau composes with alpha.

    The sign of `qvel` is ignored: back-EMF eats the same bus-voltage
    headroom in either rotation direction.

    jit-safe: `jp.clip`, no Python branch on a traced value.
    """
    w = jp.abs(qvel)
    ramp = cap * (omega_0 - w) / (omega_0 - omega_b)
    return jp.clip(ramp, 0.0, cap)


def apply_torque_envelope(tau, qvel, cap, omega_b: float, omega_0: float):
    """Clamp `tau` to the speed-dependent envelope.

    Only the DRIVING quadrant loses headroom at speed. `tau * qvel >= 0` is
    the motor doing positive work on the joint and gets the ramped limit;
    opposite signs is BRAKING (regenerative), which is not limited by
    available bus voltage the same way and keeps the full STATIC cap. It is
    exempt from the ramp, not from the cap: a braking torque past the
    actuator's own force limit is not a thing the motor can produce either.

    A stalled joint (`qvel == 0`) lands in the driving branch by the `>= 0`
    test, where the envelope is flat, so it is unaffected either way.

    jit-safe.
    """
    driving = tau * qvel >= 0.0
    limit = jp.where(driving, torque_envelope_limit(qvel, cap, omega_b, omega_0), cap)
    return jp.clip(tau, -limit, limit)


def parse_torque_envelope(spec):
    """Parse a `--torque-envelope` CLI value `"OMEGA_B,OMEGA_0"` into an
    `(omega_b, omega_0)` float tuple. `None` passes through unchanged (the
    CLI default: no envelope, flat cap).

    Raises `ValueError` on a malformed spec or a non-positive ramp width.
    The linear segment runs from `omega_b` down to zero at `omega_0`, so
    `omega_0 > omega_b >= 0` is required or `torque_envelope_limit` divides
    by zero (or ramps the wrong way).
    """
    if spec is None:
        return None
    parts = str(spec).split(",")
    if len(parts) != 2:
        raise ValueError(f"--torque-envelope must be 'OMEGA_B,OMEGA_0', got {spec!r}")
    try:
        omega_b, omega_0 = float(parts[0]), float(parts[1])
    except ValueError:
        raise ValueError(f"--torque-envelope must be 'OMEGA_B,OMEGA_0', got {spec!r}") from None
    if not 0 <= omega_b < omega_0:
        raise ValueError(
            f"--torque-envelope requires 0 <= OMEGA_B < OMEGA_0, got {omega_b},{omega_0}"
        )
    return omega_b, omega_0


# -- grid cell naming --------------------------------------------------------


def envelope_tag(torque_envelope) -> str:
    """The filename segment for one envelope setting: `"none"` for a flat
    cap, else `"<OMEGA_B>-<OMEGA_0>"`. `%g` so `5.0` writes as `5` and the
    tag stays free of the `_` and `.json` this convention anchors on."""
    if torque_envelope is None:
        return "none"
    omega_b, omega_0 = torque_envelope
    return f"{omega_b:g}-{omega_0:g}"


def cell_name(alpha: float, lag_tau: float, torque_envelope=None) -> str:
    """The filename one grid cell's battery JSON goes under.

    `battery_a<alpha>_lag<ms>ms_env<tag>.json`, w01-tek's own convention
    (`hpc/stiff_grid.job`). This function is the single writer of it and
    `eval/grid_report.py::parse_cell_name` the single reader;
    `tests/unit/test_grid.py` round-trips them so the two cannot drift.

    The lag is written in whole milliseconds, so two lags under 0.5 ms apart
    share a cell name. That is the resolution the sweep axis is specified at
    (0, 5, 10 ms); a cell finer than that has to name its own `--out`.
    """
    return (
        f"battery_a{alpha:g}_lag{round(lag_tau * 1000)}ms"
        f"_env{envelope_tag(torque_envelope)}.json"
    )


# -- the explicit-PD rollout path --------------------------------------------


def torque_mode_model(mj_model: mujoco.MjModel) -> mujoco.MjModel:
    """A deep copy of `mj_model` whose actuators are torque passthroughs.

    `gain = 1`, `bias = 0`: whatever is written to `ctrl` becomes joint
    torque directly, with no PD math. The substep loop below computes
    `kp*(ctrl - qpos) - kd*qvel` and the lag filter itself in JAX, and needs
    a model that will not re-interpret its already-computed torque as a
    position setpoint.

    `ctrllimited` goes too. The source actuators are position servos whose
    `ctrlrange` is in RADIANS; left on, `mj_fwdActuation` would clip a
    torque (N*m, a different scale entirely) to that range before gain/bias
    ever ran. The substep loop clips to the actuator force limit itself
    before writing ctrl, so no ctrl-side limit belongs here.

    `forcerange` is deliberately KEPT: it is the same limit the loop already
    clips to (alpha-scaled, when alpha was applied), so it is a no-op that
    costs nothing and fails closed if the loop's own clip ever regresses.
    """
    m = copy.deepcopy(mj_model)
    m.actuator_gainprm[:, 0] = 1.0
    m.actuator_gainprm[:, 1:] = 0.0
    m.actuator_biasprm[:, :] = 0.0
    m.actuator_ctrllimited[:] = 0
    return m


def explicit_pd_substeps(
    mjx_model_torque, qadr, vadr, kp, kd, limit, coeff, data, ctrl, tau_applied,
    n_substeps: int, envelope=None,
):
    """`jax.lax.scan` over one control period's physics substeps, with the
    PD torque computed explicitly instead of by the model's actuator
    gain/bias.

    Each substep reads `qpos`/`qvel` at its own start -- the same state
    mujoco's `mj_fwdActuation` would have used -- forms
    `kp*(ctrl - qpos) - kd*qvel`, clips it to the actuator force limit,
    passes it through the first-order lag, optionally clamps it to the
    speed-dependent envelope, writes it to `ctrl` and steps.

    `envelope`, when not None, is an `(omega_b, omega_0)` pair applied LAST:
    the torque the plant actually produces can never exceed what the joint's
    current speed allows. `None` skips the clamp entirely, so the traced
    program is identical to the pre-envelope one -- this is a Python branch
    on a static per-run value, not a traced one, and costs nothing unused.

    `ctrl` is one control step's position setpoint, held for every substep:
    this env has no action-delay or latency machinery (see
    `envs/joystick.py`'s module docstring), so there is no mid-period ctrl
    switch to reproduce the way w01-tek's `_step_with_latency` needs.
    """

    def _substep(carry, _):
        data, tau_applied = carry
        qvel_j = data.qvel[vadr]
        tau_pd = kp * (ctrl - data.qpos[qadr]) - kd * qvel_j
        tau_pd = jp.clip(tau_pd, -limit, limit)
        tau_applied = lag_update(tau_applied, tau_pd, coeff)
        if envelope is not None:
            omega_b, omega_0 = envelope
            tau_applied = apply_torque_envelope(tau_applied, qvel_j, limit, omega_b, omega_0)
        data = data.replace(ctrl=tau_applied)
        data = mjx.step(mjx_model_torque, data)
        return (data, tau_applied), None

    (data, tau_applied), _ = jax.lax.scan(
        _substep, (data, tau_applied), (), n_substeps
    )
    return data, tau_applied


def _require_measurement_env(env) -> None:
    """The step function below mirrors `envs/joystick.py::step` for a
    MEASUREMENT env: pushes, the no-progress cut and the mirror all off, as
    `eval/battery.py::_measurement_env_overrides` forces them.

    Those three are not reimplemented here. Silently dropping them would
    make a grid cell a different experiment from the battery it is compared
    against, so an env that still has one on is refused by name.
    """
    cfg = env._config
    for key in ("push", "no_progress", "symmetry"):
        block = cfg.get(key, None)
        if block is not None and bool(block.get("enable", False)):
            raise ValueError(
                f"the explicit-PD path needs a measurement env, but '{key}.enable' is on. "
                "Build the env through eval/battery.py's load_checkpoint_policy, which "
                "applies _measurement_env_overrides."
            )
    if env._preset.model != _PD_MODEL:
        raise ValueError(
            f"the explicit-PD path reads kp/kd off the built model's gain/bias params, "
            f"which are gains only for the '{_PD_MODEL}' actuator model; this env's preset "
            f"resolves to '{env._preset.model}'."
        )


def make_explicit_pd_rollout_fns(env, lag_tau: float, torque_envelope=None):
    """A `(reset, step)` pair for `eval/battery.py::rollout`, reproducing
    the measurement env's own reset/step except for the physics substep
    call: joint torque passes through an explicit PD loop and a first-order
    lag (time constant `lag_tau`) before being applied, instead of the
    model's built-in position actuator recomputing an instantaneous torque
    every substep.

    This is the plant-bandwidth risk `envs/joystick.py` cannot probe -- its
    actuators are ideal position servos -- without a training-path change,
    so the substitution happens here, eval-only.

    `torque_envelope`, when given, is an `(omega_b, omega_0)` pair applied
    last in each substep (see `apply_torque_envelope`). Passing one forces
    this path even at `lag_tau` 0, where `lag_coeff` makes the filter an
    explicit passthrough and the envelope is the only perturbation.

    `reset` is `env.reset` untouched, plus a per-actuator lag filter state
    (`tau_applied`) seeded to zeros in `info` -- so the lag state persists
    across control steps and zeroes at every reset.

    Requires `lag_tau > 0` or an envelope: neither set means "use the native
    env unchanged", not "use this path with a zero filter state". The
    caller branches (`eval/battery.py::run_battery`), and that branch is the
    documented divergence point between the two pipelines -- see this
    module's docstring.

    kp, kd and the force limit are read off `env.mj_model` AS IT STANDS NOW,
    so a caller that applied `apply_kt_miscalibration` first gets the
    alpha-scaled plant here too. Every cell -- alpha, lag and envelope,
    combined or alone -- goes through this one construction path, and the
    envelope's plateau is exactly that alpha-scaled cap.
    """
    if lag_tau <= 0 and torque_envelope is None:
        raise ValueError(
            "lag_tau <= 0 with no torque_envelope means the native path; "
            "eval/battery.py::run_battery branches on that rather than calling here"
        )
    _require_measurement_env(env)

    kp = jp.array(env.mj_model.actuator_gainprm[:, 0])
    kd = jp.array(-env.mj_model.actuator_biasprm[:, 2])
    limit = jp.array(env.mj_model.actuator_forcerange[:, 1])
    coeff = lag_coeff(env.sim_dt, lag_tau)
    mjx_model_torque = mjx.put_model(torque_mode_model(env.mj_model), impl=env._backend)
    n_substeps = env.n_substeps
    qadr, vadr = env._qadr, env._vadr
    nu = env.mj_model.nu

    def reset(rng):
        state = env.reset(rng)
        info = dict(state.info)
        info["tau_applied"] = jp.zeros(nu)
        return state.replace(info=info)

    def step(state, action):
        # Mirrors envs/joystick.py::step line for line, for a measurement
        # env (see _require_measurement_env): same 4-way rng split, same
        # motor targets, same contact/apex/air-time bookkeeping, same fall
        # test, same command resample, same observation. Only the physics
        # segment differs. Rewards and metrics are not computed: the battery
        # reads none of them, and `fall` -- the one value joystick.step gets
        # back from _compute_rewards -- is two comparisons, reproduced here.
        info = dict(state.info)
        rng, r_noise, r_cmd, _r_push = jax.random.split(info["rng"], 4)
        info["rng"] = rng

        motor_targets = jp.clip(
            env._actuator_model.ctrl_from_action(action, env._default_pose, env._action_scale),
            env._ctrl_lo,
            env._ctrl_hi,
        )

        data, tau_applied = explicit_pd_substeps(
            mjx_model_torque, qadr, vadr, kp, kd, limit, coeff,
            state.data, motor_targets, info["tau_applied"], n_substeps,
            envelope=torque_envelope,
        )
        info["tau_applied"] = tau_applied
        # The substep loop leaves data.ctrl holding the last substep's
        # APPLIED TORQUE (the torque-mode model's ctrl channel). Put the
        # POSITION SETPOINT back, matching what the native pipeline leaves
        # there (mjx_env.step writes the position target to ctrl every
        # substep, never a torque). rollout() reads rec["ctrl"] as a
        # setpoint to diff against qpos; left as a torque, tracking_err_rms
        # comes out inflated by the kp ratio -- the symptom w01-tek's own
        # comment records from validating against its native battery.
        data = data.replace(ctrl=motor_targets)

        contact = env._foot_contact(data)
        contact_filt = contact | info["last_contact"]

        info["swing_apex"] = jp.where(
            ~contact_filt,
            jp.maximum(info["swing_apex"], env._foot_clearance(data)),
            info["swing_apex"],
        )

        gravity = env._gravity_body(data)
        fall = (data.qpos[env._base_qadr + 2] < env._config.fall.min_height) | (
            gravity[2] > env._config.fall.max_tilt_gz
        )

        info["swing_apex"] = jp.where(contact_filt, 0.0, info["swing_apex"])
        info["feet_air_time"] = jp.where(contact_filt, 0.0, info["feet_air_time"] + env.dt)
        info["last_contact"] = contact
        info["last_last_action"] = info["last_action"]
        info["last_action"] = action
        info["last_torque"] = data.actuator_force
        info["step_count"] = info["step_count"] + 1

        phase = info["phase"] + env._phase_dt(info["command"])
        info["phase"] = jp.fmod(phase + jp.pi, 2 * jp.pi) - jp.pi

        info["steps_since_cmd"] = info["steps_since_cmd"] + 1
        resample = info["steps_since_cmd"] >= env._config.command.resample_steps
        info["command"] = jp.where(resample, env._sample_command(r_cmd), info["command"])
        info["steps_since_cmd"] = jp.where(resample, 0, info["steps_since_cmd"])

        obs = env._get_obs(data, info, r_noise)
        return state.replace(
            data=data, obs=obs, reward=jp.zeros(()), done=fall.astype(jp.float32), info=info
        )

    return reset, step
