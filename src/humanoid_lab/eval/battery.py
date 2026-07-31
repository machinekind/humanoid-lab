"""Fixed evaluation battery for a biped locomotion policy: one number-table
per run so iterations are comparable. Run: ./run.sh battery --run runs/<name>

Ported from w01-tek's wojtek_rl/battery.py (skeleton: load_checkpoint_policy
-> rollout -> scenario_result -> run_battery). Adapted from a 4-leg
quadruped to asimov_v1's 2-foot biped and this repo's env contract
(registry.make_env with robot_dir/preset_name/task.env read back from
run.json's hydra_config, mirroring sizing/collect.py's _load_run/
make_env_for_run pattern, rather than w01-tek's env_config-only rebuild).
Dropped entirely: w01-tek's 4-leg diag/lateral foot-pair correlation,
per-leg-joint-group splay/saturation, and the commanded-height concept
(asimov has no height command -- see envs/joystick.py's module docstring).

Scenarios (asimov command envelope from envs/joystick.py's default_config:
x +-0.8 m/s, y +-0.6 m/s, yaw +-0.6 rad/s), all scripted, fixed seed
(rollout()'s default seed=0), one episode each. Durations are specified in
seconds and converted to steps at the caller's ctrl_dt (see
battery_scenarios), so a run with a non-default ctrl_dt still gets
scenarios of the documented wall-clock length:
  stand         -- zero command, 6 s.
  walk_ramp     -- vx ramps 0 -> 0.6 m/s over the first 4 s, holds 0.6 for
                    the remaining 2 s. 6 s total.
  turn          -- constant vx=0.3 m/s, wz=0.4 rad/s. 6 s.
  strafe        -- constant vy=0.3 m/s. 4 s.
  walk_to_stop  -- vx=0.5 m/s for the first 3 s, then zero for the
                    remaining 3 s. 6 s total.
  spin_left     -- pure spin, wz=+0.5 rad/s held. 6 s.
  spin_right    -- pure spin, wz=-0.5 rad/s held. 6 s.

Robustness grid (port item 4.4): --alpha, --lag-tau and --torque-envelope
perturb the plant for one measurement. Any of them requires --out, so a
perturbed measurement cannot reach the canonical battery.json even by
forgetting the flag. The mechanisms live in eval/grid.py and the
aggregator in eval/grid_report.py; run_battery below holds the one branch
between the native and the explicit-PD rollout paths, documented there.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np
from ml_collections import config_dict
from mujoco import mjx

from humanoid_lab import sim_budget
from humanoid_lab.envs import symmetry
from humanoid_lab.eval import grid
from humanoid_lab.eval.gait import gait_metrics

# -- pure metric functions -------------------------------------------------
# numpy arrays in, float/dict out -- no env, jax rollout or checkpoint
# needed, so these are unit-tested directly in tests/unit/test_battery.py
# (mirrors sizing/report.py's own pure-reducer pattern).

# The reset transient every metric added by port items 4.1 to 4.3 excludes.
# rollout() starts recording on the first step after reset, and the opening
# steps are the robot falling into its pose against a command it has not had
# time to answer -- charging those to a KPI measures the reset, not the
# policy. 50 steps is 1 s at asimov's ctrl_dt of 0.02 (it is a step count,
# not a duration, exactly as w01-tek's TRACK_SETTLE_STEPS is). The pre-4.1
# metrics keep scoring the whole record: changing what they average over
# would change what an existing battery.json field means.
SETTLE_STEPS = 50


def _round_or_none(value: float | None, nd: int) -> float | None:
    """`round(value, nd)`, passing a None (nothing measured) straight through."""
    return None if value is None else round(value, nd)


def vibration_index(qvel_hist, dt: float, cutoff_hz: float = 5.0) -> float:
    """Fraction of joint-velocity FFT power above `cutoff_hz`.

    This is the training skill's smoothness gate (see this repo's
    ~/.claude/skills/training-mjx-locomotion/SKILL.md: "A gait that tracks
    velocity but vibrates (fast motor oscillation) is a reward problem, and
    it has a number. Measure the fraction of joint-velocity FFT power above
    5 Hz over an eval rollout" -- fbb_v2 scored 0.972 buzzing, 0.168 after
    the reward fix that removed it). Same construction as w01-tek
    wojtek_rl/battery.py's vibration_index; works on a (T,) single-joint
    signal or a (T, n_joints) array pooled across joints.
    """
    v = np.asarray(qvel_hist, dtype=float)
    v = v - v.mean(axis=0, keepdims=True)
    power = np.abs(np.fft.rfft(v, axis=0)) ** 2
    freqs = np.fft.rfftfreq(v.shape[0], d=dt)
    total = power[freqs > 0.0].sum()
    return float(power[freqs > cutoff_hz].sum() / max(total, 1e-12))


def foot_slip(foot_speed, contact) -> float:
    """Mean horizontal foot-site speed (m/s), pooled over every (step,
    foot) sample where `contact` is True (planted). Slip is a sliding-on-
    the-ground question, so only planted-foot samples count -- a swinging
    foot moving fast is normal gait, not slip. 0.0 (not NaN) if no foot is
    ever planted in the recorded window (e.g. a scenario that falls before
    any footfall)."""
    speed = np.asarray(foot_speed, dtype=float)
    c = np.asarray(contact, dtype=bool)
    if not c.any():
        return 0.0
    return float(speed[c].mean())


def torque_saturation_fraction(tau, cap, frac: float = 0.95) -> float | None:
    """Fraction of (step, joint) samples where |tau| exceeds `frac * cap`.

    `cap` broadcasts against `tau`'s last axis (per-joint effort limit,
    N*m). Returns None (not a spurious 0.0) when every cap entry is <= 0 --
    an unknown/unset cap should not silently read as "never saturates"."""
    t = np.abs(np.asarray(tau, dtype=float))
    c = np.asarray(cap, dtype=float)
    if c.size == 0 or not np.any(c > 0):
        return None
    return float((t > frac * c).mean())


def mech_power_mean(tau, omega) -> float:
    """Mean instantaneous mechanical power (W): sum_j |tau_j * omega_j| per
    step, averaged over steps. Same |tau*omega| construction as
    rewards/terms.py's energy() term and sizing/report.py's mech_power
    reducer, reimplemented here as a plain numpy array reducer (this
    module's pure-metric functions take arrays, not per-step jax scalars)."""
    t = np.asarray(tau, dtype=float)
    o = np.asarray(omega, dtype=float)
    per_step = np.abs(t * o)
    if per_step.ndim > 1:
        per_step = per_step.sum(axis=-1)
    return float(per_step.mean()) if per_step.size else 0.0


def antiphase_score(contact_left, contact_right) -> float:
    """Phase-opposition score for a 2-foot gait, in [0, 1]: 1.0 = perfectly
    alternating (antiphase -- the expected walking gait), 0.0 = perfectly
    in-phase (both feet up/down together, e.g. hopping or a frozen
    double-stance). Correlation of the two raw contact sequences is
    already -1 for perfect alternation and +1 for perfect unison, so the
    score is simply 0.5*(1 - corr): the simplest honest mapping onto
    [0, 1], no lag search needed for a 2-foot clock. A zero-variance
    sequence (e.g. `stand`'s always-both-planted contacts) has undefined
    correlation; by convention that returns 0.5 (neither pattern detected)
    rather than raising or returning NaN."""
    a = np.asarray(contact_left, dtype=float)
    b = np.asarray(contact_right, dtype=float)
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    if denom < 1e-9:
        return 0.5
    corr = float((a * b).sum() / denom)
    return 0.5 * (1.0 - corr)


def yaw_progress_deg(wz, dt: float, settle_steps: int = SETTLE_STEPS) -> float | None:
    """Signed yaw swept over the post-settle window, in degrees.

    `wz` is the BODY-frame yaw rate (rollout() records the gyro's z channel,
    which is what a body-mounted gyro reads and what the deployed robot would
    have). Integrating it gives the yaw turned about the robot's own vertical
    axis; for a robot that stays upright that is world yaw, and for one that
    does not, the body-frame number is the honest one -- a policy cannot spin
    about an axis it is not standing on.

    Positive is CCW (left), so the sign carries the chirality: a spin_right
    row whose progress comes out positive turned the wrong way, and one near
    zero did not turn at all. Summing the rate rather than differencing yaw
    angles also needs no unwrapping, so a multi-turn spin cannot alias.

    Returns None (not 0.0) when the record does not outlive `settle_steps`:
    nothing was measured, and a 0.0 there would read as "did not turn".
    """
    w = np.asarray(wz, dtype=float)[settle_steps:]
    if w.size == 0:
        return None
    return float(np.degrees(w.sum() * dt))


def tracking_error(ctrl_hist, qpos_hist, settle_steps: int = SETTLE_STEPS) -> dict:
    """RMS and p95 of |ctrl - qpos| over the post-settle window, rad.

    `ctrl` is the setpoint the servo was asked to hold and `qpos` the angle
    the joint reached, both (steps, actuated joints) in canonical order. This
    is the servo error that actuator stiffness work targets, and it is
    invisible to the velocity-tracking metrics: a policy can hit its
    commanded body velocity while every joint sags behind its setpoint.

    Both numbers pool over steps and joints. The p95 is what says whether the
    error is spread evenly or lives in a few joints -- an RMS over twelve
    joints hides one that has given up.

    Returns {"rms": None, "p95": None} when nothing outlives `settle_steps`.

    Only a position-servo preset makes this a servo error: under
    `ideal_torque`, ctrl is a torque in Nm and the subtraction is
    dimensionally meaningless. run.json's `actuator_gains.model` (port item
    4.3) is what tells a reader which one produced the run.
    """
    c = np.asarray(ctrl_hist, dtype=float)[settle_steps:]
    q = np.asarray(qpos_hist, dtype=float)[settle_steps:]
    if c.size == 0:
        return {"rms": None, "p95": None}
    err = np.abs(c - q)
    return {
        "rms": float(np.sqrt((err**2).mean())),
        "p95": float(np.percentile(err, 95)),
    }


def peak_over(values) -> int | None:
    """Largest of `values`, ignoring the unmeasured ones, or None if none was.

    The battery's per-step budget readings are None wherever the backend has
    no counter for them (every step, for the constraint rows on jax). None has
    to survive as None: reported as 0 it would read as a measured peak of
    zero, which is the one reading that can never happen while the robot is
    standing on a floor.
    """
    seen = [int(v) for v in values if v is not None]
    return max(seen) if seen else None


def scenario_result(
    name: str, rec: dict, fell_at: int | None, dt: float, torque_cap, n_steps: int
) -> dict:
    """One scenario's battery.json entry, computed from its rollout()."""
    steps = len(rec["vx"])
    r = {
        "fell": fell_at is not None,
        "fell_at": fell_at,
        "steps": steps,
        # Held the whole scripted duration. Written from the step budget
        # rather than from `fell` because rollout()'s early exit is not
        # guaranteed to stay the only one: today a scenario ends short only
        # by terminating, so completed == not fell, and a future exit (a
        # completion cut, a step cap) reports honestly with no schema change.
        "completed": fell_at is None and steps >= n_steps,
    }
    if steps < 10:
        # Too few samples for FFT/percentile metrics to mean anything
        # (an almost-instant fall) -- fell/fell_at/steps already say enough.
        return r

    cmd = rec["cmd"]
    achieved = np.stack([rec["vx"], rec["vy"], rec["wz"]], axis=-1)
    err = np.abs(cmd - achieved).mean(axis=0)
    r["vel_err_vx"] = round(float(err[0]), 4)
    r["vel_err_vy"] = round(float(err[1]), 4)
    r["vel_err_wz"] = round(float(err[2]), 4)

    # Yaw actually swept against yaw asked for, over the post-settle window
    # (port item 4.1). Reported on every row, not just the two spin probes:
    # it is a raw reading of the same body gyro every row already records,
    # and `turn`'s yaw budget is worth the same look. The pair is what makes
    # a spin row diagnosable -- progress alone cannot say whether a policy
    # under-turned or was barely asked to turn.
    r["yaw_progress_deg"] = _round_or_none(yaw_progress_deg(rec["wz"], dt), 2)
    r["yaw_cmd_deg"] = _round_or_none(yaw_progress_deg(cmd[:, 2], dt), 2)

    r["height_mean"] = round(float(rec["height"].mean()), 4)
    r["height_std"] = round(float(rec["height"].std()), 4)

    r["vibration"] = round(vibration_index(rec["qvel"], dt), 3)
    r["foot_slip"] = round(foot_slip(rec["foot_speed"], rec["contact"]), 4)

    sat = torque_saturation_fraction(rec["tau"], torque_cap)
    r["torque_sat_frac"] = round(sat, 4) if sat is not None else None

    r["mech_power_mean"] = round(mech_power_mean(rec["tau"], rec["qvel"]), 3)

    if rec["contact"].ndim == 2 and rec["contact"].shape[1] >= 2:
        r["antiphase_score"] = round(
            antiphase_score(rec["contact"][:, 0], rec["contact"][:, 1]), 3
        )

    # Gait KPIs (port item 4.2). Raw metrics, folded into nothing: they are
    # here because velocity tracking error scores a skimming gait and a
    # stand-and-lift farm as healthy.
    r.update(gait_metrics(rec, settle_steps=SETTLE_STEPS))

    # Servo error (port item 4.3), also raw.
    track = tracking_error(rec["ctrl"], rec["qpos"], SETTLE_STEPS)
    r["tracking_err_rms"] = _round_or_none(track["rms"], 4)
    r["tracking_err_p95"] = _round_or_none(track["p95"], 4)

    return r


# -- scenario command builders ----------------------------------------------

# asimov command envelope (envs/joystick.py's default_config `command`
# block, itself from PLAN.md's asimov docs): x +-0.8 m/s, y +-0.6 m/s,
# yaw +-0.6 rad/s. Every scenario below stays inside this envelope.
_VX_MAX, _VY_MAX, _WZ_MAX = 0.8, 0.6, 0.6

# Held yaw rate for the spin probes, rad/s. See battery_scenarios.
_SPIN_WZ = 0.5


def battery_scenarios(dt: float) -> dict:
    """name -> (cmd_at(i) -> np.ndarray([vx, vy, wz]), n_steps).

    Split out (rather than inlined in run_battery) so eval/video.py's
    --scenario can reuse the exact same scripted trajectories the battery
    measures. `dt` is the env's ctrl_dt: durations are given in seconds
    below and converted to steps here, so a run with a non-default ctrl_dt
    still gets scenarios of the documented wall-clock length.
    """

    def n_steps(seconds: float) -> int:
        return max(1, round(seconds / dt))

    def stand(_i):
        return np.array([0.0, 0.0, 0.0])

    ramp_s, ramp_hold_s = 4.0, 2.0
    ramp_steps = n_steps(ramp_s)

    def walk_ramp(i):
        vx = 0.6 * min(1.0, i / ramp_steps) if ramp_steps else 0.6
        return np.array([vx, 0.0, 0.0])

    def turn(_i):
        return np.array([0.3, 0.0, 0.4])

    def strafe(_i):
        return np.array([0.0, 0.3, 0.0])

    stop_s, stop_hold_s = 3.0, 3.0
    stop_switch = n_steps(stop_s)

    def walk_to_stop(i):
        vx = 0.5 if i < stop_switch else 0.0
        return np.array([vx, 0.0, 0.0])

    # Spin probes (port item 4.1). One held pure-yaw command per direction,
    # no translation, so the two rows differ in exactly one sign and a
    # chirality bug has nowhere to hide. 0.5 rad/s sits inside the +-0.6
    # yaw box with headroom: a probe pinned to the corner of the command
    # envelope would confound "cannot spin this way" with "was never trained
    # this close to the edge". Same 6 s as the other held scenarios; at
    # ctrl_dt 0.02 the post-settle window then asks for 2.5 rad (143 deg),
    # which is enough turning to see and short of a full revolution, so
    # nothing depends on the yaw reading unwrapping.
    def spin_left(_i):
        return np.array([0.0, 0.0, _SPIN_WZ])

    def spin_right(_i):
        return np.array([0.0, 0.0, -_SPIN_WZ])

    return {
        "stand": (stand, n_steps(6.0)),
        "walk_ramp": (walk_ramp, n_steps(ramp_s + ramp_hold_s)),
        "turn": (turn, n_steps(6.0)),
        "strafe": (strafe, n_steps(4.0)),
        "walk_to_stop": (walk_to_stop, n_steps(stop_s + stop_hold_s)),
        "spin_left": (spin_left, n_steps(6.0)),
        "spin_right": (spin_right, n_steps(6.0)),
    }


# -- checkpoint rollout (needs env + jax + a trained policy) ---------------


def _load_run(run_dir: Path) -> dict:
    run_json = run_dir / "run.json"
    if not run_json.exists():
        raise FileNotFoundError(
            f"{run_json} not found -- is {run_dir} a completed run.sh train/smoke output dir?"
        )
    run = json.loads(run_json.read_text())

    missing = [k for k in ("task", "hydra_config", "checkpoint_dir", "ppo_config") if k not in run]
    if missing:
        raise ValueError(f"{run_json} is missing required key(s) {missing}")

    hydra = run["hydra_config"]
    for section, key in (("robot", "dir"), ("actuators", "name")):
        if section not in hydra or key not in hydra[section]:
            raise ValueError(
                f"{run_json}: hydra_config.{section}.{key} missing -- cannot rebuild the env"
            )
    return run


def _find_latest_checkpoint(run: dict, run_dir: Path) -> Path:
    ckpt_dir = Path(run["checkpoint_dir"])
    if not ckpt_dir.exists():
        ckpt_dir = (run_dir / "checkpoints").resolve()
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"no checkpoints dir found for run {run_dir} (tried {ckpt_dir})")
    steps = [p for p in ckpt_dir.iterdir() if p.is_dir() and p.name.isdigit()]
    if not steps:
        raise FileNotFoundError(f"{ckpt_dir} has no <step> checkpoint subdirectories")
    return max(steps, key=lambda p: int(p.name))


def _measurement_env_overrides(run: dict) -> dict:
    """The run's own task.env overrides, plus three measurement-only
    changes: pushes disabled (they contaminate vibration/slip/stand
    metrics -- robustness is trained, not measured here, same rationale as
    w01-tek wojtek_rl/battery.py's load_checkpoint_policy), the
    command auto-resample effectively disabled (envs/joystick.py's
    step() resamples a random command every `command.resample_steps`
    control steps; left at its default that would clobber this battery's
    own scripted command trajectories mid-scenario, since rollout() drives
    the command by mutating state.info["command"] every step rather than
    through the env's own sampler), the no-progress cut disabled, and the
    mirror augmentation disabled.

    The mirror is off by the deployment-frame rule (see
    envs/symmetry.py's deployment_frame_overrides, which owns it): a
    measurement describes the frame the robot will be deployed in, and a
    battery that drew the coin the other way would report a policy's
    spin_left as its spin_right. Any future training-only stochastic
    augmentation stored in run config gets the same treatment.

    The cut is off for two reasons. A probabilistic termination is not a
    fall, and rollout() reports every `done` as one (`fell_at`), so a run
    trained with the cut on would have scenarios scored as falls that never
    happened -- on exactly the under-delivering checkpoints the battery
    exists to diagnose. And the battery drives the command from outside the
    sampler, so neither of the cut's two safety valves works here: the
    grace window and the EMA reseed both hang off the resample branch that
    the override above has just disabled, so a scripted command change
    (walk_ramp's ramp, walk_to_stop's switch) re-arms nothing and the meter
    keeps a seed taken from the env's own random reset command. A battery
    that ever wants to MEASURE the cut has to reseed info["progress_ema"]
    and zero info["steps_since_cmd"] itself whenever cmd_at changes the
    command."""
    hydra = run.get("hydra_config") or {}
    overrides = symmetry.deployment_frame_overrides((hydra.get("task") or {}).get("env") or {})
    overrides["push"] = {**(overrides.get("push") or {}), "enable": False}
    overrides["command"] = {**(overrides.get("command") or {}), "resample_steps": 10_000_000}
    overrides["no_progress"] = {**(overrides.get("no_progress") or {}), "enable": False}
    return overrides


def merged_env_overrides(run: dict, extra: dict | None = None) -> dict:
    """`_measurement_env_overrides(run)` with `extra` merged one level deep
    over it.

    One level, not recursive: every measurement-only change is a flat
    `{block: {key: value}}` edit, and a caller that wants to put a push back
    (`{"push": {"enable": True}}`) has to keep the run's own `interval_steps`
    and `vel` around it -- which a one-level merge does and a wholesale
    replacement would not.

    This is the seam `eval/video.py` uses to re-enable pushes for a render
    (see its `--push`). The battery itself never passes `extra`.
    """
    overrides = _measurement_env_overrides(run)
    for key, block in (extra or {}).items():
        current = overrides.get(key)
        overrides[key] = {**current, **block} if isinstance(current, dict) else block
    return overrides


def load_checkpoint_policy(run_dir: Path, extra_env_overrides: dict | None = None):
    """Load a run's measurement env + latest-checkpoint policy.

    Rebuilds the env exactly as train.py did: registry.make_env with the
    run's own robot/preset/task from run.json's hydra_config (the same
    reconstruction sizing/collect.py's make_env_for_run uses), plus the
    measurement-only overrides in _measurement_env_overrides. The PPO
    params come from run.json's own recorded ppo_config verbatim (not a
    replayed build_ppo_params + overrides), matching sizing/collect.py's
    _ppo_params_from_run: the network shape that produced this checkpoint
    can't drift from what train.py actually used.

    `extra_env_overrides` is merged one level deep over those measurement
    overrides (see merged_env_overrides). The battery never passes it;
    eval/video.py's `--push` does.

    Returns (run, env, ckpt_path, inf) where inf = jax.jit(policy).
    """
    from humanoid_lab import paths
    from humanoid_lab.policy_io import load_policy
    from humanoid_lab.registry import make_env

    run = _load_run(run_dir)
    hydra = run["hydra_config"]
    robot_dir = paths.REPO_ROOT / hydra["robot"]["dir"]
    preset_name = hydra["actuators"]["name"]
    actuator_overrides = hydra["actuators"].get("overrides") or {}
    env_overrides = merged_env_overrides(run, extra_env_overrides)
    env = make_env(run["task"], robot_dir, preset_name, env_overrides, actuator_overrides)

    ckpt = _find_latest_checkpoint(run, run_dir)
    ppo_params = config_dict.ConfigDict(run["ppo_config"])
    policy = load_policy(ckpt, env, ppo_params, deterministic=True)
    return run, env, ckpt, jax.jit(policy)


def rollout(env, reset, step, inf, cmd_at, n_steps: int, seed: int = 0):
    """Roll `n_steps` of `cmd_at` under `inf` in `env`.

    `reset`/`step` are jax.jit(env.reset)/jax.jit(env.step) -- passed in
    (rather than jitted here) so callers compile once and reuse the same
    jitted callables across every scenario (mirrors w01-tek
    wojtek_rl/battery.py's rollout()).

    Returns (rec, fell_at, budget): `rec` maps signal name -> np.ndarray over
    the steps taken (through and including the step that trips `done`, if
    any); `fell_at` is that step's index, or None if the scenario completed
    without falling; `budget` is the (contacts, rows) peak of this scenario,
    either of which is None where the backend has no counter for it.
    """
    rng = jax.random.PRNGKey(seed)
    state = reset(rng)
    rec = {
        "cmd": [], "vx": [], "vy": [], "wz": [], "height": [],
        "qvel": [], "contact": [], "foot_speed": [], "tau": [],
        "foot_clear": [], "foot_vz": [], "ctrl": [], "qpos": [],
    }
    nacon_seen, nefc_seen = [], []
    fell_at = None
    for i in range(n_steps):
        cmd = jp.array(cmd_at(i))
        state.info["command"] = cmd
        rng, act_rng = jax.random.split(rng)
        act, _ = inf(state.obs, act_rng)
        state = step(state, act)
        d = state.data

        linvel = np.asarray(env._local_linvel(d))
        gyro = np.asarray(env._gyro(d))
        contact = np.asarray(env._foot_contact(d))
        foot_vel = np.asarray(env._foot_linvel(d))
        foot_speed = np.hypot(foot_vel[:, 0], foot_vel[:, 1])

        rec["cmd"].append(np.asarray(cmd))
        rec["vx"].append(float(linvel[0]))
        rec["vy"].append(float(linvel[1]))
        rec["wz"].append(float(gyro[2]))
        rec["height"].append(float(np.asarray(d.qpos)[env._base_qadr + 2]))
        rec["qvel"].append(np.asarray(d.qvel[env._vadr]))
        rec["contact"].append(contact)
        rec["foot_speed"].append(foot_speed)
        rec["tau"].append(np.asarray(d.actuator_force))
        # Per-foot clearance and vertical velocity, for the gait KPIs (port
        # item 4.2). foot_vel is already in hand for foot_speed above, so
        # its z channel costs nothing extra. Both are WORLD frame: clearance
        # is a height above the ground and touchdown speed is how hard the
        # foot hits it, and the ground does not rotate with the robot. (See
        # eval/gait.py on what _foot_clearance is referenced to -- the reset
        # keyframe, not the floor.)
        rec["foot_clear"].append(np.asarray(env._foot_clearance(d)))
        rec["foot_vz"].append(foot_vel[:, 2])
        # Setpoint and angle over the actuated joints, for the servo KPI
        # (port item 4.3). d.ctrl is nu-long and env._qadr is the same
        # canonical actuated-joint order (robot/build.py injects actuators in
        # robot_spec.actuated_joints order), so the two line up column for
        # column.
        rec["ctrl"].append(np.asarray(d.ctrl))
        rec["qpos"].append(np.asarray(d.qpos)[np.asarray(env._qadr)])

        # Warp's contact and constraint-row budgets, sampled every step. The
        # rollout is a plain Python loop, so this is a numpy read per step and
        # costs nothing worth avoiding.
        nacon, nefc = sim_budget.observed_peaks(d)
        nacon_seen.append(nacon)
        nefc_seen.append(nefc)

        if bool(state.done):
            fell_at = i
            break
    budget = (peak_over(nacon_seen), peak_over(nefc_seen))
    return {k: np.array(v) for k, v in rec.items()}, fell_at, budget


def run_battery(
    run_dir: Path,
    alpha: float = 1.0,
    lag_tau: float = 0.0,
    torque_envelope=None,
) -> dict:
    """Run the fixed scenario battery against `run_dir`'s checkpoint.

    Returns the same dict main() writes (minus the `timestamp` main() adds
    at write time).

    `alpha` (Kt miscalibration), `lag_tau` (actuator lag, seconds) and
    `torque_envelope` (an `(omega_b, omega_0)` pair, or None) are the
    robustness grid's eval-only plant perturbations -- see `eval/grid.py`.
    The defaults `(1.0, 0.0, None)` reproduce the unperturbed battery
    exactly, and `eval/report.py` only ever calls it that way.

    **This function holds the grid's one documented branch.** An
    unperturbed cell steps `env.reset`/`env.step` themselves -- the NATIVE
    path, bit-for-bit what `./run.sh battery` has always run. A cell with a
    lag or an envelope steps the explicit-PD substitute instead. The two are
    not the same code, and no tiny lag value routes the baseline through the
    substitute. What ties them together is a measured tolerance as the lag
    goes to zero (`tests/integration/test_grid_env.py`), not a shared code
    path. See eval/grid.py's module docstring.
    """
    run, env, ckpt, inf = load_checkpoint_policy(run_dir)

    if alpha != 1.0:
        # In place on the model this eval process holds, then re-uploaded:
        # env.mj_model is the built, post-injection model, and `alpha`
        # models a firmware error on top of whatever preset produced it.
        # Nothing training-side ever sees this: the env was constructed
        # seconds ago inside this process, from run.json.
        grid.apply_kt_miscalibration(env.mj_model, alpha)
        env._mjx_model = mjx.put_model(env.mj_model, impl=env._backend)

    if lag_tau > 0 or torque_envelope is not None:
        reset_fn, step_fn = grid.make_explicit_pd_rollout_fns(env, lag_tau, torque_envelope)
        reset, step = jax.jit(reset_fn), jax.jit(step_fn)
    else:
        reset, step = jax.jit(env.reset), jax.jit(env.step)

    # Read AFTER the alpha scaling above, so `torque_sat_frac` is scored
    # against the cap the perturbed plant actually had.
    torque_cap = np.asarray(env.mj_model.actuator_forcerange[:, 1])

    results = {
        "run": run["run_name"],
        "checkpoint": ckpt.name,
        # Stamped on every battery.json, unperturbed ones included: a file
        # with no grid block would be indistinguishable from one written
        # before the axes existed.
        "grid": {
            "alpha": alpha,
            "lag_tau": lag_tau,
            "torque_envelope": list(torque_envelope) if torque_envelope else None,
            "path": "explicit_pd" if (lag_tau > 0 or torque_envelope is not None) else "native",
        },
    }
    nacon_seen, nefc_seen = [], []
    for name, (cmd_at, n_steps) in battery_scenarios(env.dt).items():
        rec, fell_at, (nacon, nefc) = rollout(env, reset, step, inf, cmd_at, n_steps)
        results[name] = scenario_result(name, rec, fell_at, env.dt, torque_cap, n_steps)
        nacon_seen.append(nacon)
        nefc_seen.append(nefc)

    # One warp budget block for the whole battery, same schema as run.json's
    # (sim_budget.budget_report_for_env is the single writer of both). Peaks
    # are over every scenario: which one touched the ceiling does not change
    # what the ceiling has to be.
    results["contacts"] = sim_budget.budget_report_for_env(
        env, peak_over(nacon_seen), peak_over(nefc_seen)
    )
    return results


def armed_grid_flags(alpha, lag_tau, torque_envelope) -> list[str]:
    """The robustness-grid flags this invocation actually perturbs with.

    Empty for the canonical battery -- the baseline cell is the native path
    (alpha 1.0, no lag, no envelope), so a run of it is not a grid cell and
    may write <run>/battery.json. `torque_envelope` is the raw CLI string:
    presence is what arms it, and a malformed one is rejected separately.
    """
    return [
        name for name, on in (
            ("--alpha", float(alpha) != 1.0),
            ("--lag-tau", float(lag_tau) != 0.0),
            ("--torque-envelope", torque_envelope is not None),
        ) if on
    ]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument(
        "--out", default=None, type=Path,
        help="write here instead of <run>/battery.json. REQUIRED whenever any "
        "of --alpha/--lag-tau/--torque-envelope perturbs the plant: a "
        "robustness-grid cell must never land on top of the run's canonical "
        "number table. eval/grid.py's cell_name is the filename "
        "eval/grid_report.py aggregates.",
    )
    ap.add_argument(
        "--alpha", type=float, default=1.0,
        help="Kt miscalibration: scale the built model's effective PD gains "
        "and torque cap by this factor (1.0 = no-op). See eval/grid.py's "
        "apply_kt_miscalibration.",
    )
    ap.add_argument(
        "--lag-tau", type=float, default=0.0,
        help="actuator-bandwidth time constant, seconds (0.0 = the native "
        "pipeline, unchanged). Above 0 switches to the explicit-PD substep "
        "loop with a first-order torque lag; see eval/grid.py's "
        "make_explicit_pd_rollout_fns.",
    )
    ap.add_argument(
        "--torque-envelope", default=None,
        help="'OMEGA_B,OMEGA_0' rad/s (default: none, flat cap). "
        "Speed-dependent DRIVING-torque cap: the static cap up to OMEGA_B, "
        "ramping linearly to 0 at OMEGA_0, 0 beyond; BRAKING keeps the full "
        "static cap. Forces the explicit-PD path even at --lag-tau 0.",
    )
    args = ap.parse_args()

    armed = armed_grid_flags(args.alpha, args.lag_tau, args.torque_envelope)
    if armed and args.out is None:
        # Same shape as build_model.py's "--set requires --out": the default
        # path is the canonical artifact, and a perturbed measurement must
        # not be able to land on it -- not even by forgetting a flag.
        ap.error(
            f"{', '.join(armed)} requires --out: the default <run>/battery.json is "
            "the run's canonical, unperturbed number table and a robustness-grid "
            "cell must not overwrite it. eval/grid.py's cell_name is the filename "
            "eval/grid_report.py aggregates"
        )

    try:
        torque_envelope = grid.parse_torque_envelope(args.torque_envelope)
    except ValueError as exc:
        # argparse's own exit path, so a malformed spec fails before the
        # checkpoint is even loaded rather than raising out of a rollout.
        ap.error(str(exc))

    results = run_battery(
        args.run, alpha=args.alpha, lag_tau=args.lag_tau, torque_envelope=torque_envelope
    )
    out = args.out or (args.run / "battery.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    stamped = dict(results, timestamp=datetime.now().isoformat(timespec="seconds"))
    out.write_text(json.dumps(stamped, indent=2))
    print(json.dumps(stamped, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
