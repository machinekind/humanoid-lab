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

from humanoid_lab import sim_budget

# -- pure metric functions -------------------------------------------------
# numpy arrays in, float/dict out -- no env, jax rollout or checkpoint
# needed, so these are unit-tested directly in tests/test_battery.py
# (mirrors sizing/report.py's own pure-reducer pattern).


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


def scenario_result(name: str, rec: dict, fell_at: int | None, dt: float, torque_cap) -> dict:
    """One scenario's battery.json entry, computed from its rollout()."""
    r = {"fell": fell_at is not None, "fell_at": fell_at, "steps": len(rec["vx"])}
    if r["steps"] < 10:
        # Too few samples for FFT/percentile metrics to mean anything
        # (an almost-instant fall) -- fell/fell_at/steps already say enough.
        return r

    cmd = rec["cmd"]
    achieved = np.stack([rec["vx"], rec["vy"], rec["wz"]], axis=-1)
    err = np.abs(cmd - achieved).mean(axis=0)
    r["vel_err_vx"] = round(float(err[0]), 4)
    r["vel_err_vy"] = round(float(err[1]), 4)
    r["vel_err_wz"] = round(float(err[2]), 4)

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

    return r


# -- scenario command builders ----------------------------------------------

# asimov command envelope (envs/joystick.py's default_config `command`
# block, itself from PLAN.md's asimov docs): x +-0.8 m/s, y +-0.6 m/s,
# yaw +-0.6 rad/s. Every scenario below stays inside this envelope.
_VX_MAX, _VY_MAX, _WZ_MAX = 0.8, 0.6, 0.6


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

    return {
        "stand": (stand, n_steps(6.0)),
        "walk_ramp": (walk_ramp, n_steps(ramp_s + ramp_hold_s)),
        "turn": (turn, n_steps(6.0)),
        "strafe": (strafe, n_steps(4.0)),
        "walk_to_stop": (walk_to_stop, n_steps(stop_s + stop_hold_s)),
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
    through the env's own sampler), and the no-progress cut disabled.

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
    overrides = dict((hydra.get("task") or {}).get("env") or {})
    overrides["push"] = {**(overrides.get("push") or {}), "enable": False}
    overrides["command"] = {**(overrides.get("command") or {}), "resample_steps": 10_000_000}
    overrides["no_progress"] = {**(overrides.get("no_progress") or {}), "enable": False}
    return overrides


def load_checkpoint_policy(run_dir: Path):
    """Load a run's measurement env + latest-checkpoint policy.

    Rebuilds the env exactly as train.py did: registry.make_env with the
    run's own robot/preset/task from run.json's hydra_config (the same
    reconstruction sizing/collect.py's make_env_for_run uses), plus the
    measurement-only overrides in _measurement_env_overrides. The PPO
    params come from run.json's own recorded ppo_config verbatim (not a
    replayed build_ppo_params + overrides), matching sizing/collect.py's
    _ppo_params_from_run: the network shape that produced this checkpoint
    can't drift from what train.py actually used.

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
    env_overrides = _measurement_env_overrides(run)
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


def run_battery(run_dir: Path) -> dict:
    """Run the fixed scenario battery against `run_dir`'s checkpoint.

    Returns the same dict main() writes (minus the `timestamp` main() adds
    at write time).
    """
    run, env, ckpt, inf = load_checkpoint_policy(run_dir)
    reset, step = jax.jit(env.reset), jax.jit(env.step)
    torque_cap = np.asarray(env.mj_model.actuator_forcerange[:, 1])

    results = {"run": run["run_name"], "checkpoint": ckpt.name}
    nacon_seen, nefc_seen = [], []
    for name, (cmd_at, n_steps) in battery_scenarios(env.dt).items():
        rec, fell_at, (nacon, nefc) = rollout(env, reset, step, inf, cmd_at, n_steps)
        results[name] = scenario_result(name, rec, fell_at, env.dt, torque_cap)
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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--out", default=None, type=Path)
    args = ap.parse_args()

    results = run_battery(args.run)
    out = args.out or (args.run / "battery.json")
    stamped = dict(results, timestamp=datetime.now().isoformat(timespec="seconds"))
    out.write_text(json.dumps(stamped, indent=2))
    print(json.dumps(stamped, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
