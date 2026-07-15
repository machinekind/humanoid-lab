"""Checkpoint rollout collector for the sizing task's per-joint tau/omega
telemetry (build order step 7, PLAN.md "First experiments" #2).

Run:
    python -m humanoid_lab.sizing.collect --run runs/<name> \
        [--episodes 4] [--steps 400] [--seed 0]

Rebuilds the env exactly as train.py did (same registry.make_env path,
same task/robot/preset/env overrides, read back from run.json's
hydra_config -- the one part of run.json that carries the un-resolved
per-run knobs train.py itself passed to make_env), loads the checkpoint's
policy via policy_io.load_policy, and rolls CPU episodes using the env's
own command sampling (Joystick._sample_command / command.resample_steps --
no scripted battery here, unlike w01-tek's battery.py: sizing wants the
torque/speed distribution the trained gait actually produces under its own
command envelope, not a handful of scripted scenarios).

Deviation from train.py's exact env construction: train.py additionally
overrides env_overrides["sim"]["num_envs"] to the PPO batch size, to size
the warp backend's contact buffers for the training batch (see
envs/backend.py's data_budget_kwargs). That override is never persisted
into run.json's hydra_config (train.py mutates a local dict copy, not the
Hydra cfg object), and it is a no-op on the CPU jax backend this collector
always runs on (JAX_PLATFORMS=cpu; the jax branch of make_data_fn ignores
num_envs entirely). So this collector leaves sim.num_envs at the task's own
default (1) rather than reconstructing the training batch size -- correct
for a single-world python rollout loop, and unobservable on CPU either way.

Writes <run>/sizing_data.npz:
    tau [T,12], omega [T,12], command [T,3], done [T]  (T = total kept
        steps, summed across every episode that produced at least one step)
    joint_names [12], effort_limit [12], velocity_limit [12]  (from the
        run's own actuator preset, canonical robot_spec.actuated_joints
        order)

sizing/report.py reduces this file; it needs no checkpoint or env.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import numpy as np
from ml_collections import config_dict

from humanoid_lab import paths


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
    for section, key in (("robot", "dir"), ("actuators", "name"), ("task", "name")):
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


def _ppo_params_from_run(run: dict) -> config_dict.ConfigDict:
    """Rebuild the ppo ConfigDict policy_io.load_policy needs from the
    literal numbers train.py recorded in run.json's ppo_config, instead of
    replaying train.build_ppo_params + overrides. Deliberate: the Go1 base
    config build_ppo_params starts from can drift as mujoco_playground
    upgrades, but ppo_config already recorded the exact network shape and
    normalize_observations flag used to produce this checkpoint's params
    pytree, so replaying it verbatim is the one reconstruction that cannot
    silently mismatch the checkpoint.
    """
    return config_dict.ConfigDict(run["ppo_config"])


def make_env_for_run(run: dict):
    from humanoid_lab.registry import make_env

    task = run["task"]
    hydra = run["hydra_config"]
    robot_dir = paths.REPO_ROOT / hydra["robot"]["dir"]
    preset_name = hydra["actuators"]["name"]
    actuator_overrides = hydra["actuators"].get("overrides") or {}
    env_overrides = hydra["task"].get("env") or {}
    env = make_env(task, robot_dir, preset_name, env_overrides, actuator_overrides)
    return env, robot_dir, preset_name, actuator_overrides


def _joint_limits(
    robot_dir: Path, preset_name: str, joint_names: list[str], overrides: dict | None = None
):
    from humanoid_lab.robot.presets import load_actuator_preset, resolve
    from humanoid_lab.robot.spec import load_robot_spec

    robot_spec = load_robot_spec(robot_dir)
    preset = load_actuator_preset(robot_dir, preset_name, overrides)
    params_by_joint = resolve(preset, robot_spec)
    effort_limit = np.array(
        [params_by_joint[n].effort_limit for n in joint_names], dtype=np.float64
    )
    velocity_limit = np.array(
        [
            params_by_joint[n].velocity_limit
            if params_by_joint[n].velocity_limit is not None
            else np.nan
            for n in joint_names
        ],
        dtype=np.float64,
    )
    return effort_limit, velocity_limit


def rollout_episode(env, reset_fn, step_fn, policy_fn, rng, steps: int):
    """Roll one episode: up to `steps` control steps, stopping early on a
    fall. Returns ((tau, omega, command, done), rng) -- `command` is the
    commanded velocity that was ACTIVE going into each step (the one the
    policy observed and that shaped that step's action), not the
    possibly-resampled command coming out of it.
    """
    rng, r_reset = jax.random.split(rng)
    state = reset_fn(r_reset)
    tau, omega, command, done = [], [], [], []
    for _ in range(steps):
        cmd_before = np.asarray(state.info["command"])
        rng, r_act = jax.random.split(rng)
        action, _ = policy_fn(state.obs, r_act)
        state = step_fn(state, action)
        d = state.data
        tau.append(np.asarray(d.actuator_force))
        omega.append(np.asarray(d.qvel[env._vadr]))
        command.append(cmd_before)
        is_done = bool(state.done)
        done.append(is_done)
        if is_done:
            break
    arrays = (
        np.asarray(tau, dtype=np.float64).reshape(-1, env.action_size),
        np.asarray(omega, dtype=np.float64).reshape(-1, env.action_size),
        np.asarray(command, dtype=np.float64).reshape(-1, 3),
        np.asarray(done, dtype=bool).reshape(-1),
    )
    return arrays, rng


def collect(run_dir: Path, episodes: int, steps: int, seed: int, out_path: Path | None = None) -> Path:
    run = _load_run(run_dir)
    ckpt_dir = _find_latest_checkpoint(run, run_dir)

    env, robot_dir, preset_name, actuator_overrides = make_env_for_run(run)
    joint_names = list(env.robot_spec.actuated_joints)
    effort_limit, velocity_limit = _joint_limits(robot_dir, preset_name, joint_names, actuator_overrides)

    from humanoid_lab.policy_io import load_policy

    ppo_params = _ppo_params_from_run(run)
    # deterministic=True: sizing measures the deployed mean policy's demand,
    # and the numbers must be reproducible run to run. Stochastic rollouts
    # would inflate percentiles with exploration noise no real robot sees.
    policy_fn = jax.jit(load_policy(ckpt_dir, env, ppo_params, deterministic=True))
    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)

    rng = jax.random.PRNGKey(seed)
    tau_chunks, omega_chunks, cmd_chunks, done_chunks = [], [], [], []
    for _ in range(episodes):
        (tau, omega, command, done), rng = rollout_episode(
            env, reset_fn, step_fn, policy_fn, rng, steps
        )
        if tau.shape[0] == 0:
            continue
        tau_chunks.append(tau)
        omega_chunks.append(omega)
        cmd_chunks.append(command)
        done_chunks.append(done)

    nu = env.action_size
    tau_all = np.concatenate(tau_chunks, axis=0) if tau_chunks else np.zeros((0, nu))
    omega_all = np.concatenate(omega_chunks, axis=0) if omega_chunks else np.zeros((0, nu))
    cmd_all = np.concatenate(cmd_chunks, axis=0) if cmd_chunks else np.zeros((0, 3))
    done_all = np.concatenate(done_chunks, axis=0) if done_chunks else np.zeros((0,), dtype=bool)

    out_path = out_path or (run_dir / "sizing_data.npz")
    np.savez(
        out_path,
        tau=tau_all,
        omega=omega_all,
        command=cmd_all,
        done=done_all,
        joint_names=np.array(joint_names),
        effort_limit=effort_limit,
        velocity_limit=velocity_limit,
    )
    print(
        f"sizing collect: episodes={episodes} steps_kept={tau_all.shape[0]} "
        f"checkpoint={ckpt_dir.name} -> {out_path}"
    )
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    collect(args.run, args.episodes, args.steps, args.seed)


if __name__ == "__main__":
    main()
