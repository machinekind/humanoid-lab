"""A reproducible benchmark of the Torch trainer, written to JSON.

Run it as `python -m humanoid_lab.torch_warp.bench <hydra overrides>` on
configs/torch_warp.yaml, with the same overrides as a training run. The
`bench` block sets the timed iterations (bench.iters), the timed physics
steps (bench.physics_steps) and the output path (bench.out, default
runs/bench/bench_<timestamp>.json). It builds the env and a Trainer
initialized from scratch, as ppo.main does. It writes no checkpoint and no
run dir.

The JSON holds:

- created, device, git_sha;
- hardware: gpu_name, gpu_capability, cuda_driver, warp_cuda_toolkit,
  torch_cuda, cpu, os. The five CUDA fields are None on a CPU device;
- versions: run_record.versions();
- config: robot, actuators, experiment, seed, num_envs, unroll_length,
  batch_size, num_minibatches, num_updates_per_batch, episode_length,
  steps_per_iter;
- warmup_s: construct (env construction), reset (the first reset),
  first_step (the first env.step: kernel compile and graph capture) and
  first_iteration (the first trainer.iteration), each timed on its own;
- throughput: physics_steps_per_s (control steps of env.physics.step with
  the neutral ctrl), env_steps_per_s (env.step with zero actions),
  ppo_env_steps_per_s (steps_per_iter * iters over the timed iterations),
  plus the step and iteration counts behind them;
- budgets: sim_budget.budget_report with the resolved naconmax_per_env and
  njmax, the max nefc seen, and the pool-wide nacon peak with its own
  overflow flag;
- memory: ppo.memory_mb(env) after the run;
- ignored: the set keys the trainer ignores, from ppo.check_config.

Bench numbers compare only between runs at the same seed on the same
hardware. Two Brax runs differing only in seed differed by 1.7x in steps/s,
and memory and throughput do not transfer between node classes.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from datetime import datetime
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from humanoid_lab.torch_warp import policy_io
from humanoid_lab.torch_warp import ppo as warp_ppo

CUDA_FIELDS = ("gpu_name", "gpu_capability", "cuda_driver", "warp_cuda_toolkit", "torch_cuda")


def git_sha() -> str | None:
    """HEAD of the repo checkout, or None when git is missing or fails."""
    from humanoid_lab import paths

    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=paths.REPO_ROOT, capture_output=True, text=True, check=False
        )
    except OSError:
        return None
    sha = out.stdout.strip()
    return sha if out.returncode == 0 and sha else None


def hardware(device: torch.device) -> dict:
    """GPU, driver, toolkit, CPU and OS. The CUDA fields are None on CPU."""
    out = dict.fromkeys(CUDA_FIELDS)
    if device.type == "cuda":
        import warp as wp

        out["gpu_name"] = torch.cuda.get_device_name(device)
        out["gpu_capability"] = list(torch.cuda.get_device_capability(device))
        out["cuda_driver"] = wp.get_cuda_driver_version()
        out["warp_cuda_toolkit"] = wp.get_cuda_toolkit_version()
        out["torch_cuda"] = torch.version.cuda
    out["cpu"] = platform.processor() or platform.machine()
    out["os"] = platform.platform()
    return out


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


@hydra.main(version_base=None, config_path="../../../configs", config_name="torch_warp")
def main(cfg: DictConfig) -> None:
    from humanoid_lab import paths, run_record, sim_budget
    from humanoid_lab.envs import symmetry
    from humanoid_lab.ppo_config import (
        brax_train_kwargs,
        compose_ppo_params,
        experiment_name,
        resolve_episode_length,
    )
    from humanoid_lab.torch_warp.env import WarpJoystick, check_device, env_config_of

    # -- config, as ppo.main builds it -------------------------------------------
    hydra_cfg = OmegaConf.to_container(cfg, resolve=True)
    ppo_params = compose_ppo_params(cfg)
    ignored = warp_ppo.check_config(hydra_cfg, ppo_params.to_dict())
    _training, net_kwargs, repo = brax_train_kwargs(ppo_params)
    net_kwargs = net_kwargs.to_dict() if hasattr(net_kwargs, "to_dict") else dict(net_kwargs)
    sym_coef = float(repo.get("sym_coef", 0.0))
    resolve_episode_length(ppo_params, cfg, env_config_of(hydra_cfg).episode_length)
    episode_length = int(ppo_params.episode_length)
    n = int(ppo_params.num_envs)
    seed = int(cfg.seed)
    iters = int(cfg.bench.iters)
    physics_steps = int(cfg.bench.physics_steps)
    if iters < 1 or physics_steps < 1:
        raise ValueError(f"bench.iters ({iters}) and bench.physics_steps ({physics_steps}) must be at least 1")
    device = check_device(cfg.torch_warp.device)
    is_cuda = device.type == "cuda"
    wdev = None

    def sync():
        # Device work is asynchronous on CUDA, so every timer read waits for
        # it. On CPU every call has already finished.
        if is_cuda:
            import warp as wp

            torch.cuda.synchronize(device)
            if wdev is not None:
                wp.synchronize_device(wdev)

    def timed(fn, count=1):
        sync()
        t0 = time.perf_counter()
        for _ in range(count):
            fn()
        sync()
        return time.perf_counter() - t0

    nacon_peak, nefc_peak = 0, 0

    def sample_peaks():
        # peaks() syncs on CUDA, so it runs between the timed phases only.
        nonlocal nacon_peak, nefc_peak
        nacon, nefc = env.physics.peaks()
        nacon_peak, nefc_peak = max(nacon_peak, nacon), max(nefc_peak, nefc)

    # -- warm-up -----------------------------------------------------------------
    warmup = {}
    t0 = time.perf_counter()
    env = WarpJoystick(hydra_cfg, n, episode_length, device=device, seed=seed)
    wdev = env.physics.wdev
    sync()
    warmup["construct"] = time.perf_counter() - t0
    warmup["reset"] = timed(env.reset)
    zeros = torch.zeros(n, env.action_size, device=env.device)
    warmup["first_step"] = timed(lambda: env.step(zeros))
    sample_peaks()

    # -- physics only and env.step -----------------------------------------------
    neutral = env.consts.neutral_ctrl.expand(n, -1).contiguous()
    physics_s = timed(lambda: env.physics.step(neutral), physics_steps)
    sample_peaks()
    env.reset()
    env_step_s = timed(lambda: env.step(zeros), physics_steps)
    sample_peaks()

    # -- PPO iterations ----------------------------------------------------------
    mirror_maps = None
    if sym_coef > 0:
        maps = symmetry.policy_mirror_maps(env.robot_spec, env.actor_obs_names)
        mirror_maps = tuple(torch.as_tensor(np.asarray(m), device=env.device) for m in maps)
    trainer = warp_ppo.Trainer(
        env,
        ppo_params,
        sym_coef=sym_coef,
        mirror_maps=mirror_maps,
        critic_warmup_iters=int(cfg.torch_warp.critic_warmup_iters),
        seed=seed,
    )
    trainer.load_params(
        policy_io.init_from_brax(seed, env.actor_obs_size, env.privileged_obs_size, env.action_size, net_kwargs)
    )
    warmup["first_iteration"] = timed(trainer.iteration)
    sample_peaks()
    ppo_s = timed(trainer.iteration, iters)
    sample_peaks()

    # -- record ------------------------------------------------------------------
    consts = env.consts_np
    budgets = sim_budget.budget_report("warp", None, nefc_peak, consts.naconmax_per_env, consts.njmax, n)
    # MJWarp's live nacon counts the whole pool, and budget_report compares
    # nacon_max per world. The pool peak gets its own field and flag.
    budgets["nacon_pool_max"] = int(nacon_peak)
    budgets["pool_overflow"] = bool(nacon_peak >= budgets["pool"])
    record = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "device": str(env.device),
        "git_sha": git_sha(),
        "hardware": hardware(env.device),
        "versions": run_record.versions(),
        "config": {
            "robot": hydra_cfg["robot"]["name"],
            "actuators": hydra_cfg["actuators"]["name"],
            "experiment": experiment_name(),
            "seed": seed,
            "num_envs": n,
            "unroll_length": int(ppo_params.unroll_length),
            "batch_size": int(ppo_params.batch_size),
            "num_minibatches": int(ppo_params.num_minibatches),
            "num_updates_per_batch": int(ppo_params.num_updates_per_batch),
            "episode_length": episode_length,
            "steps_per_iter": int(trainer.steps_per_iter),
        },
        "warmup_s": {k: round(v, 4) for k, v in warmup.items()},
        "throughput": {
            "physics_steps": physics_steps,
            "iters": iters,
            "physics_steps_per_s": round(n * physics_steps / physics_s, 1),
            "env_steps_per_s": round(n * physics_steps / env_step_s, 1),
            "ppo_env_steps_per_s": round(trainer.steps_per_iter * iters / ppo_s, 1),
        },
        "budgets": budgets,
        "memory": warp_ppo.memory_mb(env),
        "ignored": ignored,
    }

    out = cfg.bench.out
    if out:
        path = Path(out)
    else:
        path = paths.RUNS_DIR / "bench" / f"bench_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    _write_json(path, record)
    print(json.dumps(record["throughput"], indent=2))
    print(f"bench -> {path}")


if __name__ == "__main__":
    main()
