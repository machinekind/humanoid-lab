"""Hydra-configured Brax PPO trainer for humanoid-lab tasks.

Experiments are Hydra configs (repo-root configs/): pick a robot/task/
actuators combination and override anything from the CLI, e.g.

    ./run.sh train robot=roboto_origin task=joystick actuators=deploy_pd
    ./run.sh train ppo.num_timesteps=3e8 run_name=roboto_walk_v1

PPO hyper-parameters start from the playground's tuned Go1 config;
`task.ppo` then global `ppo` yaml/CLI blocks override them.
"""

import os

os.environ.setdefault("XLA_FLAGS", "--xla_gpu_triton_gemm_any=true")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.9")

import functools
import json
import time
from datetime import datetime

import hydra
from omegaconf import DictConfig, OmegaConf

from humanoid_lab import paths


class EarlyStop(Exception):
    """Raised inside the progress callback to abort a plateaued run."""


def plateau_stop(rewards, min_evals: int, patience: int, min_delta: float) -> bool:
    """True when `rewards` (eval rewards, oldest first) has plateaued.

    A reward counts as a new best only when it beats the running best by
    more than `min_delta` (eval noise must not reset the clock). Plateau =
    `patience` consecutive evals without such a new best, checked only once
    `max(min_evals, patience + 1)` evals exist.
    """
    if len(rewards) < max(min_evals, patience + 1):
        return False
    best, best_i = float("-inf"), 0
    for i, r in enumerate(rewards):
        if r > best + min_delta:
            best, best_i = r, i
    return len(rewards) - 1 - best_i >= patience


def record_eval(metrics: dict, num_steps: int, eval_rewards: list, last_eval: dict) -> bool:
    """Record one progress callback if it is an eval, and say whether it was.

    Only eval calls carry eval/episode_reward. Today brax's evaluator is the
    progress callback's sole producer, but ppo.log_training_metrics is one
    free-form override away, and its EpisodeMetricsLogger calls would read as
    NaN rewards -- NaN never beats the running best, so a few of them would
    fake a plateau and stop the run. A training call must therefore leave both
    the reward history and the last-eval record untouched.

    The history is appended whether or not early stopping is enabled: it costs
    a float per eval, and keeping the filter and the recording as one unit is
    what makes them testable together.
    """
    if "eval/episode_reward" not in metrics:
        return False
    last_eval["steps"], last_eval["metrics"] = num_steps, metrics
    eval_rewards.append(float(metrics["eval/episode_reward"]))
    return True


def _apply_ppo_overrides(p, overrides: dict) -> None:
    for k, v in (overrides or {}).items():
        if isinstance(v, dict):
            _apply_ppo_overrides(getattr(p, k), v)
        else:
            if isinstance(v, float) and v.is_integer() and not isinstance(
                getattr(p, k, None), float
            ):
                v = int(v)  # ppo.num_timesteps=3e8 arrives as float
            setattr(p, k, v)


def build_ppo_params(overrides, smoke: bool):
    """Playground Go1 PPO config + overrides.

    `overrides` is either a dict (hydra path) or a legacy ["k=v", ...] list
    (app.py / eval.py still call it that way).
    """
    from mujoco_playground.config import locomotion_params

    p = locomotion_params.brax_ppo_config("Go1JoystickFlatTerrain")
    p.network_factory.policy_obs_key = "state"
    p.network_factory.value_obs_key = "privileged_state"
    if smoke:
        p.num_timesteps = 100_000
        p.num_envs = 64
        p.batch_size = 64
        p.num_minibatches = 4
        p.num_evals = 2
    if isinstance(overrides, dict):
        _apply_ppo_overrides(p, overrides)
    else:
        for kv in overrides or []:
            k, v = kv.split("=", 1)
            for cast in (int, float):
                try:
                    v = cast(v)
                    break
                except ValueError:
                    continue
            setattr(p, k, v)
    return p


# Contact-budget preflight. Deliberately smaller than
# check_contacts' own defaults (200 steps x 5 seeds): this runs in front of
# every training job, and three seeds is what the fallen sweep's three
# attitudes need. It is a warning system, not the sizing measurement -- use
# `./run.sh check-contacts` for that.
PREFLIGHT_STEPS = 100
PREFLIGHT_SEEDS = 3


def _contact_preflight(env, cfg) -> dict:
    """The `contacts` block run.json carries, from a probe run before training.

    Why a probe rather than the training loop: brax's PPO loop is one jitted
    scan, so no Python-side code ever holds an `mjx.Data` while it runs and
    the live warp counters are unreachable from there. A short probe on the
    training env itself, on the real backend, measures the same per-world
    peaks -- and it runs BEFORE the job spends GPU hours, which is the point
    of a preflight: an undersized buffer is a silent wrong-physics bug, not a
    crash, so the only cheap moment to catch it is before the run.

    `smoke=true` skips the probe (a smoke run checks the pipeline, and the
    probe costs more than the training does), and `contact_preflight=false`
    turns it off. The block is written either way, with null peaks when
    nothing measured them, so run.json's shape never depends on the switch.
    """
    from humanoid_lab import check_contacts, sim_budget

    if cfg.smoke or not cfg.get("contact_preflight", True):
        return sim_budget.budget_report_for_env(env, None, None)
    measured = check_contacts.measure_env(env, PREFLIGHT_STEPS, PREFLIGHT_SEEDS)
    peak = measured["peak"]
    # nefc is only reported when the backend measured it. check_contacts
    # derives a row count on jax for its own sizing table; run.json must not
    # carry a field whose meaning changes with the backend that wrote it.
    nefc = peak["nefc_max"] if not measured["nefc_derived"] else None
    return sim_budget.budget_report_for_env(env, peak["nacon_max"], nefc)


def _wandb_group(cfg: DictConfig) -> str | None:
    group = cfg.wandb.get("group")
    if group:
        return str(group)
    from hydra.core.hydra_config import HydraConfig

    try:
        choice = HydraConfig.get().runtime.choices.get("experiment")
    except ValueError:
        return None  # main() invoked outside a Hydra app (e.g. tests)
    return None if choice in (None, "null") else choice


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    import jax  # noqa: F401  # heavy imports stay inside main so --cfg job is fast
    from brax.training.agents.ppo import networks as ppo_networks
    from brax.training.agents.ppo import train as ppo

    from humanoid_lab.dr.randomize import make_domain_randomize
    from humanoid_lab.envs.wrappers import make_wrap_env_fn
    from humanoid_lab.registry import make_env
    from humanoid_lab.robot.presets import effective_gains, load_actuator_preset

    task = cfg.task.name
    robot_dir = paths.REPO_ROOT / cfg.robot.dir
    preset_name = cfg.actuators.name
    actuator_overrides = (OmegaConf.to_container(cfg.actuators, resolve=True) or {}).get("overrides") or {}
    env_overrides = OmegaConf.to_container(cfg.task.env, resolve=True) or {}

    # PPO params resolve before the envs because the warp backend sizes its
    # contact buffer across the whole training batch, so the env config
    # needs the final num_envs at construction time.
    ppo_params = build_ppo_params({}, cfg.smoke)
    # Network group (configs/network) merges into the factory kwargs; explicit
    # ppo.network_factory.* overrides still win below.
    _apply_ppo_overrides(
        ppo_params.network_factory,
        OmegaConf.to_container(cfg.network, resolve=True) or {},
    )
    task_ppo_overrides = OmegaConf.to_container(cfg.task.ppo, resolve=True) or {}
    ppo_overrides = OmegaConf.to_container(cfg.ppo, resolve=True) or {}
    _apply_ppo_overrides(ppo_params, task_ppo_overrides)
    _apply_ppo_overrides(ppo_params, ppo_overrides)

    # The eval wrapper vmaps num_eval_envs worlds through the same env, so
    # the warp contact budget covers the larger of the two batches. This is
    # why env_overrides is mutated here instead of left to the task's own
    # sim.num_envs default: the warp backend needs the final batch size at
    # env construction time (see envs/backend.py), not at train-loop time.
    env_overrides.setdefault("sim", {})["num_envs"] = int(
        max(ppo_params.num_envs, ppo_params.get("num_eval_envs", 0))
    )
    env = make_env(task, robot_dir, preset_name, env_overrides, actuator_overrides)
    eval_env = make_env(task, robot_dir, preset_name, env_overrides, actuator_overrides)
    print(f"actor obs ({len(env.actor_obs_names)} components): {env.actor_obs_names}")

    # Gains as the compiled model holds them, for run.json. Read here rather
    # than from cfg.actuators: the yaml is what was asked for, and
    # actuators.overrides merges on top of it inside load_actuator_preset.
    # The preset is re-loaded through that same choke point purely for its
    # model name, which the mjModel does not carry.
    gains = effective_gains(
        env.mj_model.actuator_gainprm,
        env.mj_model.actuator_biasprm,
        env.robot_spec.actuated_joints,
        model=load_actuator_preset(robot_dir, preset_name, actuator_overrides).model,
        preset=preset_name,
    )
    print(f"actuator gains: {gains['model']} preset {gains['preset']}")

    contacts = _contact_preflight(env, cfg)
    print(f"contacts: {contacts}")
    if contacts["overflow"] or contacts["rows_overflow"]:
        print(
            "WARNING: the preflight already reached a warp buffer ceiling. Warp drops "
            "the overflow silently, so this run would be training against dropped "
            "contacts or unenforced constraint rows. Raise task.env.sim.naconmax_per_env "
            "/ njmax (see ./run.sh check-contacts) before trusting the result."
        )

    # Episode length follows the env config unless ppo yaml overrides it.
    # Smoke keeps episodes short too: brax's eval unrolls full episodes, so
    # a 1000-step episode dominates a CPU pipeline check.
    if "episode_length" not in task_ppo_overrides and "episode_length" not in ppo_overrides:
        ppo_params.episode_length = (
            min(200, env._config.episode_length) if cfg.smoke
            else env._config.episode_length
        )

    restore = cfg.restore
    if restore and not os.path.isabs(restore):
        restore = str(paths.PROJECT_DIR / restore)

    run_name = cfg.run_name or (
        ("smoke_" if cfg.smoke else "")
        + f"{task}_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    run_dir = paths.RUNS_DIR / run_name
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    training_params = dict(ppo_params)
    network_factory_cfg = training_params.pop("network_factory")
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks, **network_factory_cfg
    )

    # cfg.dr composes here (configs/dr); domain_rand gates whether it is
    # actually applied.
    dr_cfg = OmegaConf.to_container(cfg.dr, resolve=True)
    if cfg.domain_rand:
        training_params["randomization_fn"] = make_domain_randomize(
            env.mj_model, env.robot_spec, dr_cfg
        )
    else:
        enabled = [k for k, v in dr_cfg.items() if isinstance(v, dict) and v.get("enable")]
        if enabled:
            raise ValueError(
                f"domain_rand=false disables the whole dr block, but "
                f"cfg.dr fields {enabled} have enable=true; either set "
                f"domain_rand=true or disable those fields too"
            )

    wb = None
    if cfg.wandb.enable:
        try:
            import wandb

            wb = wandb.init(
                project=cfg.wandb.project,
                name=run_name,
                group=_wandb_group(cfg),
                config={
                    "hydra": OmegaConf.to_container(cfg, resolve=True),
                    "ppo": dict(ppo_params),
                    "env": env._config.to_dict(),
                },
            )
        except Exception as e:  # noqa: BLE001
            print(f"wandb disabled: {e}")

    t_last = [time.time(), 0]
    es = cfg.early_stop
    eval_rewards: list[float] = []
    last_eval = {"steps": 0, "metrics": {}}

    def progress(num_steps: int, metrics: dict) -> None:
        now = time.time()
        sps = (num_steps - t_last[1]) / max(now - t_last[0], 1e-9)
        t_last[0], t_last[1] = now, num_steps
        reward = metrics.get("eval/episode_reward", float("nan"))
        # avg_episode_length exposes die-and-reset reward hacking that the
        # reward number alone hides.
        ep_len = metrics.get("eval/avg_episode_length", float("nan"))
        print(
            f"steps {num_steps:>12,}  reward {reward:8.2f}  "
            f"ep_len {ep_len:6.0f}  {sps:,.0f} steps/s"
        )
        if wb is not None:
            wb.log({**metrics, "perf/steps_per_sec": sps}, step=num_steps)
        # Training-metric calls carry no eval reward and record nothing; see
        # record_eval, which tests/unit/test_early_stop.py exercises directly.
        if not record_eval(metrics, num_steps, eval_rewards, last_eval):
            return
        if es.enable and plateau_stop(
            eval_rewards, es.min_evals, es.patience, es.min_delta
        ):
            print(
                f"early stop: no reward gain > {es.min_delta} in the last "
                f"{es.patience} evals (best {max(eval_rewards):.2f}); "
                f"stopping at {num_steps:,} steps"
            )
            raise EarlyStop

    train_fn = functools.partial(
        ppo.train,
        **training_params,
        network_factory=network_factory,
        seed=cfg.seed,
        # mujoco_playground's own wrapping, except with no_progress on, where
        # it gains the respawn reseed layer (see envs/wrappers.py). Off, this
        # IS wrapper.wrap_for_brax_training.
        wrap_env_fn=make_wrap_env_fn(env._config),
        save_checkpoint_path=str(ckpt_dir),
        restore_checkpoint_path=restore,
        progress_fn=progress,
    )
    early_stopped = False
    try:
        make_inference_fn, params, metrics = train_fn(
            environment=env, eval_env=eval_env
        )
    except EarlyStop:
        # Brax saves a checkpoint at every eval, so the newest checkpoint in
        # ckpt_dir IS the early-stopped policy -- nothing is lost by cutting
        # the run here. train_fn never returned, so report from the last
        # completed eval instead of brax's return values.
        early_stopped = True
        metrics = last_eval["metrics"]

    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "run_name": run_name,
                "task": task,
                "num_timesteps": int(ppo_params.num_timesteps),
                "early_stopped": early_stopped,
                # The last eval's step count. On a normal run that is the
                # final eval, so this equals the budget brax actually ran.
                "stopped_at_steps": int(last_eval["steps"]),
                "final_reward": float(
                    metrics.get("eval/episode_reward", float("nan"))
                ),
                "checkpoint_dir": str(ckpt_dir),
                # Warp contact/constraint budgets and the preflight's peaks.
                # Same schema as battery.json's, on every backend, so a GPU
                # run and a local one diff without branching.
                "contacts": contacts,
                "env_config": env._config.to_dict(),
                "ppo_config": ppo_params.to_dict(),
                "hydra_config": OmegaConf.to_container(cfg, resolve=True),
                "actuators": OmegaConf.to_container(cfg.actuators, resolve=True),
                # What the BUILT model got, next to the `actuators` block
                # above, which is what the config asked for. The two differ
                # whenever actuators.overrides patches a gain: the override
                # never appears in the preset yaml.
                "actuator_gains": gains,
            },
            indent=2,
            default=str,
        )
    )
    print(f"done -> {run_dir}")
    if wb is not None:
        wb.finish()


if __name__ == "__main__":
    main()
