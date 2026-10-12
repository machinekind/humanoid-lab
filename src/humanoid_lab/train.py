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

import contextlib
import functools
import time
from collections.abc import Mapping
from datetime import datetime

import hydra
from omegaconf import DictConfig, OmegaConf

from humanoid_lab import paths, run_record

# tests/integration/reference_rollouts.py imports build_ppo_params and
# _apply_ppo_overrides from humanoid_lab.train, on origin/main too, so both
# stay importable here. Never edit reference_rollouts.py: it must still run
# on 042f018.
from humanoid_lab.ppo_config import (  # noqa: F401
    REPO_PPO_KEYS,
    _apply_ppo_overrides,
    brax_train_kwargs,
    build_ppo_params,
    compose_ppo_params,
    resolve_episode_length,
)


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


def _with_sym_loss(
    params,
    normalizer_params,
    data,
    rng,
    *,
    ppo_network,
    sym_coef,
    obs_key,
    obs_perm,
    obs_sign,
    act_perm,
    act_sign,
    compute_ppo_loss,
    **kwargs,
):
    """brax's PPO loss plus sym_coef * mean((loc(M s) - M loc(s))^2).

    loc is the policy mean, the first half of the distribution params. The
    mirror M applies to the raw observation, before the network normalizes
    it. Gradients flow through both branches. The metrics gain 'sym_loss',
    which brax reports as training/sym_loss.
    """
    import jax.numpy as jnp

    loss, metrics = compute_ppo_loss(
        params, normalizer_params, data, rng, ppo_network=ppo_network, **kwargs
    )
    a = ppo_network.parametric_action_distribution.param_size // 2
    obs = data.observation
    if isinstance(obs, Mapping):
        obs_m = {**obs, obs_key: obs[obs_key][..., obs_perm] * obs_sign}
    else:
        obs_m = obs[..., obs_perm] * obs_sign
    apply = ppo_network.policy_network.apply
    loc = apply(normalizer_params, params.policy, obs)[..., :a]
    loc_m = apply(normalizer_params, params.policy, obs_m)[..., :a]
    sym = jnp.mean(jnp.square(loc_m - loc[..., act_perm] * act_sign))
    return loss + sym_coef * sym, {**metrics, "sym_loss": sym}


def sym_loss_patch(sym_coef, obs_perm, obs_sign, act_perm, act_sign, obs_key):
    """A context manager that adds the mirror-symmetry loss to brax's PPO.

    brax's ppo.train takes no loss argument. It reads
    ppo_losses.compute_ppo_loss when train() runs, so the patch must wrap the
    whole train_fn call. The patch is undone when the block exits. The added
    loss consumes no RNG key, so the run's random stream is unchanged.

    A zero sym_coef returns a nullcontext and patches nothing, so the run is
    the same as without this function.

    Verified on brax 0.14.2 only. tests/unit/test_brax_sym_loss.py checks
    that ppo.train still looks the loss up by that name.
    """
    if sym_coef == 0:
        return contextlib.nullcontext()
    from unittest import mock

    import jax.numpy as jnp
    from brax.training.agents.ppo import losses as ppo_losses

    original = ppo_losses.compute_ppo_loss
    return mock.patch.object(
        ppo_losses,
        "compute_ppo_loss",
        functools.partial(
            _with_sym_loss,
            compute_ppo_loss=original,
            sym_coef=float(sym_coef),
            obs_key=obs_key,
            obs_perm=jnp.asarray(obs_perm),
            obs_sign=jnp.asarray(obs_sign),
            act_perm=jnp.asarray(act_perm),
            act_sign=jnp.asarray(act_sign),
        ),
    )


def _wandb_group(cfg: DictConfig) -> str | None:
    group = cfg.wandb.get("group")
    if group:
        return str(group)
    from humanoid_lab.ppo_config import experiment_name

    return experiment_name()


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    import jax  # heavy imports stay inside main so --cfg job is fast
    from brax.training.agents.ppo import networks as ppo_networks
    from brax.training.agents.ppo import train as ppo

    from humanoid_lab.dr.randomize import make_domain_randomize
    from humanoid_lab.dr.transform import check_dr_enabled
    from humanoid_lab.envs import symmetry
    from humanoid_lab.envs.wrappers import make_wrap_env_fn
    from humanoid_lab.registry import make_env

    task = cfg.task.name
    robot_dir = paths.REPO_ROOT / cfg.robot.dir
    preset_name = cfg.actuators.name
    actuator_overrides = (OmegaConf.to_container(cfg.actuators, resolve=True) or {}).get("overrides") or {}
    env_overrides = OmegaConf.to_container(cfg.task.env, resolve=True) or {}

    # PPO params resolve before the envs because the warp backend sizes its
    # contact buffer across the whole training batch, so the env config
    # needs the final num_envs at construction time.
    # The network group merges into the factory kwargs, then task.ppo, then
    # ppo (see ppo_config.compose_ppo_params).
    ppo_params = compose_ppo_params(cfg)

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

    # Gains as the compiled model holds them, for run.json (see
    # run_record.actuator_gains).
    hydra_cfg = OmegaConf.to_container(cfg, resolve=True)
    gains = run_record.actuator_gains(env, hydra_cfg)
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
    # Smoke keeps episodes short too (see ppo_config.resolve_episode_length).
    resolve_episode_length(ppo_params, cfg, env._config.episode_length)

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

    # REPO_PPO_KEYS leave training_params here. brax's ppo.train rejects them.
    training_params, network_factory_cfg, repo = brax_train_kwargs(ppo_params)
    sym_coef = float(repo.get("sym_coef", 0.0))
    # Maps are built only for a nonzero coefficient. A robot without a
    # JOINT_SIGNS entry (asimov_v1) then raises a KeyError naming it, and
    # still trains at 0.
    mirror_maps = (None, None, None, None)
    if sym_coef != 0:
        mirror_maps = symmetry.policy_mirror_maps(env.robot_spec, env.actor_obs_names)
        print(f"symmetry loss on, coef {sym_coef}")
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
        check_dr_enabled(cfg.domain_rand, dr_cfg)

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
        # brax attaches the last epoch's training metrics to every eval call
        # after the first; the LR moves under ppo.learning_rate_schedule
        # ADAPTIVE_KL, and kl_mean is what drives it.
        opt = "".join(
            f"  {label} {float(metrics[key]):.3g}"
            for label, key in (
                ("lr", "training/learning_rate"),
                ("kl", "training/kl_mean"),
                ("sym", "training/sym_loss"),
            )
            if key in metrics
        )
        print(
            f"steps {num_steps:>12,}  reward {reward:8.2f}  "
            f"ep_len {ep_len:6.0f}  {sps:,.0f} steps/s{opt}"
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
        # mujoco_playground's own wrapping, plus a respawn reseed layer when
        # no_progress is on or a gait-symmetry scale is nonzero (see
        # envs/wrappers.py). With neither, this IS
        # wrapper.wrap_for_brax_training.
        wrap_env_fn=make_wrap_env_fn(env._config),
        save_checkpoint_path=str(ckpt_dir),
        restore_checkpoint_path=restore,
        progress_fn=progress,
    )
    early_stopped = False
    try:
        # With sym_coef 0 the patch is a nullcontext and this is the plain call.
        with sym_loss_patch(
            sym_coef, *mirror_maps, network_factory_cfg.get("policy_obs_key", "state")
        ):
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

    # Written once, after training. jobs/train_chain.sh reads a fresh
    # run.json as proof that the run finished.
    record = run_record.build_run_record(
        env,
        hydra_cfg,
        ppo_params,
        run_name=run_name,
        checkpoint_dir=ckpt_dir,
        contacts=contacts,
        env_config=env._config.to_dict(),
        actuator_gains=gains,
        trainer=run_record.trainer_block("brax_ppo", env._backend, jax.devices()[0]),
        results={
            "early_stopped": early_stopped,
            "stopped_at_steps": int(last_eval["steps"]),
            "final_reward": float(metrics.get("eval/episode_reward", float("nan"))),
        },
        # A missing source run never stops a Brax run.
        warm_start=(
            run_record.warm_start_block(restore, hydra_cfg, strict=False) if restore else None
        ),
    )
    run_record.write_run(run_dir, record)
    print(f"done -> {run_dir}")
    if wb is not None:
        wb.finish()


if __name__ == "__main__":
    main()
