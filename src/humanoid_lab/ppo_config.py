"""The PPO hyperparameters both trainers use.

They start from the playground's tuned Go1 config. The network group, then
`task.ppo`, then the global `ppo` block override it.

REPO_PPO_KEYS are keys the repo owns. Brax's ppo.train does not take them.

Importing this module sets no environment variable. train.py sets the XLA
variables at its own import, and that stays there.
"""

from omegaconf import OmegaConf

# Popped before brax's ppo.train. They stay in run.json's ppo_config, and the
# Torch trainer reads them too.
REPO_PPO_KEYS = frozenset({"sym_coef"})


def experiment_name() -> str | None:
    """The `experiment=` choice of the running Hydra app, or None.

    Both trainers use it as the default wandb group, and bench records it.
    """
    from hydra.core.hydra_config import HydraConfig

    try:
        choice = HydraConfig.get().runtime.choices.get("experiment")
    except ValueError:
        return None  # main() invoked outside a Hydra app (e.g. tests)
    return None if choice in (None, "null") else str(choice)


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
    # Not in the playground config; set to brax's default so ppo.gae_lambda
    # overrides land on a visible key.
    p.gae_lambda = 0.95
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


def compose_ppo_params(cfg):
    """The PPO params as both trainers assemble them.

    The REPO_PPO_KEYS stay in the returned ConfigDict.
    """
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
    return ppo_params


def resolve_episode_length(ppo_params, cfg, env_episode_length) -> None:
    """Set ppo_params.episode_length from the env unless task.ppo or ppo sets it."""
    task_ppo_overrides = OmegaConf.to_container(cfg.task.ppo, resolve=True) or {}
    ppo_overrides = OmegaConf.to_container(cfg.ppo, resolve=True) or {}
    # Episode length follows the env config unless ppo yaml overrides it.
    # Smoke keeps episodes short too: brax's eval unrolls full episodes, so
    # a 1000-step episode dominates a CPU pipeline check.
    if "episode_length" not in task_ppo_overrides and "episode_length" not in ppo_overrides:
        ppo_params.episode_length = (
            min(200, env_episode_length) if cfg.smoke
            else env_episode_length
        )


def brax_train_kwargs(ppo_params):
    """Split ppo_params into brax's ppo.train kwargs, the factory kwargs and the repo keys.

    Returns (training_params, network_factory_cfg, repo). training_params is
    dict(ppo_params) without 'network_factory' and without REPO_PPO_KEYS.
    repo holds the REPO_PPO_KEYS that are present.
    """
    training_params = dict(ppo_params)
    network_factory_cfg = training_params.pop("network_factory")
    repo = {k: training_params.pop(k) for k in sorted(REPO_PPO_KEYS) if k in training_params}
    return training_params, network_factory_cfg, repo
