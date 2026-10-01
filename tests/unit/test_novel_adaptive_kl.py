"""configs/experiment/novel_adaptive_kl.yaml: yolo_v4 with brax's adaptive-KL
learning-rate schedule and clip 0.2. Pins the composed values, that the
overlay differs from yolo_v4 only in its ppo block, that train.py forwards
every ppo key under a name brax's `ppo.train` accepts, and that brax's
controller moves the LR inside the yaml's bounds on the optimizer state
train.py's settings produce. Model-free: no env, no network, no rollout.
"""

from __future__ import annotations

import inspect

import jax.numpy as jnp
import optax
import pytest
from brax.training.agents.ppo import optimizer as ppo_optimizer
from brax.training.agents.ppo import train as ppo
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from humanoid_lab import paths
from humanoid_lab.train import _apply_ppo_overrides, build_ppo_params

EXPERIMENT = "novel_adaptive_kl"
SCHEDULE_KEYS = {
    "learning_rate_schedule": "ADAPTIVE_KL",
    "desired_kl": 0.01,
    "learning_rate_schedule_min_lr": 1.0e-5,
    "learning_rate_schedule_max_lr": 3.0e-3,
    "clipping_epsilon": 0.2,
}
# Keys train.main passes to ppo.train next to dict(ppo_params).
MAIN_KWARGS = {
    "network_factory",
    "randomization_fn",
    "seed",
    "wrap_env_fn",
    "save_checkpoint_path",
    "restore_checkpoint_path",
    "progress_fn",
    "environment",
    "eval_env",
}


def _compose(experiment=EXPERIMENT):
    with initialize_config_dir(version_base=None, config_dir=str(paths.CONFIGS_DIR)):
        return compose(config_name="config", overrides=[f"experiment={experiment}"])


def _ppo_params(cfg):
    """The ppo params exactly as train.main assembles them."""
    p = build_ppo_params({}, smoke=False)
    _apply_ppo_overrides(
        p.network_factory, OmegaConf.to_container(cfg.network, resolve=True) or {}
    )
    _apply_ppo_overrides(p, OmegaConf.to_container(cfg.task.ppo, resolve=True) or {})
    _apply_ppo_overrides(p, OmegaConf.to_container(cfg.ppo, resolve=True) or {})
    return p


def test_novel_adaptive_kl_composes_the_schedule_block():
    cfg = _compose()
    assert cfg.robot.name == "roboto_origin"
    assert cfg.actuators.name == "deploy_pd"
    assert cfg.domain_rand is True
    # Unquoted in the yaml; Hydra reads it as a str.
    assert cfg.ppo.learning_rate_schedule == "ADAPTIVE_KL"
    for key, value in SCHEDULE_KEYS.items():
        if isinstance(value, float):
            assert cfg.ppo[key] == pytest.approx(value), key
    assert cfg.ppo.learning_rate == pytest.approx(1.0e-4)
    assert cfg.ppo.num_evals == 24
    assert cfg.ppo.num_timesteps == pytest.approx(1.0e9)
    # Reads state.info['time_out'], which no wrapper in the training stack sets.
    assert "bootstrap_on_timeout" not in cfg.ppo


def test_novel_adaptive_kl_is_yolo_v4_plus_the_ppo_keys_only():
    ours = OmegaConf.to_container(_compose(), resolve=True)
    base = OmegaConf.to_container(_compose("yolo_v4"), resolve=True)
    ours_ppo, base_ppo = ours.pop("ppo"), base.pop("ppo")
    assert ours == base
    assert {k: v for k, v in ours_ppo.items() if k not in SCHEDULE_KEYS} == base_ppo
    assert set(ours_ppo) - set(base_ppo) == set(SCHEDULE_KEYS)


def test_novel_adaptive_kl_reaches_brax_as_the_enum():
    p = _ppo_params(_compose())
    assert ppo_optimizer.LRSchedule(p.learning_rate_schedule) is ppo_optimizer.LRSchedule.ADAPTIVE_KL
    assert p.learning_rate == pytest.approx(1.0e-4)
    assert p.desired_kl == pytest.approx(0.01)
    assert p.learning_rate_schedule_min_lr == pytest.approx(1.0e-5)
    assert p.learning_rate_schedule_max_lr == pytest.approx(3.0e-3)
    assert p.clipping_epsilon == pytest.approx(0.2)
    # _apply_ppo_overrides turns integral floats into ints; none of these is.
    for key in ("desired_kl", "learning_rate_schedule_min_lr", "learning_rate_schedule_max_lr", "clipping_epsilon"):
        assert isinstance(p[key], float), key
    assert p.max_grad_norm == pytest.approx(1.0)
    assert p.num_evals == 24
    assert "bootstrap_on_timeout" not in p


def test_every_forwarded_ppo_key_is_a_brax_train_parameter():
    """A misspelled key would reach ppo.train as an unexpected kwarg and fail
    only after the env is built on the training host."""
    params = inspect.signature(ppo.train).parameters
    forwarded = set(dict(_ppo_params(_compose()))) | MAIN_KWARGS
    assert forwarded - set(params) == set()
    assert params["bootstrap_on_timeout"].default is False


def test_brax_controller_stays_inside_the_yaml_bounds():
    """brax wraps adam in inject_hyperparams under ADAPTIVE_KL and chains it
    behind clip_by_global_norm when max_grad_norm is set; the controller reads
    the LR from the chain's last state. From 1e-4 a low KL reaches the 3e-3
    cap on the ninth minibatch step (1e-4 * 1.5**8 = 2.56e-3), and a high KL
    walks it down to the 1e-5 floor."""
    p = _ppo_params(_compose())
    optimizer = optax.chain(
        optax.clip_by_global_norm(p.max_grad_norm),
        optax.inject_hyperparams(optax.adam)(learning_rate=p.learning_rate),
    )
    state = optimizer.init({"w": jnp.zeros(3)})

    def step(state, kl):
        return ppo_optimizer.adaptive_kl_learning_rate(
            state,
            jnp.asarray(kl),
            p.desired_kl,
            min_learning_rate=p.learning_rate_schedule_min_lr,
            max_learning_rate=p.learning_rate_schedule_max_lr,
        )

    low_kl, high_kl = p.desired_kl / 4, p.desired_kl * 4
    for _ in range(8):
        state, lr = step(state, low_kl)
    assert float(lr) == pytest.approx(1.0e-4 * 1.5**8, rel=1e-5)
    state, lr = step(state, low_kl)
    assert float(lr) == pytest.approx(p.learning_rate_schedule_max_lr, rel=1e-6)
    # Inside the dead band the LR holds.
    state, lr = step(state, p.desired_kl)
    assert float(lr) == pytest.approx(p.learning_rate_schedule_max_lr, rel=1e-6)
    for _ in range(20):
        state, lr = step(state, high_kl)
    assert float(lr) == pytest.approx(p.learning_rate_schedule_min_lr, rel=1e-6)
    assert float(state[-1].hyperparams["learning_rate"]) == pytest.approx(float(lr))
