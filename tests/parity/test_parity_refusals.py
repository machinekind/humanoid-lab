"""Construction and config refusals of the Torch trainer.

Each thing the Torch trainer does not implement fails at construction or in
the config ledger (torch_warp.ppo.check_config), with a message that says
what to do. The PR description lists each refusal with its test.
"""

from __future__ import annotations

import inspect

import pytest
from brax.training.agents.ppo import networks as brax_networks
from brax.training.agents.ppo import train as brax_train
from hydra import compose, initialize_config_dir

from humanoid_lab import paths
from humanoid_lab.ppo_config import compose_ppo_params

torch = pytest.importorskip("torch")

from humanoid_lab.torch_warp import env as warp_env
from humanoid_lab.torch_warp import ppo as warp_ppo


def _hydra(env=None, robot_dir="robots/roboto_origin", actuators="deploy_pd"):
    return {
        "task": {"name": "joystick", "env": env or {}},
        "robot": {"dir": robot_dir},
        "actuators": {"name": actuators, "overrides": {}},
        "domain_rand": False,
        "dr": {},
    }


def test_sim_backend_jax_is_refused():
    with pytest.raises(ValueError, match="sim.backend=jax"):
        warp_env.WarpJoystick(_hydra({"sim": {"backend": "jax"}}), 2, 10, device="cpu")


def test_missing_budgets_are_refused():
    # tests/data/toy_robot has no sim_budget block.
    hydra = _hydra({"reset_keyframe": "standing"}, robot_dir="tests/data/toy_robot", actuators="pd_test")
    with pytest.raises(ValueError, match="check-contacts"):
        warp_env.WarpJoystick(hydra, 2, 10, device="cpu")


def test_the_sizing_task_is_refused():
    # Sizing subclasses Joystick. Its telemetry is per-step metrics, which
    # the Torch env does not emit.
    hydra = _hydra()
    hydra["task"]["name"] = "sizing"
    with pytest.raises(ValueError, match="sizing stays a JAX rollout"):
        warp_env.WarpJoystick(hydra, 2, 10, device="cpu")


def test_a_cuda_device_without_cuda_is_refused():
    if torch.cuda.is_available():
        pytest.skip("this host has CUDA")
    with pytest.raises(RuntimeError, match="download.pytorch.org"):
        warp_env.check_device("cuda:0")
    with pytest.raises(RuntimeError, match="download.pytorch.org"):
        warp_env.WarpJoystick(_hydra(), 2, 10, device="cuda:0")


# -- the config ledger ---------------------------------------------------------
# Config-only checks: Hydra composes, nothing is built.

# A value the Torch loop does not implement, for each refused key.
REFUSED_PPO = {
    "action_repeat": 2,
    "reward_scaling": 0.5,
    "clipping_epsilon_value": 0.2,
    "normalize_observations_mode": "ema",
    "normalize_observations_std_eps": 1e-6,
    "normalize_until_count": 1000,
    "bootstrap_on_timeout": True,
    "use_distributional_critic": True,
    "learning_rate_schedule": "ADAPTIVE_KL",
    "vision": True,
    "augment_pixels": True,
    "restore_value_fn": False,
    "max_devices_per_host": 2,
    "normalize_observations": False,
    "normalize_advantage": False,
}
REFUSED_NETWORK = {
    "activation": "relu",
    "distribution_type": "normal",
    "layer_norm": True,
    "policy_network_kernel_init_fn": "orthogonal",
    "value_network_kernel_init_fn": "orthogonal",
    "mean_kernel_init_fn": "orthogonal",
    "policy_network_kernel_init_kwargs": {"scale": 0.01},
    "value_network_kernel_init_kwargs": {"scale": 0.01},
    "mean_kernel_init_kwargs": {"scale": 0.01},
    "use_distributional_critic": True,
    "policy_obs_key": "privileged_state",
    "value_obs_key": "state",
}


def _composed(overrides=()):
    with initialize_config_dir(version_base=None, config_dir=str(paths.CONFIGS_DIR)):
        cfg = compose(config_name="torch_warp", overrides=list(overrides))
    return cfg, compose_ppo_params(cfg)


def test_the_default_config_passes_with_no_ignored_key():
    cfg, params = _composed()
    assert warp_ppo.check_config(cfg, params) == {}


@pytest.mark.parametrize("key", sorted(REFUSED_PPO))
def test_each_refused_ppo_key_is_named(key):
    cfg, params = _composed()
    params[key] = REFUSED_PPO[key]
    with pytest.raises(ValueError, match=rf"ppo\.{key}="):
        warp_ppo.check_config(cfg, params)


def test_every_refusal_is_listed_at_once():
    cfg, params = _composed(["++task.env.sim.backend=jax", "early_stop.enable=true"])
    params.action_repeat = 2
    params.reward_scaling = 2.0
    with pytest.raises(ValueError) as e:
        warp_ppo.check_config(cfg, params)
    for name in ("ppo.action_repeat", "ppo.reward_scaling", "task=", "early_stop="):
        assert name in str(e.value)


def test_adaptive_kl_is_refused():
    cfg, params = _composed(["experiment=novel_adaptive_kl"])
    with pytest.raises(ValueError, match="learning_rate_schedule") as e:
        warp_ppo.check_config(cfg, params)
    # The ADAPTIVE_KL knobs are ignored, so only the schedule itself is refused.
    assert "desired_kl" not in str(e.value)


def test_ignored_keys_come_back_with_their_reason():
    cfg, params = _composed()
    params.desired_kl = 0.02
    params.network_factory.init_noise_std = 0.5
    ignored = warp_ppo.check_config(cfg, params)
    assert set(ignored) == {"ppo.desired_kl", "ppo.network_factory.init_noise_std"}
    assert "ADAPTIVE_KL" in ignored["ppo.desired_kl"]


@pytest.mark.parametrize("key", sorted(REFUSED_NETWORK))
def test_each_refused_network_key_is_named(key):
    cfg, params = _composed()
    params.network_factory[key] = REFUSED_NETWORK[key]
    with pytest.raises(ValueError, match=rf"ppo\.network_factory\.{key}="):
        warp_ppo.check_config(cfg, params)


@pytest.mark.parametrize("name", ["silu", "swish"])
def test_an_activation_string_is_refused(name):
    # make_ppo_networks needs a callable, so the string would crash later.
    cfg, params = _composed([f"++network.activation={name}"])
    with pytest.raises(ValueError, match=r"ppo\.network_factory\.activation=.*brax's default"):
        warp_ppo.check_config(cfg, params)


def test_brax_default_activation_is_accepted():
    default = inspect.signature(brax_networks.make_ppo_networks).parameters["activation"].default
    cfg, params = _composed()
    params.network_factory.activation = default
    warp_ppo.check_config(cfg, params)


def test_early_stop_is_refused():
    cfg, params = _composed(["early_stop.enable=true"])
    with pytest.raises(ValueError, match="early_stop.*eval reward"):
        warp_ppo.check_config(cfg, params)


def test_sim_backend_jax_is_refused_by_the_ledger():
    cfg, params = _composed(["++task.env.sim.backend=jax"])
    with pytest.raises(ValueError, match="sim.backend=jax"):
        warp_ppo.check_config(cfg, params)


def test_an_unknown_ppo_key_is_refused():
    cfg, params = _composed(["++ppo.made_up_knob=1"])
    with pytest.raises(ValueError, match=r"ppo\.made_up_knob: unknown key"):
        warp_ppo.check_config(cfg, params)
    cfg, params = _composed(["++ppo.network_factory.made_up_knob=1"])
    with pytest.raises(ValueError, match=r"ppo\.network_factory\.made_up_knob: unknown key"):
        warp_ppo.check_config(cfg, params)


def test_the_ledger_covers_every_brax_parameter():
    _, params = _composed()
    keys = set(inspect.signature(brax_train.train).parameters) | set(params.keys())
    assert keys <= set(warp_ppo.PPO_LEDGER), sorted(keys - set(warp_ppo.PPO_LEDGER))


def test_the_ledger_covers_every_network_kwarg():
    keys = set(inspect.signature(brax_networks.make_ppo_networks).parameters)
    assert keys <= set(warp_ppo.NETWORK_LEDGER), sorted(keys - set(warp_ppo.NETWORK_LEDGER))


def test_the_ledger_covers_every_top_level_key():
    for overrides in ([], ["experiment=roboto_gait_v40"]):
        cfg, _ = _composed(overrides)
        missing = set(cfg.keys()) - set(warp_ppo.TOP_LEVEL_LEDGER)
        assert not missing, sorted(missing)
    for entries in (warp_ppo.PPO_LEDGER, warp_ppo.NETWORK_LEDGER, warp_ppo.TOP_LEVEL_LEDGER):
        for key, entry in entries.items():
            assert entry.status in (warp_ppo.IMPLEMENTED, warp_ppo.IGNORED, warp_ppo.REFUSED), key
            if entry.status != warp_ppo.IMPLEMENTED:
                assert entry.reason, key
            if entry.status == warp_ppo.REFUSED:
                assert entry.allowed is not None, key


def test_a_batch_that_does_not_split_into_unrolls_is_refused():
    cfg, params = _composed(["++ppo.num_envs=100"])
    # 256 * 32 = 8192 trajectories do not split into whole unrolls of 100 worlds.
    with pytest.raises(ValueError, match="multiple of num_envs"):
        warp_ppo.check_config(cfg, params)
    with pytest.raises(ValueError, match="multiple of num_envs"):
        warp_ppo.loop_sizes(params)
