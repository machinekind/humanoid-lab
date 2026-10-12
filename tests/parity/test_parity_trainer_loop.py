"""The Torch PPO loop (torch_warp.ppo.Trainer) on the CPU warp device.

roboto_origin deploy_pd, N=4, unroll 5, batch 4, 2 minibatches, so one
iteration is 2 unrolls of 5 steps. The networks are small and start from
brax's init through policy_io.init_from_brax.
"""

from __future__ import annotations

import math

import pytest
from hydra import compose, initialize_config_dir

from humanoid_lab import paths
from humanoid_lab.envs import symmetry
from humanoid_lab.ppo_config import compose_ppo_params

torch = pytest.importorskip("torch")

from humanoid_lab.torch_warp import policy_io
from humanoid_lab.torch_warp import ppo as warp_ppo
from humanoid_lab.torch_warp.env import WarpJoystick

N = 4
EPISODE_LENGTH = 8
HYDRA = {
    "task": {"name": "joystick", "env": {}},
    "robot": {"dir": "robots/roboto_origin"},
    "actuators": {"name": "deploy_pd", "overrides": {}},
    "domain_rand": False,
    "dr": {},
}


def _ppo_params(**changes):
    with initialize_config_dir(version_base=None, config_dir=str(paths.CONFIGS_DIR)):
        cfg = compose(config_name="torch_warp")
    p = compose_ppo_params(cfg)
    p.num_envs = N
    p.unroll_length = 5
    p.batch_size = 4
    p.num_minibatches = 2
    p.num_updates_per_batch = 2
    p.num_timesteps = 400
    p.num_evals = 2
    p.num_resets_per_eval = 1
    p.episode_length = EPISODE_LENGTH
    p.network_factory.policy_hidden_layer_sizes = (16, 16)
    p.network_factory.value_hidden_layer_sizes = (16, 16)
    for k, v in changes.items():
        setattr(p, k, v)
    return p


@pytest.fixture(scope="module")
def env():
    return WarpJoystick(HYDRA, N, EPISODE_LENGTH, device="cpu", seed=0)


def _trainer(env, params=None, **kwargs):
    params = params or _ppo_params()
    tr = warp_ppo.Trainer(env, params, **kwargs)
    kw = params.network_factory.to_dict()
    tr.load_params(policy_io.init_from_brax(0, env.actor_obs_size, env.privileged_obs_size, env.action_size, kw))
    return tr


def _flat(net):
    return torch.cat([p.detach().reshape(-1).clone() for p in net.parameters()])


def test_one_iteration_runs_and_counts_steps(env):
    tr = _trainer(env)
    assert tr.n_unrolls == 2
    assert tr.steps_per_iter == N * 5 * 2
    losses = tr.iteration()
    assert set(losses) == set(warp_ppo.LOSS_KEYS)
    assert all(isinstance(v, torch.Tensor) for v in losses.values())
    assert tr.step == tr.steps_per_iter
    assert tr.it == 1
    stats = tr.read_stats()
    for k in (*warp_ppo.LOSS_KEYS, "reward_x1000", "falls_per_1M", "done_ep_len"):
        assert math.isfinite(stats[k]), k
    # 10 steps per iteration with episodes of 8: every world finished one.
    assert stats["done_ep_len"] <= EPISODE_LENGTH
    assert set(stats["terms"]) == set(env.term_names)
    assert all(math.isfinite(v) for v in stats["terms"].values())
    assert stats["sym_loss"] == 0.0
    # The normalizers saw the whole batch.
    assert tr.norm_s.count == tr.norm_p.count == tr.steps_per_iter


def test_iters_per_reset_uses_brax_ceil(env, monkeypatch):
    # 1010 / (40 steps per iteration) = 25.25. brax takes the ceiling. The PR-head Torch runner rounded to 25.
    sizes = warp_ppo.loop_sizes(_ppo_params(num_timesteps=1010))
    assert sizes["iters_per_reset"] == 26
    assert warp_ppo.Trainer(env, _ppo_params(num_timesteps=1010)).iters_per_reset == 26
    # num_evals 3 and 2 resets per eval: 1010 / (2 * 2 * 40) = 6.3.
    assert warp_ppo.loop_sizes(_ppo_params(num_timesteps=1010, num_evals=3, num_resets_per_eval=2))[
        "iters_per_reset"
    ] == 7

    calls = []
    real_reset = env.reset

    def counting_reset():
        calls.append(1)
        return real_reset()

    monkeypatch.setattr(env, "reset", counting_reset)
    # 40 timesteps: a reset after every iteration, on top of the first one.
    tr = _trainer(env, _ppo_params(num_timesteps=40))
    assert tr.iters_per_reset == 1
    tr.iteration()
    tr.iteration()
    assert len(calls) == 3
    # brax resets per epoch only when num_resets_per_eval > 0.
    calls.clear()
    tr = _trainer(env, _ppo_params(num_timesteps=40, num_resets_per_eval=0))
    tr.iteration()
    tr.iteration()
    assert len(calls) == 1


def test_critic_warmup_freezes_the_policy(env):
    maps = symmetry.policy_mirror_maps(env.robot_spec, env.actor_obs_names)
    tr = _trainer(env, critic_warmup_iters=1, sym_coef=1.0, mirror_maps=maps)
    policy0, value0 = _flat(tr.policy), _flat(tr.value)
    losses = tr.iteration()
    assert torch.equal(_flat(tr.policy), policy0)
    assert not torch.equal(_flat(tr.value), value0)
    assert float(losses["sym_loss"]) == 0.0
    # The second iteration trains the policy.
    tr.iteration()
    assert not torch.equal(_flat(tr.policy), policy0)


def test_sym_coef_reports_a_sym_loss(env):
    with pytest.raises(ValueError, match="mirror_maps"):
        warp_ppo.Trainer(env, _ppo_params(), sym_coef=1.0)
    maps = symmetry.policy_mirror_maps(env.robot_spec, env.actor_obs_names)
    tr = _trainer(env, sym_coef=1.0, mirror_maps=maps)
    tr.iteration()
    stats = tr.read_stats()
    assert math.isfinite(stats["sym_loss"])
    assert stats["sym_loss"] > 0.0


def test_state_dict_round_trips(env, tmp_path):
    tr = _trainer(env, seed=3)
    tr.iteration()
    path = tmp_path / "trainer_state.pt"
    policy_io.save_trainer_state(path, tr.state_dict())
    state = policy_io.load_trainer_state(path, "cpu")

    other = warp_ppo.Trainer(env, _ppo_params(), seed=99)
    other.load_state_dict(state)
    assert (other.step, other.it) == (tr.step, tr.it)
    assert torch.equal(_flat(other.policy), _flat(tr.policy))
    assert torch.equal(_flat(other.value), _flat(tr.value))
    for a, b in ((other.norm_s, tr.norm_s), (other.norm_p, tr.norm_p)):
        assert a.count == b.count
        assert torch.equal(a.mean, b.mean) and torch.equal(a.svar, b.svar) and torch.equal(a.std, b.std)
    sa, sb = other.opt.state_dict(), tr.opt.state_dict()
    assert sa["param_groups"] == sb["param_groups"]
    for k in sb["state"]:
        for name in ("exp_avg", "exp_avg_sq", "step"):
            assert torch.equal(sa["state"][k][name], sb["state"][k][name])
    # The loop generator continues where it stopped.
    assert torch.equal(
        torch.rand(5, generator=other.gen),
        torch.rand(5, generator=tr.gen),
    )
