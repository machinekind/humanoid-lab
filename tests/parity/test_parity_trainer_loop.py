"""The Torch PPO loop (torch_warp.ppo.Trainer) on the CPU warp device.

roboto_origin deploy_pd, N=4, unroll 5, batch 4, 2 minibatches, so one
iteration is 2 unrolls of 5 steps. The networks are small and start from
brax's init through policy_io.init_from_brax.
"""

from __future__ import annotations

import functools
import types

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


def test_each_iteration_counts_its_steps(env):
    tr = _trainer(env)
    assert tr.n_unrolls == 2
    assert tr.steps_per_iter == N * 5 * 2
    for it in (1, 2):
        tr.iteration()
        assert tr.it == it
        assert tr.step == it * tr.steps_per_iter
        # The normalizers saw every collected step.
        assert tr.norm_s.count == tr.norm_p.count == it * tr.steps_per_iter


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


def _brax_sym_loss(tr, obs_s, obs_p, maps):
    """train._with_sym_loss on the trainer's networks, normalizers and observations."""
    import jax.numpy as jp
    from brax.training import types as brax_types
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import networks as ppo_networks

    from humanoid_lab import train

    kw = _ppo_params().network_factory.to_dict()
    net = ppo_networks.make_ppo_networks(
        {"state": (tr.S,), "privileged_state": (tr.P,)},
        tr.A,
        preprocess_observations_fn=running_statistics.normalize,
        **kw,
    )

    def arr(t):
        return jp.asarray(t.detach().cpu().numpy())

    norm = running_statistics.RunningStatisticsState(
        count=brax_types.UInt64(hi=0, lo=tr.norm_s.count),
        mean={"state": arr(tr.norm_s.mean), "privileged_state": arr(tr.norm_p.mean)},
        summed_variance={"state": arr(tr.norm_s.svar), "privileged_state": arr(tr.norm_p.svar)},
        std={"state": arr(tr.norm_s.std), "privileged_state": arr(tr.norm_p.std)},
    )
    data = types.SimpleNamespace(observation={"state": arr(obs_s), "privileged_state": arr(obs_p)})
    obs_perm, obs_sign, act_perm, act_sign = (jp.asarray(m) for m in maps)
    _loss, metrics = train._with_sym_loss(
        types.SimpleNamespace(policy=policy_io.to_brax(tr.policy)),
        norm,
        data,
        None,
        ppo_network=net,
        sym_coef=1.0,
        obs_key=kw.get("policy_obs_key", "state"),
        obs_perm=obs_perm,
        obs_sign=obs_sign,
        act_perm=act_perm,
        act_sign=act_sign,
        compute_ppo_loss=lambda *a, **k: (0.0, {}),
    )
    return float(metrics["sym_loss"])


def _record_update(update, batches, name, x):
    batches[name] = x.clone()
    update(x)


def test_sym_loss_matches_brax(env):
    with pytest.raises(ValueError, match="mirror_maps"):
        warp_ppo.Trainer(env, _ppo_params(), sym_coef=1.0)
    maps = symmetry.policy_mirror_maps(env.robot_spec, env.actor_obs_names)
    # A zero learning rate keeps the policy fixed through the update. Every
    # minibatch then reads the same networks, and the reported loss is the
    # mean over the whole batch.
    tr = _trainer(env, _ppo_params(learning_rate=0.0), sym_coef=1.0, mirror_maps=maps)
    batches = {}
    for name in ("norm_s", "norm_p"):
        norm = getattr(tr, name)
        norm.update = functools.partial(_record_update, norm.update, batches, name)
    losses = tr.iteration()
    want = _brax_sym_loss(tr, batches["norm_s"], batches["norm_p"], maps)
    assert want > 0
    assert float(losses["sym_loss"]) == pytest.approx(want, rel=1e-5)


def test_state_dict_round_trips(env, tmp_path):
    tr = _trainer(env, seed=3)
    tr.iteration()
    path = tmp_path / "trainer_state.pt"
    policy_io.save_trainer_state(path, tr.state_dict())
    # A second save replaces the file and leaves no temporary file behind.
    policy_io.save_trainer_state(path, tr.state_dict())
    assert sorted(p.name for p in tmp_path.iterdir()) == ["trainer_state.pt"]
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
