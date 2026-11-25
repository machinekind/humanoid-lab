"""Torch policy_io against the Brax checkpoint format.

A checkpoint goes Brax -> Torch -> Brax and every step is read back through
the loaders the repo already uses: humanoid_lab.policy_io.load_policy and
the exporter's flatten_params plus runtime.forward. No env is built. The
observation and action sizes come from a namespace.
"""

from __future__ import annotations

import functools
import json
from types import SimpleNamespace

import jax
import jax.numpy as jp
import numpy as np
import pytest
from brax.training.acme import running_statistics, specs
from brax.training.agents.ppo import checkpoint as ppo_checkpoint
from brax.training.agents.ppo import networks as ppo_networks
from ml_collections import config_dict

from humanoid_lab import policy_io
from humanoid_lab.export import policy as export_policy
from humanoid_lab.export import runtime

torch = pytest.importorskip("torch")

from humanoid_lab.torch_warp import policy_io as tpio

S, P, A = 24, 31, 5
NET = {
    "policy_hidden_layer_sizes": [16, 8],
    "value_hidden_layer_sizes": [16, 8],
    "policy_obs_key": "state",
    "value_obs_key": "privileged_state",
}
# Above 2**32, so the hi word of brax's UInt64 count is used.
BIG_COUNT = 5_000_000_123


def _obs_size(s=S, p=P):
    return {"state": (s,), "privileged_state": (p,)}


def _inference(step_dir, s=S, p=P):
    env = SimpleNamespace(observation_size=_obs_size(s, p), action_size=A)
    ppo = config_dict.ConfigDict({"network_factory": NET, "normalize_observations": True})
    return policy_io.load_policy(str(step_dir), env, ppo)


def _brax_actions(step_dir, obs, s=S, p=P):
    inf = _inference(step_dir, s, p)
    priv = np.zeros((obs.shape[0], p), np.float32)
    act, _ = jax.vmap(lambda o, q: inf({"state": o, "privileged_state": q}, jax.random.PRNGKey(0)))(obs, priv)
    return np.asarray(act)


def _torch_actions(policy, norm_s, obs):
    with torch.no_grad():
        out = policy(norm_s(torch.as_tensor(obs)))
    return torch.tanh(out[:, :A]).numpy()


def _nets(s=S, p=P):
    pol = tpio.mlp([s, *NET["policy_hidden_layer_sizes"], 2 * A])
    val = tpio.mlp([p, *NET["value_hidden_layer_sizes"], 1])
    return pol, val


def _obs(n=16, width=S, seed=1):
    return np.random.default_rng(seed).uniform(-2, 2, (n, width)).astype(np.float32)


@pytest.fixture
def brax_ckpt(tmp_path):
    """A brax-written checkpoint with real statistics and a count above 2**32."""
    obs_size = _obs_size()
    factory = functools.partial(ppo_networks.make_ppo_networks, **NET)
    net = factory(obs_size, A, preprocess_observations_fn=running_statistics.normalize)
    kp, kv, ks = jax.random.split(jax.random.PRNGKey(0), 3)
    pp, vp = net.policy_network.init(kp), net.value_network.init(kv)
    norm = running_statistics.init_state({k: specs.Array(v, jp.dtype("float32")) for k, v in obs_size.items()})
    batch = {k: jax.random.uniform(ks, (64, *v), minval=-3, maxval=3) for k, v in obs_size.items()}
    norm = running_statistics.update(norm, batch)
    # Scale the statistics up to a large count. std stays the same.
    scale = np.float32(BIG_COUNT / 64)
    norm = norm.replace(
        count=type(norm.count)(hi=np.uint32(BIG_COUNT >> 32), lo=np.uint32(BIG_COUNT & 0xFFFFFFFF)),
        summed_variance={k: np.asarray(v) * scale for k, v in norm.summed_variance.items()},
    )
    std = {k: np.sqrt(np.asarray(v) / np.float32(BIG_COUNT)) for k, v in norm.summed_variance.items()}
    norm = norm.replace(std={k: np.clip(v, 1e-6, 1e6).astype(np.float32) for k, v in std.items()})
    root = tmp_path / "src"
    config = ppo_checkpoint.network_config(
        observation_size=obs_size, action_size=A, normalize_observations=True, network_factory=factory
    )
    ppo_checkpoint.save(root, 3, (norm, pp, vp), config)
    return root / f"{3:012d}"


def _from_ckpt(step_dir, grow_state=0):
    pol, val = _nets(S + grow_state)
    norm_s, norm_p = tpio.from_brax(policy_io.load_params(str(step_dir)), pol, val, grow_state=grow_state)
    return pol, val, norm_s, norm_p


def test_brax_to_torch_matches_load_policy(brax_ckpt):
    pol, _, norm_s, norm_p = _from_ckpt(brax_ckpt)
    assert norm_s.count == BIG_COUNT and norm_p.count == BIG_COUNT
    obs = _obs()
    diff = np.abs(_brax_actions(brax_ckpt, obs) - _torch_actions(pol, norm_s, obs)).max()
    assert diff < 1e-6, diff


def test_torch_to_brax_round_trips_through_load_policy_and_export(brax_ckpt, tmp_path):
    pol, val, norm_s, norm_p = _from_ckpt(brax_ckpt)
    gen = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for prm in [*pol.parameters(), *val.parameters()]:
            prm.add_(0.05 * torch.randn(prm.shape, generator=gen))
    rng = np.random.default_rng(2)
    norm_s.update(torch.as_tensor(rng.uniform(-1, 4, (100, S)).astype(np.float32)))
    norm_p.update(torch.as_tensor(rng.uniform(-1, 4, (100, P)).astype(np.float32)))

    step_dir = tpio.save_brax(tmp_path / "out", 7, norm_s, norm_p, pol, val, NET, A)
    assert step_dir.name == f"{7:012d}"
    obs = _obs()
    want = _torch_actions(pol, norm_s, obs)
    assert np.abs(_brax_actions(step_dir, obs) - want).max() < 1e-6

    params = policy_io.load_params(str(step_dir))
    weights = export_policy.flatten_params(params, "state")
    got = np.stack([runtime.forward(weights, o, A) for o in obs])
    assert np.abs(got - want).max() < 1e-5

    count = int(params[0].count.hi) * 2**32 + int(params[0].count.lo)
    assert count == BIG_COUNT + 100 == norm_s.count

    saved = json.loads((step_dir / tpio.NETWORK_CONFIG).read_text(encoding="utf-8"))
    assert {k: list(v) for k, v in saved["observation_size"].items()} == {"state": [S], "privileged_state": [P]}


def test_trailing_growth_is_accepted_and_keeps_actions(brax_ckpt):
    sizes = {"gyro": 3, "joint_pos": 21, "heading_err": 1}
    k = tpio.check_layout(["gyro", "joint_pos"], ["gyro", "joint_pos", "heading_err"], sizes, "state")
    assert k == 1

    pol0, _, norm_s0, _ = _from_ckpt(brax_ckpt)
    pol, _, norm_s, norm_p = _from_ckpt(brax_ckpt, grow_state=k)
    assert norm_s.size == S + k and norm_p.size == P
    np.testing.assert_allclose(norm_s.std[S:].numpy(), 0.2, rtol=1e-5)
    np.testing.assert_array_equal(norm_s.std[:S].numpy(), norm_s0.std.numpy())

    obs = _obs()
    extra = np.random.default_rng(5).uniform(-3, 3, (obs.shape[0], k)).astype(np.float32)
    grown = _torch_actions(pol, norm_s, np.concatenate([obs, extra], axis=-1))
    assert np.abs(grown - _torch_actions(pol0, norm_s0, obs)).max() < 1e-6
    assert np.abs(grown - _brax_actions(brax_ckpt, obs)).max() < 1e-6


def test_mid_list_growth_is_refused():
    sizes = {"gyro": 3, "joint_pos": 21, "heading_err": 1, "command": 3}
    with pytest.raises(ValueError, match="heading_err") as err:
        old, new = ["gyro", "joint_pos", "command"], ["gyro", "heading_err", "joint_pos", "command"]
        tpio.check_layout(old, new, sizes, "state")
    assert "state" in str(err.value)
    for new in (["gyro", "command"], ["joint_pos", "gyro", "command"], ["gyro", "joint_pos", "command", "gyro"]):
        with pytest.raises(ValueError):
            tpio.check_layout(["gyro", "joint_pos", "command"], new, sizes, "state")


def test_the_normalizer_has_std_one_at_count_zero():
    # brax's init_state converts to count 0 and std 1.
    for norm in tpio.from_brax(tpio.init_from_brax(3, S, P, A, NET), *_nets()):
        assert norm.count == 0
        assert torch.equal(norm.std, torch.ones_like(norm.std))

    n = tpio.Normalizer.zeros(4)
    assert n.count == 0
    assert torch.equal(n.std, torch.ones(4))
    n.grow(2)
    assert torch.equal(n.std, torch.ones(6))
    x = torch.arange(12.0).reshape(2, 6)
    assert torch.equal(n(x), x)

    # Two Welford updates match brax's.
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(5, 6)).astype(np.float32), rng.normal(size=(7, 6)).astype(np.float32)
    n.update(torch.as_tensor(a))
    n.update(torch.as_tensor(b))
    rs = running_statistics.init_state(specs.Array((6,), jp.dtype("float32")))
    rs = running_statistics.update(running_statistics.update(rs, jp.asarray(a)), jp.asarray(b))
    np.testing.assert_allclose(n.mean.numpy(), np.asarray(rs.mean), rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(n.std.numpy(), np.asarray(rs.std), rtol=1e-6)
    assert n.count == 12


def test_network_kwargs_must_match_the_checkpoint(brax_ckpt):
    tpio.check_network_kwargs(brax_ckpt, NET)
    tpio.check_network_kwargs(brax_ckpt, {**NET, "policy_hidden_layer_sizes": (16, 8)})
    with pytest.raises(ValueError) as err:
        tpio.check_network_kwargs(
            brax_ckpt, {**NET, "policy_hidden_layer_sizes": [32, 8], "value_obs_key": "state"}
        )
    assert "policy_hidden_layer_sizes" in str(err.value)
    assert "value_obs_key" in str(err.value)

    path = brax_ckpt / tpio.NETWORK_CONFIG
    saved = json.loads(path.read_text(encoding="utf-8"))
    saved["network_factory_kwargs"]["activation"] = "relu"
    path.write_text(json.dumps(saved), encoding="utf-8")
    with pytest.raises(ValueError, match="activation"):
        tpio.check_network_kwargs(brax_ckpt, NET)


def test_save_brax_refuses_an_existing_step(brax_ckpt):
    pol, val, norm_s, norm_p = _from_ckpt(brax_ckpt)
    with pytest.raises(FileExistsError):
        tpio.save_brax(brax_ckpt.parent, 3, norm_s, norm_p, pol, val, NET, A)
