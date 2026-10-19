"""The Torch PPO math against brax's own functions, model-free.

Trainer._gae against brax ppo.losses.compute_gae, and log_prob and the
entropy base term against brax's NormalTanhDistribution. Inputs are random
float32. brax is time-major, the Torch trainer is [batch, T].
"""

from __future__ import annotations

import types

import jax.numpy as jnp
import numpy as np
import pytest
from brax.training import distribution
from brax.training.agents.ppo import losses
from tolerances import PPO_MATH

torch = pytest.importorskip("torch")

from humanoid_lab.torch_warp import ppo as warp_ppo

B, T, A = 6, 9, 5
GAMMA, LAM = 0.97, 0.95


def test_gae_matches_brax():
    rng = np.random.default_rng(0)
    v = rng.normal(size=(B, T)).astype(np.float32)
    v_boot = rng.normal(size=(B,)).astype(np.float32)
    r = rng.normal(size=(B, T)).astype(np.float32)
    done = (rng.uniform(size=(B, T)) < 0.25).astype(np.float32)
    trunc = done * (rng.uniform(size=(B, T)) < 0.5).astype(np.float32)
    assert done.any() and trunc.any() and (done * (1 - trunc)).any()

    trainer = types.SimpleNamespace(gamma=GAMMA, lam=LAM, T=T)
    vs, adv = warp_ppo.Trainer._gae(
        trainer, *(torch.tensor(x) for x in (v, v_boot, r, done, trunc))
    )
    want_vs, want_adv = losses.compute_gae(
        truncation=jnp.asarray(trunc.T),
        termination=jnp.asarray((done * (1 - trunc)).T),
        rewards=jnp.asarray(r.T),
        values=jnp.asarray(v.T),
        bootstrap_value=jnp.asarray(v_boot),
        lambda_=LAM,
        discount=GAMMA,
    )
    np.testing.assert_allclose(vs.numpy(), np.asarray(want_vs).T, **PPO_MATH)
    np.testing.assert_allclose(adv.numpy(), np.asarray(want_adv).T, **PPO_MATH)


def test_log_prob_and_entropy_match_brax():
    rng = np.random.default_rng(1)
    logits = rng.normal(size=(7, 2 * A)).astype(np.float32)
    raw = rng.normal(size=(7, A)).astype(np.float32)
    dist = distribution.NormalTanhDistribution(event_size=A)

    loc, scale = warp_ppo.dist_params(torch.tensor(logits))
    got = warp_ppo.log_prob(loc, scale, torch.tensor(raw))
    want = dist.log_prob(jnp.asarray(logits), jnp.asarray(raw))
    np.testing.assert_allclose(got.numpy(), np.asarray(want), **PPO_MATH)

    # entropy() is the normal entropy plus tanh_log_det of one sample, as
    # in brax's NormalTanhDistribution.entropy. A second generator at the
    # same seed replays the sample.
    ent = warp_ppo.entropy(loc, scale, torch.Generator().manual_seed(3))
    noise = torch.randn(loc.shape, generator=torch.Generator().manual_seed(3))
    base = ent - warp_ppo.tanh_log_det(loc + scale * noise).sum(-1)
    want_base = dist.create_dist(jnp.asarray(logits)).entropy().sum(-1)
    np.testing.assert_allclose(base.numpy(), np.asarray(want_base), **PPO_MATH)

    x = torch.tensor(raw)
    want_det = dist._postprocessor.forward_log_det_jacobian(jnp.asarray(raw))
    np.testing.assert_allclose(warp_ppo.tanh_log_det(x).numpy(), np.asarray(want_det), **PPO_MATH)
