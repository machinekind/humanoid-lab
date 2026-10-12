"""The Brax mirror-symmetry loss behind ppo.sym_coef.

train.sym_loss_patch swaps brax's ppo_losses.compute_ppo_loss for
train._with_sym_loss while ppo.train runs. These tests guard the brax lookup
the patch relies on and check the added loss on stub networks. Eager jax
only.
"""

import contextlib
import inspect
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
from brax.training.agents.ppo import losses as ppo_losses
from brax.training.agents.ppo import train as brax_ppo_train

from humanoid_lab import train

B, T, S, A = 2, 3, 4, 2

# Observation mirror: swap components 0 and 1, negate component 2.
OBS_PERM = np.array([1, 0, 2, 3])
OBS_SIGN = np.array([1.0, 1.0, -1.0, 1.0], dtype=np.float32)
# Action mirror: swap the two actions.
ACT_PERM = np.array([1, 0])
ACT_SIGN = np.array([1.0, 1.0], dtype=np.float32)


def _maps():
    return OBS_PERM, OBS_SIGN, ACT_PERM, ACT_SIGN


def _stub_loss(params, normalizer_params, data, rng, *, ppo_network, **kwargs):
    return jnp.float32(1.0), {"x": 0}


def _stub_network(w):
    return SimpleNamespace(
        parametric_action_distribution=SimpleNamespace(param_size=2 * A),
        policy_network=SimpleNamespace(apply=lambda n, p, obs: obs["state"] @ w),
    )


def _data(rng):
    return SimpleNamespace(
        observation={
            "state": jnp.asarray(rng.normal(size=(B, T, S)), dtype=jnp.float32),
            "privileged_state": jnp.asarray(rng.normal(size=(B, T, 7)), dtype=jnp.float32),
        }
    )


def _call(w, data, coef):
    return train._with_sym_loss(
        SimpleNamespace(policy=None),
        None,
        data,
        None,
        ppo_network=_stub_network(jnp.asarray(w)),
        sym_coef=coef,
        obs_key="state",
        obs_perm=jnp.asarray(OBS_PERM),
        obs_sign=jnp.asarray(OBS_SIGN),
        act_perm=jnp.asarray(ACT_PERM),
        act_sign=jnp.asarray(ACT_SIGN),
        compute_ppo_loss=_stub_loss,
    )


def test_brax_train_looks_the_loss_up_at_call_time():
    # The patch works only while ppo.train reads the module attribute when it
    # runs. A brax upgrade that binds the loss some other way breaks it.
    assert "ppo_losses.compute_ppo_loss" in inspect.getsource(brax_ppo_train.train)


def test_a_zero_sym_coef_patches_nothing():
    original = ppo_losses.compute_ppo_loss
    ctx = train.sym_loss_patch(0.0, *_maps(), "state")
    assert isinstance(ctx, contextlib.nullcontext)
    with ctx:
        assert ppo_losses.compute_ppo_loss is original
    assert ppo_losses.compute_ppo_loss is original


def test_a_nonzero_sym_coef_patches_only_inside_the_block():
    original = ppo_losses.compute_ppo_loss
    with train.sym_loss_patch(0.5, *_maps(), "state"):
        patched = ppo_losses.compute_ppo_loss
        assert patched is not original
        assert patched.keywords["compute_ppo_loss"] is original
        assert patched.keywords["sym_coef"] == 0.5
    assert ppo_losses.compute_ppo_loss is original


def test_the_wrapper_adds_the_mirror_loss():
    rng = np.random.default_rng(0)
    w = rng.normal(size=(S, 2 * A)).astype(np.float32)
    data = _data(rng)
    coef = 0.7
    loss, metrics = _call(w, data, coef)

    s = np.asarray(data.observation["state"])
    loc = (s @ w)[..., :A]
    loc_m = ((s[..., OBS_PERM] * OBS_SIGN) @ w)[..., :A]
    sym = np.mean(np.square(loc_m - loc[..., ACT_PERM] * ACT_SIGN))
    assert sym > 0
    np.testing.assert_allclose(float(loss), 1.0 + coef * sym, rtol=1e-6)
    np.testing.assert_allclose(float(metrics["sym_loss"]), sym, rtol=1e-6)
    assert metrics["x"] == 0


def test_an_equivariant_policy_has_zero_sym_loss():
    # loc(M s) == M loc(s) when W maps the mirrored inputs onto the mirrored
    # outputs: rows 0 and 1 swap with the action columns, row 2 is zero
    # because its sign flips while the actions do not.
    w = np.zeros((S, 2 * A), dtype=np.float32)
    w[0, :A] = [0.3, -0.8]
    w[1, :A] = [-0.8, 0.3]
    w[3, :A] = [0.5, 0.5]
    w[:, A:] = np.random.default_rng(1).normal(size=(S, A))  # the std half is ignored
    loss, metrics = _call(w, _data(np.random.default_rng(2)), 2.0)
    np.testing.assert_allclose(float(metrics["sym_loss"]), 0.0, atol=1e-6)
    np.testing.assert_allclose(float(loss), 1.0, atol=1e-5)
