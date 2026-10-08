"""Tests for the sim.backend flag: resolution and warp data budget kwargs.

This suite runs with JAX_PLATFORMS=cpu, so "auto" must resolve to jax here.
Warp needs CUDA, so its runtime behavior is validated on a GPU box instead.
Env-instantiating cases (backend actually wired into a running env) land
once the env classes exist.
"""

import jax
import pytest

from humanoid_lab.envs import backend
from humanoid_lab.envs.backend import data_budget_kwargs, make_data_fn, resolve_backend


def test_resolve_backend_passes_explicit_values_through():
    assert resolve_backend("jax") == "jax"
    assert resolve_backend("warp") == "warp"


def test_resolve_backend_auto_is_jax_on_a_cpu_host():
    assert jax.default_backend() == "cpu"
    assert resolve_backend("auto") == "jax"


def test_resolve_backend_rejects_unknown_values():
    with pytest.raises(ValueError):
        resolve_backend("cuda")


def test_budget_kwargs_empty_for_jax():
    assert data_budget_kwargs("jax", 32, 320, 4096) == {}


def test_budget_kwargs_scale_naconmax_only():
    kw = data_budget_kwargs("warp", 32, 320, 4096)
    assert kw == {"naconmax": 32 * 4096, "njmax": 320}


def test_no_ccd_budget_adds_no_kwarg():
    assert data_budget_kwargs("warp", 32, 320, 4096, None) == data_budget_kwargs(
        "warp", 32, 320, 4096
    )
    assert data_budget_kwargs("jax", 32, 320, 4096, 16) == {}


def test_the_ccd_budget_is_a_pool_like_naconmax():
    kw = data_budget_kwargs("warp", 32, 320, 4096, 16)
    assert kw == {"naconmax": 32 * 4096, "njmax": 320, "naccdmax": 16 * 4096}


def _recording_make_data(monkeypatch):
    calls = []

    def record(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(backend.mjx, "make_data", record)
    return calls


def test_make_data_fn_hands_the_ccd_budget_to_warp(monkeypatch):
    calls = _recording_make_data(monkeypatch)
    mj_model, mjx_model = object(), object()
    make_data_fn("warp", mj_model, mjx_model, 32, 320, 4096, 16)()
    make_data_fn("warp", mj_model, mjx_model, 32, 320, 4096)()
    make_data_fn("jax", mj_model, mjx_model, 32, 320, 4096, 16)()
    (args, kwargs), (_, default), (jax_args, jax_kwargs) = calls
    assert args == (mj_model,)
    assert kwargs == {"impl": "warp", "naconmax": 32 * 4096, "njmax": 320, "naccdmax": 16 * 4096}
    assert "naccdmax" not in default
    assert jax_args == (mjx_model,) and jax_kwargs == {}
