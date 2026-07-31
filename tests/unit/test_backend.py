"""Tests for the sim.backend flag: resolution and warp data budget kwargs.

This suite runs with JAX_PLATFORMS=cpu, so "auto" must resolve to jax here.
Warp needs CUDA, so its runtime behavior is validated on a GPU box instead.
Env-instantiating cases (backend actually wired into a running env) land
once the env classes exist.
"""

import jax
import pytest

from humanoid_lab.envs.backend import data_budget_kwargs, resolve_backend


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
