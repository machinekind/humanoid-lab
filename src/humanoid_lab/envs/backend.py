"""Robot-agnostic MJX backend plumbing.

Ported from wojtek_rl/base.py: only the sim.backend resolution and warp
data-budget helpers. The env base class (w01-tekEnv there) is robot-specific
plumbing and lands in a later step alongside the RobotSpec loader.
"""

import jax
from mujoco import mjx


def resolve_backend(backend: str) -> str:
    """Resolve a sim.backend value to "jax" or "warp".

    "auto" picks warp when jax runs on a GPU and the vendored MJWarp
    imports, and jax otherwise. Explicit values pass through. "warp" on a
    host without CUDA fails later in put_model, and that failure should
    stay loud.
    """
    if backend in ("jax", "warp"):
        return backend
    if backend != "auto":
        raise ValueError(f"sim.backend must be jax, warp or auto, got {backend!r}")
    try:
        from mujoco.mjx import warp as mjxw

        warp_ok = bool(mjxw.WARP_INSTALLED)
    except Exception:
        warp_ok = False
    return "warp" if warp_ok and jax.default_backend() == "gpu" else "jax"


def data_budget_kwargs(
    backend: str, naconmax_per_env: int, njmax: int, num_envs: int
) -> dict:
    """make_data buffer kwargs for the resolved backend.

    Warp reserves fixed buffer space for contacts and constraints before the
    simulation runs. The jax backend sizes its own buffers on the fly, so it
    takes no kwargs.

    naconmax sizes one shared contact pool for the whole batch of envs. It
    is naconmax_per_env multiplied by num_envs.

    njmax sizes the constraint rows for a single world. Every env in the
    batch gets its own njmax rows, so this number never multiplies by
    num_envs.

    If a buffer is too small, warp drops the overflow silently instead of
    raising an error. The measured numbers behind the defaults are in
    docs/plans/mjwarp-phase0-report.md §4.
    """
    if backend != "warp":
        return {}
    return {
        "naconmax": int(naconmax_per_env) * int(num_envs),
        "njmax": int(njmax),
    }


def make_data_fn(backend, mj_model, mjx_model, naconmax_per_env, njmax, num_envs):
    """Return a zero-argument callable that builds a fresh mjx.Data on the backend.

    The warp branch applies the buffer budgets from data_budget_kwargs. The
    jax branch stays byte-for-byte the call the envs made before the backend
    flag existed.
    """
    if backend == "warp":
        kwargs = data_budget_kwargs("warp", naconmax_per_env, njmax, num_envs)
        return lambda: mjx.make_data(mj_model, impl="warp", **kwargs)
    return lambda: mjx.make_data(mjx_model)
