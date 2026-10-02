"""Robot-agnostic MJX backend plumbing.

Holds the sim.backend resolution and the warp data-budget helpers only. The
robot-specific env base class lives in envs/base.py alongside the RobotSpec
loader.
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
    backend: str,
    naconmax_per_env: int,
    njmax: int,
    num_envs: int,
    naccdmax_per_env: int | None = None,
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

    naccdmax sizes the CCD scratch that MJWarp allocates on every collision
    call of a model with convex pairs (see sim_budget.ccd_slot_bytes). It is
    one pool for the whole batch too, naccdmax_per_env multiplied by
    num_envs. MJWarp refuses a naccdmax above naconmax. None leaves the
    kwarg out, and MJWarp then sizes the scratch to the naconmax pool.

    If a buffer is too small, warp drops the overflow instead of raising an
    error. MJWarp prints a message from the device to file descriptor 1,
    which Python's sys.stdout never sees (fd_capture.py captures it). The
    measured numbers behind the defaults are in envs/joystick.py's `sim`
    block; `./run.sh check-contacts` re-measures them, and `./run.sh
    check-terrain` gates a terrain recipe's budgets.
    """
    if backend != "warp":
        return {}
    kwargs = {
        "naconmax": int(naconmax_per_env) * int(num_envs),
        "njmax": int(njmax),
    }
    if naccdmax_per_env is not None:
        kwargs["naccdmax"] = int(naccdmax_per_env) * int(num_envs)
    return kwargs


def make_data_fn(
    backend, mj_model, mjx_model, naconmax_per_env, njmax, num_envs, naccdmax_per_env=None
):
    """Return a zero-argument callable that builds a fresh mjx.Data on the backend.

    The warp branch applies the buffer budgets from data_budget_kwargs. The
    jax branch stays byte-for-byte the call the envs made before the backend
    flag existed.
    """
    if backend == "warp":
        kwargs = data_budget_kwargs(
            "warp", naconmax_per_env, njmax, num_envs, naccdmax_per_env
        )
        return lambda: mjx.make_data(mj_model, impl="warp", **kwargs)
    return lambda: mjx.make_data(mjx_model)
