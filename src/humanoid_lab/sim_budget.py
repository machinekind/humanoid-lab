"""Warp contact and constraint budgets: accounting, measurement, reporting.

Two fixed-size buffers decide whether a warp run is simulating what its
config says it is.

`naconmax_per_env` sizes ONE shared contact pool for the whole batch:
`mjx.make_data(..., naconmax=naconmax_per_env * num_envs)` allocates it up
front, so the product is a real device-memory line item -- w01-tek's 256-per-
env pool is what ran a 4096-env job out of device memory. Contacts past the
pool are dropped SILENTLY.

`njmax` sizes the constraint rows of a single world and never multiplies by
the env count. Rows past it apply no force, with no warning anywhere: no
counter reports it, no exception is raised, the policy just trains against a
robot whose feet half-pass through the floor. That is why the peaks below get
recorded even when nothing is wrong.

The jax backend sizes its own buffers on the fly and has neither budget, so
it can never overflow. It also has no live counters: its `_impl.ncon` and
`_impl.nefc` are the buffer sizes fixed at make_data time (asimov_v1:
ncon 499, nefc 2025 = 2 friction + 27 limit + 499*4 contact rows), identical
whatever the robot is doing. What IS measurable there is the number of
contacts actually penetrating, `active_contacts` below.

Ported from w01-tek's wojtek_rl/terrain_scan.py (`nacon_of`/`nefc_of` at
:520-535 and the `result["contacts"]` block at :940-965). One deliberate
divergence: w01-tek's `nacon` counter is batch-wide, so it compares against
the whole pool. Every peak here is PER WORLD, so `overflow` compares against
`naconmax_per_env` and `rows_overflow` against `njmax`. `pool` is reported
alongside as the memory number, not as a threshold.
"""

from __future__ import annotations

import numpy as np

# mujoco.mjtCone values, inlined so the pure helpers here import nothing.
CONE_PYRAMIDAL = 0
CONE_ELLIPTIC = 1


def rows_per_contact(cone: int, dim: int) -> int:
    """Constraint rows one contact of condim `dim` costs under `cone`.

    A pyramidal cone linearizes the friction cone into 2*(dim-1) rows -- 4 at
    condim 3, 6 at condim 4 (w01-tek's own example), 10 at condim 6 -- and a
    frictionless contact (dim 1) still costs its one normal row. An elliptic
    cone costs dim rows flat.
    """
    if cone == CONE_PYRAMIDAL:
        return max(1, 2 * (int(dim) - 1))
    if cone == CONE_ELLIPTIC:
        return int(dim)
    raise ValueError(f"unknown mjtCone value {cone!r}: expected 0 (pyramidal) or 1 (elliptic)")


def recommend_budget(peak: int, headroom: float, step: int) -> int:
    """A budget that clears `peak` by `headroom`, rounded up to `step`.

    w01-tek sized its own pool at about 7x its measured peak (12 contacts ->
    88). Rounding is upward so the returned number never sits under the
    headroom it claims, and a zero peak still returns one full step: a
    zero-length buffer would drop every contact there is.
    """
    want = max(1, int(np.ceil(float(peak) * float(headroom))))
    return int(np.ceil(want / step) * step)


def active_contacts(dist) -> int:
    """Contacts actually penetrating in `dist`, an mjx contact distance array.

    mjx keeps the contact array at a fixed length and fills the unused slots
    with candidate pairs that are apart, so `len(dist)` is the buffer size and
    means nothing. Negative distance is the contact.
    """
    return int((np.asarray(dist) < 0).sum())


def live_peaks(data) -> tuple[int | None, int | None]:
    """(nacon, nefc) read off a warp `mjx.Data`, or (None, None) elsewhere.

    Warp's `_impl.nacon` is the live count of the shared pool and its
    `_impl.nefc` is one row count per world, so the peak is the max over
    worlds. The jax impl carries an `nefc` too, but as a 0-d buffer size that
    never moves -- reporting it as a peak would make every jax run look like
    it had overflowed, so a 0-d value reads as "not measured" (None), never as
    a number. w01-tek's `nefc_of` makes the same ndim check and returns 0; None
    is used here so "no measurement" and "measured zero" stay distinguishable
    in the json.
    """
    impl = getattr(data, "_impl", None)
    if impl is None:
        return None, None

    def peak(value):
        if value is None or getattr(value, "ndim", 0) == 0:
            return None
        return int(np.asarray(value).max())

    nacon = getattr(impl, "nacon", None)
    # nacon is a single pool-wide scalar on warp, so it is read directly
    # rather than through peak()'s per-world max.
    nacon = None if nacon is None else int(np.asarray(nacon).max())
    return nacon, peak(getattr(impl, "nefc", None))


def budget_report(
    backend: str,
    nacon_max,
    nefc_max,
    naconmax_per_env: int,
    njmax: int,
    num_envs: int,
) -> dict:
    """The `contacts` block run.json and battery.json carry.

    The same keys on both backends, so a GPU run and a CPU run diff cleanly
    and no reader has to branch on `backend`; the peaks are None where nothing
    measured them. Every value is a plain Python type: these go through
    json.dumps, and a numpy scalar would either raise or be stringified by a
    `default=str`.

    Both overflow flags are backend-gated. The jax backend has no fixed
    buffers, so a peak past a warp budget is not an overflow there -- it means
    the warp run of the same config WOULD have dropped contacts, which is
    check_contacts' job to say, not this block's.
    """
    nacon = None if nacon_max is None else int(nacon_max)
    nefc = None if nefc_max is None else int(nefc_max)
    is_warp = backend == "warp"
    return {
        "backend": backend,
        "nacon_max": nacon,
        "naconmax_per_env": int(naconmax_per_env),
        "num_envs": int(num_envs),
        # What make_data allocates for the whole batch. Reported for the
        # device-memory arithmetic, never compared against a per-world peak.
        "pool": int(naconmax_per_env) * int(num_envs),
        # >= not >: at the budget the buffer is full and the next contact is
        # already gone, silently.
        "overflow": bool(is_warp and nacon is not None and nacon >= naconmax_per_env),
        "nefc_max": nefc,
        "njmax": int(njmax),
        "rows_overflow": bool(is_warp and nefc is not None and nefc >= njmax),
    }
