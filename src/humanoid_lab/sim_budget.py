"""Warp contact and constraint budgets: accounting, measurement, reporting.

Two fixed-size buffers decide whether a warp run is simulating what its
config says it is.

`naconmax_per_env` sizes ONE shared contact pool for the whole batch:
`mjx.make_data(..., naconmax=naconmax_per_env * num_envs)` allocates it up
front, so the product is a real device-memory line item. A 256-per-env pool
has run a 4096-env job out of device memory. Contacts past the pool are
dropped SILENTLY.

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

Every peak reported here is PER WORLD, so `overflow` compares against
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
    condim 3, 6 at condim 4, 10 at condim 6 -- and a
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

    The inherited sizing rule is about 7x the measured peak (12 contacts ->
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
    a number. None rather than 0, so "no measurement" and "measured zero"
    stay distinguishable in the json.
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


def contact_dist(data):
    """The contact distance array off an mjx `Data`, or None if it has none."""
    contact = getattr(getattr(data, "_impl", None), "contact", None)
    return None if contact is None else contact.dist


def observed_peaks(data) -> tuple[int | None, int | None]:
    """(contacts, rows) for one step, as a caller in a step loop should record.

    Warp's counters win where they exist. On jax the contact count is still a
    real measurement -- the penetrating entries of the padded contact array --
    while the row count is not measurable at all, so it stays None. Deriving a
    row count from the contact count is check_contacts' job, where the
    derivation can be labelled as one; a run.json field must not carry a
    number whose meaning changes with the backend that wrote it.
    """
    nacon, nefc = live_peaks(data)
    if nacon is None:
        dist = contact_dist(data)
        nacon = None if dist is None else active_contacts(dist)
    return nacon, nefc


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

    The budgets themselves may be None on the jax backend (a robot with no
    recorded sim_budget); warp cannot construct without them (envs/base.py
    refuses), so a warp report always carries real numbers.
    """
    nacon = None if nacon_max is None else int(nacon_max)
    nefc = None if nefc_max is None else int(nefc_max)
    nacon_budget = None if naconmax_per_env is None else int(naconmax_per_env)
    njmax_budget = None if njmax is None else int(njmax)
    is_warp = backend == "warp"
    return {
        "backend": backend,
        "nacon_max": nacon,
        "naconmax_per_env": nacon_budget,
        "num_envs": int(num_envs),
        # What make_data allocates for the whole batch. Reported for the
        # device-memory arithmetic, never compared against a per-world peak.
        "pool": None if nacon_budget is None else nacon_budget * int(num_envs),
        # >= not >: at the budget the buffer is full and the next contact is
        # already gone, silently.
        "overflow": bool(is_warp and nacon is not None and nacon >= nacon_budget),
        "nefc_max": nefc,
        "njmax": njmax_budget,
        "rows_overflow": bool(is_warp and nefc is not None and nefc >= njmax_budget),
    }


def budget_report_for_env(env, nacon_max, nefc_max) -> dict:
    """`budget_report` with the budgets the env actually resolved (sim
    config value if set, else the robot's own robot.yaml sim_budget).

    The single adapter train.py and eval/battery.py both call, so the two
    files cannot disagree about what a run was configured with.
    """
    return budget_report(
        env._backend, nacon_max, nefc_max,
        env._naconmax_per_env, env._njmax, env._config.sim.num_envs,
    )


# -- traced telemetry: the counters inside a jitted training step ----------

# Fraction of a budget at which the telemetry warns.
TELEMETRY_WARN_FRAC = 0.9
# The three counters, in the order traced_counters returns them.
TELEMETRY_COUNTERS = ("nefc", "nacon", "ncollision")


def traced_counters(data):
    """(nefc, nacon, ncollision) of a batched warp `mjx.Data`, as traced
    int32 scalars.

    `nefc` is one row count per world, and the max over worlds is
    returned. `nacon` and `ncollision` count the batch's one shared pool.
    `ncollision` counts broadphase candidate pairs, which share the
    naconmax buffer with the contacts. `jp.max` reads a pool scalar and a
    copy broadcast across worlds alike. jax data has no live counters (its
    impl has no `nacon`), and every counter reads 0 there.

    jax.numpy is imported here, so the module's host helpers stay free of
    it."""
    import jax.numpy as jp

    impl = getattr(data, "_impl", None)
    if impl is None or not hasattr(impl, "nacon"):
        zero = jp.zeros((), jp.int32)
        return zero, zero, zero
    return tuple(jp.max(jp.asarray(getattr(impl, k))).astype(jp.int32) for k in TELEMETRY_COUNTERS)


def telemetry_caps(env) -> tuple[int, int, int]:
    """The budgets of the three TELEMETRY_COUNTERS for `env`: its resolved
    njmax for rows, and its naconmax pool for nacon and ncollision. The
    pool is `budget_report`'s: naconmax_per_env x num_envs."""
    pool = int(env._naconmax_per_env) * int(env._config.sim.num_envs)
    return int(env._njmax), pool, pool


def telemetry_step(prev_peaks, counters, caps):
    """(peaks, fracs, fire) after one step, traced.

    `prev_peaks` and `counters` hold the three TELEMETRY_COUNTERS. `caps`
    are their budgets as static ints: the resolved njmax for rows, and the
    naconmax pool (naconmax_per_env x num_envs) for both pool counters.
    Peaks are running maxima, and `fracs` are the peaks over the caps.
    `fire` is True for a counter whose peak reached TELEMETRY_WARN_FRAC of
    its cap on this step and not before. Peaks never fall, so each counter
    fires at most once until a reset zeroes the peaks."""
    import jax.numpy as jp

    prev = jp.asarray(prev_peaks, jp.int32)
    peaks = jp.maximum(prev, jp.asarray(counters, jp.int32))
    cap = jp.asarray(caps, jp.float32)
    threshold = TELEMETRY_WARN_FRAC * cap
    fire = (peaks >= threshold) & (threshold > prev)
    return peaks, peaks / cap, fire


def telemetry_warning(peaks, caps, fire) -> str:
    """The warning lines for the counters `fire` flags, host side."""
    budget = {"nefc": "njmax", "nacon": "the naconmax pool", "ncollision": "the naconmax pool"}
    advice = {
        "nefc": "Constraint rows past njmax apply no force. Raise task.env.sim.njmax.",
        "nacon": "Contacts past the pool are dropped. Raise task.env.sim.naconmax_per_env.",
        "ncollision": (
            "Broadphase candidates share the contact pool, and candidates past it are "
            "dropped. Raise task.env.sim.naconmax_per_env."
        ),
    }
    lines = []
    for name, peak, cap, hit in zip(TELEMETRY_COUNTERS, np.asarray(peaks), caps, np.asarray(fire)):
        if hit:
            lines.append(
                f"WARNING: warp {name} peaked at {int(peak)} of {budget[name]} {int(cap)}, "
                f"over {TELEMETRY_WARN_FRAC:.0%}. {advice[name]}"
            )
    return "\n".join(lines)


# MJWarp's EPA scratch bounds (mujoco_warp/_src/types.py): faces EPA may add
# per iteration, and the longest horizon it tracks.
_EPA_FACES_PER_ITER = 5
_EPA_HORIZON = 24
# Scratch element sizes in MJWarp, which stores float32: a vec3, and an int
# or a float.
_VEC3_BYTES = 12
_SCALAR_BYTES = 4
# EPA's iteration count when every convex pair of a model is box-box.
_BOX_BOX_EPA_ITERATIONS = 16


def ccd_slot_bytes(ccd_iterations: int, box_box: bool, all_convex_box_box: bool = False) -> int:
    """Bytes of CCD scratch per `naccdmax` slot, as MJWarp allocates them.

    MJWarp allocates this scratch on every collision call of a model with
    convex pairs, sized by `naccdmax` (a pool for the whole batch), outside
    the XLA memory pool. A model without convex pairs skips it. A flat floor
    pairs with primitive colliders only. Heightfield and box-box pairs are
    convex.

    EPA scratch comes first. EPA runs E = `ccd_iterations` iterations, or
    16 when every convex pair of the model is box-box. Per slot it holds
    10 + 2E vertices (a vec3 and an int each), 6 + 5E faces (an int, a vec3
    and a float each) and a 24-entry horizon of ints. Box-box pairs add
    484 B of multi-contact scratch: 4-sided clipping polygons and 3
    candidate normals. That holds with multi-contact enabled, MuJoCo's
    default, and no mesh pairs. At 35 iterations a slot is 4,996 B, and
    5,480 B with box-box pairs.
    """
    e = _BOX_BOX_EPA_ITERATIONS if all_convex_box_box else int(ccd_iterations)
    vertices = (10 + 2 * e) * (_VEC3_BYTES + _SCALAR_BYTES)
    faces = (6 + _EPA_FACES_PER_ITER * e) * (_SCALAR_BYTES + _VEC3_BYTES + _SCALAR_BYTES)
    horizon = _EPA_HORIZON * _SCALAR_BYTES
    total = vertices + faces + horizon
    if box_box:
        polygon, degree = 4, 3
        # Per polygon side: two vec3 in each of the polygon and its clipped
        # copy, one vec3 in each of the two contact faces, and the side's
        # plane normal (vec3) and distance (float).
        total += polygon * (4 * _VEC3_BYTES + 2 * _VEC3_BYTES + _VEC3_BYTES + _SCALAR_BYTES)
        # Per candidate: two normal indices (ints), two normals and an edge
        # vertex (vec3 each).
        total += degree * (2 * _SCALAR_BYTES + 3 * _VEC3_BYTES)
    return total
