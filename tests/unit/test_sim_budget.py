"""Tests for sim_budget.py: the warp contact/constraint budget accounting.

Everything here is pure dict and array arithmetic. The live counters
(`data._impl.nacon`, `data._impl.nefc`, `data._impl.ncollision`) exist only
on the warp backend. Stepping warp needs CUDA, so a stub object stands in for
`data._impl` and supplies the counter values. The stubs' field names are
checked against the real warp Data class, which imports on CPU.
"""

from __future__ import annotations

import dataclasses
import types

import numpy as np
import pytest

from humanoid_lab import sim_budget

# The two schema keys a reader has to be able to find in run.json/battery.json
# whatever backend produced them.
SCHEMA_KEYS = {
    "backend",
    "nacon_max",
    "naconmax_per_env",
    "num_envs",
    "pool",
    "overflow",
    "nefc_max",
    "njmax",
    "rows_overflow",
}


# -- rows_per_contact -------------------------------------------------------


@pytest.mark.parametrize(
    "cone, dim, rows",
    [
        (sim_budget.CONE_PYRAMIDAL, 1, 1),
        (sim_budget.CONE_PYRAMIDAL, 3, 4),
        (sim_budget.CONE_PYRAMIDAL, 4, 6),
        (sim_budget.CONE_PYRAMIDAL, 6, 10),
        (sim_budget.CONE_ELLIPTIC, 3, 3),
        (sim_budget.CONE_ELLIPTIC, 6, 6),
    ],
)
def test_rows_per_contact_matches_mujocos_own_accounting(cone, dim, rows):
    """A pyramidal cone costs 2*(dim-1) rows per contact and never fewer than
    one; an elliptic cone costs dim. These are the numbers njmax is spent
    in: one contact costs 6 rows at condim=4 with a pyramidal cone, the
    dim=4 row of this table."""
    assert sim_budget.rows_per_contact(cone, dim) == rows


def test_rows_per_contact_rejects_an_unknown_cone():
    with pytest.raises(ValueError, match="mjtCone"):
        sim_budget.rows_per_contact(99, 3)


# -- active_contacts --------------------------------------------------------


def test_active_contacts_counts_penetrating_pairs_only():
    """mjx keeps a fixed-size contact array and pads it with candidate pairs
    that are not touching, so the array length is the buffer size, not a
    measurement. A contact is live when its distance is negative."""
    dist = np.array([0.6, -1e-4, 0.0, -0.02, 3.0])
    assert sim_budget.active_contacts(dist) == 2


def test_active_contacts_is_zero_on_an_all_clear_buffer():
    assert sim_budget.active_contacts(np.array([0.5, 0.1, 0.0])) == 0


# -- live_peaks: the warp-only counters -------------------------------------


def _stub_data(impl):
    return types.SimpleNamespace(_impl=impl)


def test_live_peaks_reads_the_warp_counters():
    """Warp's `nacon` is one number for the whole shared pool and its `nefc`
    is one row count per world, so the peak is the max over worlds."""
    impl = types.SimpleNamespace(nacon=np.int32(17), nefc=np.array([40, 61, 12]))
    assert sim_budget.live_peaks(_stub_data(impl)) == (17, 61)


def test_live_peaks_ignores_the_jax_backends_static_buffer_sizes():
    """The jax impl carries `nefc` too, but as a scalar buffer size fixed at
    make_data time -- 2025 rows on asimov, whatever the robot is doing. A
    0-d value is therefore not a measurement and must read as None, not as a
    peak that would make every run look like it overflowed."""
    impl = types.SimpleNamespace(nefc=np.int64(2025), ncon=499)
    assert sim_budget.live_peaks(_stub_data(impl)) == (None, None)


def test_live_peaks_tolerates_a_data_object_with_no_impl():
    assert sim_budget.live_peaks(types.SimpleNamespace()) == (None, None)


# -- budget_report: the run.json / battery.json block -----------------------


def test_budget_report_has_the_same_keys_on_both_backends():
    """A remote GPU run and a local CPU run must produce the same shape, so a
    reader (and a future diff of two runs) never has to branch on backend."""
    jax_block = sim_budget.budget_report("jax", None, None, 32, 320, 4096)
    warp_block = sim_budget.budget_report("warp", 12, 90, 32, 320, 4096)
    assert set(jax_block) == set(warp_block) == SCHEMA_KEYS


def test_budget_report_records_the_pool_as_the_product():
    """Warp allocates one contact pool for the whole batch at make_data time,
    so the budget times the env count is a real device-memory line item."""
    block = sim_budget.budget_report("warp", 12, 90, 32, 320, 4096)
    assert block["pool"] == 32 * 4096
    assert block["naconmax_per_env"] == 32
    assert block["num_envs"] == 4096
    assert block["njmax"] == 320


def test_budget_report_flags_contact_overflow_on_warp():
    """A per-env peak at the budget leaves no slot for one more contact, so
    >= is the flag, not >."""
    assert sim_budget.budget_report("warp", 32, 10, 32, 320, 1)["overflow"] is True
    assert sim_budget.budget_report("warp", 31, 10, 32, 320, 1)["overflow"] is False


def test_budget_report_flags_row_overflow_on_warp():
    """The second budget, and the worse one: rows past njmax apply no force,
    and nothing raises."""
    assert sim_budget.budget_report("warp", 1, 320, 32, 320, 1)["rows_overflow"] is True
    assert sim_budget.budget_report("warp", 1, 319, 32, 320, 1)["rows_overflow"] is False


def test_budget_report_never_flags_overflow_on_jax():
    """The jax backend sizes its own buffers on the fly and has no budget to
    overflow, so a peak past the configured warp budget is not an overflow
    there -- it is a warning that the warp run would have dropped contacts,
    and check_contacts is where that gets said."""
    block = sim_budget.budget_report("jax", 999, 9999, 32, 320, 1)
    assert block["overflow"] is False
    assert block["rows_overflow"] is False
    assert block["nacon_max"] == 999


def test_budget_report_flags_are_plain_bools_and_peaks_are_plain_ints():
    """These land in json.dumps: a numpy bool or a jax scalar is not
    serializable, and json.dumps(default=str) would quietly stringify it."""
    block = sim_budget.budget_report("warp", np.int32(40), np.int64(400), 32, 320, 2)
    assert type(block["overflow"]) is bool
    assert type(block["rows_overflow"]) is bool
    assert type(block["nacon_max"]) is int
    assert type(block["nefc_max"]) is int


def test_budget_report_keeps_missing_peaks_as_null():
    block = sim_budget.budget_report("jax", None, None, 32, 320, 1)
    assert block["nacon_max"] is None
    assert block["nefc_max"] is None
    assert block["backend"] == "jax"


# -- recommend_budget -------------------------------------------------------


def test_recommend_budget_clears_the_peak_at_the_stated_headroom():
    """The sizing rule is about 7x the measured peak, and the rounding is
    upward, so the recommendation never lands under the headroom it
    claims."""
    assert sim_budget.recommend_budget(12, headroom=7.0, step=8) >= 12 * 7.0
    assert sim_budget.recommend_budget(12, headroom=7.0, step=8) % 8 == 0


def test_recommend_budget_rounds_up_to_the_step():
    assert sim_budget.recommend_budget(10, headroom=7.0, step=8) == 72  # 70 -> 72
    assert sim_budget.recommend_budget(8, headroom=4.0, step=32) == 32  # 32 exactly


def test_recommend_budget_never_returns_zero_for_a_zero_peak():
    """A regime that measured nothing must not recommend a zero-length buffer;
    warp would then drop every contact."""
    assert sim_budget.recommend_budget(0, headroom=7.0, step=8) == 8


# -- observed_peaks: what a caller in a step loop records -------------------


def test_observed_peaks_prefers_the_warp_counters():
    impl = types.SimpleNamespace(
        nacon=np.int32(9),
        nefc=np.array([31, 44]),
        contact=types.SimpleNamespace(dist=np.array([-1.0] * 20)),
    )
    assert sim_budget.observed_peaks(_stub_data(impl)) == (9, 44)


def test_observed_peaks_counts_contacts_when_there_are_no_counters():
    """The jax fallback: contacts are countable, rows are not. Reporting a
    number for the rows here would be reporting the static buffer size."""
    impl = types.SimpleNamespace(
        nefc=np.int64(2025),
        contact=types.SimpleNamespace(dist=np.array([0.4, -0.01, -0.2, 1.0])),
    )
    assert sim_budget.observed_peaks(_stub_data(impl)) == (2, None)


def test_observed_peaks_is_all_none_with_neither_counters_nor_contacts():
    assert sim_budget.observed_peaks(_stub_data(types.SimpleNamespace())) == (None, None)


# -- budget_report_for_env: the one adapter train.py and battery.py use -----


def _stub_env(backend="jax", naconmax_per_env=224, njmax=1120, num_envs=1):
    # _naconmax_per_env/_njmax mirror envs/base.py's resolved attributes
    # (sim config value if set, else the robot.yaml sim_budget).
    return types.SimpleNamespace(
        _backend=backend,
        _naconmax_per_env=naconmax_per_env,
        _njmax=njmax,
        _config=types.SimpleNamespace(sim=types.SimpleNamespace(num_envs=num_envs)),
    )


def test_budget_report_for_env_reads_the_budgets_the_env_resolved():
    """Both writers of the block read the same three numbers from the same
    place, so run.json and battery.json can never disagree about what the run
    was configured with."""
    block = sim_budget.budget_report_for_env(_stub_env(num_envs=4096), 40, None)
    assert block["naconmax_per_env"] == 224
    assert block["njmax"] == 1120
    assert block["num_envs"] == 4096
    assert block["pool"] == 224 * 4096
    assert block["nacon_max"] == 40
    assert block["backend"] == "jax"


def test_budget_report_for_env_flags_overflow_on_a_warp_env():
    block = sim_budget.budget_report_for_env(
        _stub_env(backend="warp", naconmax_per_env=32, njmax=320), 32, 320
    )
    assert block["overflow"] is True
    assert block["rows_overflow"] is True


# -- ccd_slot_bytes -----------------------------------------------------------


def test_ccd_slot_bytes_matches_the_allocation():
    """MJWarp's per-slot CCD scratch at 35 EPA iterations, the iteration
    count both robots compile with: 4,996 B of EPA scratch, plus 484 B of
    multi-contact scratch when the model has box-box pairs."""
    assert sim_budget.ccd_slot_bytes(35, box_box=False) == 4996
    assert sim_budget.ccd_slot_bytes(35, box_box=True) == 5480
    # Every term grows with the iteration count: 2 vertices of 16 B and 5
    # faces of 20 B per iteration.
    assert sim_budget.ccd_slot_bytes(36, box_box=False) - 4996 == 2 * 16 + 5 * 20


def test_a_box_box_only_model_runs_sixteen_epa_iterations():
    assert sim_budget.ccd_slot_bytes(35, box_box=True, all_convex_box_box=True) == (
        sim_budget.ccd_slot_bytes(16, box_box=True)
    )


# -- pool_report ----------------------------------------------------------------


def test_pool_report_compares_against_the_pool():
    """The pool counters count the whole batch. 500 contacts over 8 worlds
    is well past a 100 per-env budget, but inside the 800-slot pool, so
    nothing overflows. Broadphase candidates share the pool, so the larger
    of the two counters is the demand. Rows stay per world, against
    njmax."""
    report = sim_budget.pool_report("warp", 500, 300, 900, 100, 1000, 8)
    assert report["pool"] == 800
    assert report["fill_pool"] == pytest.approx(500 / 800)
    assert report["fill_rows"] == pytest.approx(0.9)
    assert report["overflow"] is False and report["rows_overflow"] is False

    assert sim_budget.pool_report("warp", 500, 801, 900, 100, 1000, 8)["overflow"] is True
    assert sim_budget.pool_report("warp", 801, 10, 900, 100, 1000, 8)["overflow"] is True
    # MJWarp drops past the buffer only when the count exceeds it. Here
    # both counters sit exactly at their buffers, so neither flag is set.
    r = sim_budget.pool_report("warp", 800, 800, 1000, 100, 1000, 8)
    assert r["overflow"] is False and r["rows_overflow"] is False
    assert sim_budget.pool_report("warp", 1, 1, 1001, 100, 1000, 8)["rows_overflow"] is True


def test_pool_report_is_unmeasured_off_warp():
    """jax has no pool counters and no fixed buffers: no fill, no flag."""
    report = sim_budget.pool_report("jax", None, None, None, 512, 4096, 16)
    assert report["fill_pool"] is None and report["fill_rows"] is None
    assert report["overflow"] is False and report["rows_overflow"] is False
    assert report["pool"] == 512 * 16
    assert sim_budget.pool_report("jax", 10_000, 10_000, 10_000, 1, 1, 1)["overflow"] is False


# -- traced telemetry ----------------------------------------------------------


def test_traced_counters_on_jax_and_warp_shaped_data():
    """jax data has no live counters and reads zeros, whatever buffer sizes
    its impl carries. On warp data `nefc` is per world and reads its max.
    The pool counters read the same as a scalar or as a copy broadcast
    across worlds."""
    jax_impl = types.SimpleNamespace(nefc=np.full(4, 2025), ncon=np.full(4, 499))
    assert [int(c) for c in sim_budget.traced_counters(_stub_data(jax_impl))] == [0, 0, 0]
    assert [int(c) for c in sim_budget.traced_counters(types.SimpleNamespace())] == [0, 0, 0]

    pool = types.SimpleNamespace(nefc=np.array([40, 61, 12]), nacon=np.array([17]), ncollision=np.int32(90))
    broadcast = types.SimpleNamespace(
        nefc=np.array([40, 61, 12]), nacon=np.full(3, 17), ncollision=np.full((3, 1), 90)
    )
    for impl in (pool, broadcast):
        counters = sim_budget.traced_counters(_stub_data(impl))
        assert [int(c) for c in counters] == [61, 17, 90]
        assert all(np.asarray(c).shape == () and np.asarray(c).dtype == np.int32 for c in counters)


def test_the_counter_names_are_the_warp_data_fields():
    """The stubs above use the real field names. A rename of `nacon` would
    turn warp telemetry into silent zeros, because its absence marks jax
    data. A rename of `nefc` or `ncollision` would raise only at trace
    time on a GPU host."""
    from mujoco.mjx._src import types as jax_types
    from mujoco.mjx.warp import types as warp_types

    assert set(sim_budget.TELEMETRY_COUNTERS) <= {f.name for f in dataclasses.fields(warp_types.DataWarp)}
    assert "nacon" not in {f.name for f in dataclasses.fields(jax_types.DataJAX)}


def test_telemetry_caps_are_njmax_and_the_pool():
    """Rows have one world's budget, njmax. Contacts and broadphase
    candidates share the pool: naconmax_per_env times the env count."""
    env = _stub_env(backend="warp", naconmax_per_env=32, njmax=320, num_envs=64)
    assert sim_budget.telemetry_caps(env) == (320, 2048, 2048)
    assert sim_budget.telemetry_caps(env)[1] == sim_budget.budget_report_for_env(env, None, None)["pool"]


CAPS = (320, 1000, 1000)


def test_telemetry_peaks_are_monotone():
    peaks = np.zeros(3, np.int32)
    seen = []
    for counters in ([100, 50, 80], [90, 400, 70], [10, 10, 10], [200, 300, 500]):
        peaks, _, _ = sim_budget.telemetry_step(peaks, counters, CAPS)
        seen.append(np.asarray(peaks).tolist())
    assert seen == [[100, 50, 80], [100, 400, 80], [100, 400, 80], [200, 400, 500]]


def test_telemetry_fires_once_at_the_first_90pct_crossing():
    """The rows reach 288 of 320 on the second step and fire there. A later,
    higher peak does not fire again. The pool counters never reach 900."""
    peaks = np.zeros(3, np.int32)
    fired = []
    for counters in ([280, 10, 10], [288, 899, 10], [319, 10, 10], [320, 10, 10]):
        peaks, _, fire = sim_budget.telemetry_step(peaks, counters, CAPS)
        fired.append(np.asarray(fire).tolist())
    assert fired == [[False] * 3, [True, False, False], [False] * 3, [False] * 3]
    _, _, fire = sim_budget.telemetry_step(peaks, [0, 900, 950], CAPS)
    assert np.asarray(fire).tolist() == [False, True, True]


def test_telemetry_fractions_are_the_peaks_over_their_own_caps():
    caps = (320, 2048, 2048)
    _, fracs, _ = sim_budget.telemetry_step(np.zeros(3, np.int32), [160, 1024, 512], caps)
    np.testing.assert_allclose(np.asarray(fracs), [0.5, 0.5, 0.25])


def test_telemetry_warning_names_each_fired_counter_and_its_budget():
    text = sim_budget.telemetry_warning([300, 950, 100], CAPS, [True, True, False])
    lines = text.splitlines()
    assert len(lines) == 2
    assert "nefc peaked at 300 of njmax 320" in lines[0] and "task.env.sim.njmax" in lines[0]
    assert "nacon peaked at 950 of the naconmax pool 1000" in lines[1]
    assert sim_budget.telemetry_warning([0, 0, 0], CAPS, [False] * 3) == ""
