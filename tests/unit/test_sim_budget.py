"""Tests for sim_budget.py: the warp contact/constraint budget accounting.

Everything here is pure dict and array arithmetic. The two live counters
(`data._impl.nacon`, `data._impl.nefc`) only exist on the warp backend, which
needs CUDA, so the warp branch is exercised with a stub object standing in for
`data._impl` -- structure, not physics.
"""

from __future__ import annotations

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
    one; an elliptic cone costs dim. These are the numbers njmax is spent in,
    and w01-tek's note that "one contact costs 6 rows at condim=4 with a
    pyramidal cone" is the dim=4 row of this table."""
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
    """At the per-env budget the pool is full and warp drops the overflow
    silently, so >= is the flag, not >."""
    assert sim_budget.budget_report("warp", 32, 10, 32, 320, 1)["overflow"] is True
    assert sim_budget.budget_report("warp", 31, 10, 32, 320, 1)["overflow"] is False


def test_budget_report_flags_row_overflow_on_warp():
    """The second budget, and the worse one: rows past njmax apply no force
    and nothing warns anywhere."""
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
    """w01-tek sized its own pool at about 7x the measured peak. The rounding
    is upward, so the recommendation never lands under the headroom it
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
    sim = types.SimpleNamespace(
        naconmax_per_env=naconmax_per_env, njmax=njmax, num_envs=num_envs
    )
    return types.SimpleNamespace(
        _backend=backend, _config=types.SimpleNamespace(sim=sim)
    )


def test_budget_report_for_env_reads_the_budgets_off_the_env_config():
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
