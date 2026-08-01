"""Tests for check_contacts.py, the contact-budget measurement.

The measurement itself builds a model and steps MJX, so it lives here. The
pure arithmetic it reports against (`sim_budget.recommend_budget`) is unit
tested in tests/unit/test_sim_budget.py.

The second test is a standing guard: it fails when someone adds collision
geometry to a robot without resizing the configured budgets. That failure
means the budgets are undersized, not that the test is too strict -- on the
warp backend the contacts past the budget are dropped with no warning at
all.
"""

from __future__ import annotations

import pytest

from humanoid_lab import check_contacts, paths
from humanoid_lab.envs.joystick import default_config

# Short by CLI standards: every measured peak lands inside the first 80
# control steps, and three seeds is enough to cover the fallen sweep's three
# attitudes. The recorded measurement in default_config's comment uses the
# CLI's own longer defaults.
TEST_STEPS = 100
TEST_SEEDS = 3

ROBOTS = [("asimov_v1", "sizing_ideal"), ("roboto_origin", "deploy_pd")]


@pytest.fixture(scope="module")
def measured():
    return {
        robot: check_contacts.measure_robot(
            paths.ROBOTS_DIR / robot, preset, steps=TEST_STEPS, seeds=TEST_SEEDS
        )
        for robot, preset in ROBOTS
    }


@pytest.mark.parametrize("robot, preset", ROBOTS, ids=[r for r, _ in ROBOTS])
def test_every_regime_reports_a_peak(measured, robot, preset):
    """Three regimes, all of them measuring something.

    No ordering is asserted between them, and that is a finding rather than a
    gap: neither robot's home keyframe is held up by a neutral action under
    the stock preset gains, so the "standing" regime is itself a collapse, and
    its opening steps -- both feet flat, every sole capsule loaded -- often
    carry the highest contact count of the three. The fallen regime is
    measured because early training is mostly fallen robots, not because it is
    guaranteed to be the worst."""
    result = measured[robot]
    regimes = result["regimes"]
    assert set(regimes) == set(check_contacts.REGIMES)
    for name, r in regimes.items():
        assert r["nacon_max"] > 0, f"{name} measured no contacts at all"
        assert r["nefc_max"] >= r["nacon_max"], f"{name} rows below contacts"
        assert r["steps"] == TEST_STEPS
        assert r["seeds"] == TEST_SEEDS

    # The fallen regime really did put the robot down.
    assert regimes["fallen"]["height_end"] < 0.5


@pytest.mark.parametrize("robot, preset", ROBOTS, ids=[r for r, _ in ROBOTS])
def test_the_configured_budgets_hold_the_measured_peaks_with_headroom(
    measured, robot, preset
):
    """The configured budgets must still cover the measured peaks at the
    headroom the config comment claims. Adding foot geometry, raising condim,
    or injecting collision primitives moves the peaks; this test is what makes
    that a build failure instead of a silent contact drop on the next GPU
    run."""
    cfg = default_config().sim
    peak_contacts = measured[robot]["peak"]["nacon_max"]
    peak_rows = measured[robot]["peak"]["nefc_max"]

    assert peak_contacts * check_contacts.HEADROOM <= cfg.naconmax_per_env, (
        f"{robot}: {peak_contacts} peak contacts at {check_contacts.HEADROOM}x headroom "
        f"needs naconmax_per_env >= {peak_contacts * check_contacts.HEADROOM}, config has "
        f"{cfg.naconmax_per_env}"
    )
    assert peak_rows * check_contacts.HEADROOM <= cfg.njmax, (
        f"{robot}: {peak_rows} peak constraint rows at {check_contacts.HEADROOM}x headroom "
        f"needs njmax >= {peak_rows * check_contacts.HEADROOM}, config has {cfg.njmax}"
    )


def test_the_peak_block_is_the_max_over_the_regimes(measured):
    """One number per budget, taken over every regime -- sizing off a single
    regime is the mistake the fallen probe exists to prevent."""
    for result in measured.values():
        regimes = result["regimes"]
        assert result["peak"]["nacon_max"] == max(r["nacon_max"] for r in regimes.values())
        assert result["peak"]["nefc_max"] == max(r["nefc_max"] for r in regimes.values())


def test_the_result_records_what_produced_it(measured):
    """The measurement lands in a config comment, so it has to carry its own
    provenance: which robot, which preset, which backend, how many steps."""
    result = measured["asimov_v1"]
    assert result["robot"] == "asimov_v1"
    assert result["preset"] == "sizing_ideal"
    assert result["backend"] == "jax"
    assert result["steps"] == TEST_STEPS
    assert result["seeds"] == TEST_SEEDS
    # On jax the rows are derived from the active contacts, not read off a
    # counter, and the report must say so rather than pass a derivation off as
    # a measurement.
    assert result["nefc_derived"] is True
    assert result["rows_per_contact"] >= 1
