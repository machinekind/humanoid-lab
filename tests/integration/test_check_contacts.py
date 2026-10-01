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

import functools

import pytest

from humanoid_lab import check_contacts, paths
from humanoid_lab.robot.spec import load_robot_spec

# Short by CLI standards. 100 control steps x 4 seeds reproduce the CLI's
# overall peak for every robot and preset, which is all the budget guard
# reads. Roboto Origin needs both: deploy_pd's 32 contacts land standing at
# step 91, and sizing_ideal's 38 land fallen at seed 3, the fallen sweep's
# second face-down draw. Some per-regime peaks land later (step 103, or seed
# 4, under the CLI defaults), so the per-regime numbers here can run below
# the CLI's; the tests only check that each regime measured something. The
# budgets recorded in the robot.yamls use the CLI's own longer defaults.
TEST_STEPS = 100
TEST_SEEDS = 4

# Every robot directory, like test_robot_conformance: adding a robot means
# adding a directory, not editing this list. Each robot is measured under
# every preset it has. The preset that sets the budget differs per robot
# (Roboto Origin peaks at 38 contacts under sizing_ideal and 32 under
# deploy_pd), so the guard takes the maximum over presets.
ROBOTS = sorted(p.parent.name for p in paths.ROBOTS_DIR.glob("*/robot.yaml"))


def _presets(robot: str) -> list[str]:
    return sorted(p.stem for p in (paths.ROBOTS_DIR / robot / "actuators").glob("*.yaml"))


def _first_preset(robot: str) -> str:
    return _presets(robot)[0]


CASES = [(robot, preset) for robot in ROBOTS for preset in _presets(robot)]


@functools.cache
def _measured(robot: str, preset: str) -> dict:
    """One measurement per (robot, preset), taken on first use, so a partial
    run measures only the cases it selects."""
    return check_contacts.measure_robot(
        paths.ROBOTS_DIR / robot, preset, steps=TEST_STEPS, seeds=TEST_SEEDS
    )


@pytest.mark.parametrize(("robot", "preset"), CASES)
def test_every_regime_reports_a_peak(robot, preset):
    """Three regimes, all of them measuring something.

    No ordering is asserted between them, and that is a finding rather than a
    gap: neither robot's home keyframe is held up by a neutral action under
    the stock preset gains, so the "standing" regime is itself a collapse, and
    its opening steps -- both feet flat, every sole capsule loaded -- often
    carry the highest contact count of the three. The fallen regime is
    measured because early training is mostly fallen robots, not because it is
    guaranteed to be the worst."""
    regimes = _measured(robot, preset)["regimes"]
    assert set(regimes) == set(check_contacts.REGIMES)
    for name, r in regimes.items():
        assert r["nacon_max"] > 0, f"{name} measured no contacts at all"
        assert r["nefc_max"] >= r["nacon_max"], f"{name} rows below contacts"
        assert r["steps"] == TEST_STEPS
        assert r["seeds"] == TEST_SEEDS

    # The fallen regime really did put the robot down.
    assert regimes["fallen"]["height_end"] < 0.5


@pytest.mark.parametrize("robot", ROBOTS)
def test_the_recorded_budgets_hold_the_measured_peaks_with_headroom(robot):
    """robot.yaml's sim_budget must still cover the measured peaks at the
    headroom rule. Adding foot geometry, raising condim, or injecting
    collision primitives moves the peaks; this test is what makes that a
    build failure instead of a silent contact drop on the next GPU run."""
    budget = load_robot_spec(paths.ROBOTS_DIR / robot).sim_budget
    assert budget, f"{robot}: robot.yaml records no sim_budget block to guard"
    peaks = [_measured(robot, preset)["peak"] for preset in _presets(robot)]
    peak_contacts = max(peak["nacon_max"] for peak in peaks)
    peak_rows = max(peak["nefc_max"] for peak in peaks)

    assert peak_contacts * check_contacts.HEADROOM <= budget["naconmax_per_env"], (
        f"{robot}: {peak_contacts} peak contacts at {check_contacts.HEADROOM}x headroom "
        f"needs naconmax_per_env >= {peak_contacts * check_contacts.HEADROOM}, robot.yaml has "
        f"{budget['naconmax_per_env']}"
    )
    assert peak_rows * check_contacts.HEADROOM <= budget["njmax"], (
        f"{robot}: {peak_rows} peak constraint rows at {check_contacts.HEADROOM}x headroom "
        f"needs njmax >= {peak_rows * check_contacts.HEADROOM}, robot.yaml has {budget['njmax']}"
    )


@pytest.mark.parametrize(("robot", "preset"), CASES)
def test_the_peak_block_is_the_max_over_the_regimes(robot, preset):
    """One number per budget, taken over every regime -- sizing off a single
    regime is the mistake the fallen probe exists to prevent."""
    result = _measured(robot, preset)
    regimes = result["regimes"]
    assert result["peak"]["nacon_max"] == max(r["nacon_max"] for r in regimes.values())
    assert result["peak"]["nefc_max"] == max(r["nefc_max"] for r in regimes.values())


def test_the_result_records_what_produced_it():
    """The measurement lands in a config comment, so it has to carry its own
    provenance: which robot, which preset, which backend, how many steps."""
    preset = _first_preset("asimov_v1")
    result = _measured("asimov_v1", preset)
    assert result["robot"] == "asimov_v1"
    assert result["preset"] == preset
    assert result["backend"] == "jax"
    assert result["steps"] == TEST_STEPS
    assert result["seeds"] == TEST_SEEDS
    # On jax the rows are derived from the active contacts, not read off a
    # counter, and the report must say so rather than pass a derivation off as
    # a measurement.
    assert result["nefc_derived"] is True
    assert result["rows_per_contact"] >= 1
