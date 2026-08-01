"""Typo guards for the shared motor catalogs under motors/.

`motors/encos.yaml` and `motors/mab.yaml` are hardware datasheets, not robot
data: they belong to no robot directory and no actuator preset, and the
sizing report reads them by name. So they are not part of the robot
conformance suite, which only asserts things derived from a robot's own
robot.yaml.

A few numbers from PLAN.md's "v0 leg motors (ENCOS)" and "MAB Robotics
actuator survey" tables are pinned here so a catalog typo fails a test.
Carried over unchanged from the robot-named test file these two checks used
to sit in; they read yaml and touch no model, which is why they live in
tests/unit.
"""

from __future__ import annotations

import yaml

from humanoid_lab import paths


def test_encos_motor_catalog_matches_plan():
    with (paths.REPO_ROOT / "motors" / "encos.yaml").open() as f:
        encos = yaml.safe_load(f)

    # PLAN.md "v0 leg motors": knee CAN tau clamp is +/-70 (tighter than the
    # 75 Nm datasheet peak) -- a prime spot for a catalog typo to hide.
    assert encos["knee"]["can_clamp_nm"] == 70
    assert encos["hip_pitch"]["tau_peak_nm"] == 120
    assert encos["ankle"]["joints"] == ["ankle_pitch", "ankle_roll"]


def test_mab_motor_catalog_matches_plan():
    with (paths.REPO_ROOT / "motors" / "mab.yaml").open() as f:
        mab = yaml.safe_load(f)

    # PLAN.md "MAB Robotics actuator survey" table.
    assert mab["MA-p-45-36"]["price_eur_list"] == 357
    assert mab["MA-p-100-30"]["torque_peak_nm"] == 150
