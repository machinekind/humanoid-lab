"""Build-order step 5 gate: the three asimov_v1 actuator presets compile and
their per-joint compiled values match PLAN.md's tables.

CPU-fast on purpose (module-scoped compile fixtures, no MJX): this checks
injected gainprm/biasprm/forcerange/armature against PLAN.md's "v0 leg
motors (ENCOS)", "Gain schemes", and "v1 armatures" tables, one representative
(left-side) joint per group, for sizing_ideal, encos_datasheet, and
deploy_pd. It also sanity-checks the motor catalogs (motors/encos.yaml,
motors/mab.yaml) against a couple of numbers so a catalog typo fails a test.
"""

from __future__ import annotations

import yaml
import pytest

from humanoid_lab import paths
from humanoid_lab.robot.build import build_spec, compile_spec

ROBOT_DIR = paths.ROBOTS_DIR / "asimov_v1"

# group -> representative left-side joint name (robot.yaml's joint_groups).
LEFT_JOINT = {
    "hip_pitch": "left_hip_pitch_joint",
    "hip_roll": "left_hip_roll_joint",
    "hip_yaw": "left_hip_yaw_joint",
    "knee": "left_knee_joint",
    "ankle_pitch": "left_ankle_pitch_joint",
    "ankle_roll": "left_ankle_roll_joint",
}

# v1 XML armatures (PLAN.md "v1 armatures"), same across all three presets:
# a preset overrides the effort/kp/kd, not the physical inertia.
ARMATURE = {
    "hip_pitch": 0.095625,
    "hip_roll": 0.11,
    "hip_yaw": 0.038,
    "knee": 0.0339552,
    "ankle_pitch": 0.0565056,
    "ankle_roll": 0.0565056,
}

# kp = armature * (2*pi*10 Hz)^2, the v0 sim scheme (PLAN.md "Gain schemes")
# applied to v1 armatures; shared by sizing_ideal and encos_datasheet.
SIM_SCHEME_KP = {
    "hip_pitch": 377.5,
    "hip_roll": 434.3,
    "hip_yaw": 150.0,
    "knee": 134.1,
    "ankle_pitch": 223.1,
    "ankle_roll": 223.1,
}

# Per-preset (kp, kd, effort_limit) tables, PLAN.md "v0 leg motors (ENCOS)"
# and "Gain schemes".
PRESET_TABLES = {
    "sizing_ideal": {
        # 2x ENCOS datasheet peak torque (PLAN.md "v0 leg motors").
        "hip_pitch": (SIM_SCHEME_KP["hip_pitch"], 5.0, 240.0),
        "hip_roll": (SIM_SCHEME_KP["hip_roll"], 5.0, 180.0),
        "hip_yaw": (SIM_SCHEME_KP["hip_yaw"], 5.0, 120.0),
        "knee": (SIM_SCHEME_KP["knee"], 5.0, 150.0),
        "ankle_pitch": (SIM_SCHEME_KP["ankle_pitch"], 5.0, 72.0),
        "ankle_roll": (SIM_SCHEME_KP["ankle_roll"], 5.0, 72.0),
    },
    "encos_datasheet": {
        # ENCOS datasheet peak torque, verbatim.
        "hip_pitch": (SIM_SCHEME_KP["hip_pitch"], 5.0, 120.0),
        "hip_roll": (SIM_SCHEME_KP["hip_roll"], 5.0, 90.0),
        "hip_yaw": (SIM_SCHEME_KP["hip_yaw"], 5.0, 60.0),
        "knee": (SIM_SCHEME_KP["knee"], 5.0, 75.0),
        "ankle_pitch": (SIM_SCHEME_KP["ankle_pitch"], 5.0, 36.0),
        "ankle_roll": (SIM_SCHEME_KP["ankle_roll"], 5.0, 36.0),
    },
    "deploy_pd": {
        # v0 firmware gains after gain identification (kp/kd); effort_limit
        # is the CAN command clamp (tighter than datasheet peak for knee/ankle).
        "hip_pitch": (12.8, 0.8, 120.0),
        "hip_roll": (328.0, 5.0, 90.0),
        "hip_yaw": (212.2, 5.0, 60.0),
        "knee": (64.2, 2.7, 70.0),
        "ankle_pitch": (19.3, 3.3, 30.0),
        "ankle_roll": (18.1, 0.9, 30.0),
    },
}


@pytest.fixture(scope="module", params=list(PRESET_TABLES))
def built_model(request):
    """One compiled model per preset, module-scoped: (preset_name, model)."""
    preset_name = request.param
    spec = build_spec(ROBOT_DIR, preset_name)
    return preset_name, compile_spec(spec)


def test_preset_injects_expected_gains_and_armature(built_model):
    preset_name, model = built_model
    table = PRESET_TABLES[preset_name]

    for group_name, joint_name in LEFT_JOINT.items():
        kp, kd, effort = table[group_name]

        actuator = model.actuator(joint_name)
        assert actuator.gainprm[0] == pytest.approx(kp), (preset_name, group_name)
        assert actuator.biasprm[1] == pytest.approx(-kp), (preset_name, group_name)
        assert actuator.biasprm[2] == pytest.approx(-kd), (preset_name, group_name)
        assert tuple(actuator.forcerange) == pytest.approx((-effort, effort)), (
            preset_name,
            group_name,
        )

        joint = model.joint(joint_name)
        dof_addr = joint.dofadr[0]
        assert model.dof_armature[dof_addr] == pytest.approx(ARMATURE[group_name]), (
            preset_name,
            group_name,
        )


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
