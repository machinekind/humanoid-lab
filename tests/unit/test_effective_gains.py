"""The gain stamp run.json carries (port item 4.3).

`effective_gains` reads kp and kd back off a BUILT model's actuator params,
so what run.json records is what the physics ran with -- after preset
loading and after `actuators.overrides` merging. run.json's
`actuator_gains` block IS this function's return value, so the field-set
assertions here are the run.json schema test.

Pure: arrays and names in, a dict out. The proof that the numbers survive a
real injection (including an override) is
tests/integration/test_injection.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from humanoid_lab.robot.presets import effective_gains

_JOINTS = ["hip_pitch_joint", "knee_joint"]


def _pd_params(kps, kds):
    """gainprm/biasprm as actuators/models.py's PositionPD.inject writes them."""
    gainprm = np.zeros((len(kps), 10))
    biasprm = np.zeros((len(kps), 10))
    for i, (kp, kd) in enumerate(zip(kps, kds)):
        gainprm[i, 0] = kp
        biasprm[i, 1] = -kp
        biasprm[i, 2] = -kd
    return gainprm, biasprm


def test_the_block_carries_every_field_a_reader_needs():
    gainprm, biasprm = _pd_params([50.0, 60.0], [2.0, 3.0])

    block = effective_gains(gainprm, biasprm, _JOINTS, model="pd", preset="pd_test")

    assert set(block) == {"preset", "model", "joints", "kp", "kd"}
    assert block["preset"] == "pd_test"
    assert block["model"] == "pd"
    assert block["joints"] == _JOINTS


def test_kp_and_kd_come_off_the_actuator_params():
    gainprm, biasprm = _pd_params([50.0, 60.0], [2.0, 3.0])

    block = effective_gains(gainprm, biasprm, _JOINTS, model="pd", preset="pd_test")

    assert block["kp"] == pytest.approx([50.0, 60.0])
    assert block["kd"] == pytest.approx([2.0, 3.0])


def test_the_gains_are_per_actuator_not_one_number():
    """A preset with per-group gains injects a different kp per joint. One
    scalar would hide exactly the group the run was tuning."""
    gainprm, biasprm = _pd_params([50.0, 60.0], [2.0, 3.0])

    block = effective_gains(gainprm, biasprm, _JOINTS, model="pd", preset="pd_test")

    assert len(block["kp"]) == len(_JOINTS)
    assert block["kp"][0] != block["kp"][1]


def test_the_joint_order_is_the_column_order_of_the_gains():
    gainprm, biasprm = _pd_params([50.0, 60.0], [2.0, 3.0])

    block = effective_gains(gainprm, biasprm, _JOINTS, model="pd", preset="pd_test")

    assert dict(zip(block["joints"], block["kp"])) == {
        "hip_pitch_joint": 50.0,
        "knee_joint": 60.0,
    }


def test_an_ideal_torque_preset_is_stamped_with_its_model_name():
    """Its gainprm[0] is 1.0 and it has no bias term at all, so these are not
    PD gains. They are stamped anyway -- the model name is what makes the
    numbers readable, and a run.json whose shape depended on the actuator
    model would need branching at every reader."""
    gainprm = np.zeros((2, 10))
    gainprm[:, 0] = 1.0
    biasprm = np.zeros((2, 10))

    block = effective_gains(gainprm, biasprm, _JOINTS, model="ideal_torque", preset="sizing_ideal")

    assert block["model"] == "ideal_torque"
    assert block["kp"] == pytest.approx([1.0, 1.0])
    assert block["kd"] == pytest.approx([0.0, 0.0])


def test_the_values_are_json_native_floats():
    """run.json is written with json.dumps; numpy scalars would land there
    only through the `default=str` fallback, as quoted strings."""
    gainprm, biasprm = _pd_params([50.0], [2.0])

    block = effective_gains(gainprm, biasprm, ["knee_joint"], model="pd", preset="pd_test")

    assert all(type(v) is float for v in block["kp"] + block["kd"])


def test_a_gain_table_that_does_not_match_the_joint_list_raises():
    gainprm, biasprm = _pd_params([50.0, 60.0, 70.0], [2.0, 3.0, 4.0])

    with pytest.raises(ValueError, match="3 actuators"):
        effective_gains(gainprm, biasprm, _JOINTS, model="pd", preset="pd_test")
