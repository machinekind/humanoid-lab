"""Construction refusals of the promoted task.env knobs, model-free.

Each check is the module-level function the env constructor calls, run on a
plain config. Later packages add their knobs to this file.
"""

from types import SimpleNamespace

import pytest
from ml_collections import config_dict

from humanoid_lab.envs.joystick import (
    FROZEN_PHASE_OBS,
    Joystick,
    check_command_ramp,
    check_frozen_phase_obs,
    check_kept_terms,
    default_config,
)
from humanoid_lab.rewards import terms


def _command(**ramp):
    command = default_config().command
    command.ramp.update(ramp)
    return command


# -- command.ramp --------------------------------------------------------------


def test_the_default_ramp_is_off_and_passes():
    command = default_config().command
    assert (command.ramp.lin, command.ramp.yaw) == (0.0, 0.0)
    check_command_ramp(command)


def test_both_rates_set_pass():
    check_command_ramp(_command(lin=0.5, yaw=1.0))


@pytest.mark.parametrize("lin, yaw", [(0.5, 0.0), (0.0, 1.0)])
def test_one_rate_set_is_refused_and_named(lin, yaw):
    with pytest.raises(ValueError) as err:
        check_command_ramp(_command(lin=lin, yaw=yaw))
    assert "command.ramp.lin" in str(err.value)
    assert "command.ramp.yaw" in str(err.value)


@pytest.mark.parametrize("lin, yaw", [(-0.5, 1.0), (0.5, -1.0), (-0.5, -1.0)])
def test_a_negative_rate_is_refused(lin, yaw):
    with pytest.raises(ValueError, match="command.ramp"):
        check_command_ramp(_command(lin=lin, yaw=yaw))


def test_a_config_without_a_ramp_block_passes():
    check_command_ramp(config_dict.ConfigDict({"resample_steps": 250}))
    check_command_ramp({})


# -- the landing and lean terms (reward.step_length, foot_reach, knee_extend, forward_lean) --

FULL = {
    "step_length": {"target_a": 0.05, "target_b": 0.2},
    "foot_reach": {"target_a": 0.05, "target_b": 0.15},
    "knee_extend": {"flex_lo": 0.1, "flex_hi": 0.6},
    "forward_lean": {"max_deg": 5.0, "vx_max": 1.0, "sigma": 0.05},
}


def _reward(scales, **params):
    """default_config's reward block with these scales and parameter overrides."""
    reward = default_config().reward
    reward.scales.update(scales)
    for term, block in params.items():
        reward[term].update(block)
    return reward


def _check(reward, n_feet=2, freq=(1.0, 2.0)):
    gait = default_config().gait
    gait.freq = freq
    check_kept_terms(reward, gait, n_feet)


def test_the_defaults_pass():
    _check(default_config().reward)
    _check(default_config().reward, n_feet=4, freq=(0.0, 0.0))


def test_every_term_with_its_parameters_passes():
    _check(_reward({t: 1.0 for t in FULL}, **FULL))
    _check(_reward({"step_length": 1.0}, step_length={"target": "linear", **FULL["step_length"]}))


@pytest.mark.parametrize(
    "term, missing",
    [
        ("foot_reach", ("target_a", "target_b")),
        ("knee_extend", ("flex_lo", "flex_hi")),
        ("forward_lean", ("max_deg", "vx_max", "sigma")),
    ],
)
def test_a_required_parameter_left_none_is_refused_and_named(term, missing):
    for key in missing:
        params = {term: {**FULL[term], key: None}}
        with pytest.raises(ValueError) as err:
            _check(_reward({term: 0.5}, **params))
        assert f"reward.{term}.{key} = None" in str(err.value)
        assert f"reward.scales.{term} = 0.5" in str(err.value)
    # At scale 0 the same config passes.
    _check(_reward({term: 0.0}, **{term: {k: None for k in missing}}))


def test_a_linear_step_length_needs_both_coefficients():
    for key in ("target_a", "target_b"):
        params = {"step_length": {"target": "linear", **FULL["step_length"], key: None}}
        with pytest.raises(ValueError, match=f"reward.step_length.{key} = None"):
            _check(_reward({"step_length": 1.0}, **params))
    # The clock target does not read them.
    _check(_reward({"step_length": 1.0}))


def test_an_unknown_step_length_target_is_refused():
    with pytest.raises(ValueError, match="reward.step_length.target = 'stride'"):
        _check(_reward({"step_length": 1.0}, step_length={"target": "stride"}))


@pytest.mark.parametrize("freq", [(0.0, 0.0), (0.0, 2.0)])
def test_a_clock_step_length_with_a_zero_rest_frequency_is_refused(freq):
    with pytest.raises(ValueError) as err:
        _check(_reward({"step_length": 1.0}), freq=freq)
    msg = str(err.value)
    assert "gait.freq" in msg and "NaN" in msg and '"linear"' in msg
    # The linear target does not read the clock.
    _check(_reward({"step_length": 1.0}, step_length={"target": "linear", **FULL["step_length"]}), freq=freq)


@pytest.mark.parametrize("x_min", [terms.FOOT_REACH_MIN_TARGET, 0.05])
def test_foot_reach_x_min_at_or_above_the_floor_is_refused(x_min):
    with pytest.raises(ValueError, match=f"reward.foot_reach.x_min = {x_min}"):
        _check(_reward({"foot_reach": 1.0}, foot_reach={**FULL["foot_reach"], "x_min": x_min}))
    _check(_reward({"foot_reach": 1.0}, foot_reach={**FULL["foot_reach"], "x_min": 0.01}))


@pytest.mark.parametrize("term", ["step_length", "foot_reach", "knee_extend"])
def test_a_landing_term_needs_two_feet(term):
    with pytest.raises(ValueError, match=f"reward.scales.{term} = 1.0 pairs the two feet"):
        _check(_reward({term: 1.0}, **{term: FULL[term]}), n_feet=4)


def test_forward_lean_runs_on_any_foot_count():
    _check(_reward({"forward_lean": 1.0}, forward_lean=FULL["forward_lean"]), n_feet=4)


def test_yaml_ints_pass():
    """A YAML int stays an int in a None-defaulted field. The check reads it as given."""
    _check(_reward({"knee_extend": 1}, knee_extend={"flex_lo": 0, "flex_hi": 1}))


def test_knee_extend_without_a_knee_group_is_refused():
    stub = SimpleNamespace(
        _robot_spec=SimpleNamespace(joint_groups={}, foot_sites=["left_foot", "right_foot"], actuated_joints=[])
    )
    with pytest.raises(ValueError, match="reward.scales.knee_extend needs a 'knee' joint group"):
        Joystick._foot_ordered_group_qidx(stub, "knee", "knee_extend")
    with pytest.raises(ValueError, match="reward.scales.knee_stance needs a 'knee' joint group"):
        Joystick._foot_ordered_group_qidx(stub, "knee")


# -- gait.frozen_phase_obs -------------------------------------------------------


def test_the_default_frozen_phase_obs_is_the_legacy_cos_sin():
    assert default_config().gait.frozen_phase_obs == "cos_sin"
    check_frozen_phase_obs(default_config().gait)


@pytest.mark.parametrize("mode", FROZEN_PHASE_OBS)
def test_both_frozen_phase_obs_values_pass(mode):
    gait = default_config().gait
    gait.frozen_phase_obs = mode
    check_frozen_phase_obs(gait)


@pytest.mark.parametrize("mode", ["offsets", "zero", ""])
def test_an_unknown_frozen_phase_obs_is_refused_and_named(mode):
    gait = default_config().gait
    gait.frozen_phase_obs = mode
    with pytest.raises(ValueError, match=r"gait\.frozen_phase_obs.*cos_sin.*zeros"):
        check_frozen_phase_obs(gait)
