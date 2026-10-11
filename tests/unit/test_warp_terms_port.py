"""The shared port of the promoted recipe against the code that trained v40.

tests/data/fixtures/warp_terms.npz holds inputs and the outputs of the PR-head
Torch runner (wtrain/wenv.py at d8f56f7), written by
tests/data/fixtures/generate_warp_terms.py. These tests feed the same inputs
to the shared functions and compare: the four landing and lean terms with the
land_frac record, the speed-scaled lift, the straight command draw and the
command ramp. The test reads numpy arrays only. It needs no torch.

The port differs from wenv in three intended ways:
- The base position of the foot offsets is xpos[root_body]. wenv read the
  free joint's qpos. The fixture puts the same base position in both, so this
  does not show here.
- The heading uses the IMU quaternion. wenv read the free joint's qpos. The
  fixture's quaternion is one array, so this does not show either.
- The straight draw's zeroed axes are +0.0. wenv multiplied by [1, 0, 0],
  which leaves -0.0 where vy or wz was negative. The straight test asserts
  this.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from task_toys import toy_consts, toy_replace

from humanoid_lab.envs import joystick, progress, signals
from humanoid_lab.rewards import compose

FIXTURE = Path(__file__).resolve().parents[1] / "data" / "fixtures" / "warp_terms.npz"

# tests/parity/tolerances.TASK, restated: tests/unit does not import the
# parity package.
RTOL = ATOL = 1e-5

TERM_CASES = ["clock_pair", "linear_plain", "lift"]
NU = 4
NV = 10


@pytest.fixture(scope="module")
def data():
    return np.load(FIXTURE)


def _case(data, name):
    prefix = f"{name}/"
    arrays = {k[len(prefix) :]: data[k] for k in data.files if k.startswith(prefix) and k != f"{name}/config"}
    return arrays, json.loads(str(data[f"{name}/config"]))


def test_fixture_records_every_case(data):
    meta = json.loads(str(data["_meta"]))
    assert meta["source_sha"].startswith("d8f56f7")
    assert meta["cases"] == sorted(TERM_CASES + ["straight", "ramp"])


def _cfg(config):
    """default_config with the case's terms, their parameters, freq and lift.

    The mapping is the one run_record.migrate_legacy applies to a legacy run.
    """
    cfg = joystick.default_config()
    cfg.gait.freq = tuple(config["freq"])
    for term, scale in config["scales"].items():
        cfg.reward.scales[term] = scale
    for term, spec in config["extra"].items():
        params = {k: v for k, v in spec.items() if k != "scale"}
        if term == "step_length":
            params["target"] = "linear" if "target_a" in params else "clock"
        cfg.reward.scales[term] = spec["scale"]
        cfg.reward[term].update(params)
    lift = config["run"].get("warp_lift")
    if lift:
        cfg.gait.lift.update(v_full=lift["v_full"], min_frac=lift["min_frac"])
    return cfg


def _consts(cfg, config):
    flags = {f"{t}_on": bool(cfg.reward.scales[t]) for t in compose.KEPT_TERMS}
    return toy_replace(
        toy_consts("numpy"),
        foot_site_ids=np.array([0, 1]),
        root_body=0,
        knee_qidx=np.array(config["knee_qidx"]),
        deadband=float(config["deadband"]),
        cmd_vmax=np.float32(config["cmd_vmax"]),
        land_frac_on=compose.land_frac_on(cfg.reward),
        **flags,
    )


def _shared_rewards(x, cfg, consts):
    n, nf = x["in_command"].shape[0], consts.n_feet
    qpos = x["in_qpos"]
    quat = x["in_quat"]
    xpos = np.zeros((n, 1, 3), np.float32)
    xpos[:, 0, :] = x["in_base_pos"]
    r = {
        "quat": quat,
        "gyro": x["in_gyro"],
        "linvel": x["in_linvel"],
        "gravity": x["in_gravity"],
        "qpos_act": qpos[:, consts.qadr],
        "qvel_act": x["in_qvel"][:, consts.vadr],
        "qacc_act": np.zeros((n, NU), np.float32),
        "actuator_force": x["in_actuator_force"],
        "base_height": qpos[:, 2],
        "foot_clearance": x["in_foot_clearance"],
        "foot_linvel": x["in_foot_linvel"],
        "foot_pos_base": signals.foot_pos_base(x["in_site_xpos"], xpos, np.array([0, 1]), 0, quat),
    }
    zeros = np.zeros((n, nf), np.float32)
    info = {
        "command": x["in_command"],
        "phase": x["in_phase"],
        "last_torque": x["in_last_torque"],
        "last_action": x["in_last_action"],
        "last_last_action": x["in_last_last_action"],
        "feet_air_time": x["in_feet_air_time"],
        "feet_contact_time": zeros,
        "last_contact": np.zeros((n, nf), bool),
        "swing_apex": x["in_swing_apex"],
        "last_apex": zeros,
        "air_dur_ema": zeros,
        "stance_dur_ema": zeros,
        "land_frac": x["in_land_frac"],
    }
    return compose.compute_rewards(
        r, info, x["in_action"], x["in_first_contact"], x["in_contact"], consts, cfg
    )


@pytest.mark.parametrize("name", TERM_CASES)
def test_terms_match_wenv(data, name):
    x, config = _case(data, name)
    cfg = _cfg(config)
    consts = _consts(cfg, config)
    rewards, fall, land = _shared_rewards(x, cfg, consts)
    assert config["terms"], "the case records no term"
    for term in config["terms"]:
        want = x[f"t_{term}"]
        assert np.count_nonzero(want), f"{name}: wenv pays 0 for {term} on every row"
        np.testing.assert_allclose(np.asarray(rewards[term]), want, rtol=RTOL, atol=ATOL, err_msg=f"{name}: {term}")
    np.testing.assert_array_equal(np.asarray(fall), x["fall"])
    if consts.land_frac_on:
        np.testing.assert_allclose(np.asarray(land), x["land_frac_now"], rtol=RTOL, atol=ATOL)
    else:
        assert land is None


def test_the_cases_cover_both_targets_and_the_pair_min(data):
    pair = _cfg(_case(data, "clock_pair")[1])
    plain = _cfg(_case(data, "linear_plain")[1])
    assert pair.reward.step_length.target == "clock" and plain.reward.step_length.target == "linear"
    assert compose.land_frac_on(pair.reward) and not compose.land_frac_on(plain.reward)
    assert _cfg(_case(data, "lift")[1]).gait.lift.v_full == 0.5


def test_straight_matches_wenv(data):
    x, config = _case(data, "straight")
    draws = {k: x[k] for k in ("vx", "vy", "wz", "zero_on", "straight_on", "pure_slow_on", "pure_slow_vx")}
    got = np.asarray(progress.command_from_draws(draws, config["command"]))
    want = x["command"]
    np.testing.assert_array_equal(got, want)
    zero = got == 0.0
    # wenv leaves -0.0 on some zeroed axes. The shared draw gives +0.0.
    assert np.any(np.signbit(want) & zero)
    assert not np.any(np.signbit(got) & zero)


def test_ramp_matches_wenv(data):
    x, config = _case(data, "ramp")
    lin, yaw = float(config["ramp"]["lin"]), float(config["ramp"]["yaw"])
    assert not np.any(x["reset_command"])
    np.testing.assert_array_equal(x["command_before"][0], x["reset_command"])
    assert x["resampled"].any() and not x["resampled"].all()
    for k in range(config["steps"]):
        got = np.asarray(progress.ramp_command(x["command_before"][k], x["cmd_target"][k], lin, yaw, config["dt"]))
        np.testing.assert_allclose(got, x["command_after"][k], rtol=RTOL, atol=ATOL, err_msg=f"step {k}")
        if k + 1 < config["steps"]:
            np.testing.assert_array_equal(x["command_before"][k + 1], x["command_after"][k])
