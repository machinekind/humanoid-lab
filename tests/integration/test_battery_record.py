"""The real battery rollout record against RECORD_FIELDS.

No other test runs the real battery rollout, so this file pins the producer
side of RECORD_FIELDS. The unit tests in tests/unit/test_battery.py pin the
consumer side on a synthetic record.

Every test reuses one set of jitted callables, so nothing compiles twice.
"""

import dataclasses
from types import SimpleNamespace

import jax
import jax.numpy as jp
import numpy as np
import pytest

from humanoid_lab import paths
from humanoid_lab.eval import battery
from humanoid_lab.registry import make_env

N_STEPS = 20


@pytest.fixture(scope="module")
def roll():
    overrides = battery.merged_env_overrides(
        {"hydra_config": {"task": {"env": {}}}}, {"sim": {"backend": "jax"}}
    )
    env = make_env("joystick", paths.ROBOTS_DIR / "roboto_origin", "deploy_pd", overrides)
    reset, step = jax.jit(env.reset), jax.jit(env.step)

    def inf(obs, rng):
        return jp.zeros(env.action_size), {}

    cmd_at = battery.battery_scenarios(env.dt, env._config.command)["walk_ramp"][0]
    rec, meta, fell_at, budget = battery.rollout(env, reset, step, inf, cmd_at, N_STEPS)
    return SimpleNamespace(
        env=env, reset=reset, step=step, inf=inf, cmd_at=cmd_at,
        rec=rec, meta=meta, fell_at=fell_at, budget=budget,
    )


def test_the_meta_carries_dt_and_the_foot_sides(roll):
    assert roll.meta == {"dt": float(roll.env.dt), "foot_sides": ("left", "right")}


def test_the_new_fields_read_the_right_signals(roll):
    rec = roll.rec
    # roboto has a knee group, so the optional knee_q is recorded.
    assert "knee_q" in rec
    norms = np.linalg.norm(rec["base_quat"], axis=-1)
    assert np.all(np.abs(norms - 1.0) < 1e-3)
    # xpos and qpos are one physics substep apart.
    assert np.all(np.abs(rec["base_pos"][:, 2] - rec["height"]) < 0.01)
    # roboto's foot-ordered knee columns in the actuated-joint arrays.
    np.testing.assert_array_equal(rec["knee_q"], rec["qpos"][:, [3, 9]])
    # foot_clear is the foot site's z minus a fixed sole offset per foot.
    assert np.ptp(rec["foot_pos"][..., 2] - rec["foot_clear"], axis=0).max() < 1e-6
    for key, value in rec.items():
        if np.issubdtype(value.dtype, np.floating):
            assert np.isfinite(value).all(), key


def test_a_robot_without_a_knee_group_records_no_knee_q(roll, monkeypatch):
    spec = roll.env.robot_spec
    groups = {name: joints for name, joints in spec.joint_groups.items() if name != "knee"}
    monkeypatch.setattr(roll.env, "_robot_spec", dataclasses.replace(spec, joint_groups=groups))
    rec, _, _, _ = battery.rollout(roll.env, roll.reset, roll.step, roll.inf, roll.cmd_at, 5)

    assert "knee_q" not in rec
    battery.check_record(rec, battery.record_dims(roll.env))


def test_scenario_result_scores_the_real_record(roll):
    env = roll.env
    rec = roll.rec
    torque_cap = np.asarray(env.mj_model.actuator_forcerange[:, 1])
    r = battery.scenario_result(
        "walk_ramp", rec, roll.fell_at, env.dt, torque_cap, N_STEPS, meta=roll.meta
    )

    # Zero actions hold the stand for the whole scenario.
    assert roll.fell_at is None
    assert r["fell"] is False
    assert r["completed"] is True
    assert r["steps"] == N_STEPS
    vx_err = np.abs(rec["cmd"][:, 0] - rec["vx"]).mean()
    assert r["vel_err_vx"] == pytest.approx(vx_err, abs=1e-4)
    assert r["height_mean"] == pytest.approx(rec["height"].mean(), abs=1e-4)
    # 20 rows is shorter than the 1 s settle window, so the gait shape goes
    # unscored. Every key is still present, as None or a zero count, and the
    # per-foot keys take their names from the meta's foot sides.
    assert N_STEPS < battery.settle_steps(env.dt)
    assert r["swings"] == 0
    assert r["speed_frac"] is None
    assert r["landing_x_m_left"] is None
    assert r["knee_td_rad_right"] is None
