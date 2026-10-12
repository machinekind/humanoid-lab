"""WarpJoystick in lockstep with the wrapped JAX env.

The JAX env runs first, inside the training wrappers: vmap, EpisodeWrapper,
BraxAutoResetWrapper and the reseed layers. Its post-step physics and its
draws are recorded. WarpJoystick then runs on that physics and those draws
and keeps its own bookkeeping. Its info is never copied from JAX.

Every step compares done, truncation, reward, the reward terms, both
observations, every info key, the ctrl and the respawn write. Floats use
tolerances.TASK. Bools and ints must match exactly, dtype included.

Eight worlds run 400 steps with 50-step episodes. Worlds 0-3 get full-size
random actions and fall. Worlds 4-7 get small actions and time out. Five
configs cover command resampling and ramping, the gait trackers, the
heading reference, the landing record, the no-progress cut, symmetry and
pushes. test_the_scenario_covers_its_events proves each config reaches its
events.
"""

from __future__ import annotations

import copy
import types

import jax
import jax.numpy as jp
import numpy as np
import parity_report
import pytest
from tolerances import TASK

from humanoid_lab.envs import gait_state, joystick
from humanoid_lab.rewards import compose

torch = pytest.importorskip("torch")

import replay

N = 8
STEPS = 400
L = 50  # episode length
# Worlds 0-3 fall and worlds 4-7 time out.
AMPLITUDES = [1.0] * 4 + [0.1] * 4
# Fresh uniform actions on every step average out and rarely topple the
# robot. Each action is held for HOLD steps instead, with a per-step jitter
# of JITTER times the world's amplitude on top.
HOLD = 25
JITTER = 0.25

# name -> (compose overrides, task.env overrides).
CONFIGS = {
    "stock": (["actuators=deploy_pd"], {"command": {"resample_steps": 20}}),
    "recipe": (["experiment=roboto_gait_v40"], {"command": {"resample_steps": 20}}),
    "recipe_ramp_straight_lift": (
        ["experiment=roboto_gait_v40"],
        {
            "command": {"resample_steps": 20, "ramp": {"lin": 0.5, "yaw": 1.0}, "straight_prob": 0.3},
            "gait": {"lift": {"v_full": 0.5, "min_frac": 0.25}},
        },
    ),
    "no_progress_symmetry_push": (
        ["actuators=deploy_pd"],
        {
            "command": {"resample_steps": 20},
            # p_max is the default 0.02. At 0.5 the cut ends every episode within
            # about 10 steps, and no world reaches its timeout.
            "no_progress": {"enable": True, "grace_sec": 0.1, "ema_sec": 0.2, "risk_below": 0.9, "p_max": 0.02},
            "symmetry": {"enable": True},
            "push": {
                "enable": True,
                "interval_steps_range": [5, 15],
                "vel_z": 0.3,
                "ang_vel_rp": 0.5,
                "ang_vel_yaw": 0.7,
            },
            # The income scale turns on the 'gait' reseed layer.
            "reward": {"scales": {"gait_symmetry_income": 0.1}},
        },
    ),
    "heading": (
        ["actuators=deploy_pd"],
        {
            "command": {"resample_steps": 20, "pure_wz_prob": 0.3, "straight_prob": 0.3},
            "obs": {"state": [*joystick.default_config().obs.state, "heading_err"]},
        },
    ),
}
ACTION_SEEDS = {
    "stock": 11,
    "recipe": 12,
    "recipe_ramp_straight_lift": 13,
    "no_progress_symmetry_push": 14,
    "heading": 15,
}
# Steps of a sabotage replay. The first timeout is at step 50.
SABOTAGE_STEPS = 150

_CASES = {}


def _actions(action_size, seed) -> np.ndarray:
    """float32 (STEPS, N, action_size): held random actions plus a per-step jitter."""
    held = replay.random_actions(-(-STEPS // HOLD), N, action_size, AMPLITUDES, seed)
    jitter = replay.random_actions(STEPS, N, action_size, AMPLITUDES, seed + 100) * JITTER
    return np.clip(np.repeat(held, HOLD, axis=0)[:STEPS] + jitter, -1.0, 1.0).astype(np.float32)


def _build(name, device):
    overrides, env_overrides = CONFIGS[name]
    hydra = replay.hydra_dict(overrides, env=env_overrides, domain_rand=False)
    env = replay.jax_env(hydra)
    actions = _actions(env.action_size, ACTION_SEEDS[name])
    record = replay.record_jax(env, actions, L, seed=ACTION_SEEDS[name])
    run = replay.replay_torch(record, hydra, device, L)
    return types.SimpleNamespace(
        name=name, hydra=hydra, env=env, actions=actions, record=record, run=run, device=device
    )


@pytest.fixture(scope="module")
def device(parity_device):
    return replay.torch_device(parity_device)


@pytest.fixture(scope="module")
def cases(device):
    """name -> the cached case. The record is built before any test body runs."""

    def get(name):
        if name not in _CASES:
            _CASES[name] = _build(name, device)
        return _CASES[name]

    yield get
    _CASES.clear()


@pytest.fixture(scope="module", params=list(CONFIGS))
def case(request, cases):
    return cases(request.param)


# -- the lockstep comparison ----------------------------------------------------------
def assert_lockstep(name, record, run):
    """Raise at the first mismatch between the JAX record and the Torch run, in time order."""
    check = f"env[{name}]"

    def cmp(key, expected, actual, where, angle=False):
        replay.compare(check, key, expected, actual, where=f"{name} {where} {key}", angle=angle)

    def cmp_info(jax_info, torch_info, where):
        assert set(torch_info) == set(jax_info), (
            f"{check} {name} {where} info keys: only JAX {sorted(set(jax_info) - set(torch_info))}, "
            f"only Torch {sorted(set(torch_info) - set(jax_info))}"
        )
        for k in sorted(jax_info):
            cmp(k, jax_info[k], torch_info[k], where, angle=k in replay.ANGLE_KEYS)

    # Reset.
    cmp("state", record.reset_obs["state"], run.reset_state, "reset")
    cmp("privileged_state", record.reset_obs["privileged_state"], run.reset_priv, "reset")
    cmp_info(replay.task_info(record.reset_info), run.reset_info, "reset")
    for f in ("qpos", "qvel", "ctrl"):
        cmp(f"reset_write.{f}", record.reset_data[f], run.reset_write[f], "reset")

    jax_terms = {k.removeprefix("reward/") for k in record.steps[0].metrics if k.startswith("reward/")}
    for t, (st, ts) in enumerate(zip(record.steps, run.steps, strict=False)):
        where = f"step {t}"
        cmp("done", st.done, ts.done, where)
        cmp("truncation", st.truncation, ts.truncation, where)
        cmp("reward", st.reward, ts.reward, where)

        assert list(ts.terms) == run.env.term_names, f"{check} {name} {where} terms: order differs from term_names"
        assert set(ts.terms) == jax_terms, (
            f"{check} {name} {where} terms: only JAX {sorted(jax_terms - set(ts.terms))}, "
            f"only Torch {sorted(set(ts.terms) - jax_terms)}"
        )
        for k in ts.terms:
            cmp(f"terms.{k}", st.metrics[f"reward/{k}"], ts.terms[k], where)

        # Done worlds carry the first obs on both sides.
        cmp("state", st.obs["state"], ts.state, where)
        cmp("privileged_state", st.obs["privileged_state"], ts.priv, where)

        cmp_info(st.info, ts.info, where)

        # mjx_env.step writes ctrl into the post-step data.
        cmp("ctrl", st.physics["ctrl"], ts.step_log["ctrl"], where)

        # WarpJoystick.step restores the done worlds on every step.
        restore = ts.restore_log
        assert restore is not None, f"{check} {name} {where} restore: WarpJoystick made no restore write"
        cmp("restore.mask", st.done, restore["mask"], where)
        m = st.done
        if m.any():
            for f in ("qpos", "qvel", "ctrl"):
                # BraxAutoResetWrapper's first data.
                cmp(f"restore.{f}", record.reset_data[f][m], restore[f][m], where)


def test_reset_and_every_step_match_jax(case):
    assert len(case.run.steps) == STEPS
    assert_lockstep(case.name, case.record, case.run)


# -- coverage ---------------------------------------------------------------------------
def _stack(record, key):
    return np.stack([st.info[key] for st in record.steps])


def test_the_scenario_covers_its_events(case):
    record, name = case.record, case.name
    done = np.stack([st.done for st in record.steps])
    trunc = np.stack([st.truncation for st in record.steps])
    assert (done & ~trunc).any(), f"{name}: no fall"
    assert trunc.any(), f"{name}: no timeout"

    since = _stack(record, "steps_since_cmd")
    assert ((since[1:] == 0) & ~done[1:]).any(), f"{name}: no command resample on a live world"

    if name in ("recipe", "recipe_ramp_straight_lift"):
        land = _stack(record, "land_frac")
        assert (land != 0).any(), f"{name}: land_frac is never nonzero"
        assert done.any()
        for t in range(STEPS):
            assert (land[t][done[t]] == 0).all(), f"{name} step {t}: a done world keeps its land_frac"
        yaw_ref = _stack(record, "yaw_ref")
        assert (yaw_ref != yaw_ref[0]).any(), f"{name}: yaw_ref never changes"

    if name == "recipe_ramp_straight_lift":
        command, target = _stack(record, "command"), _stack(record, "cmd_target")
        assert (command != target).any(), f"{name}: the ramp never lags the target"
        straight = (target[..., 1] == 0) & (target[..., 2] == 0) & (target[..., 0] != 0)
        assert straight.any(), f"{name}: no straight command"

    if name == "no_progress_symmetry_push":
        cuts = sum(float(st.metrics["no_progress_cut"].sum()) for st in record.steps)
        assert cuts > 0, f"{name}: the no-progress cut never fires"
        mirror = _stack(record, "mirror")
        assert mirror.any() and not mirror.all(), f"{name}: mirror holds one value only"

    if name == "heading":
        yaw_ref = _stack(record, "yaw_ref")
        live = ~done[1:]
        moved = yaw_ref[1:] != yaw_ref[:-1]
        assert (moved & live).any(), f"{name}: yaw_ref never moves on a live world"
        assert (~moved & live).any(), f"{name}: yaw_ref never holds on a live world"
        assert done.any(), f"{name}: no done world"


# -- the kick ---------------------------------------------------------------------------
def test_kicks_match_the_jax_kicks(cases):
    c = cases("no_progress_symmetry_push")
    env, record, run = c.env, c.record, c.run
    check = "env[no_progress_symmetry_push]"
    step_draws = jax.jit(jax.vmap(lambda k: joystick.jax_step_draws(k, env.config)))
    nv = env.mj_model.nv
    bv = int(env._consts.base_vadr)
    kicked = False
    for t, (st, ts) in enumerate(zip(record.steps, run.steps, strict=True)):
        where = f"no_progress_symmetry_push step {t}"
        info_prev = record.reset_info if t == 0 else record.steps[t - 1].info
        info_prev = {k: jp.asarray(v) for k, v in info_prev.items()}
        draws = step_draws(jp.asarray(st.rng_before))
        _, _, ctrl_j, kick_j = gait_state.pre_physics(
            info_prev, jp.asarray(st.action), env._consts, env.config, draws["push"]
        )
        # The oracle must reproduce the ctrl the JAX step wrote.
        replay.compare(check, "kick.ctrl", st.physics["ctrl"], np.asarray(ctrl_j), where=f"{where} kick.ctrl")

        expected = np.zeros((N, nv), dtype=np.float32)
        expected[:, bv : bv + 2] = np.asarray(kick_j["planar"])
        if kick_j["z"] is not None:
            expected[:, bv + 2] = np.asarray(kick_j["z"])
        if kick_j["ang"] is not None:
            expected[:, bv + 3 : bv + 6] = np.asarray(kick_j["ang"])
        got = ts.step_log["qvel_in"] - ts.step_log["qvel_ref"]
        replay.compare(check, "kick", expected, got, where=f"{where} kick", tol=TASK)
        kicked = kicked or bool((expected != 0).any())
    assert kicked, "no step had a nonzero kick"


# -- sabotage ---------------------------------------------------------------------------
@pytest.fixture
def sabotage_report():
    """Keep the sabotaged errors out of the parity report."""
    saved = copy.deepcopy(parity_report.RESULTS)
    yield
    parity_report.RESULTS.clear()
    parity_report.RESULTS.update(saved)


def test_the_lockstep_catches_a_skipped_respawn_clear(cases, monkeypatch, sabotage_report):
    c = cases("heading")
    monkeypatch.setattr(gait_state, "clear_done", lambda info, done, flags, values: dict(info))
    run = replay.replay_torch(c.record, c.hydra, c.device, L, steps=SABOTAGE_STEPS)
    with pytest.raises(AssertionError, match="yaw_ref"):
        assert_lockstep(c.name, c.record, run)


def test_the_lockstep_catches_an_off_by_one_episode_counter(cases, sabotage_report):
    c = cases("stock")

    def hook(tenv):
        tenv.episode_length = L + 1

    run = replay.replay_torch(c.record, c.hydra, c.device, L, steps=SABOTAGE_STEPS, hook=hook)
    with pytest.raises(AssertionError, match="done|truncation"):
        assert_lockstep(c.name, c.record, run)


def test_the_lockstep_catches_torch_only_reward_drift(cases, monkeypatch, sabotage_report):
    c = cases("stock")
    total_reward = compose.total_reward
    monkeypatch.setattr(compose, "total_reward", lambda *a, **kw: total_reward(*a, **kw) * 1.001)
    run = replay.replay_torch(c.record, c.hydra, c.device, L, steps=SABOTAGE_STEPS)
    with pytest.raises(AssertionError, match="reward"):
        assert_lockstep(c.name, c.record, run)
