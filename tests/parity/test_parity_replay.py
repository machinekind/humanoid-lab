"""The replay helpers the lockstep, physics, DR, RNG and closed-loop checks share.

The JAX recorder must reproduce the JAX key chain and keep the physics from
before the respawn. ReplayDraws must give the JAX adapters' draws in
TorchDraws' structure. ReplayPhysics and ReplayDraws must cover every name
torch_warp/env.py uses. The full lockstep comparison is test_parity_env.py.
"""

from __future__ import annotations

import inspect
import re

import jax
import numpy as np
import pytest

from humanoid_lab import paths, registry
from humanoid_lab.dr import randomize, transform
from humanoid_lab.envs import joystick, progress

torch = pytest.importorskip("torch")

import replay

from humanoid_lab.torch_warp.env import TorchDraws

N = 4
STEPS = 12
EPISODE_LENGTH = 5
ACTION_SIZE = 6
ACTOR_OBS_SIZE = 20
ALL_ON_DR = {
    "com_offset": {"enable": True},
    "joint_gains": {"enable": True},
    "dof": {"enable": True},
    "foot_friction": {"enable": True},
    "motor_strength": {"enable": True},
    "base_mass_add": {"enable": True},
}
DR_DIMS = {"nbody": 9, "nu": 4, "nv": 10, "nfoot": 2}


@pytest.fixture(scope="module")
def device(parity_device):
    return replay.torch_device(parity_device)


@pytest.fixture(scope="module")
def stock():
    hydra = replay.hydra_dict(["actuators=deploy_pd"], env={"command": {"resample_steps": 4}}, domain_rand=False)
    env = replay.jax_env(hydra)
    actions = replay.random_actions(STEPS, N, env.action_size, [1.0, 1.0, 0.05, 0.05], seed=1)
    record = replay.record_jax(env, actions, episode_length=EPISODE_LENGTH)
    return hydra, env, record


def _all_on_cfg():
    cfg = joystick.default_config()
    registry._apply_overrides(
        cfg,
        {
            "push": {"interval_steps_range": (5, 15), "vel_z": 0.3, "ang_vel_rp": 0.5, "ang_vel_yaw": 0.7},
            "symmetry": {"enable": True},
            "no_progress": {"enable": True},
            "command": {prob_key: 0.2 for _, _, prob_key, _ in progress.COMMAND_DRAWS},
        },
    )
    return cfg


def _all_off_cfg():
    cfg = joystick.default_config()
    registry._apply_overrides(cfg, {"push": {"enable": False}})
    assert not cfg.symmetry.enable and not cfg.no_progress.enable
    return cfg


def _dr_table():
    return transform.DR_DRAWS(transform.resolved_cfg(ALL_ON_DR), DR_DIMS)


def _assert_equal_tree(expected, actual, where):
    """expected is eager JAX, actual is ReplayDraws output as numpy."""
    if expected is None or actual is None:
        assert expected is None and actual is None, where
        return
    if isinstance(expected, dict):
        assert set(expected) == set(actual), f"{where}: {sorted(expected)} != {sorted(actual)}"
        for k in expected:
            _assert_equal_tree(expected[k], actual[k], f"{where}.{k}")
        return
    e = np.asarray(expected)
    assert e.dtype == actual.dtype, f"{where}: {e.dtype} != {actual.dtype}"
    np.testing.assert_array_equal(actual, e, err_msg=where)


def _structure(tree):
    """Nested keys, per-world shapes, dtypes and None positions."""
    if tree is None:
        return None
    if isinstance(tree, dict):
        return {k: _structure(v) for k, v in tree.items()}
    return (tuple(tree.shape[1:]), str(tree.dtype))


# -- the recorder ----------------------------------------------------------------
def test_the_recorder_reproduces_the_jax_key_chain(stock):
    _, env, record = stock
    next_rng = jax.jit(jax.vmap(lambda k: joystick.jax_step_draws(k, env.config)["rng"]))
    for t in range(STEPS - 1):
        got = np.asarray(next_rng(record.steps[t].rng_before))
        np.testing.assert_array_equal(got, record.steps[t + 1].rng_before, err_msg=f"step {t}")


def test_the_physics_frame_is_the_data_before_the_respawn(stock):
    _, _, record = stock
    assert any(st.done.any() for st in record.steps), "no done world in the recording"
    respawn_moved = False
    for t, st in enumerate(record.steps):
        live = ~st.done
        for f in replay.FRAME_FIELDS:
            np.testing.assert_array_equal(st.physics[f][live], st.data[f][live], err_msg=f"step {t} {f}")
        if st.done.any():
            np.testing.assert_array_equal(st.data["qpos"][st.done], record.reset_data["qpos"][st.done])
            respawn_moved |= bool((st.physics["qpos"][st.done] != st.data["qpos"][st.done]).any())
    assert respawn_moved, "no done world has a pre-respawn qpos that differs from its first qpos"


# -- the replay draws ------------------------------------------------------------------
def test_replay_draws_match_the_jax_adapters(device):
    cfg = _all_on_cfg()
    keys = jax.random.split(jax.random.PRNGKey(5), N)
    draws = replay.ReplayDraws(device, dr_keys=keys)

    expect = dict(jax.vmap(lambda k: joystick.jax_reset_draws(k, cfg, ACTION_SIZE))(keys))
    expect.pop("rng")
    draws.set_reset_keys(keys)
    got = replay.to_numpy(draws.reset(N, cfg, ACTION_SIZE))
    assert "push_countdown" in got and "mirror_u" in got
    _assert_equal_tree(expect, got, "reset")

    eager = jax.vmap(lambda k: joystick.jax_step_draws(k, cfg))(keys)
    noise = jax.vmap(lambda k: jax.random.uniform(k, (ACTOR_OBS_SIZE,), minval=-1.0, maxval=1.0))(eager["noise_key"])
    expect = {"noise": noise, "command": eager["command"], "push": eager["push"], "u_term": eager["u_term"]}
    draws.set_step_keys(keys)
    got = replay.to_numpy(draws.step(N, cfg, ACTOR_OBS_SIZE))
    assert got["push"] is not None and got["u_term"] is not None
    _assert_equal_tree(expect, got, "step")

    table = _dr_table()
    expect = jax.vmap(lambda k: randomize.jax_dr_draws(k, table))(keys)
    _assert_equal_tree(expect, replay.to_numpy(draws.dr(N, table)), "dr")


def test_replay_draws_keys_are_one_shot(device):
    cfg = _all_off_cfg()
    keys = jax.random.split(jax.random.PRNGKey(5), N)
    draws = replay.ReplayDraws(device)
    draws.set_step_keys(keys)
    draws.step(N, cfg, ACTOR_OBS_SIZE)
    with pytest.raises(AssertionError, match="set_step_keys"):
        draws.step(N, cfg, ACTOR_OBS_SIZE)
    with pytest.raises(AssertionError, match="set_reset_keys"):
        draws.reset(N, cfg, ACTION_SIZE)


@pytest.mark.parametrize("make_cfg", [_all_on_cfg, _all_off_cfg], ids=["all_on", "all_off"])
def test_replay_draws_have_the_torch_draws_structure(device, make_cfg):
    cfg = make_cfg()
    keys = jax.random.split(jax.random.PRNGKey(5), N)
    rd = replay.ReplayDraws(device, dr_keys=keys)
    td = TorchDraws(0, device)

    rd.set_reset_keys(keys)
    assert _structure(rd.reset(N, cfg, ACTION_SIZE)) == _structure(td.reset(N, cfg, ACTION_SIZE))
    rd.set_step_keys(keys)
    got, want = rd.step(N, cfg, ACTOR_OBS_SIZE), td.step(N, cfg, ACTOR_OBS_SIZE)
    assert _structure(got) == _structure(want)
    for t in (got, want):
        assert (t["push"] is None) == (not cfg.push.enable)
        assert (t["u_term"] is None) == (not cfg.no_progress.enable)
    table = _dr_table()
    assert _structure(rd.dr(N, table)) == _structure(td.dr(N, table))


def test_replay_draws_cover_every_jax_draw_adapter():
    found = set()
    for module in (joystick, randomize):
        for name, obj in vars(module).items():
            if re.fullmatch(r"jax_\w+_draws", name) and callable(obj) and inspect.getmodule(obj) is module:
                found.add(name)
    assert found == set(replay.ReplayDraws.ADAPTERS)


# -- the replay physics ----------------------------------------------------------------
def test_replay_objects_cover_the_env_interface(stock, device):
    _, env, _ = stock
    text = (paths.REPO_ROOT / "src" / "humanoid_lab" / "torch_warp" / "env.py").read_text(encoding="utf-8")
    physics = replay.ReplayPhysics(env.mj_model, N, device)
    draws = replay.ReplayDraws(device)
    used = {"physics": set(re.findall(r"\bphysics\.(\w+)", text)), "draws": set(re.findall(r"\bdraws\.(\w+)", text))}
    assert {"data", "step", "reset_worlds", "restore_worlds"} <= used["physics"]
    assert {"reset", "step"} <= used["draws"]
    for label, obj in (("physics", physics), ("draws", draws)):
        for name in sorted(used[label]):
            if name.startswith("__"):
                continue
            assert hasattr(obj, name), f"env.py uses {label}.{name}, which {type(obj).__name__} lacks"


def test_replay_physics_loads_frames_in_place_and_logs_writes(stock, device):
    _, env, record = stock
    physics = replay.ReplayPhysics(env.mj_model, N, device)
    ids = {f: id(getattr(physics.data, f)) for f in vars(physics.data)}
    nv, nu = env.mj_model.nv, env.mj_model.nu
    zeros_q = torch.zeros(N, env.mj_model.nq, device=device)

    physics.queue(record.reset_data)
    physics.queue(record.steps[0].physics)
    physics.reset_worlds(zeros_q, torch.zeros(N, nv, device=device), torch.zeros(N, nu, device=device))
    np.testing.assert_array_equal(physics.data.qpos.cpu().numpy(), record.reset_data["qpos"])
    loaded_qvel = physics.data.qvel.clone()

    kick = torch.zeros(N, nv, device=device)
    kick[:, 0] = 0.25
    physics.data.qvel += kick
    ctrl = torch.full((N, nu), 0.5, device=device)
    physics.step(ctrl)
    np.testing.assert_array_equal(physics.data.qpos.cpu().numpy(), record.steps[0].physics["qpos"])
    assert {f: id(getattr(physics.data, f)) for f in vars(physics.data)} == ids

    log = physics.pop_log()
    assert [k for k, _ in log] == ["reset", "step"]
    write = log[1][1]
    np.testing.assert_array_equal(write["ctrl"], ctrl.cpu().numpy())
    np.testing.assert_array_equal(write["qvel_ref"], loaded_qvel.cpu().numpy())
    np.testing.assert_allclose(write["qvel_in"] - write["qvel_ref"], kick.cpu().numpy(), rtol=0, atol=1e-6)
    assert physics.pop_log() == []

    before = {f: getattr(physics.data, f).clone() for f in ("qpos", "qvel", "ctrl")}
    mask = torch.tensor([True, False, False, True], device=device)
    first_q = torch.full((N, env.mj_model.nq), 7.0, device=device)
    physics.restore_worlds(mask, first_q, torch.zeros(N, nv, device=device), torch.zeros(N, nu, device=device))
    q = physics.data.qpos
    assert (q[mask] == 7.0).all()
    assert torch.equal(q[~mask], before["qpos"][~mask])
    assert torch.equal(physics.data.qvel[~mask], before["qvel"][~mask])
    assert torch.equal(physics.data.ctrl[~mask], before["ctrl"][~mask])
    assert (physics.data.qacc_warmstart[mask] == 0).all() and (physics.data.time[mask] == 0).all()
    assert [k for k, _ in physics.pop_log()] == ["restore"]

    with pytest.raises(AssertionError, match="no JAX frame queued"):
        physics.step(ctrl)


# -- the Torch replay -------------------------------------------------------------------
def test_a_warp_joystick_runs_on_replay(stock, device):
    hydra, _, record = stock
    run = replay.replay_torch(record, hydra, device, EPISODE_LENGTH)
    assert len(run.steps) == STEPS
    s, p = run.env.actor_obs_size, run.env.privileged_obs_size
    for t, (ts, js) in enumerate(zip(run.steps, record.steps, strict=True)):
        assert ts.state.dtype == np.float32 and ts.state.shape == (N, s), t
        assert ts.priv.dtype == np.float32 and ts.priv.shape == (N, p), t
        assert ts.reward.shape == (N,), t
        assert ts.done.dtype == np.bool_, t
        np.testing.assert_array_equal(ts.done, js.done, err_msg=f"step {t}")
        assert ts.step_log is not None and ts.restore_log is not None, t
