"""WarpJoystick glue on the CPU warp device: roboto_origin deploy_pd, N=4.

This file checks the env's own pieces: the obs sizes, the step contract,
the constants conversion, the draw dict structure, the episode counter, the
respawn and the DR install. Lockstep parity with the JAX env, physics parity
and the RNG statistics are commit 5's tests.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest
from test_parity_task_math import torch_consts

from humanoid_lab.dr import randomize, transform
from humanoid_lab.envs import gait_state
from humanoid_lab.envs.joystick import default_config, jax_reset_draws, jax_step_draws

torch = pytest.importorskip("torch")

from humanoid_lab.torch_warp import env as warp_env
from humanoid_lab.torch_warp.physics import WarpPhysics, wp

N = 4
CPU = torch.device("cpu")
HEADING_STATE = [*default_config().obs.state, "heading_err"]
RESPAWN_ENV = {
    "obs": {"state": HEADING_STATE},
    "symmetry": {"enable": True},
    "reward": {"scales": {"gait_symmetry_income": 0.1}},
}
ALL_PROBS = {
    "pure_wz_prob": 0.1,
    "pure_vy_prob": 0.1,
    "pure_slow_prob": 0.1,
    "pure_fast_prob": 0.1,
    "pure_back_prob": 0.1,
    "straight_prob": 0.1,
}
ALL_GATES = {
    "command": ALL_PROBS,
    "push": {"enable": True, "interval_steps_range": [5, 15], "vel_z": 0.3, "ang_vel_rp": 0.5, "ang_vel_yaw": 0.7},
    "symmetry": {"enable": True},
    "no_progress": {"enable": True},
}


def _all_dr(joint_gains):
    return {
        "com_offset": {"enable": True},
        "joint_gains": {"enable": joint_gains},
        "dof": {"enable": True},
        "foot_friction": {"enable": True},
        "motor_strength": {"enable": True},
        "base_mass_add": {"enable": True},
    }


def _hydra(env=None, domain_rand=False, dr=None):
    return {
        "task": {"name": "joystick", "env": env or {}},
        "robot": {"dir": "robots/roboto_origin"},
        "actuators": {"name": "deploy_pd", "overrides": {}},
        "domain_rand": domain_rand,
        "dr": dr or {},
    }


def _make(env=None, episode_length=1000, **kw):
    hydra = _hydra(env, kw.pop("domain_rand", False), kw.pop("dr", None))
    return warp_env.WarpJoystick(hydra, N, episode_length, device="cpu", **kw)


def _zeros(env):
    return torch.zeros(N, env.action_size)


@pytest.fixture(scope="module")
def stock():
    return _make(episode_length=3)


@pytest.fixture(scope="module")
def respawn_env():
    return _make(RESPAWN_ENV, episode_length=50)


def test_obs_sizes_match_the_jax_env(stock, respawn_env):
    for env in (stock, respawn_env):
        assert env.jenv.observation_size == {
            "state": (env.actor_obs_size,),
            "privileged_state": (env.privileged_obs_size,),
        }


def test_reset_and_step_follow_the_contract(stock):
    state, priv = stock.reset()
    for t, size in ((state, stock.actor_obs_size), (priv, stock.privileged_obs_size)):
        assert t.dtype == torch.float32 and tuple(t.shape) == (N, size)
    state, priv, reward, done, truncation, terms = stock.step(_zeros(stock))
    assert state.dtype == torch.float32 and tuple(state.shape) == (N, stock.actor_obs_size)
    assert priv.dtype == torch.float32 and tuple(priv.shape) == (N, stock.privileged_obs_size)
    assert reward.dtype == torch.float32 and tuple(reward.shape) == (N,)
    for t in (done, truncation):
        assert t.dtype == torch.bool and tuple(t.shape) == (N,)
    assert list(terms) == stock.term_names
    for name, v in terms.items():
        assert v.dtype == torch.float32 and tuple(v.shape) == (N,), name
    assert torch.isfinite(reward).all()


def test_term_names_match_the_reward_dict(stock, respawn_env):
    for env in (stock, respawn_env):
        env.reset()
        terms = env.step(_zeros(env))[-1]
        assert list(terms) == env.term_names
    assert "gait_symmetry_income" in respawn_env.term_names
    assert "gait_symmetry_income" not in stock.term_names


def test_torch_constants_match_the_parity_conversion(stock):
    jenv = stock.jenv
    ours = warp_env.torch_constants(jenv.resolved_constants(), "cpu")
    ref = torch_consts(jenv._consts)
    for f in ref.__dataclass_fields__:
        if f == "floor_geom_id":
            # resolved_constants() resolves it. The JAX record leaves it None.
            assert ours.floor_geom_id == randomize._find_floor_geom_id(jenv.mj_model, None)
            continue
        a, b = getattr(ours, f), getattr(ref, f)
        if isinstance(b, torch.Tensor):
            assert isinstance(a, torch.Tensor), f
            assert a.device == CPU, f
            if b.dtype.is_floating_point:
                assert a.dtype == torch.float32, f
            elif b.dtype == torch.bool:
                assert a.dtype == torch.bool, f
            else:
                assert a.dtype == torch.int64, f
            np.testing.assert_array_equal(a.numpy(), b.numpy(), err_msg=f)
        elif f == "actuator_model":
            assert a is b
        else:
            assert a == b, f


def _structure(x):
    """Keys, per-world shapes and dtypes, with the leading world axis kept."""
    if x is None:
        return None
    if isinstance(x, dict):
        return {k: _structure(v) for k, v in x.items()}
    return tuple(x.shape), str(x.dtype).replace("torch.", "")


def _drop(d, *keys):
    return {k: v for k, v in d.items() if k not in keys}


def test_draw_dicts_match_the_jax_adapters():
    cfg = warp_env.env_config_of(_hydra(ALL_GATES))
    nu, actor = 12, 50
    keys = jax.random.split(jax.random.PRNGKey(0), N)
    draws = warp_env.TorchDraws(0, CPU)

    jax_reset = jax.vmap(lambda k: jax_reset_draws(k, cfg, nu))(keys)
    torch_reset = draws.reset(N, cfg, nu)
    assert _structure(torch_reset) == _structure(_drop(jax_reset, "rng"))
    assert set(torch_reset["command"]) >= {f"{n}_on" for n in ("pure_wz", "straight")}
    assert "push_countdown" in torch_reset and "mirror_u" in torch_reset

    jax_step = jax.vmap(lambda k: jax_step_draws(k, cfg))(keys)
    torch_step = draws.step(N, cfg, actor)
    assert _structure(_drop(torch_step, "noise")) == _structure(_drop(jax_step, "rng", "noise_key"))
    assert _structure(torch_step["noise"]) == ((N, actor), "float32")
    assert set(torch_step["push"]) == {"dir", "next_countdown", "vel_z", "ang"}

    dims = {"nbody": 9, "nu": nu, "nv": nu + 6, "nfoot": 2}
    for joint_gains in (True, False):
        table = transform.DR_DRAWS(transform.resolved_cfg(_all_dr(joint_gains)), dims)
        jax_dr = jax.vmap(lambda k, t=table: randomize.jax_dr_draws(k, t))(keys)
        assert _structure(draws.dr(N, table)) == _structure(jax_dr)


def test_timeouts_and_falls_set_done_and_truncation(stock):
    stock.reset()
    for i in range(3):
        _, _, _, done, truncation, _ = stock.step(_zeros(stock))
        assert bool(done.all()) == (i == 2)
    assert truncation.all()

    # A fall on the timeout step is done without truncation.
    stock.reset()
    for _ in range(2):
        stock.step(_zeros(stock))
    stock.physics.data.qpos[0, stock.consts.base_qadr + 2] = 0.05
    _, _, _, done, truncation, _ = stock.step(_zeros(stock))
    assert done.all()
    assert truncation.tolist() == [False, True, True, True]

    stock.reset()
    stock.physics.data.qpos[0, stock.consts.base_qadr + 2] = 0.05
    _, _, _, done, truncation, _ = stock.step(_zeros(stock))
    assert done.tolist() == [True, False, False, False]
    assert not truncation.any()


def test_the_counter_restarts_after_done(stock):
    stock.reset()
    for _ in range(3):
        stock.step(_zeros(stock))
    assert stock.steps.dtype == torch.float32
    assert stock.steps.tolist() == [3.0] * N and stock.done.all()
    _, _, _, done, _, _ = stock.step(_zeros(stock))
    assert stock.steps.tolist() == [1.0] * N
    assert not done.any()


def test_a_respawn_restores_the_first_state_and_clears_the_layers(respawn_env):
    env = respawn_env
    assert env.flags == ("gait", "heading")
    state0, priv0 = env.reset()
    first = env.first
    command0 = env.info["command"].clone()
    g = torch.Generator().manual_seed(0)
    action = torch.rand(N, env.action_size, generator=g) * 2 - 1
    env.step(action)
    assert (env.info["feet_contact_time"] > 0).any(dim=-1).all()

    # World 0 falls on the next step. The others keep walking.
    env.physics.data.qpos[0, env.consts.base_qadr + 2] = 0.05
    state, priv, _, done, _, _ = env.step(action)
    assert done.tolist() == [True, False, False, False]

    torch.testing.assert_close(env.physics.data.qpos[0], first["qpos"][0], rtol=0, atol=0)
    assert not torch.equal(env.physics.data.qpos[1], first["qpos"][1])
    torch.testing.assert_close(state[0], state0[0], rtol=0, atol=0)
    torch.testing.assert_close(priv[0], priv0[0], rtol=0, atol=0)
    assert not torch.equal(state[1], state0[1])
    # The respawn keeps the command.
    torch.testing.assert_close(env.info["command"], command0, rtol=0, atol=0)
    torch.testing.assert_close(env.info["yaw_ref"][0], first["yaw"][0], rtol=0, atol=0)
    for k in gait_state.GAIT_TRACKER_KEYS:
        assert not env.info[k][0].any(), k
    assert env.info["feet_contact_time"][1:].any(dim=-1).all()


def test_info_holds_no_physics_view(respawn_env):
    env = respawn_env
    env.reset()
    env.step(_zeros(env))
    torque = env.info["last_torque"].clone()
    assert env.info["last_torque"].data_ptr() != env.physics.data.actuator_force.data_ptr()
    env.step(_zeros(env))
    torch.testing.assert_close(env.info["last_last_action"], _zeros(env))
    assert not torch.equal(env.info["last_torque"], torque)


def test_dr_is_per_world_and_leaves_the_jax_model_alone():
    env = _make(domain_rand=True, dr={"foot_friction": {"enable": True}})
    mass = wp.to_torch(env.physics.m.body_mass)
    assert tuple(mass.shape) == (N, env.mj_model.nbody)
    for i in range(1, N):
        assert not torch.equal(mass[0], mass[i])
    feet = np.asarray(env.consts_np.foot_geom_ids)
    assert env.physics.mj_model is not env.jenv.mj_model
    assert (env.physics.mj_model.geom_priority[feet] == 1).all()
    # MuJoCo's default priority is 0. The JAX env's model keeps it.
    assert (env.jenv.mj_model.geom_priority[feet] == 0).all()
    env.reset()
    assert torch.isfinite(env.step(_zeros(env))[2]).all()


def test_dr_off_with_an_enabled_field_is_refused():
    with pytest.raises(ValueError, match="domain_rand=false"):
        _make(dr={"foot_friction": {"enable": True}})


class _NoCutDraws(warp_env.TorchDraws):
    calls = 0

    def step(self, n, cfg, actor_obs_size):
        d = super().step(n, cfg, actor_obs_size)
        d["u_term"] = torch.ones(n, device=self.device)
        self.calls += 1
        return d


def test_draws_and_physics_are_injectable(stock):
    c = stock.consts_np
    phys = WarpPhysics(stock.mj_model, N, CPU, c.naconmax_per_env, c.njmax, stock.n_substeps)
    draws = _NoCutDraws(0, CPU)
    env = _make({"no_progress": {"enable": True}}, physics=phys, draws=draws)
    assert env.physics is phys and env.draws is draws
    env.reset()
    for _ in range(3):
        env.step(_zeros(env))
    assert draws.calls == 3


def test_streams_are_seeded_per_key():
    keys = [row[:3] for row in warp_env.STREAMS]
    assert len(set(keys)) == len(keys)
    draws = warp_env.TorchDraws(0, CPU)
    seeds = [draws._seed(k) for k in keys]
    assert len(set(seeds)) == len(seeds)
    assert all(0 <= s < 2**63 for s in seeds)

    # Every stream an all-gates run opens is a documented one.
    cfg = warp_env.env_config_of(_hydra(ALL_GATES))
    draws.reset(N, cfg, 12)
    draws.step(N, cfg, 50)
    dims = {"nbody": 9, "nu": 12, "nv": 18, "nfoot": 2}
    for joint_gains in (True, False):
        draws.dr(N, transform.DR_DRAWS(transform.resolved_cfg(_all_dr(joint_gains)), dims))
    assert set(draws._gens) == set(keys)


def test_the_draw_state_round_trips():
    cfg = warp_env.env_config_of(_hydra(ALL_GATES))
    draws = warp_env.TorchDraws(3, CPU)
    draws.step(N, cfg, 50)
    saved = draws.state_dict()
    want = draws.step(N, cfg, 50)
    other = warp_env.TorchDraws(3, CPU)
    other.load_state_dict(saved)
    got = other.step(N, cfg, 50)
    torch.testing.assert_close(got["noise"], want["noise"], rtol=0, atol=0)
    torch.testing.assert_close(got["push"]["next_countdown"], want["push"]["next_countdown"], rtol=0, atol=0)
