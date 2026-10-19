"""WarpJoystick glue on the CPU warp device: roboto_origin deploy_pd, N=4.

This file checks the env's own pieces: the draw dict structure, the done
and truncation flags, and the DR refusal. test_parity_env.py checks
lockstep parity with the JAX env, test_parity_physics.py checks physics
parity and test_parity_rng.py checks the RNG statistics.
"""

from __future__ import annotations

import jax
import pytest

from humanoid_lab.dr import randomize, transform
from humanoid_lab.envs.joystick import jax_reset_draws, jax_step_draws

torch = pytest.importorskip("torch")

from humanoid_lab.torch_warp import env as warp_env

N = 4
CPU = torch.device("cpu")
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


def test_dr_off_with_an_enabled_field_is_refused():
    with pytest.raises(ValueError, match="domain_rand=false"):
        _make(dr={"foot_friction": {"enable": True}})
