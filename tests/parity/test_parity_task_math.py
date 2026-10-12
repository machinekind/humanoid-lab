"""Parity of the shared task math: JAX arrays against Torch tensors.

Every array function in the shared modules runs twice on the same values:
once on JAX arrays and once on Torch tensors on the --parity-device
(cpu by default). The shared modules are
envs/{signals,progress,gait_state,push,symmetry}, rewards/{terms,compose}
and dr/transform. Float outputs must agree within tolerances.TASK, and every
float Torch output must be float32. Bool and int outputs must match exactly,
dtype included.

The inputs come from one JAX env with every optional gate on: symmetry, the
no-progress cut, heading_err, the ranged push with every knob, the four
style terms plus both distance terms, and the four landing and lean terms
with pair_min on. A jitted single-world rollout gives 8
snapshots of data and info, stacked into a batch of 8 worlds. The draws come
from the JAX draw adapters. A few inputs are set by hand so that both sides
of a branch occur in the batch. Each function gets identical inputs on both
sides: a function's input is never the other backend's output.

A failure here that comes from shared code is fixed in the shared module,
never in this file.
"""

from __future__ import annotations

import dataclasses
import inspect
import math
import types

import jax
import jax.numpy as jp
import numpy as np
import parity_report
import pytest
from tolerances import TASK

from humanoid_lab import paths
from humanoid_lab.dr import randomize, transform
from humanoid_lab.envs import constants, gait_state, progress, push, signals, symmetry
from humanoid_lab.envs.joystick import (
    default_config,
    jax_command_draws,
    jax_push_draws,
    jax_reset_draws,
    jax_step_draws,
)
from humanoid_lab.registry import make_env
from humanoid_lab.rewards import compose, terms

torch = pytest.importorskip("torch")

import replay

N = 8
STEPS = 40
SNAPSHOT_EVERY = STEPS // N

STYLE_SCALES = {
    "knee_stance": -0.1,
    "gait_symmetry": -0.1,
    "gait_symmetry_income": 0.1,
    "feet_apex_min": 0.1,
    "feet_distance": 0.1,
    "knee_distance": 0.1,
}

# Hand-set commands: forward, diagonal, a stand, a command inside the
# deadband, a pure turn and a turning walk.
COMMANDS = np.array(
    [
        [0.5, 0.0, 0.0],
        [0.3, 0.2, 0.0],
        [0.4, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.02, 0.01, 0.0],
        [0.0, 0.0, 0.8],
        [0.6, -0.2, 0.3],
        [0.0, 0.0, 0.0],
    ],
    np.float32,
)
FLAGS = np.array([True, False, True, False, False, True, True, False])

# Shared-module functions that take no data arrays: they build maps, masks,
# tables or flags from config and names. They have nothing to compare.
ARRAY_FREE = {
    "compose.land_frac_on",
    "gait_state.gait_dur_ema_on",
    "gait_state.reseed_flags",
    "progress.ramp_on",
    "push.push_draw_names",
    "push.push_ranges",
    "symmetry.axis_mirror_sign",
    "symmetry.joint_mirror",
    "symmetry.foot_mirror",
    "symmetry.component_mirror",
    "symmetry.obs_mirror",
    "symmetry.policy_mirror_maps",
    "transform.field_cfg",
    "transform.resolved_cfg",
    "transform.check_dr_enabled",
    "transform.DR_DRAWS",
    "transform.dr_masks",
}

SHARED_MODULES = {
    "constants": constants,
    "signals": signals,
    "progress": progress,
    "gait_state": gait_state,
    "push": push,
    "symmetry": symmetry,
    "terms": terms,
    "compose": compose,
    "transform": transform,
}


# -- conversion -----------------------------------------------------------


# The torch device of the Torch inputs. The world fixture sets it from
# --parity-device before it converts any input.
TORCH_DEVICE = {"device": torch.device("cpu")}


def to_torch(x):
    """Torch tensors of the same values on TORCH_DEVICE: float arrays become float32,
    int arrays keep their int dtype, bool arrays become torch.bool. Python
    scalars, strings and None pass through, and dicts and tuples convert
    leaf by leaf."""
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    if isinstance(x, dict):
        return {k: to_torch(v) for k, v in x.items()}
    if isinstance(x, tuple):
        return tuple(to_torch(v) for v in x)
    a = np.asarray(x)
    if a.dtype.kind == "f":
        return torch.tensor(a, dtype=torch.float32, device=TORCH_DEVICE["device"])
    if a.dtype.kind == "b":
        return torch.tensor(a, dtype=torch.bool, device=TORCH_DEVICE["device"])
    if a.dtype.kind == "i":
        return torch.tensor(a, device=TORCH_DEVICE["device"])
    raise TypeError(f"no torch conversion for dtype {a.dtype}")


def to_jax(x):
    """JAX arrays of numpy inputs, with the same pass-through rules."""
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    if isinstance(x, dict):
        return {k: to_jax(v) for k, v in x.items()}
    if isinstance(x, tuple):
        return tuple(to_jax(v) for v in x)
    return jp.asarray(x)


def torch_consts(consts):
    """env._consts with every array field converted. Python scalars, dicts,
    tuples and the actuator model are unchanged."""
    arrays = {
        f.name: to_torch(getattr(consts, f.name))
        for f in dataclasses.fields(consts)
        if isinstance(getattr(consts, f.name), (np.ndarray, jax.Array))
    }
    return dataclasses.replace(consts, **arrays)


def assert_parity(j, t, where="out"):
    if j is None:
        assert t is None, f"{where}: JAX gives None, Torch gives {type(t).__name__}"
        return
    if isinstance(j, dict):
        assert isinstance(t, dict), f"{where}: Torch gives {type(t).__name__}, not a dict"
        assert list(t) == list(j), f"{where}: keys differ, JAX {list(j)}, Torch {list(t)}"
        for k in j:
            assert_parity(j[k], t[k], f"{where}/{k}")
        return
    if isinstance(j, tuple):
        assert isinstance(t, tuple) and len(t) == len(j), f"{where}: tuple shapes differ"
        for i, (a, b) in enumerate(zip(j, t)):
            assert_parity(a, b, f"{where}[{i}]")
        return
    assert isinstance(t, torch.Tensor), f"{where}: Torch gives {type(t).__name__}, not a tensor"
    a = np.asarray(j)
    b = t.detach().cpu().numpy()
    assert b.shape == a.shape, f"{where}: shape {b.shape} vs JAX {a.shape}"
    if a.dtype.kind == "f":
        assert t.dtype == torch.float32, f"{where}: Torch dtype {t.dtype}, expected float32"
        assert np.isfinite(a).all(), f"{where}: JAX output is not finite"
        parity_report.record(
            "task_math", where.split("/")[0].split("[")[0], a, b, tol=parity_report.tol_text(TASK)
        )
        np.testing.assert_allclose(b, a, err_msg=where, **TASK)
    else:
        assert b.dtype == a.dtype, f"{where}: Torch dtype {b.dtype} vs JAX {a.dtype}"
        np.testing.assert_array_equal(b, a, err_msg=where)


# -- inputs ---------------------------------------------------------------


def _env():
    state = tuple(default_config().obs.state) + ("heading_err",)
    return make_env(
        "joystick",
        paths.ROBOTS_DIR / "roboto_origin",
        "deploy_pd",
        {
            "sim": {"backend": "jax"},
            "command": {"ramp": {"lin": 0.5, "yaw": 1.0}},
            "gait": {"lift": {"v_full": 0.5, "min_frac": 0.25}},
            "symmetry": {"enable": True},
            "no_progress": {"enable": True},
            "obs": {"state": state},
            "push": {"interval_steps_range": (5, 15), "vel_z": 0.3, "ang_vel_rp": 0.5, "ang_vel_yaw": 0.7},
            "reward": {
                "scales": {
                    **STYLE_SCALES,
                    "step_length": 1.0,
                    "foot_reach": 1.0,
                    "knee_extend": 0.5,
                    "forward_lean": 0.5,
                },
                # roboto's home knee is 0.3 rad, and flexion is positive.
                "step_length": {"pair_min": True},
                "foot_reach": {"target_a": 0.05, "target_b": 0.15, "pair_min": True},
                "knee_extend": {"flex_lo": 0.1, "flex_hi": 0.6, "td_mult": 0.5, "pair_min": True},
                "forward_lean": {"max_deg": 5.0, "vx_max": 1.0, "sigma": 0.05},
            },
        },
    )


def _rollout(env):
    """8 snapshots of a jitted single-world rollout, stacked on a leading axis."""
    reset, step = jax.jit(env.reset), jax.jit(env.step)
    key = jax.random.PRNGKey(7)
    key, k_reset = jax.random.split(key)
    state = reset(k_reset)
    data, info = [], []
    for i in range(1, STEPS + 1):
        key, k_act = jax.random.split(key)
        state = step(state, jax.random.uniform(k_act, (env.action_size,), minval=-0.5, maxval=0.5))
        if i % SNAPSHOT_EVERY == 0:
            data.append({f: getattr(state.data, f) for f in signals.DATA_FIELDS})
            info.append({k: v for k, v in state.info.items() if k != "rng"})

    def stack(*xs):
        return jp.stack(xs)

    return jax.tree.map(stack, *data), jax.tree.map(stack, *info)


def _term_arrays(rng, batch, nu, nf):
    def f(*shape, lo=-1.0, hi=1.0):
        return rng.uniform(lo, hi, batch + shape).astype(np.float32)

    def dur():
        """Mode durations, some still unrecorded (0)."""
        return (f(nf, lo=0.0, hi=0.6) * (rng.uniform(size=batch + (nf,)) > 0.2)).astype(np.float32)

    return {
        "scalar": f(),
        "pos": f(lo=0.0, hi=2.0),
        "xy": f(2),
        "xy_b": f(2),
        "joints": f(nu),
        "joints_b": f(nu),
        "joints_c": f(nu),
        "feet": f(nf),
        "feet_pos": f(nf, lo=0.0, hi=1.0),
        "feet_pos_b": f(nf, lo=0.0, hi=1.0),
        "feet_xy": f(nf, 2),
        "contact_f": (rng.uniform(size=batch + (nf,)) > 0.5).astype(np.float32),
        "contact_b": rng.uniform(size=batch + (nf,)) > 0.5,
        "fall": rng.uniform(size=batch) > 0.5,
        "dur": dur(),
        "dur_b": dur(),
        "ones": np.ones(nu, np.float32),
        "cap": np.full(nu, 30.0, np.float32),
        "lo": np.full(nu, -1.0, np.float32),
        "hi": np.full(nu, 1.0, np.float32),
        "target": f(lo=-0.1, hi=0.4),
        "feet_x": f(nf, lo=-0.3, hi=0.4),
        "frac": f(nf, lo=0.0, hi=1.0),
        "contact_c": rng.uniform(size=batch + (nf,)) > 0.5,
        "knee": f(nf, lo=0.0, hi=1.2),
        "gx": f(lo=-0.3, hi=0.3),
        "vx": f(lo=-0.5, hi=1.2),
    }


def _dr_inputs(env, keys):
    """name -> (base fields, masks, draws) for the default DR config and for
    every optional field on."""
    m = env.mj_model
    floor_id = randomize._find_floor_geom_id(m, None)
    root_id = randomize._find_base_body_id(m)
    foot_ids = np.array([m.geom(n).id for n in env._robot_spec.foot_geoms])
    dims = {"nbody": m.nbody, "nu": m.nu, "nv": m.nv, "nfoot": len(foot_ids)}
    masks = transform.dr_masks(m.ngeom, m.nbody, floor_id, root_id, foot_ids)
    fields = ("geom_friction", "body_mass", "actuator_gainprm", "actuator_biasprm", "actuator_forcerange")
    extra = ("body_ipos", "dof_damping", "dof_armature", "dof_frictionloss")
    all_on = {
        k: {"enable": True}
        for k in ("com_offset", "joint_gains", "dof", "foot_friction", "motor_strength", "base_mass_add")
    }
    out = {}
    for name, dr_cfg in (("defaults", None), ("all_on", all_on)):
        table = transform.DR_DRAWS(transform.resolved_cfg(dr_cfg), dims)
        draws = jax.vmap(lambda k, table=table: randomize.jax_dr_draws(k, table))(keys)
        base = {f: np.asarray(getattr(env.mjx_model, f)) for f in fields + (extra if dr_cfg else ())}
        out[name] = (base, masks, draws)
    return out


def _all_pure(command_cfg):
    c = command_cfg.copy_and_resolve_references()
    for _name, _idx, prob_key, _range_key in progress.COMMAND_DRAWS:
        c[prob_key] = 0.2
    return c


def _assert_both_branches(b):
    """The update, push and mirror cases are only worth something if the
    batch holds both outcomes of each branch."""

    def mixed(x):
        x = np.asarray(x)
        return x.any() and not x.all()

    _info, _rewards, _done, aux = gait_state.update(b.update_info, b.r, b.action, b.consts, b.cfg, b.update_draws)
    assert mixed(aux["cut"])
    assert mixed(np.asarray(b.update_info["steps_since_cmd"]) + 1 >= b.cfg.command.resample_steps)
    push_now, _ = push.push_schedule(b.cfg.push, b.step_count, b.countdown, b.push_draws["next_countdown"])
    assert mixed(push_now)
    fixed_now, _ = push.push_schedule(b.fixed_push, b.step_count)
    assert mixed(fixed_now)
    assert mixed(FLAGS)
    assert mixed(b.command_draws["zero_on"])
    for name, *_ in progress.COMMAND_DRAWS:
        assert mixed(b.command_draws[f"{name}_on"]), name


@pytest.fixture(scope="module")
def world(parity_device):
    """The inputs of every case, once as JAX arrays and once as Torch tensors."""
    TORCH_DEVICE["device"] = replay.torch_device(parity_device)
    env = _env()
    cfg = env._config
    consts = env._consts
    data, info = _rollout(env)
    for k, v in {**data, **info}.items():
        assert np.isfinite(np.asarray(v, dtype=np.float64)).all(), f"rollout {k} is not finite"

    keys = jax.random.split(jax.random.PRNGKey(11), N)
    np_rng = np.random.default_rng(3)
    nu, nf = env.action_size, consts.n_feet

    r = signals.readings(types.SimpleNamespace(**data), consts)
    contact = r["contact"]
    contact_filt = contact | info["last_contact"]
    first_contact = (info["feet_air_time"] > 0) & contact_filt
    flags = jp.asarray(FLAGS)
    cat = signals.catalog(r, info, consts)
    actor = list(env.actor_obs_names)
    noise = jax.random.uniform(keys[0], signals.gather(cat, actor).shape, minval=-1.0, maxval=1.0)
    action = jp.asarray(np_rng.uniform(-1.0, 1.0, (N, nu)).astype(np.float32))
    rewards, _fall, _land = compose.compute_rewards(r, info, action, first_contact, contact, consts, cfg)

    pure_cfg = _all_pure(cfg.command)
    command_draws = jax.vmap(lambda k: jax_command_draws(k, pure_cfg))(jax.random.split(keys[1], 256))
    reset_draws = jax.vmap(lambda k: jax_reset_draws(k, cfg, nu))(keys)
    step_draws = jax.vmap(lambda k: jax_step_draws(k, cfg))(keys)

    # update: hand-set commands, counters and meter, so the batch holds worlds
    # that turn and do not, are armed and not, are cut and not, and resample
    # and do not. The meter at -1 puts every armed world at p_max.
    grace = math.ceil(cfg.no_progress.grace_sec / env.dt) + 20
    resample = cfg.command.resample_steps
    update_info = {
        **info,
        "command": jp.asarray(COMMANDS),
        "steps_since_cmd": jp.asarray(
            [0, 50, grace, grace, grace, grace, resample - 1, resample - 1], dtype=jp.int32
        ),
        "progress_ema": jp.full((N,), -1.0, jp.float32),
    }
    update_draws = {
        "command": step_draws["command"],
        "u_term": jp.asarray([0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.01, 0.0], jp.float32),
    }

    # pre_physics and the push: mixed mirror flags and countdowns, so some
    # worlds are pushed and mirrored and some are not.
    countdown = jp.asarray([0, 1, 0, 5, -1, 2, 0, 3], dtype=jp.int32)
    pre_info = {**info, "mirror": flags, "push_countdown": countdown}
    fixed_push = default_config().push
    fixed_push.interval_steps = 7
    fixed_draws = jax.vmap(lambda k: jax_push_draws(k, fixed_push))(keys)
    step_count = jp.asarray([0, 6, 7, 13, 20, 27, 34, 41], dtype=jp.int32)

    terms_a = _term_arrays(np_rng, (2, 3), nu, nf)
    jax_in = {
        "data": data,
        "info": info,
        "r": r,
        "contact": contact,
        "first_contact": first_contact,
        "flags": flags,
        "cat": cat,
        "noise": noise,
        "action": action,
        "rewards": rewards,
        "commands": jp.asarray(COMMANDS),
        "vec3": jp.asarray(np_rng.uniform(-2.0, 2.0, (N, 3)).astype(np.float32)),
        "angles": jp.asarray(np_rng.uniform(-10.0, 10.0, (N,)).astype(np.float32)),
        "yaw_ref": jp.asarray(np_rng.uniform(-4.0, 4.0, (N,)).astype(np.float32)),
        "ratio": jp.asarray(np_rng.uniform(-1.0, 2.0, (N,)).astype(np.float32)),
        "phase": jp.asarray(np_rng.uniform(-math.pi, math.pi, (N,)).astype(np.float32)),
        "steps": jp.asarray(np_rng.integers(0, 300, (N,)).astype(np.int32)),
        "done": flags,
        "command_draws": command_draws,
        "reset_draws": {k: v for k, v in reset_draws.items() if k != "rng"},
        "update_info": update_info,
        "update_draws": update_draws,
        "pre_info": pre_info,
        "push_draws": step_draws["push"],
        "countdown": countdown,
        "step_count": step_count,
        "fixed_draws": fixed_draws,
        "terms": to_jax(terms_a),
        "dr": {name: (to_jax(b), to_jax(m), d) for name, (b, m, d) in _dr_inputs(env, keys).items()},
    }
    # compose with landings: the rollout's random policy lands few feet ahead
    # of the base, so the landing terms get hand-set foot positions, landing
    # flags, air times and a previous land_frac. The commands mix fwd on and
    # off. Drawn last, so every input above keeps its values.
    jax_in["land_r"] = {
        **r,
        "foot_pos_base": jp.asarray(np_rng.uniform(-0.2, 0.4, (N, nf, 3)).astype(np.float32)),
    }
    jax_in["land_info"] = {
        **update_info,
        "feet_air_time": jp.asarray(np_rng.uniform(0.0, 0.5, (N, nf)).astype(np.float32)),
        "land_frac": jp.asarray(np_rng.uniform(0.0, 1.0, (N, nf, 3)).astype(np.float32)),
    }
    jax_in["land_first_contact"] = jp.asarray(np_rng.uniform(size=(N, nf)) > 0.4)
    torch_in = {k: to_torch(v) for k, v in jax_in.items()}
    jax_in["data"] = types.SimpleNamespace(**jax_in["data"])
    torch_in["data"] = types.SimpleNamespace(**torch_in["data"])
    jax_in["consts"] = consts
    torch_in["consts"] = torch_consts(consts)
    for b in (jax_in, torch_in):
        b.update(env=env, cfg=cfg, actor=actor, pure_cfg=pure_cfg, fixed_push=fixed_push)
    jax_ns = types.SimpleNamespace(**jax_in)
    _assert_both_branches(jax_ns)
    return types.SimpleNamespace(jax=jax_ns, torch=types.SimpleNamespace(**torch_in))


# -- cases ----------------------------------------------------------------

CASES = {}


def case(name):
    """Register a case: fn(b) runs one shared function on backend inputs b.
    The name is module.function, plus :variant when one function has more
    than one case."""

    def register(fn):
        CASES[name] = fn
        return fn

    return register


def _mirror(b):
    c = b.consts
    return (b.flags, c.state_perm, c.state_sign, c.priv_perm, c.priv_sign)


# signals
case("signals.quat_inv")(lambda b: signals.quat_inv(b.r["quat"]))
case("signals.quat_rotate")(lambda b: signals.quat_rotate(b.r["quat"], b.vec3))
case("signals.quat_rotate_inv")(lambda b: signals.quat_rotate_inv(b.r["quat"], b.vec3))
case("signals.quat:sensor")(
    lambda b: signals.quat(b.data.sensordata, b.data.qpos, b.consts.sensor_adr, b.consts.base_qadr)
)
case("signals.quat:qpos")(lambda b: signals.quat(b.data.sensordata, b.data.qpos, {}, b.consts.base_qadr))
case("signals.gyro:sensor")(
    lambda b: signals.gyro(b.data.sensordata, b.data.qvel, b.consts.sensor_adr, b.consts.base_vadr)
)
case("signals.gyro:qvel")(lambda b: signals.gyro(b.data.sensordata, b.data.qvel, {}, b.consts.base_vadr))
case("signals.linvel:sensor")(
    lambda b: signals.linvel(
        b.data.sensordata, b.data.qvel, b.r["quat"], b.consts.sensor_adr, b.consts.base_vadr
    )
)
case("signals.linvel:qvel")(
    lambda b: signals.linvel(b.data.sensordata, b.data.qvel, b.r["quat"], {}, b.consts.base_vadr)
)
case("signals.gravity_body")(lambda b: signals.gravity_body(b.r["quat"]))
case("signals.base_yaw")(lambda b: signals.base_yaw(b.r["quat"]))
case("signals.wrap_angle")(lambda b: signals.wrap_angle(b.angles))
case("signals.heading_err")(lambda b: signals.heading_err(b.yaw_ref, b.r["yaw"]))
case("signals.foot_clearance")(
    lambda b: signals.foot_clearance(
        b.data.site_xpos, b.consts.foot_site_ids, b.consts.foot_site_sole_offset
    )
)
case("signals.foot_contact")(
    lambda b: signals.foot_contact(
        b.data.geom_xpos, b.consts.foot_geom_ids, b.consts.foot_geom_radius, b.consts.foot_geom_member
    )
)
case("signals.foot_linvel")(
    lambda b: signals.foot_linvel(
        b.data.site_xpos,
        b.data.subtree_com,
        b.data.cdof,
        b.data.qvel,
        b.consts.foot_site_ids,
        b.consts.foot_root_body,
        b.consts.foot_dof_mask,
    )
)
case("signals.foot_pos_base")(
    lambda b: signals.foot_pos_base(
        b.data.site_xpos, b.data.xpos, b.consts.foot_site_ids, b.consts.root_body, b.r["quat"]
    )
)
case("signals.body_y_sep")(lambda b: signals.body_y_sep(b.data.xpos, b.consts.ankle_pair, b.r["quat"]))
case("signals.fall")(
    lambda b: signals.fall(
        b.r["base_height"], b.r["gravity"][..., 2], b.cfg.fall.min_height, b.cfg.fall.max_tilt_gz
    )
)
case("signals.readings")(lambda b: signals.readings(b.data, b.consts))
case("signals.catalog")(lambda b: signals.catalog(b.r, b.info, b.consts))
case("signals.catalog:no_heading")(
    lambda b: signals.catalog(b.r, {k: v for k, v in b.info.items() if k != "yaw_ref"}, b.consts)
)
case("signals.catalog:frozen_zeros")(
    lambda b: signals.catalog(b.r, b.info, dataclasses.replace(b.consts, frozen_phase_zeros=True))
)
case("signals.gather")(lambda b: signals.gather(b.cat, b.actor))
case("signals.noise_scales")(lambda b: signals.noise_scales(b.cat, b.actor, b.cfg.obs_noise))
case("signals.add_noise")(
    lambda b: signals.add_noise(
        signals.gather(b.cat, b.actor), b.noise, signals.noise_scales(b.cat, b.actor, b.cfg.obs_noise)
    )
)
case("signals.observe:plain")(lambda b: signals.observe(b.cat, b.actor, b.cfg.obs.privileged))
case("signals.observe:noise_mirror")(
    lambda b: signals.observe(
        b.cat, b.actor, b.cfg.obs.privileged, noise=b.noise, obs_noise=b.cfg.obs_noise, mirror=_mirror(b)
    )
)

# progress
case("progress.served")(lambda b: progress.served(b.r["linvel"][..., :2], b.r["gyro"][..., 2], b.commands))
case("progress.hazard")(
    lambda b: progress.hazard(b.ratio, b.cfg.no_progress.risk_below, b.cfg.no_progress.p_max)
)
case("progress.armed")(
    lambda b: progress.armed(
        progress.cmd_speed(b.commands), b.steps, b.consts.dt, b.cfg.no_progress.grace_sec, b.consts.deadband
    )
)
case("progress.cmd_speed")(lambda b: progress.cmd_speed(b.commands))
case("progress.clock_freq")(
    lambda b: progress.clock_freq(progress.cmd_speed(b.commands), b.consts.cmd_vmax, b.cfg.gait.freq)
)
case("progress.phase_dt")(
    lambda b: progress.phase_dt(b.commands, b.consts.cmd_vmax, b.cfg.gait.freq, b.consts.dt, b.consts.deadband)
)
case("progress.leg_phases")(lambda b: progress.leg_phases(b.phase))
case("progress.gait_targets")(
    lambda b: progress.gait_targets(b.phase, b.cfg.gait.duty, b.cfg.gait.swing_height)
)
case("progress.command_from_draws")(lambda b: progress.command_from_draws(b.command_draws, b.pure_cfg))
case("progress.ramp_command")(lambda b: progress.ramp_command(b.commands, b.vec3, 0.5, 1.0, b.consts.dt))
case("progress.fwd_mask")(lambda b: progress.fwd_mask(b.commands, b.consts.deadband))
case("progress.lift_scale")(lambda b: progress.lift_scale(b.commands, 0.5, 0.25))

# terms, on random inputs with batch shape (2, 3)
_TERM_ARGS = {
    "tracking_err_lin": lambda a: (a["xy"], a["xy_b"]),
    "tracking_err_ang": lambda a: (a["scalar"], a["pos"]),
    "tracking_kernel": lambda a: (a["pos"], 0.25),
    "tracking_rel_sigma": lambda a: (a["scalar"], 0.25, 0.1),
    "tracking_far_blend": lambda a: (a["pos"], a["pos"], 0.3, 2.0),
    "lin_vel_z": lambda a: (a["scalar"],),
    "ang_vel_xy": lambda a: (a["xy"],),
    "orientation": lambda a: (a["xy"],),
    "torques": lambda a: (a["joints"],),
    "torque_rate": lambda a: (a["joints"], a["joints_b"]),
    "action_rate": lambda a: (a["joints"], a["joints_b"]),
    "action_accel": lambda a: (a["joints"], a["joints_b"], a["joints_c"]),
    "energy": lambda a: (a["joints"], a["joints_b"]),
    "pose": lambda a: (a["joints"], a["joints_b"], a["ones"]),
    "feet_air_time": lambda a: (a["dur"], a["contact_f"], 0.0, 0.1),
    "feet_air_time:cap": lambda a: (a["dur"], a["contact_f"], 0.3, 0.1),
    "feet_slip": lambda a: (a["feet_xy"], a["contact_f"]),
    "feet_apex": lambda a: (a["feet_pos"], a["contact_f"], 0.05),
    "feet_apex_min": lambda a: (a["feet_pos"], a["feet_pos_b"], a["contact_f"], 0.05),
    "feet_landing": lambda a: (a["feet"], a["feet_pos"], 0.3),
    "feet_phase": lambda a: (a["feet_pos"], a["feet_pos_b"], 0.1),
    "stand_still": lambda a: (a["joints"], a["joints_b"], a["joints_c"], 0.2),
    "termination": lambda a: (a["fall"],),
    "torque_limit": lambda a: (a["joints"] * 40.0, a["cap"], 0.8),
    "pose_l1": lambda a: (a["joints"], a["joints_b"], a["ones"]),
    "joint_pos_limits": lambda a: (a["joints"] * 2.0, a["lo"], a["hi"]),
    "joint_vel": lambda a: (a["joints"],),
    "joint_acc": lambda a: (a["joints"],),
    "upward": lambda a: (a["scalar"],),
    "distance_band": lambda a: (a["pos"], 0.2, 0.5),
    "feet_contact_without_cmd": lambda a: (a["contact_b"], a["scalar"]),
    "feet_air_time_biped": lambda a: (a["dur"], a["dur_b"], a["contact_b"], 0.4),
    "knee_stance": lambda a: (a["feet"], a["contact_f"], 0.3),
    "gait_symmetry": lambda a: (a["dur"], a["dur_b"], 0.1, 1.0),
    "gait_symmetry_income": lambda a: (a["dur"], a["dur_b"], a["contact_b"], 0.1, 1.0),
    "step_length": lambda a: (a["feet_x"], a["contact_b"], a["target"], 1.5),
    "step_length:pair": lambda a: (a["feet_x"], a["contact_b"], a["target"], 1.0, True, a["frac"]),
    "foot_reach": lambda a: (a["feet_x"], a["contact_b"], a["target"], -0.05, 2.0),
    "foot_reach:pair": lambda a: (a["feet_x"], a["contact_b"], a["target"], 0.0, 2.0, True, a["frac"]),
    "knee_extend": lambda a: (
        a["knee"], a["feet_x"], a["contact_c"], a["contact_b"], a["dur"], 0.2, 1.0, 0.0, 0.2, 0.5
    ),
    "knee_extend:pair": lambda a: (
        a["knee"], a["feet_x"], a["contact_c"], a["contact_b"], a["dur"], 0.2, 1.0, 0.05, 0.2, 0.5, True, a["frac"]
    ),
    "forward_lean": lambda a: (a["gx"], a["vx"], 8.0, 1.0, 0.05),
}
for _name, _args in _TERM_ARGS.items():
    _fname = _name.split(":")[0]
    case(f"terms.{_name}")(lambda b, fname=_fname, args=_args: getattr(terms, fname)(*args(b.terms)))

# compose
case("compose.compute_rewards")(
    lambda b: compose.compute_rewards(b.r, b.info, b.action, b.first_contact, b.contact, b.consts, b.cfg)
)
case("compose.compute_rewards:landing")(
    lambda b: compose.compute_rewards(
        b.land_r, b.land_info, b.action, b.land_first_contact, b.contact, b.consts, b.cfg
    )
)
case("compose.total_reward")(
    lambda b: compose.total_reward(b.rewards, b.cfg.reward.scales, b.consts.dt)
)

# gait_state
for _flags in (("progress",), ("gait",), ("heading",), ("land",), gait_state.LAYERS):
    case(f"gait_state.clear_done:{'+'.join(_flags)}")(
        lambda b, flags=_flags: gait_state.clear_done(
            b.info,
            b.done,
            flags,
            {"progress_ema": progress.cmd_speed(b.info["command"]), "yaw_ref": b.r["yaw"]},
        )
    )
case("gait_state.reset_offset")(
    lambda b: gait_state.reset_offset(b.reset_draws["pose_noise"], 0.05)
)
case("gait_state.reset_state")(
    lambda b: gait_state.reset_state(
        progress.command_from_draws(b.reset_draws["command"], b.cfg.command),
        b.consts,
        b.cfg,
        yaw=b.r["yaw"],
        push_countdown=b.reset_draws["push_countdown"],
        mirror_u=b.reset_draws["mirror_u"],
    )
)
case("gait_state.pre_physics")(
    lambda b: gait_state.pre_physics(b.pre_info, b.action, b.consts, b.cfg, b.push_draws)
)
case("gait_state.update")(
    lambda b: gait_state.update(b.update_info, b.r, b.action, b.consts, b.cfg, b.update_draws)
)

# push
case("push.push_schedule:ranged")(
    lambda b: push.push_schedule(b.cfg.push, b.step_count, b.countdown, b.push_draws["next_countdown"])
)
case("push.push_schedule:fixed")(lambda b: push.push_schedule(b.fixed_push, b.step_count))
case("push.push_kick:all_knobs")(lambda b: push.push_kick(b.cfg.push, b.flags, b.push_draws))
case("push.push_kick:planar")(lambda b: push.push_kick(b.fixed_push, b.flags, b.fixed_draws))

# symmetry
case("symmetry.mirror:obs")(
    lambda b: symmetry.mirror(
        signals.gather(b.cat, b.actor), b.flags, b.consts.state_perm, b.consts.state_sign
    )
)
case("symmetry.mirror:action")(
    lambda b: symmetry.mirror(b.action, b.flags, b.consts.act_perm, b.consts.act_sign)
)

# dr
case("transform.dr_fields:defaults")(lambda b: transform.dr_fields(*b.dr["defaults"]))
case("transform.dr_fields:all_on")(lambda b: transform.dr_fields(*b.dr["all_on"]))


# -- tests ----------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(CASES))
def test_torch_matches_jax(world, name):
    fn = CASES[name]
    assert_parity(fn(world.jax), fn(world.torch), name)


def test_every_shared_array_function_has_a_case():
    covered = {n.split(":")[0] for n in CASES} | {"constants.xp_of"}
    missing = []
    for label, mod in SHARED_MODULES.items():
        for fname, obj in vars(mod).items():
            if fname.startswith("_") or not inspect.isfunction(obj) or obj.__module__ != mod.__name__:
                continue
            qual = f"{label}.{fname}"
            if qual not in covered and qual not in ARRAY_FREE:
                missing.append(qual)
    assert not missing, (
        f"shared functions with no parity case: {missing}. Add a case, or list "
        "the function in ARRAY_FREE if it takes no data arrays."
    )
