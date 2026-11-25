"""The origin/main reference rollouts, shared by the generator and the test.

tests/data/fixtures/ pins what the JAX path of origin/main (042f018) computes
for the code the Torch/shared-module refactor touches: the train layout
(DR, brax wrappers, vmap), respawn reseeds, DR draws, single-world
mechanisms, the pure command draws, the composed PPO params and a tiny brax
training run. This module defines those rollouts once so the generator and
`test_reference_fixtures.py` cannot drift apart.

It uses only APIs that exist on both origin/main and the refactored tree
(registry.make_env, make_domain_randomize, make_wrap_env_fn, Joystick
reset/step/_sample_command, train.build_ppo_params, hydra compose). It is not
a `test_*.py` file, so pytest never collects it.
"""

from __future__ import annotations

import dataclasses
import functools
import json
import platform
import subprocess
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path

import jax
import numpy as np

from humanoid_lab import paths

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "data" / "fixtures"
PPO_PARAMS_PATH = FIXTURE_DIR / "ppo_params.json"
BRAX_TRAIN_PATH = FIXTURE_DIR / "brax_train.npz"

BASE_SHA = "042f0180bbe0984e75bb177682456bd477493861"

_JAX = {"sim": {"backend": "jax"}}

_ALL_ON_FIELDS = (
    "com_offset",
    "joint_gains",
    "dof",
    "foot_friction",
    "motor_strength",
    "base_mass_add",
)
_ALL_ON = {k: {"enable": True} for k in _ALL_ON_FIELDS}
_DR_CFGS = {
    "defaults": None,
    "all_on": _ALL_ON,
    "gains_off": {**_ALL_ON, "joint_gains": {"enable": False}},
    "odd_ranges": {
        **_ALL_ON,
        "base": {
            "floor_friction": [0.3, 1.7],
            "base_mass": [0.55, 1.45],
            "link_mass": [0.85, 1.15],
            "gain_fallback": [0.75, 1.33],
        },
        "motor_strength": {"enable": True, "range": [0.37, 1.13]},
        "base_mass_add": {"enable": True, "kg": 2.7},
    },
}

_NO_PROGRESS_SINGLE = {"enable": True, "grace_sec": 0.2, "p_max": 0.5}
_NO_PROGRESS_RESPAWN = {"enable": True, "grace_sec": 0.1, "p_max": 0.5}


@dataclass(frozen=True)
class ReferenceCase:
    name: str
    kind: str  # train_layout | respawn | dr_draws | single
    robot: str
    preset: str = "deploy_pd"
    env_overrides: dict = field(default_factory=dict)
    steps: int = 0
    backend: str = "jax"
    dr: str = "defaults"  # dr_draws: key into _DR_CFGS


def _train_layout(robot: str, backend: str) -> ReferenceCase:
    return ReferenceCase(
        name=f"train_layout__{robot}__{backend}",
        kind="train_layout",
        robot=robot,
        env_overrides={"sim": {"backend": backend, "num_envs": 8}},
        steps=64,
        backend=backend,
    )


def _respawn(variant: str, backend: str) -> ReferenceCase:
    overrides = {
        "sim": {"backend": backend, "num_envs": 4},
        "command": {"resample_steps": 15},
    }
    if variant in ("no_progress", "both"):
        overrides["no_progress"] = dict(_NO_PROGRESS_RESPAWN)
    if variant in ("gait_income", "both"):
        overrides["reward"] = {"scales": {"gait_symmetry_income": 0.5}}
    return ReferenceCase(
        name=f"respawn__{variant}__{backend}",
        kind="respawn",
        robot="roboto_origin",
        env_overrides=overrides,
        steps=80,
        backend=backend,
    )


def _single(name: str, robot: str, steps: int, **overrides) -> ReferenceCase:
    return ReferenceCase(
        name=name, kind="single", robot=robot, env_overrides={**_JAX, **overrides}, steps=steps
    )


def _stand(name: str, command: dict) -> ReferenceCase:
    return _single(
        name,
        "roboto_origin",
        80,
        no_progress=dict(_NO_PROGRESS_SINGLE),
        command={"resample_steps": 10, **command},
    )


_ROBOTO_MECHANISMS = {
    "symmetry": {"enable": True, "mirror_prob": 1.0},
    "no_progress": dict(_NO_PROGRESS_SINGLE),
    "real_pose_ref": True,
    "obs_noise": {"gyro": 0.05, "joint_vel": 0.2},
    "reward": {
        "orientation_tol_deg": 20.0,
        "tracking_relative": True,
        "tracking_product": True,
        "tracking_far_weight": 0.2,
        "shaping_tracking_gate": True,
        "scales": {
            "knee_stance": -0.1,
            "gait_symmetry": -0.5,
            "gait_symmetry_income": 0.5,
            "feet_apex_min": 0.5,
            "feet_apex": 0.5,
            "feet_landing": -0.1,
            "feet_air_time_biped": 0.5,
            "torque_limit": -0.1,
            "torque_rate": -1e-4,
            "pose_l1": -0.1,
            "joint_pos_limits": -1.0,
            "joint_vel": -1e-4,
            "joint_acc": -1e-7,
            "upward": 0.1,
            "feet_distance": 0.1,
            "knee_distance": 0.1,
            "feet_contact_without_cmd": 0.1,
        },
    },
}

_ASIMOV_MECHANISMS = {
    "no_progress": dict(_NO_PROGRESS_SINGLE),
    "gait": {"air_time_cap": 0.5},
    "reward": {
        "orientation_tol_deg": 10.0,
        "tracking_relative": True,
        "tracking_far_weight": 0.2,
        "scales": {
            "feet_apex": 0.5,
            "feet_landing": -0.1,
            "feet_air_time_biped": 0.5,
            "gait_symmetry": -0.5,
            "knee_stance": -0.1,
            "feet_distance": 0.1,
            "knee_distance": 0.1,
        },
    },
}

CASES = (
    *(_train_layout(r, b) for r in ("roboto_origin", "asimov_v1") for b in ("jax", "warp")),
    *(_respawn(v, b) for v in ("no_progress", "gait_income", "both") for b in ("jax", "warp")),
    *(
        ReferenceCase(
            name=f"dr_draws__{r}__{d}",
            kind="dr_draws",
            robot=r,
            env_overrides=dict(_JAX),
            dr=d,
        )
        for r in ("roboto_origin", "asimov_v1")
        for d in _DR_CFGS
    ),
    _stand("stand__zero", {"zero_prob": 1.0}),
    _stand("stand__inside", {"vx": (-0.04, 0.04), "vy": (-0.04, 0.04), "wz": (-0.04, 0.04)}),
    _stand("stand__straddle", {"vx": (-0.08, 0.08), "vy": (-0.08, 0.08), "wz": (-0.08, 0.08)}),
    _single(
        "push__range",
        "roboto_origin",
        80,
        push={
            "interval_steps_range": (5, 15),
            "vel_z": 0.3,
            "ang_vel_rp": 0.5,
            "ang_vel_yaw": 0.7,
        },
    ),
    _single("push__fixed", "roboto_origin", 80, push={"interval_steps": 10, "vel_z": 0.3}),
    _single("mechanisms__roboto_origin", "roboto_origin", 80, **_ROBOTO_MECHANISMS),
    _single("mechanisms__asimov_v1", "asimov_v1", 80, **_ASIMOV_MECHANISMS),
    _single("frozen_clock__roboto_origin", "roboto_origin", 40, gait={"freq": (0.0, 0.0)}),
)
CASES = (*CASES, ReferenceCase(name="pure_draws", kind="pure_draws", robot="roboto_origin"))

CASE_IDS = tuple(c.name for c in CASES)
EXTRA_NAMES = ("ppo_params", "brax_train")
ALL_NAMES = (*CASE_IDS, *EXTRA_NAMES)


def case_path(name: str) -> Path:
    return FIXTURE_DIR / f"{name}.npz"


def source_sha() -> str:
    return subprocess.run(
        ["git", "-C", str(paths.REPO_ROOT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def host_meta() -> dict:
    return {
        "_platform": sys.platform,
        "_machine": platform.machine(),
        "_source_sha": source_sha(),
    }


def _make_env(case: ReferenceCase):
    from humanoid_lab.registry import make_env

    return make_env("joystick", paths.ROBOTS_DIR / case.robot, case.preset, case.env_overrides)


def _key_str(k) -> str:
    for attr in ("key", "name", "idx"):
        if hasattr(k, attr):
            return str(getattr(k, attr))
    return str(k)


def _leaf_np(x) -> np.ndarray:
    if hasattr(x, "dtype") and jax.dtypes.issubdtype(x.dtype, jax.dtypes.prng_key):
        x = jax.random.key_data(x)
    return np.asarray(x)


def _info_leaves(info, skip_first_data: bool) -> dict[str, np.ndarray]:
    out = {}
    flat, _ = jax.tree_util.tree_flatten_with_path(info)
    for path, leaf in flat:
        parts = [_key_str(k) for k in path]
        if skip_first_data and "AutoResetWrapper_first_data" in parts:
            continue
        out["info/" + "/".join(parts)] = _leaf_np(leaf)
    return out


def _stacked(per_step: dict[str, list]) -> dict[str, np.ndarray]:
    return {k: np.stack(v) for k, v in per_step.items()}


def _record_batched(w, state, actions, steps: int, env) -> dict[str, np.ndarray]:
    step = jax.jit(w.step)
    metric_names = sorted(state.metrics)
    rec = {"reward": [], "done": [], "obs_state": [], "obs_privileged_state": [], "metrics": []}
    for t in range(steps):
        state = step(state, actions[t])
        rec["reward"].append(np.asarray(state.reward))
        rec["done"].append(np.asarray(state.done))
        rec["obs_state"].append(np.asarray(state.obs["state"]))
        rec["obs_privileged_state"].append(np.asarray(state.obs["privileged_state"]))
        rec["metrics"].append(
            np.stack([np.asarray(state.metrics[k]) for k in metric_names], axis=-1)
        )
    out = _stacked({k: v for k, v in rec.items() if k != "metrics"})
    out["metric_names"] = np.array(metric_names)
    out["metric_values"] = np.stack(rec["metrics"])
    out.update(_info_leaves(state.info, skip_first_data=True))
    out["final_qpos"] = np.asarray(state.data.qpos)
    out["final_qvel"] = np.asarray(state.data.qvel)
    return out


def _rollout_train_layout(case: ReferenceCase) -> dict[str, np.ndarray]:
    from humanoid_lab.dr.randomize import make_domain_randomize
    from humanoid_lab.envs.wrappers import make_wrap_env_fn

    env = _make_env(case)
    rand = functools.partial(
        make_domain_randomize(env.mj_model, env.robot_spec, None),
        rng=jax.random.split(jax.random.PRNGKey(101), 8),
    )
    w = make_wrap_env_fn(env._config)(
        env, episode_length=20, action_repeat=1, randomization_fn=rand
    )
    state = jax.jit(w.reset)(jax.random.split(jax.random.PRNGKey(202), 8))
    actions = jax.random.uniform(
        jax.random.PRNGKey(303), (case.steps, 8, env.action_size), minval=-1.0, maxval=1.0
    )
    return _record_batched(w, state, actions, case.steps, env)


def _rollout_respawn(case: ReferenceCase) -> dict[str, np.ndarray]:
    from humanoid_lab.envs.wrappers import make_wrap_env_fn

    env = _make_env(case)
    w = make_wrap_env_fn(env._config)(env, episode_length=20, action_repeat=1)
    state = jax.jit(w.reset)(jax.random.split(jax.random.PRNGKey(212), 4))
    actions = jax.random.uniform(
        jax.random.PRNGKey(313), (case.steps, 4, env.action_size), minval=-1.0, maxval=1.0
    )
    return _record_batched(w, state, actions, case.steps, env)


_FALLBACK_FIELDS = (
    "geom_friction",
    "body_mass",
    "actuator_gainprm",
    "actuator_biasprm",
    "actuator_forcerange",
    "body_ipos",
    "dof_damping",
    "dof_armature",
    "dof_frictionloss",
)


def _rollout_dr_draws(case: ReferenceCase) -> dict[str, np.ndarray]:
    from humanoid_lab.dr.randomize import make_domain_randomize

    env = _make_env(case)
    dr_cfg = _DR_CFGS[case.dr]
    model, in_axes = make_domain_randomize(env.mj_model, env.robot_spec, dr_cfg)(
        env.mjx_model, jax.random.split(jax.random.PRNGKey(404), 16)
    )
    if dataclasses.is_dataclass(in_axes):
        names = [f.name for f in dataclasses.fields(in_axes)]
    else:
        names = list(_FALLBACK_FIELDS)
    batched = sorted(
        n
        for n in names
        if isinstance(getattr(in_axes, n, None), int)
        and getattr(in_axes, n) == 0
        # Size fields such as ncam also read 0 here; a drawn field is batched.
        and np.ndim(getattr(model, n)) > 0
        and np.shape(getattr(model, n))[0] == 16
    )
    out = {"in_axes_fields": np.array(batched), "geom_priority": np.asarray(model.geom_priority)}
    for n in batched:
        out[f"field/{n}"] = np.asarray(getattr(model, n))
    return out


def _rollout_single(case: ReferenceCase) -> dict[str, np.ndarray]:
    env = _make_env(case)
    actions = 0.2 * jax.random.uniform(
        jax.random.PRNGKey(7), (case.steps, env.action_size), minval=-1.0, maxval=1.0
    )
    step = jax.jit(env.step)
    state = env.reset(jax.random.PRNGKey(2026))
    metric_names = sorted(state.metrics)
    rewards, dones, obs_state, obs_priv, metrics = [], [], [], [], []
    for t in range(case.steps):
        state = step(state, actions[t])
        rewards.append(np.asarray(state.reward))
        dones.append(np.asarray(state.done))
        obs_state.append(np.asarray(state.obs["state"]))
        obs_priv.append(np.asarray(state.obs["privileged_state"]))
        metrics.append(np.asarray([np.asarray(state.metrics[k]) for k in metric_names]))
    out = {
        "reward": np.stack(rewards),
        "done": np.stack(dones),
        "obs_state": np.stack(obs_state),
        "obs_privileged_state": np.stack(obs_priv),
        "metric_names": np.array(metric_names),
        "metric_values": np.stack(metrics),
        "final_rng": np.asarray(jax.random.key_data(state.info["rng"])),
        "final_phase": np.asarray(state.info["phase"]),
        "final_command": np.asarray(state.info["command"]),
    }
    out.update(_info_leaves(state.info, skip_first_data=False))
    return out


def _rollout_pure_draws(case: ReferenceCase) -> dict[str, np.ndarray]:
    from humanoid_lab.envs.joystick import Joystick, default_config

    cfg = default_config()
    c = cfg.command
    c.pure_wz_prob = 0.15
    c.pure_vy_prob = 0.15
    c.pure_slow_prob = 0.2
    c.pure_fast_prob = 0.2
    c.pure_back_prob = 0.2
    c.zero_prob = 0.1
    stub = types.SimpleNamespace(_config=cfg)
    draw = jax.jit(jax.vmap(lambda k: Joystick._sample_command(stub, k)))
    return {"commands": np.asarray(draw(jax.random.split(jax.random.PRNGKey(505), 4096)))}


_ROLLOUTS = {
    "train_layout": _rollout_train_layout,
    "respawn": _rollout_respawn,
    "dr_draws": _rollout_dr_draws,
    "single": _rollout_single,
    "pure_draws": _rollout_pure_draws,
}


def rollout(case: ReferenceCase) -> dict[str, np.ndarray]:
    return _ROLLOUTS[case.kind](case)


def _compose(overrides):
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(version_base=None, config_dir=str(paths.CONFIGS_DIR)):
        return compose(config_name="config", overrides=overrides)


def experiment_stems() -> list[str]:
    return sorted(
        p.stem
        for p in (paths.CONFIGS_DIR / "experiment").glob("*.yaml")
        if not p.stem.startswith("_")
    )


def _composed_ppo(overrides) -> dict:
    from omegaconf import OmegaConf

    from humanoid_lab.train import _apply_ppo_overrides, build_ppo_params

    cfg = _compose(overrides)
    p = build_ppo_params({}, cfg.smoke)
    _apply_ppo_overrides(p.network_factory, OmegaConf.to_container(cfg.network, resolve=True) or {})
    _apply_ppo_overrides(p, OmegaConf.to_container(cfg.task.ppo, resolve=True) or {})
    _apply_ppo_overrides(p, OmegaConf.to_container(cfg.ppo, resolve=True) or {})
    return json.loads(json.dumps(p.to_dict()))


def ppo_params() -> dict:
    out = {stem: _composed_ppo([f"experiment={stem}"]) for stem in experiment_stems()}
    out["__smoke__"] = _composed_ppo(["smoke=true"])
    return out


def brax_train() -> dict[str, np.ndarray]:
    from brax.training.agents.ppo import networks as ppo_networks
    from brax.training.agents.ppo import train as ppo

    from humanoid_lab.dr.randomize import make_domain_randomize
    from humanoid_lab.envs.wrappers import make_wrap_env_fn
    from humanoid_lab.registry import make_env

    env = make_env(
        "joystick",
        paths.ROBOTS_DIR / "roboto_origin",
        "deploy_pd",
        {"sim": {"backend": "jax", "num_envs": 8}, "episode_length": 20},
    )
    nf = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=(32, 32),
        value_hidden_layer_sizes=(32, 32),
        policy_obs_key="state",
        value_obs_key="privileged_state",
    )
    last = {"step": None, "metrics": None}

    def progress(step, metrics):
        last["step"], last["metrics"] = step, metrics

    _, params, _ = ppo.train(
        environment=env,
        num_timesteps=2000,
        num_envs=8,
        episode_length=20,
        batch_size=8,
        num_minibatches=2,
        unroll_length=10,
        num_updates_per_batch=1,
        learning_rate=3e-4,
        normalize_observations=True,
        seed=3,
        run_evals=False,
        num_evals=2,
        randomization_fn=make_domain_randomize(env.mj_model, env.robot_spec, None),
        wrap_env_fn=make_wrap_env_fn(env._config),
        network_factory=nf,
        progress_fn=progress,
    )
    out = {}
    flat, _ = jax.tree_util.tree_flatten_with_path(params)
    for path, leaf in flat:
        out["param/" + "/".join(_key_str(k) for k in path)] = np.asarray(leaf)
    for name, value in sorted(last["metrics"].items()):
        if "sps" in name or "walltime" in name:
            continue
        out[f"metric/{name}"] = np.asarray(value)
    out["final_step"] = np.asarray(last["step"])
    return out
