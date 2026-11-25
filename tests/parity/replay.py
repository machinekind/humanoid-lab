"""Replay helpers for the lockstep checks between the JAX env and WarpJoystick.

The JAX wrapped env runs first and is recorded. WarpJoystick then runs on
two replay objects. ReplayPhysics feeds it the JAX env's post-step mjx.Data
arrays and records the ctrl and the writes the Torch env makes. ReplayDraws
feeds it the JAX draws. It gets them by calling the JAX draw adapters on the
JAX keys. The five adapters are jax_reset_draws, jax_step_draws,
jax_command_draws and jax_push_draws in envs/joystick.py, and jax_dr_draws in
dr/randomize.py.

Torch's info is never copied from JAX. Only physics arrays and random draws
cross over. Everything else WarpJoystick computes itself, so its info, obs,
rewards and done flags can be compared with the JAX ones.

The lockstep, physics, DR, RNG and closed-loop parity tests share this
module.
"""

from __future__ import annotations

import collections
import copy
import dataclasses
import types

import jax
import jax.numpy as jp
import numpy as np
import parity_report
import pytest
import torch
from hydra import compose as hydra_compose
from hydra import initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from tolerances import TASK

from humanoid_lab import paths, registry
from humanoid_lab.dr import randomize
from humanoid_lab.envs import joystick, signals, wrappers
from humanoid_lab.torch_warp.env import WarpJoystick

# The mjx.Data fields one replayed frame carries.
FRAME_FIELDS = signals.DATA_FIELDS + ("ctrl", "qacc_warmstart")

# Info keys the JAX wrappers and the recorder add. brax's EpisodeWrapper adds
# steps, truncation, episode_done and episode_metrics. playground's
# BraxAutoResetWrapper adds the AutoResetWrapper_ keys. record_jax adds
# replay_data.
WRAPPER_INFO = frozenset({"rng", "steps", "truncation", "episode_done", "episode_metrics", "replay_data"})

# Info keys compared modulo 2*pi. A float tie at the wrap point can put the
# same angle 2*pi apart on the two backends.
ANGLE_KEYS = frozenset({"phase", "yaw_ref"})


# -- small helpers ---------------------------------------------------------------
def task_info(info) -> dict:
    """info without the wrapper and recorder keys."""
    return {k: v for k, v in info.items() if k not in WRAPPER_INFO and not k.startswith("AutoResetWrapper_")}


def torch_device(name) -> torch.device:
    """The torch device for a parity run. A CUDA device skips the test when torch sees no GPU."""
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        pytest.skip(f"--parity-device={name} needs CUDA, and torch.cuda.is_available() is False")
    return device


def to_numpy(tree):
    """Tensors and jax arrays as numpy arrays. Dicts recurse and None passes through."""
    if tree is None:
        return None
    if isinstance(tree, dict):
        return {k: to_numpy(v) for k, v in tree.items()}
    if isinstance(tree, torch.Tensor):
        return tree.detach().cpu().numpy()
    return np.asarray(tree)


# -- Hydra and env helpers ----------------------------------------------------------
def _merge(base, extra):
    """Deep-merge extra into base. A dict merges into a dict, and any other value replaces."""
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _merge(base[k], v)
        else:
            base[k] = copy.deepcopy(v)
    return base


def hydra_dict(overrides=(), env=None, dr=None, **top) -> dict:
    """The torch_warp config as a plain dict, on the cpu device.

    env merges into task.env and dr into dr. Each keyword sets a top-level
    key, for example domain_rand=False.
    """
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=str(paths.CONFIGS_DIR)):
        cfg = hydra_compose(config_name="torch_warp", overrides=["torch_warp.device=cpu", *overrides])
    d = OmegaConf.to_container(cfg, resolve=True)
    if env:
        d["task"]["env"] = _merge(d["task"].get("env") or {}, env)
    if dr:
        d["dr"] = _merge(d.get("dr") or {}, dr)
    d.update(top)
    return d


def jax_env(hydra) -> joystick.Joystick:
    """The JAX reference env for a hydra dict, on the jax backend.

    It is built the way WarpJoystick builds its constants env, but as its own
    object. WarpJoystick.jenv is never the JAX reference.
    """
    overrides = copy.deepcopy(hydra["task"].get("env") or {})
    overrides["sim"] = {**(overrides.get("sim") or {}), "backend": "jax"}
    return registry.make_env(
        hydra["task"]["name"],
        paths.REPO_ROOT / hydra["robot"]["dir"],
        hydra["actuators"]["name"],
        overrides,
        hydra["actuators"].get("overrides") or {},
    )


def random_actions(steps, num_envs, action_size, amplitudes, seed) -> np.ndarray:
    """float32 (steps, num_envs, action_size) uniform actions, one amplitude per world."""
    rng = np.random.default_rng(seed)
    a = rng.uniform(-1.0, 1.0, (steps, num_envs, action_size)) * np.asarray(amplitudes)[None, :, None]
    return a.astype(np.float32)


# -- the JAX recorder -----------------------------------------------------------------
@dataclasses.dataclass
class JaxStep:
    rng_before: np.ndarray  # (N, 2) task key before the step
    action: np.ndarray
    obs: dict  # 'state' and 'privileged_state'
    reward: np.ndarray
    done: np.ndarray  # bool, state.done > 0
    truncation: np.ndarray  # bool, info['truncation'] > 0
    info: dict  # task_info, numpy
    metrics: dict
    data: dict  # FRAME_FIELDS of the wrapped data: done worlds hold the first data
    physics: dict  # FRAME_FIELDS of the post-step data before the respawn


@dataclasses.dataclass
class JaxRecord:
    env: object
    cfg: object
    reset_keys: np.ndarray  # (N, 2) keys the Joystick reset got
    reset_obs: dict
    reset_info: dict
    reset_data: dict  # FRAME_FIELDS
    steps: list


def _frame(data) -> dict:
    return {f: getattr(data, f) for f in FRAME_FIELDS}


def _stash(state):
    return state.replace(info={**state.info, "replay_data": _frame(state.data)})


def record_jax(env, actions, episode_length, seed=0) -> JaxRecord:
    """Run the JAX wrapped env on actions (steps, N, nu) and record every step.

    DR stays off. The reseed wrappers come from wrappers.make_wrap_env_fn.
    """
    n = actions.shape[1]
    # BraxAutoResetWrapper replaces a done world's data with the first data
    # after the step and keeps info. The stash puts the true post-step data
    # in info, because WarpJoystick reads it before its own respawn. The
    # patch sits on the instance, because BraxDomainRandomizationVmapWrapper
    # calls env.unwrapped and would bypass a wrapper class. Reset is patched
    # too, so the info pytree keeps one structure and jit traces the step once.
    orig_reset, orig_step = env.reset, env.step
    env.reset = lambda rng: _stash(orig_reset(rng))
    env.step = lambda s, a: _stash(orig_step(s, a))
    try:
        wenv = wrappers.make_wrap_env_fn(env.config)(env, episode_length=episode_length, action_repeat=1)
        reset, step = jax.jit(wenv.reset), jax.jit(wenv.step)
        rng = jax.random.split(jax.random.PRNGKey(seed), n)
        # BraxAutoResetWrapper.reset splits each key and takes rng_key[..., 1].
        # That indexes the LAST axis of the (N, 2, 2) split. [:, 1] would be
        # another key.
        reset_keys = jax.vmap(jax.random.split)(rng)[..., 1]
        s = reset(rng)
        expect_rng = jax.vmap(lambda k: joystick.jax_reset_draws(k, env.config, env.action_size)["rng"])(reset_keys)
        assert np.array_equal(np.asarray(expect_rng), np.asarray(s.info["rng"])), (
            "reset_keys must be jax.vmap(jax.random.split)(rng)[..., 1], the key BraxAutoResetWrapper.reset passes"
        )
        record = JaxRecord(
            env=env,
            cfg=env.config,
            reset_keys=np.asarray(reset_keys),
            reset_obs=to_numpy(dict(s.obs)),
            reset_info=to_numpy(task_info(s.info)),
            reset_data=to_numpy(_frame(s.data)),
            steps=[],
        )
        for t in range(actions.shape[0]):
            rng_before = np.asarray(s.info["rng"])
            s = step(s, jp.asarray(actions[t]))
            record.steps.append(
                JaxStep(
                    rng_before=rng_before,
                    action=np.asarray(actions[t]),
                    obs=to_numpy(dict(s.obs)),
                    reward=np.asarray(s.reward),
                    done=np.asarray(s.done) > 0,
                    truncation=np.asarray(s.info["truncation"]) > 0,
                    info=to_numpy(task_info(s.info)),
                    metrics=to_numpy(dict(s.metrics)),
                    data=to_numpy(_frame(s.data)),
                    physics=to_numpy(s.info["replay_data"]),
                )
            )
    finally:
        del env.reset, env.step
    _assert_stash(record)
    return record


def _assert_stash(record) -> None:
    """The stash holds the data from before the respawn.

    A live world's stashed frame is its wrapped data. A done world's wrapped
    data is the first data, and its stash keeps the data the step made.
    """
    for t, st in enumerate(record.steps):
        live = ~st.done
        for f in FRAME_FIELDS:
            assert np.array_equal(st.physics[f][live], st.data[f][live]), f"step {t} {f}: the stash differs"
        assert np.array_equal(st.data["qpos"][st.done], record.reset_data["qpos"][st.done]), (
            f"step {t}: a done world's wrapped data is not its first data"
        )


# -- the replay physics -------------------------------------------------------------
class ReplayPhysics:
    """The physics interface of WarpPhysics, fed from recorded JAX frames.

    Each reset_worlds or step loads the next queued frame into the data
    views in place, so every view keeps its identity. The writes the Torch
    env makes are logged. qvel_in minus qvel_ref of a step entry is the kick
    Torch added to data.qvel.
    """

    def __init__(self, mj_model, num_envs, device):
        m, n = mj_model, int(num_envs)
        self.mj_model = mj_model
        self.num_envs = n
        self.device = torch.device(device)
        shapes = {
            "qpos": (n, m.nq),
            "qvel": (n, m.nv),
            "qacc": (n, m.nv),
            "qacc_warmstart": (n, m.nv),
            "sensordata": (n, m.nsensordata),
            "actuator_force": (n, m.nu),
            "ctrl": (n, m.nu),
            "xpos": (n, m.nbody, 3),
            "subtree_com": (n, m.nbody, 3),
            "site_xpos": (n, m.nsite, 3),
            "geom_xpos": (n, m.ngeom, 3),
            "cdof": (n, m.nv, 6),
            "time": (n,),
        }
        missing = set(FRAME_FIELDS) - set(shapes)
        assert not missing, f"ReplayPhysics has no shape for {sorted(missing)}"
        self.data = types.SimpleNamespace(
            **{f: torch.zeros(s, dtype=torch.float32, device=self.device) for f, s in shapes.items()}
        )
        self.dr_fields = None
        self._frames = collections.deque()
        self._log = []
        self._qvel_ref = self.data.qvel.clone()

    def queue(self, frame) -> None:
        """Queue one frame: FRAME_FIELDS -> numpy array with a leading N axis."""
        self._frames.append(frame)

    def _load(self):
        assert self._frames, "ReplayPhysics: no JAX frame queued for this call"
        frame = self._frames.popleft()
        for f in FRAME_FIELDS:
            view = getattr(self.data, f)
            view.copy_(torch.as_tensor(np.array(frame[f]), dtype=view.dtype, device=view.device))

    def reset_worlds(self, qpos, qvel, ctrl):
        self._log.append(("reset", {"qpos": qpos.clone(), "qvel": qvel.clone(), "ctrl": ctrl.clone()}))
        self._load()
        self._qvel_ref = self.data.qvel.clone()

    def step(self, ctrl):
        self._log.append(
            ("step", {"ctrl": ctrl.clone(), "qvel_in": self.data.qvel.clone(), "qvel_ref": self._qvel_ref.clone()})
        )
        self._load()
        self._qvel_ref = self.data.qvel.clone()

    def restore_worlds(self, mask, qpos, qvel, ctrl):
        """WarpPhysics's semantics: write the masked worlds, zero their warmstart and clock."""
        self._log.append(
            ("restore", {"mask": mask.clone(), "qpos": qpos.clone(), "qvel": qvel.clone(), "ctrl": ctrl.clone()})
        )
        d, m = self.data, mask[:, None]
        d.qpos.copy_(torch.where(m, qpos, d.qpos))
        d.qvel.copy_(torch.where(m, qvel, d.qvel))
        d.ctrl.copy_(torch.where(m, ctrl, d.ctrl))
        d.qacc_warmstart.copy_(torch.where(m, torch.zeros_like(d.qacc_warmstart), d.qacc_warmstart))
        d.time.copy_(torch.where(mask, torch.zeros_like(d.time), d.time))
        self._qvel_ref = d.qvel.clone()

    def set_dr(self, fields):
        self.dr_fields = {k: v.clone() for k, v in fields.items()}

    def base_fields(self, names):
        """The unbatched mj_model fields as float32 tensors on the device, as WarpPhysics gives them."""
        return {
            n: torch.as_tensor(np.asarray(getattr(self.mj_model, n), dtype=np.float32), device=self.device)
            for n in names
        }

    def peaks(self):
        return 0, 0

    def pop_log(self) -> list:
        """The logged writes since the last call, as (kind, dict of numpy)."""
        out = [(kind, to_numpy(d)) for kind, d in self._log]
        self._log = []
        return out


# -- the replay draws -----------------------------------------------------------------
def _to_torch(tree, device):
    if tree is None:
        return None
    if isinstance(tree, dict):
        return {k: _to_torch(v, device) for k, v in tree.items()}
    # np.array copies. Arrays from jax are read-only, and torch warns on those.
    a = np.array(tree)
    if a.dtype.kind == "f":
        return torch.as_tensor(a.astype(np.float32), device=device)
    if a.dtype.kind == "b":
        return torch.as_tensor(a, dtype=torch.bool, device=device)
    return torch.as_tensor(a, device=device)


class ReplayDraws:
    """The draws interface of TorchDraws, made by the JAX adapters on JAX keys.

    The reset and step keys are one-shot. Set them before each WarpJoystick
    reset or step. The DR keys are given at construction.
    """

    def __init__(self, device, dr_keys=None):
        self.device = torch.device(device)
        self.dr_keys = dr_keys
        self._reset_keys = None
        self._step_keys = None
        self._fns = {}

    def set_reset_keys(self, keys) -> None:
        self._reset_keys = jp.asarray(np.asarray(keys, dtype=np.uint32))

    def set_step_keys(self, keys) -> None:
        self._step_keys = jp.asarray(np.asarray(keys, dtype=np.uint32))

    def _take(self, attr, setter, n):
        keys = getattr(self, attr)
        assert keys is not None, f"ReplayDraws: call {setter} before this draw"
        setattr(self, attr, None)
        assert keys.shape[0] == n, f"ReplayDraws: {keys.shape[0]} keys for {n} worlds"
        return keys

    def _fn(self, name, make):
        # cfg and the sizes are fixed for one env, so each method jits once.
        fn = self._fns.get(name)
        if fn is None:
            fn = self._fns[name] = jax.jit(make())
        return fn

    def reset(self, n, cfg, action_size):
        keys = self._take("_reset_keys", "set_reset_keys", n)
        fn = self._fn("reset", lambda: jax.vmap(lambda k: joystick.jax_reset_draws(k, cfg, action_size)))
        d = dict(fn(keys))
        d.pop("rng")
        return _to_torch(d, self.device)

    def step(self, n, cfg, actor_obs_size):
        keys = self._take("_step_keys", "set_step_keys", n)
        fn = self._fn("step", lambda: jax.vmap(lambda k: joystick.jax_step_draws(k, cfg)))
        d = fn(keys)
        # base.HumanoidEnv._observe's actor noise draw, per world.
        noise_fn = self._fn(
            "noise",
            lambda: jax.vmap(lambda k: jax.random.uniform(k, (actor_obs_size,), minval=-1.0, maxval=1.0)),
        )
        out = {
            "noise": noise_fn(d["noise_key"]),
            "command": d["command"],
            "push": d["push"],
            "u_term": d["u_term"],
        }
        return _to_torch(out, self.device)

    def dr(self, n, table):
        assert self.dr_keys is not None, "ReplayDraws: dr needs dr_keys at construction"
        keys = jp.asarray(np.asarray(self.dr_keys, dtype=np.uint32))
        assert keys.shape[0] == n, f"ReplayDraws: {keys.shape[0]} DR keys for {n} worlds"
        return _to_torch(jax.vmap(lambda k: randomize.jax_dr_draws(k, table))(keys), self.device)

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state) -> None:
        pass


# -- the Torch replay -------------------------------------------------------------------
@dataclasses.dataclass
class TorchStep:
    state: np.ndarray
    priv: np.ndarray
    reward: np.ndarray
    done: np.ndarray
    truncation: np.ndarray
    terms: dict
    info: dict
    step_log: dict  # the 'step' write: ctrl, qvel_in, qvel_ref
    restore_log: dict | None  # the 'restore' write


@dataclasses.dataclass
class TorchRun:
    env: WarpJoystick
    reset_state: np.ndarray
    reset_priv: np.ndarray
    reset_info: dict
    reset_write: dict  # the 'reset' write: qpos, qvel, ctrl
    steps: list


def _entry(log, kind):
    found = [d for k, d in log if k == kind]
    assert len(found) <= 1, f"{len(found)} '{kind}' writes in one call"
    return found[0] if found else None


def replay_torch(record, hydra, device, episode_length, steps=None, hook=None) -> TorchRun:
    """Run WarpJoystick on the recorded JAX physics and draws.

    hydra must have domain_rand False. hook(tenv) runs after construction,
    before the reset.
    """
    assert not hydra.get("domain_rand"), "replay_torch replays no DR. Set domain_rand=False"
    n = record.reset_keys.shape[0]
    physics = ReplayPhysics(record.env.mj_model, n, device)
    draws = ReplayDraws(device)
    tenv = WarpJoystick(hydra, n, episode_length, device=device, physics=physics, draws=draws)
    if hook is not None:
        hook(tenv)

    draws.set_reset_keys(record.reset_keys)
    physics.queue(record.reset_data)
    state, priv = tenv.reset()
    run = TorchRun(
        env=tenv,
        reset_state=to_numpy(state),
        reset_priv=to_numpy(priv),
        reset_info=to_numpy(tenv.info),
        reset_write=_entry(physics.pop_log(), "reset"),
        steps=[],
    )
    for st in record.steps[: len(record.steps) if steps is None else steps]:
        draws.set_step_keys(st.rng_before)
        physics.queue(st.physics)
        state, priv, reward, done, truncation, terms = tenv.step(torch.as_tensor(st.action, device=device))
        log = physics.pop_log()
        run.steps.append(
            TorchStep(
                state=to_numpy(state),
                priv=to_numpy(priv),
                reward=to_numpy(reward),
                done=to_numpy(done),
                truncation=to_numpy(truncation),
                terms=to_numpy(dict(terms)),
                info=to_numpy(tenv.info),
                step_log=_entry(log, "step"),
                restore_log=_entry(log, "restore"),
            )
        )
    return run


# -- comparison ---------------------------------------------------------------------
def _worst_world(err) -> str:
    err = np.asarray(err)
    if err.ndim == 0:
        return "scalar"
    per_world = err.reshape(err.shape[0], -1).max(axis=1) if err.size else err
    return f"world {int(np.argmax(per_world))}" if per_world.size else "empty"


def compare(check, metric, expected, actual, where=None, tol=TASK, angle=False):
    """Assert actual (Torch, numpy) matches expected (JAX, numpy).

    Floats use tol and are recorded in parity_report. Ints and bools must be
    equal. Dicts recurse. With angle, floats compare modulo 2*pi.
    """
    where = metric if where is None else where
    tag = f"{check} {where}"
    if expected is None or actual is None:
        assert expected is None and actual is None, f"{tag}: JAX is {expected!r}, Torch is {actual!r}"
        return
    if isinstance(expected, dict) or isinstance(actual, dict):
        assert isinstance(expected, dict) and isinstance(actual, dict), f"{tag}: only one side is a dict"
        assert set(expected) == set(actual), f"{tag}: JAX keys {sorted(expected)}, Torch keys {sorted(actual)}"
        for k in sorted(expected):
            compare(check, metric, expected[k], actual[k], where=f"{where}.{k}", tol=tol, angle=angle)
        return
    e, a = np.asarray(expected), np.asarray(actual)
    assert e.shape == a.shape, f"{tag}: JAX shape {e.shape}, Torch shape {a.shape}"
    if e.dtype.kind == "f" or a.dtype.kind == "f":
        assert e.dtype == np.float32 and a.dtype == np.float32, f"{tag}: JAX {e.dtype}, Torch {a.dtype}"
        assert np.isfinite(e).all(), f"{tag}: JAX is not finite at {_worst_world(~np.isfinite(e))}"
        if angle:
            e64, a64 = e.astype(np.float64), a.astype(np.float64)
            wrapped = signals.wrap_angle(a64 - e64)
            # The report gets the wrapped error, so a 2*pi tie does not show as an error.
            parity_report.record(check, metric, e64, e64 + wrapped, tol=parity_report.tol_text(tol))
            err = np.abs(wrapped)
            bound = tol["atol"] + tol["rtol"] * np.abs(e64)
            assert (err <= bound).all(), (
                f"{tag}: angle error {float(err.max()):.3g} at {_worst_world(err - bound)} exceeds {tol}"
            )
            return
        parity_report.record(check, metric, e, a, tol=parity_report.tol_text(tol))
        diff = np.abs(a.astype(np.float64) - e.astype(np.float64))
        np.testing.assert_allclose(a, e, err_msg=f"{tag}, worst at {_worst_world(diff)}", **tol)
        return
    assert e.dtype == a.dtype, f"{tag}: JAX {e.dtype}, Torch {a.dtype}"
    np.testing.assert_array_equal(a, e, err_msg=f"{tag}, first mismatch at {_worst_world(a != e)}")
