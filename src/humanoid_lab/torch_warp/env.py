"""The joystick task on Torch tensors and MJWarp physics.

WarpJoystick keeps no task math. It builds the JAX Joystick once, as a
constants env, so every shared construction check runs, and reads its
resolved constants. Each step then makes the calls Joystick.step makes, on
the whole batch at once: gait_state.pre_physics, the physics,
signals.readings, gait_state.update, compose.total_reward and
signals.observe.

Three pieces are the Torch side's own:

- TorchDraws, the RNG adapter. It makes the draws the JAX adapters in
  envs/joystick.py and dr/randomize.py make, with the same dict keys and
  per-world shapes, from one torch.Generator per stream.
- The episode counter. JAX gets it from brax's EpisodeWrapper and
  BraxAutoResetWrapper, so it lives here and not in shared code.
- The respawn. A done world gets its first physics state and its first
  observation back, keeps its command, and has the reseed layers of
  gait_state.clear_done applied, as the JAX wrappers do.
"""

from __future__ import annotations

import copy
import dataclasses
import types

import numpy as np
import torch
import warp as wp

from humanoid_lab import paths, registry
from humanoid_lab.dr import transform
from humanoid_lab.envs import gait_state, joystick, progress, push, signals
from humanoid_lab.envs.constants import TaskConstants
from humanoid_lab.rewards import compose
from humanoid_lab.torch_warp.physics import WarpPhysics

TORCH_INDEX_INSTALL = (
    "python -m pip install torch --index-url https://download.pytorch.org/whl/cu126"
)

# Draw domains of TorchDraws. See its docstring for the streams in each.
RESET = 1
STEP = 2
RESET_COMMAND = 3
STEP_COMMAND = 4
PUSH = 5
DR = 6

# The JAX adapters fold in 0x100 + the table index (COMMAND_DRAWS,
# PUSH_DRAWS and DR_DRAWS). The Torch streams reuse that number as the index.
FOLD = 0x100


def _command_streams(domain, label):
    rows = [
        (domain, 0, 0, f"{label} vx"),
        (domain, 1, 0, f"{label} vy"),
        (domain, 2, 0, f"{label} wz"),
        (domain, 3, 0, f"{label} zero_on"),
    ]
    for name, idx, _prob_key, range_key in progress.COMMAND_DRAWS:
        rows.append((domain, FOLD + idx, 0, f"{label} {name}_on"))
        if range_key:
            rows.append((domain, FOLD + idx, 1, f"{label} {name}_vx"))
    return rows


# Every stream TorchDraws can open: (domain, index, sub, draw).
STREAMS = (
    (RESET, 2, 0, "reset pose_noise"),
    (RESET, 3, 0, "reset push_countdown"),
    (RESET, 4, 0, "reset mirror_u"),
    (STEP, 1, 0, "step noise"),
    (STEP, 4, 0, "step u_term"),
    *_command_streams(RESET_COMMAND, "reset command"),
    *_command_streams(STEP_COMMAND, "step command"),
    (PUSH, 0, 0, "push dir"),
    *[(PUSH, FOLD + idx, 0, f"push {name}") for name, idx, _, _ in push.PUSH_DRAWS if idx is not None],
    *[(DR, i, 0, f"dr split {i}") for i in range(5)],
    (DR, FOLD + 1, 0, "dr gain_scale"),
    (DR, FOLD + 1, 1, "dr kd_scale"),
    (DR, FOLD + 2, 0, "dr com_offset"),
    (DR, FOLD + 3, 0, "dr dof_damping"),
    (DR, FOLD + 3, 1, "dr dof_armature"),
    (DR, FOLD + 3, 2, "dr dof_frictionloss"),
    (DR, FOLD + 4, 0, "dr foot_scale"),
    (DR, FOLD + 5, 0, "dr motor_scale"),
    (DR, FOLD + 6, 0, "dr mass_add"),
)


# -- helpers ------------------------------------------------------------------
def check_device(name) -> torch.device:
    """The torch device for torch_warp.device. A CUDA device needs a CUDA
    torch build, a visible GPU and warp's CUDA runtime."""
    device = torch.device(name)
    if device.type == "cuda":
        if torch.version.cuda is None or not torch.cuda.is_available():
            raise RuntimeError(
                f"torch_warp.device={name} needs a CUDA build of torch and a visible GPU. "
                f"This torch is {torch.__version__} (CUDA {torch.version.cuda}). "
                f"Install CUDA torch from the PyTorch index: {TORCH_INDEX_INSTALL}"
            )
        if not wp.is_cuda_available():
            raise RuntimeError(
                f"torch_warp.device={name}: warp finds no CUDA device. Check the NVIDIA driver."
            )
    elif device.type != "cpu":
        raise ValueError(f"torch_warp.device must be cuda or cpu, got {name!r}")
    return device


def check_jax_on_cpu() -> None:
    """JAX builds the constants env and the checkpoints, and must stay off the GPU."""
    import jax

    if jax.default_backend() != "cpu":
        raise RuntimeError(
            f"JAX runs on {jax.default_backend()}, so it would take the GPU memory torch "
            "and warp need. Set JAX_PLATFORMS=cpu before the first jax import "
            "(humanoid_lab.torch_warp sets it when imported first)."
        )


def torch_constants(consts, device) -> TaskConstants:
    """consts with every numpy array field as a tensor on the device.

    Float arrays become float32, bool arrays torch.bool and int arrays
    int64. Python scalars, tuples, dicts, None and the actuator model are
    unchanged. A numpy mask next to a CUDA tensor fails, and torch.where
    rejects a numpy condition, so every array converts once, here.
    """
    out = {}
    for f in dataclasses.fields(consts):
        v = getattr(consts, f.name)
        if isinstance(v, np.ndarray):
            out[f.name] = _tensor(v, device)
    return dataclasses.replace(consts, **out)


def _tensor(a, device):
    # torch.tensor copies. The numpy arrays of resolved_constants() can be
    # read-only views of jax buffers.
    a = np.asarray(a)
    if a.dtype.kind == "f":
        return torch.tensor(a, dtype=torch.float32, device=device)
    if a.dtype.kind == "b":
        return torch.tensor(a, dtype=torch.bool, device=device)
    if a.dtype.kind in "iu":
        return torch.tensor(a, dtype=torch.int64, device=device)
    raise TypeError(f"no tensor conversion for dtype {a.dtype}")


def env_config_of(hydra):
    """The task's env ConfigDict from a plain Hydra dict, with no model built."""
    return registry.env_config(hydra["task"]["name"], hydra["task"].get("env"))


def component_sizes(consts_np, mj_model, heading_obs) -> dict:
    """name -> width of each catalog component, from the one shared catalog.

    signals.readings and signals.catalog run once on numpy zeros of the
    DATA_FIELDS shapes.
    """
    m = mj_model
    f32 = np.float32
    data = types.SimpleNamespace(
        qpos=np.zeros(m.nq, f32),
        qvel=np.zeros(m.nv, f32),
        qacc=np.zeros(m.nv, f32),
        sensordata=np.zeros(m.nsensordata, f32),
        actuator_force=np.zeros(m.nu, f32),
        xpos=np.zeros((m.nbody, 3), f32),
        subtree_com=np.zeros((m.nbody, 3), f32),
        site_xpos=np.zeros((m.nsite, 3), f32),
        geom_xpos=np.zeros((m.ngeom, 3), f32),
        cdof=np.zeros((m.nv, 6), f32),
    )
    info = {"last_action": np.zeros(m.nu, f32), "command": np.zeros(3, f32), "phase": np.zeros((), f32)}
    if heading_obs:
        info["yaw_ref"] = np.zeros((), f32)
    cat = signals.catalog(signals.readings(data, consts_np), info, consts_np)
    return {name: int(v.shape[-1]) for name, v in cat.items()}


# -- the RNG adapter ------------------------------------------------------------
class TorchDraws:
    """The Torch RNG adapter: the draws of the JAX adapters, one generator per stream.

    Each stream is a torch.Generator on the device, seeded with
    SeedSequence([seed, domain, index, sub]). That seed is the same on every
    OS. A generator is made on first use and cached by its key. A stream's
    samples never depend on which other streams exist, so turning an
    optional draw on or off leaves every other draw unchanged.

    index is the JAX key position: the split position for a split key, and
    0x100 + the table index for a fold-in draw. sub tells apart two draws
    made from one JAX key: the bernoulli (0) and the range draw (1) of a
    pure command draw, or the sub-key position of a split DR fold.

    | domain            | index        | sub  | draw                               |
    |-------------------|--------------|------|------------------------------------|
    | RESET 1           | 2            | 0    | pose_noise                         |
    | RESET 1           | 3            | 0    | push_countdown (ranged push)       |
    | RESET 1           | 4            | 0    | mirror_u (symmetry)                |
    | STEP 2            | 1            | 0    | noise (actor obs noise)            |
    | STEP 2            | 4            | 0    | u_term (no_progress)               |
    | RESET_COMMAND 3,  | 0, 1, 2      | 0    | vx, vy, wz                         |
    | STEP_COMMAND 4    | 3            | 0    | zero_on                            |
    |                   | 0x100 + idx  | 0    | <name>_on, COMMAND_DRAWS idx 1-6   |
    |                   | 0x100 + idx  | 1    | <name>_vx (pure_slow, fast, back)  |
    | PUSH 5            | 0            | 0    | dir                                |
    |                   | 0x101        | 0    | next_countdown                     |
    |                   | 0x102        | 0    | vel_z                              |
    |                   | 0x103        | 0    | ang                                |
    | DR 6              | 0-4          | 0    | floor_friction, base_scale,        |
    |                   |              |      | link_scale, gain_scalar, kd_scalar |
    |                   | 0x101        | 0, 1 | gain_scale, kd_scale               |
    |                   | 0x102        | 0    | com_offset                         |
    |                   | 0x103        | 0-2  | dof_damping, armature, frictionloss|
    |                   | 0x104        | 0    | foot_scale                         |
    |                   | 0x105        | 0    | motor_scale                        |
    |                   | 0x106        | 0    | mass_add                           |

    STREAMS lists the same keys. A uniform draw is lo + (hi - lo) * rand,
    a bernoulli is rand < p, and a randint is int32 like jax.random.randint.
    Torch samples are never compared with JAX samples. Only the dict keys
    and the shapes match the JAX adapters, so the shared code reads both.
    """

    def __init__(self, seed, device):
        self.seed = int(seed)
        self.device = torch.device(device)
        self._gens = {}

    def _seed(self, key) -> int:
        state = np.random.SeedSequence([self.seed, *key]).generate_state(1, np.uint64)[0]
        return int(state % np.uint64(2**63))

    def _gen(self, key):
        g = self._gens.get(key)
        if g is None:
            g = torch.Generator(device=self.device)
            g.manual_seed(self._seed(key))
            self._gens[key] = g
        return g

    def _bound(self, b):
        if isinstance(b, (list, tuple)):
            return torch.tensor([float(v) for v in b], dtype=torch.float32, device=self.device)
        return float(b)

    def _rand(self, key, shape):
        return torch.rand(shape, generator=self._gen(key), device=self.device, dtype=torch.float32)

    def uniform(self, key, shape, lo, hi):
        lo, hi = self._bound(lo), self._bound(hi)
        return lo + (hi - lo) * self._rand(key, shape)

    def bernoulli(self, key, shape, p):
        return self._rand(key, shape) < float(p)

    def randint(self, key, shape, lo, hi):
        """An int32 in [lo, hi], both inclusive."""
        return torch.randint(
            int(lo), int(hi) + 1, shape, generator=self._gen(key), device=self.device, dtype=torch.int32
        )

    def command(self, domain, n, command_cfg):
        """The draws behind n commands: jax_command_draws' keys, shapes (n,)."""
        c = command_cfg
        draws = {
            "vx": self.uniform((domain, 0, 0), (n,), c.vx[0], c.vx[1]),
            "vy": self.uniform((domain, 1, 0), (n,), c.vy[0], c.vy[1]),
            "wz": self.uniform((domain, 2, 0), (n,), c.wz[0], c.wz[1]),
        }
        for name, idx, prob_key, range_key in progress.COMMAND_DRAWS:
            p = c.get(prob_key, 0.0)
            if p:
                draws[f"{name}_on"] = self.bernoulli((domain, FOLD + idx, 0), (n,), p)
                if range_key:
                    lo, hi = c[range_key]
                    draws[f"{name}_vx"] = self.uniform((domain, FOLD + idx, 1), (n,), lo, hi)
        draws["zero_on"] = self.bernoulli((domain, 3, 0), (n,), c.zero_prob)
        return draws

    def push(self, n, push_cfg):
        """The draws behind n pushes: jax_push_draws' keys."""
        ranges = push.push_ranges(push_cfg)
        draws = {"dir": self.uniform((PUSH, 0, 0), (n, 2), -1.0, 1.0)}
        for name, idx, shape, _ in push.PUSH_DRAWS:
            if name == "dir" or name not in ranges:
                continue
            key = (PUSH, FOLD + idx, 0)
            lo, hi = ranges[name]
            if name == "next_countdown":
                draws[name] = self.randint(key, (n, *shape), lo, hi)
            else:
                draws[name] = self.uniform(key, (n, *shape), lo, hi)
        return draws

    def reset(self, n, cfg, action_size):
        """The draws behind n resets: jax_reset_draws' keys without 'rng'."""
        draws = {
            "command": self.command(RESET_COMMAND, n, cfg.command),
            "pose_noise": self.uniform((RESET, 2, 0), (n, action_size), -1.0, 1.0),
        }
        if cfg.push.enable and cfg.push.get("interval_steps_range", None):
            lo, hi = (int(v) for v in cfg.push.interval_steps_range)
            draws["push_countdown"] = self.randint((RESET, 3, 0), (n,), lo, hi)
        if cfg.symmetry.enable:
            draws["mirror_u"] = self._rand((RESET, 4, 0), (n,))
        return draws

    def step(self, n, cfg, actor_obs_size):
        """The draws behind n steps: jax_step_draws' keys, with the actor
        noise itself under 'noise' in place of 'rng' and 'noise_key'."""
        return {
            "noise": self.uniform((STEP, 1, 0), (n, actor_obs_size), -1.0, 1.0),
            "command": self.command(STEP_COMMAND, n, cfg.command),
            "push": self.push(n, cfg.push) if cfg.push.enable else None,
            "u_term": self._rand((STEP, 4, 0), (n,)) if cfg.no_progress.enable else None,
        }

    def dr(self, n, table):
        """The draws of a transform.DR_DRAWS table, shapes (n, *shape)."""
        draws = {}
        for name, (shape, lo, hi, slot) in table.items():
            if slot[0] == "split":
                key = (DR, slot[1], 0)
            else:
                _, idx, pos, _n = slot
                key = (DR, FOLD + idx, pos or 0)
            draws[name] = self.uniform(key, (n, *shape), lo, hi)
        return draws

    def state_dict(self) -> dict:
        return {"/".join(str(k) for k in key): g.get_state() for key, g in self._gens.items()}

    def load_state_dict(self, state) -> None:
        for name, s in state.items():
            self._gen(tuple(int(k) for k in name.split("/"))).set_state(s)


# -- the env ----------------------------------------------------------------------
class WarpJoystick:
    """The joystick task on a batch of MJWarp worlds, stepped from Torch.

    The env protocol the PPO loop uses:

    - attributes num_envs, action_size, actor_obs_size, privileged_obs_size,
      device, term_names, actor_obs_names and env_config;
    - reset() -> (state, priv): float32 (N, actor_obs_size) and
      (N, privileged_obs_size);
    - step(action) -> (state, priv, reward, done, truncation, terms). action
      is (N, action_size), already squashed into [-1, 1]. reward is float32
      (N,), done and truncation are bool (N,), and terms maps each name in
      term_names to its unscaled (N,) reward term.

    hydra is a plain dict: OmegaConf.to_container(cfg, resolve=True), or a
    run.json hydra_config. physics and draws are injectable for tests/parity.
    """

    def __init__(self, hydra, num_envs, episode_length, device="cuda:0", seed=0, physics=None, draws=None):
        self.device = check_device(device)
        env_hydra = hydra["task"].get("env") or {}
        if (env_hydra.get("sim") or {}).get("backend") == "jax":
            raise ValueError(
                "task.env.sim.backend=jax: the Torch trainer steps MJWarp physics. "
                "Leave sim.backend unset, or train with the Brax trainer (./run.sh train)."
            )
        self.num_envs = n = int(num_envs)
        self.episode_length = int(episode_length)

        # The constants env: one JAX Joystick on the jax backend, so every
        # shared construction check runs and nothing is put on the GPU.
        overrides = copy.deepcopy(env_hydra)
        overrides["sim"] = {**(overrides.get("sim") or {}), "backend": "jax", "num_envs": 1}
        self.constants_env_overrides = {"sim.backend": "jax", "sim.num_envs": 1}
        actuators = hydra.get("actuators") or {}
        self.jenv = registry.make_env(
            hydra["task"]["name"],
            paths.REPO_ROOT / hydra["robot"]["dir"],
            actuators["name"],
            overrides,
            actuators.get("overrides") or {},
        )
        if type(self.jenv) is not joystick.Joystick:
            raise ValueError(
                f"the Torch trainer runs the joystick task only, got {type(self.jenv).__name__}. "
                "The Torch env emits no per-step metrics, so sizing stays a JAX rollout."
            )
        check_jax_on_cpu()

        self.cfg = self.jenv.config
        self.consts_np = self.jenv.resolved_constants()
        if self.consts_np.naconmax_per_env is None or self.consts_np.njmax is None:
            raise ValueError(
                f"MJWarp needs contact budgets, and robot '{self.jenv.robot_spec.name}' has no "
                "sim_budget block in its robot.yaml (and the sim config sets none). Measure with "
                "./run.sh check-contacts and record naconmax_per_env/njmax in robot.yaml"
            )
        self.consts = torch_constants(self.consts_np, self.device)

        self.mj_model = self.jenv.mj_model
        self.robot_spec = self.jenv.robot_spec
        self.dt = float(self.jenv.dt)
        self.n_substeps = int(self.jenv.n_substeps)
        self.action_size = int(self.jenv.action_size)

        self.actor_obs_names = list(self.jenv.actor_obs_names)
        self.priv_obs_names = list(self.cfg.obs.privileged)
        # Joystick.__init__ decides whether the task has a heading observation.
        self.heading_obs = bool(self.consts_np.heading_obs)
        self.sizes = component_sizes(self.consts_np, self.mj_model, self.heading_obs)
        self.actor_obs_size = sum(self.sizes[k] for k in self.actor_obs_names)
        self.privileged_obs_size = sum(self.sizes[k] for k in self.priv_obs_names)
        self.term_names = joystick.reward_term_names(self.cfg.reward.scales)

        env_config = self.cfg.to_dict()
        env_config["sim"] = {**env_config["sim"], "backend": "warp", "num_envs": n}
        self.env_config = env_config

        self.draws = draws or TorchDraws(seed, self.device)

        # Domain randomization through dr/transform, as dr/randomize.py does it.
        self.domain_rand = bool(hydra.get("domain_rand"))
        dr_cfg = hydra.get("dr") or {}
        fields = None
        foot_priority = False
        if not self.domain_rand:
            transform.check_dr_enabled(False, dr_cfg)
        else:
            rc = transform.resolved_cfg(dr_cfg)
            foot_priority = bool(rc["foot_friction"]["enable"])
            m = self.mj_model
            dims = {"nbody": m.nbody, "nu": m.nu, "nv": m.nv, "nfoot": len(self.consts_np.foot_geom_ids)}
            table = transform.DR_DRAWS(rc, dims)
            masks = {
                k: _tensor(v, self.device)
                for k, v in transform.dr_masks(
                    m.ngeom,
                    m.nbody,
                    self.consts_np.floor_geom_id,
                    self.consts_np.root_body,
                    self.consts_np.foot_geom_ids,
                ).items()
            }

        self.physics = physics or WarpPhysics(
            self.mj_model,
            n,
            self.device,
            self.consts_np.naconmax_per_env,
            self.consts_np.njmax,
            self.n_substeps,
            foot_geom_ids=self.consts_np.foot_geom_ids,
            foot_priority=foot_priority,
        )
        if self.domain_rand:
            # make_domain_randomize's base field list.
            names = ["geom_friction", "body_mass", "actuator_gainprm", "actuator_biasprm", "actuator_forcerange"]
            if "com_offset" in table:
                names.append("body_ipos")
            if "dof_damping" in table:
                names += ["dof_damping", "dof_armature", "dof_frictionloss"]
            base = self.physics.base_fields(names)
            fields = transform.dr_fields(base, masks, self.draws.dr(n, table))
            self.physics.set_dr(fields)

        self.flags = gait_state.reseed_flags(self.cfg)
        c = self.consts
        self._mirror_maps = (
            (c.state_perm, c.state_sign, c.priv_perm, c.priv_sign) if self.cfg.symmetry.enable else None
        )
        self.info = None
        self.first = None
        self.steps = None
        self.done = None

    # -- helpers ------------------------------------------------------------
    def _mirror(self, info):
        if self._mirror_maps is None:
            return None
        return (info["mirror"], *self._mirror_maps)

    def _readings(self):
        """signals.readings as a snapshot.

        Some readings are the physics views themselves or slices of them,
        and the next physics step overwrites those in place. info keeps
        readings across steps (last_torque is actuator_force), so each one
        is cloned.
        """
        return {k: v.clone() for k, v in signals.readings(self.physics.data, self.consts).items()}

    # -- protocol -------------------------------------------------------------
    def reset(self):
        n, c, cfg = self.num_envs, self.consts, self.cfg
        d = self.draws.reset(n, cfg, self.action_size)
        command = progress.command_from_draws(d["command"], cfg.command)

        qpos = c.reset_qpos.expand(n, -1).clone()
        qpos[:, c.qadr] += gait_state.reset_offset(d["pose_noise"], float(cfg.get("reset_noise", 0.0)))
        qvel = torch.zeros(n, self.mj_model.nv, dtype=torch.float32, device=self.device)
        ctrl = c.neutral_ctrl.expand(n, -1).clone()
        self.physics.reset_worlds(qpos, qvel, ctrl)

        r = self._readings()
        info = gait_state.reset_state(
            command,
            c,
            cfg,
            yaw=r["yaw"] if self.heading_obs else None,
            push_countdown=d.get("push_countdown"),
            mirror_u=d.get("mirror_u"),
        )
        # No noise: the JAX reset passes no rng to the observation.
        obs = signals.observe(
            signals.catalog(r, info, c), self.actor_obs_names, self.priv_obs_names, mirror=self._mirror(info)
        )
        state, priv = obs["state"], obs["privileged_state"]
        self.first = {
            "qpos": qpos.clone(),
            "qvel": qvel.clone(),
            "ctrl": ctrl.clone(),
            "state": state.clone(),
            "priv": priv.clone(),
            "yaw": r["yaw"].clone(),
        }
        self.info = info
        # brax's EpisodeWrapper keeps steps as a float.
        self.steps = torch.zeros(n, dtype=torch.float32, device=self.device)
        self.done = torch.zeros(n, dtype=torch.bool, device=self.device)
        return state, priv

    def step(self, action):
        n, c, cfg, first = self.num_envs, self.consts, self.cfg, self.first

        # BraxAutoResetWrapper zeroes the counter at the start of the step
        # after done.
        steps = torch.where(self.done, torch.zeros_like(self.steps), self.steps)

        d = self.draws.step(n, cfg, self.actor_obs_size)
        info, action, ctrl, kick = gait_state.pre_physics(self.info, action, c, cfg, d["push"])
        if kick is not None:
            # Joystick.step's write order.
            qvel = self.physics.data.qvel
            bv = c.base_vadr
            qvel[:, bv : bv + 2] += kick["planar"]
            if kick["z"] is not None:
                qvel[:, bv + 2] += kick["z"]
            if kick["ang"] is not None:
                qvel[:, bv + 3 : bv + 6] += kick["ang"]

        self.physics.step(ctrl)
        r = self._readings()
        info, rewards, env_done, _aux = gait_state.update(info, r, action, c, cfg, d)
        reward = compose.total_reward(rewards, cfg.reward.scales, self.dt)
        obs = signals.observe(
            signals.catalog(r, info, c),
            self.actor_obs_names,
            self.priv_obs_names,
            noise=d["noise"],
            obs_noise=cfg.obs_noise,
            mirror=self._mirror(info),
        )

        # brax's EpisodeWrapper.
        steps = steps + 1.0
        timeout = steps >= self.episode_length
        done = env_done | timeout
        truncation = timeout & ~env_done

        # BraxAutoResetWrapper(full_reset=False): done worlds get the first
        # physics state and the first observation, and keep their command.
        self.physics.restore_worlds(done, first["qpos"], first["qvel"], first["ctrl"])
        state = torch.where(done[:, None], first["state"], obs["state"])
        priv = torch.where(done[:, None], first["priv"], obs["privileged_state"])
        if self.flags:
            # One call with every flag equals the stacked JAX reseed
            # wrappers, because the layers touch different keys.
            info = gait_state.clear_done(
                info,
                done,
                self.flags,
                {"progress_ema": progress.cmd_speed(info["command"]), "yaw_ref": first["yaw"]},
            )

        self.info = info
        self.steps = steps
        self.done = done
        return state, priv, reward, done, truncation, rewards

    def state_dict(self) -> dict:
        return {"draws": self.draws.state_dict()}

    def load_state_dict(self, state) -> None:
        self.draws.load_state_dict(state["draws"])
