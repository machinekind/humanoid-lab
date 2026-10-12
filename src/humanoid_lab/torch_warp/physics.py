"""MJWarp physics for the Torch env, stepped from Torch tensors.

WarpPhysics owns one MJWarp model and one batch of data.

- put_model runs on a deep copy of the MuJoCo model, so the caller's model is
  never mutated. make_data gets the resolved contact and constraint budgets
  through envs/backend.py's data_budget_kwargs.
- Graph capture runs on CUDA only. Each graph is captured once on a private
  warp stream and launches on torch's current stream, so torch ops and warp
  launches stay ordered on one stream.
  On the CPU device MJWarp runs without capture.
- The data fields are exposed as wp.to_torch views. Writing a view writes the
  warp buffer, with no copy.
- set_dr installs per-world model fields. The reset and respawn writes go
  through reset_worlds and restore_worlds.

The physics interface WarpJoystick uses is: data, step, reset_worlds,
restore_worlds, set_dr and peaks. Commit 5's ReplayPhysics implements the
same names.
"""

from __future__ import annotations

import copy
import types

import numpy as np
import torch
import warp as wp
from mujoco.mjx.third_party import mujoco_warp as mjw
from mujoco.mjx.third_party.mujoco_warp._src import types as mjw_types

from humanoid_lab.envs import backend, signals

# The warp dtype of each model field that domain randomization may replace.
# The set matches dr/randomize.py's per-world fields.
DR_DTYPES = {
    "geom_friction": wp.vec3,
    "body_mass": wp.float32,
    "actuator_gainprm": mjw_types.vec10f,
    "actuator_biasprm": mjw_types.vec10f,
    "actuator_forcerange": wp.vec2,
    "body_ipos": wp.vec3,
    "dof_damping": wp.float32,
    "dof_armature": wp.float32,
    "dof_frictionloss": wp.float32,
}

# Data fields exposed as torch views. nacon and nefc are the live budget
# counters that peaks() reads.
_VIEW_FIELDS = (*signals.DATA_FIELDS, "ctrl", "qacc_warmstart", "time", "nacon", "nefc")


class WarpPhysics:
    """One MJWarp model and a batch of num_envs worlds on one device."""

    def __init__(
        self,
        mj_model,
        num_envs,
        device,
        naconmax_per_env,
        njmax,
        n_substeps,
        foot_geom_ids=None,
        foot_priority=False,
    ):
        if naconmax_per_env is None or njmax is None:
            raise ValueError(
                "MJWarp needs contact budgets. Set task.env.sim.naconmax_per_env "
                "and njmax, or measure with ./run.sh check-contacts and record "
                "naconmax_per_env/njmax in robot.yaml"
            )
        self.num_envs = int(num_envs)
        self.n_substeps = int(n_substeps)
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())

        self.mj_model = copy.deepcopy(mj_model)
        if foot_priority:
            # dr/randomize.py: foot geoms win friction outright. A static
            # field, so it is set on the host model before put_model.
            self.mj_model.geom_priority[np.asarray(foot_geom_ids)] = 1

        wp.init()
        self.wdev = wp.get_device(str(self.device))
        is_cuda = self.device.type == "cuda"
        if self.wdev.is_cuda != is_cuda or (is_cuda and self.wdev.ordinal != self.device.index):
            raise ValueError(f"warp device {self.wdev} does not match torch device {self.device}")

        with wp.ScopedDevice(self.wdev):
            self.m = mjw.put_model(self.mj_model)
            self.d = mjw.make_data(
                self.mj_model,
                nworld=self.num_envs,
                **backend.data_budget_kwargs("warp", naconmax_per_env, njmax, self.num_envs),
            )

        views = {f: wp.to_torch(getattr(self.d, f)) for f in _VIEW_FIELDS}
        for f, v in views.items():
            setattr(self, f, v)
        # signals.readings reads this namespace. The env adds kicks to
        # data.qvel in place.
        self.data = types.SimpleNamespace(
            **{f: views[f] for f in (*signals.DATA_FIELDS, "ctrl", "qacc_warmstart", "time")}
        )

        self._keep = []
        self._graph = None
        self._fwd_graph = None
        self._graph_checked = False
        # CUDA cannot capture on torch's default stream, which is the legacy
        # null stream. Capture records on this private stream instead, and
        # every launch runs on torch's current stream.
        self._capture_stream = wp.Stream(self.wdev) if is_cuda else None

    # -- domain randomization --------------------------------------------------
    def base_fields(self, names):
        """The unbatched mj_model fields as float32 tensors on the device."""
        return {
            n: torch.as_tensor(np.asarray(getattr(self.mj_model, n), dtype=np.float32), device=self.device)
            for n in names
        }

    def set_dr(self, fields):
        """Install per-world model fields. Each tensor has leading dim num_envs.

        The kernels read field[worldid % field.shape[0]], so a leading dim of
        num_envs gives each world its own value. Derived constants such as
        body_subtreemass are never recomputed. The JAX path does not
        recompute them either, and parity depends on that.
        """
        if self._graph is not None or self._fwd_graph is not None:
            raise RuntimeError("set_dr after graph capture: the captured graph holds the old field pointers")
        for name, t in fields.items():
            if t.shape[0] != self.num_envs:
                raise ValueError(f"DR field {name} has leading dim {t.shape[0]}, expected {self.num_envs}")
            t = t.to(self.device).contiguous()
            self._keep.append(t)
            setattr(self.m, name, wp.from_torch(t, dtype=DR_DTYPES[name]))

    # -- stepping --------------------------------------------------------------
    def _run(self, fn, attr):
        """Run fn once. On CUDA the first call compiles, then captures a graph,
        and later calls launch that graph on torch's current stream."""
        if not self.wdev.is_cuda:
            with wp.ScopedDevice(self.wdev):
                fn()
            return
        stream = wp.stream_from_torch(self.device)
        graph = getattr(self, attr)
        if graph is None:
            with wp.ScopedDevice(self.wdev), wp.ScopedStream(stream):
                fn()  # compiles the kernels, and does this call's work
            # MJWarp passes no stream to its launches, so they go to the
            # device's current stream. ScopedStream makes the capture stream
            # current. Without it the launches run eagerly on another stream
            # and the graph records nothing.
            with wp.ScopedStream(self._capture_stream), wp.ScopedCapture() as cap:
                fn()
            setattr(self, attr, cap.graph)
            return
        with wp.ScopedStream(stream):
            wp.capture_launch(graph, stream=stream)

    def _substeps(self):
        for _ in range(self.n_substeps):
            mjw.step(self.m, self.d)

    def _forward(self):
        mjw.forward(self.m, self.d)

    def step(self, ctrl):
        """Write ctrl into every world, then run n_substeps physics steps."""
        self.ctrl.copy_(ctrl)
        check = self._graph is not None and not self._graph_checked
        if check:
            t0 = self.time.clone()
        self._run(self._substeps, "_graph")
        if check:
            # The first launch of the captured graph must advance the clock.
            # An empty capture would freeze physics without an error.
            self._graph_checked = True
            if not bool((self.time > t0).all()):
                raise RuntimeError("captured physics graph did nothing: the clock did not advance")

    def forward(self):
        self._run(self._forward, "_fwd_graph")

    # -- reset and respawn writes ------------------------------------------------
    def reset_worlds(self, qpos, qvel, ctrl):
        """Write every world's state, zero the warmstart and the clock, then forward."""
        self.qpos.copy_(qpos)
        self.qvel.copy_(qvel)
        self.ctrl.copy_(ctrl)
        self.qacc_warmstart.zero_()
        self.time.zero_()
        self.forward()

    def restore_worlds(self, mask, qpos, qvel, ctrl):
        """Write the state of the masked worlds only.

        No forward runs. The next step recomputes from qpos and qvel, as the
        JAX respawn restores its first data.
        """
        m = mask[:, None]
        self.qpos.copy_(torch.where(m, qpos, self.qpos))
        self.qvel.copy_(torch.where(m, qvel, self.qvel))
        self.ctrl.copy_(torch.where(m, ctrl, self.ctrl))
        self.qacc_warmstart.copy_(torch.where(m, torch.zeros_like(self.qacc_warmstart), self.qacc_warmstart))
        self.time.copy_(torch.where(mask, torch.zeros_like(self.time), self.time))

    def peaks(self):
        """(contacts in the shared pool, max constraint rows in one world).

        This syncs with the device on CUDA. Call it at log time only.
        """
        return int(self.nacon.max().item()), int(self.nefc.max().item())
