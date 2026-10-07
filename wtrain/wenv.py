"""Torch + MJWarp port of envs/joystick.py, for training on a Windows box.

JAX has no CUDA build on Windows, so the MJX training env cannot use the GPU
here. This module runs the same task on MJWarp directly: the copy vendored
inside mujoco-mjx 3.10.0 (mujoco.mjx.third_party.mujoco_warp, warp-lang
1.13.0), which is the physics the keeper checkpoints were trained on.

Every constant (address tables, default pose, action scale, ctrl clip, foot
tables, reward weights, config) is read off the original JAX env built from
the run's own run.json, so nothing is restated here. What is restated is the
per-step math of Joystick.step/_compute_rewards/_build_obs, the playground
EpisodeWrapper + BraxAutoResetWrapper(full_reset=False) semantics, train.py's
GaitReseedWrapper, and dr/randomize.py. wtrain/parity.py checks the math
against the JAX env term by term.

Supported config: the locomotion-v1 / noclock-v1 shape (symmetry off,
no_progress off, absolute tracking kernels, no far blend, no product gate,
no shaping gate, no orientation cone, real_pose_ref off). Anything else
refuses in __init__ instead of silently diverging.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
import warp as wp
from mujoco.mjx.third_party import mujoco_warp as mjw
from mujoco.mjx.third_party.mujoco_warp._src import types as mjw_types

from humanoid_lab import paths

SPEED_DEADBAND = 0.05  # envs/progress.py
YAW_SPEED_WEIGHT = 0.3  # envs/progress.py
_PHASE_OFFSETS = (0.0, math.pi)  # envs/joystick.py

# Reward terms this port computes. A term with a nonzero scale outside this
# set refuses at construction.
_TERMS = (
    "tracking_lin_vel", "tracking_ang_vel", "lin_vel_z", "ang_vel_xy", "orientation",
    "torques", "torque_rate", "action_rate", "action_accel", "energy", "pose",
    "feet_air_time", "feet_slip", "feet_phase", "stand_still", "termination",
    "torque_limit", "feet_apex", "feet_landing", "pose_l1", "joint_pos_limits",
    "joint_vel", "joint_acc", "upward", "feet_distance", "knee_distance",
    "feet_contact_without_cmd", "feet_air_time_biped", "gait_symmetry_income",
)


def load_run(run_dir) -> dict:
    return json.loads((Path(run_dir) / "run.json").read_text())


def make_jax_env(run: dict, num_envs: int = 1):
    """The original Joystick env from run.json, on the jax backend (CPU).

    Used for its constants and as the parity oracle; never stepped in
    training."""
    from humanoid_lab.registry import make_env

    hydra = run["hydra_config"]
    env_overrides = json.loads(json.dumps(hydra["task"]["env"]))
    env_overrides.setdefault("sim", {})["backend"] = "jax"
    env_overrides["sim"]["num_envs"] = num_envs
    robot_dir = paths.REPO_ROOT / hydra["robot"]["dir"]
    return make_env(
        run["task"], robot_dir, hydra["actuators"]["name"], env_overrides,
        hydra["actuators"].get("overrides") or {},
    )


def _t(x, dev, dtype=torch.float32):
    return torch.as_tensor(np.asarray(x), dtype=dtype, device=dev)


def quat_rotate_inv(q, v):
    """Rotate v (…,3) by the inverse of unit quaternion q (…,4) wxyz."""
    w, xyz = q[..., :1], q[..., 1:]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v - w * t + torch.cross(xyz, t, dim=-1)


class WarpJoystick:
    def __init__(self, run: dict, num_envs: int, seed: int = 0, dr: bool | None = None,
                 device: str = "cuda:0", njmax: int | None = None, naconmax_per_env: int | None = None):
        self.run = run
        self.num_envs = N = num_envs
        self.dev = dev = torch.device(device)
        self.gen = torch.Generator(device=dev)
        self.gen.manual_seed(seed)

        E = make_jax_env(run)
        self.jenv = E
        cfg = E._config
        self.cfg = cfg
        self._check_supported(cfg)

        mjm = E.mj_model
        self.mjm = mjm
        self.n_substeps = int(E.n_substeps)
        self.dt = float(E.dt)
        self.nu = nu = mjm.nu
        self.n_feet = E._n_feet

        # -- constants, straight off the JAX env --------------------------------
        self.qadr = _t(E._qadr, dev, torch.long)
        self.vadr = _t(E._vadr, dev, torch.long)
        self.default_pose = _t(E._default_pose, dev)
        self.pose_anchor = _t(E._pose_anchor, dev)
        self.action_scale = _t(E._action_scale, dev)
        self.ctrl_lo = _t(E._ctrl_lo, dev)
        self.ctrl_hi = _t(E._ctrl_hi, dev)
        self.neutral_ctrl = _t(E._neutral_ctrl, dev)
        self.reset_qpos = _t(E._reset_qpos, dev)
        self.base_qadr, self.base_vadr = int(E._base_qadr), int(E._base_vadr)
        sa = E._sensor_adr
        for k in ("gyro", "quat", "linvel"):
            if k not in sa:
                raise ValueError(f"port expects a '{k}' sensor")
        self.s_gyro, self.s_quat, self.s_linvel = sa["gyro"], sa["quat"], sa["linvel"]
        self.foot_site_ids = _t(E._foot_site_ids, dev, torch.long)
        self.foot_geom_ids = _t(E._foot_geom_ids, dev, torch.long)
        self.foot_geom_radius = _t(E._foot_geom_radius, dev)
        self.foot_geom_foot_idx = _t(E._foot_geom_foot_idx, dev, torch.long)
        self.foot_sole_offset = _t(E._foot_site_sole_offset, dev)
        self.pose_l1_weight = _t(E._pose_l1_weight, dev)
        self.pose_weight = _t(E._pose_weight, dev)
        self.soft_lo = _t(E._soft_lo_j, dev)
        self.soft_hi = _t(E._soft_hi_j, dev)
        self.torque_cap = _t(E._torque_cap, dev)
        self.ankle_pair = E._ankle_pair
        self.knee_pair = E._knee_pair
        self.cmd_vmax = float(E._cmd_vmax)

        # Foot point-velocity jacobian mask (mjx.support.jac): dofs whose body
        # is the foot site's body or one of its ancestors.
        site_body = np.asarray(E._foot_site_body)
        mask = np.zeros((self.n_feet, mjm.nv), dtype=np.float32)
        for f, b in enumerate(site_body):
            anc = set()
            while True:
                anc.add(int(b))
                if b == 0:
                    break
                b = mjm.body_parentid[b]
            mask[f] = [1.0 if int(mjm.dof_bodyid[i]) in anc else 0.0 for i in range(mjm.nv)]
        self.foot_dof_mask = _t(mask, dev)
        self.foot_root = _t(mjm.body_rootid[site_body], dev, torch.long)

        rw = cfg.reward
        self.scales = {k: float(v) for k, v in dict(rw.scales).items()}
        for k, v in self.scales.items():
            if v and k not in _TERMS:
                raise ValueError(f"reward term '{k}' (scale {v}) is not ported")
        self.terms = [k for k in _TERMS if self.scales.get(k, 0.0)]

        # Warp-only reward terms, configured by run.json's top-level
        # `warp_reward` block (outside hydra task.env, so the JAX env that
        # eval/video rebuild from the same run.json never sees them).
        #   knee_swing {scale, target}: per swing leg, knee flexion beyond the
        #     anchor as a fraction of (target - anchor), clipped to [0, 1].
        #   arm_swing {scale, k, sigma}: contralateral arm swing,
        #     exp(-sum((d_arm_pitch[side] - k * d_thigh_pitch[other])^2) / sigma).
        #     On this robot +arm_pitch swings the arm back and -thigh_pitch
        #     swings the leg forward, so a forward left leg with a forward
        #     right arm makes both offsets negative: k > 0.
        #     Optional k_elbow: the elbow also flexes on the forward swing,
        #     flex_target = k_elbow * max(0, -d_arm_pitch), flex = straight - q.
        #   elbow_hyperext {scale (negative), straight}: L1 excursion of the
        #     elbow past straight (q > straight bends the forearm backward).
        #   forward_lean {scale, max_deg, vx_max, sigma}: while commanded
        #     forward, exp(-(g_body_x - sin(target))^2 / sigma) with target =
        #     max_deg * clip(cmd_vx / vx_max, 0, 1). g_body_x = +sin(pitch) for a
        #     nose-down lean (base x is forward).
        #   push_off {scale, p_ref, behind}: per stance foot that sits more
        #     than `behind` m behind the base (base frame) while commanded
        #     forward, the ankle-pitch plantarflexion power
        #     max(tau, 0) * max(qdot, 0) / p_ref, clipped to [0, 1]
        #     (+ankle_pitch is toes-down on this robot).
        #   step_length {scale, [target_a, target_b], [power]}: on each landing while
        #     commanded forward, how far the landing foot sits ahead of the
        #     other foot (base frame x) as a fraction of the target step,
        #     clipped to [0, 1]: it asks for the stride, it does not pay for
        #     overreaching it. Target: target_a + target_b * cmd_vx when given
        #     (clock-free runs), else the step the gait clock implies,
        #     cmd_vx / (2 * freq).
        #   foot_reach {scale, target_a, target_b, [power], [x_min]}: on each landing while
        #     commanded forward, how far ahead of the base (base frame x) the
        #     landing foot touches down, as a fraction of target_a + target_b *
        #     cmd_vx, clipped to [0, 1] and raised to `power` (default 2). A foot
        #     that lands under or behind the pelvis is a catch, not a step.
        #   hip_swing {scale, target}: per swing leg while moving, hip flexion
        #     beyond the anchor (-thigh_pitch is forward on this robot) as a
        #     fraction of `target` rad (scaled by the speed lift factor),
        #     clipped to [0, 1].
        #   knee_swing's optional hip_target: pay min(knee fraction, hip
        #     fraction) instead of the knee alone, so the flexion counts only
        #     with the thigh swinging forward -- a bare knee bend in swing kicks
        #     the shank backward (heel to the glutes).
        #   swing_back {scale (negative)}: per swing foot, the squared backward
        #     speed of the foot relative to the base, in the base frame.
        #   progress {scale}: while commanded forward, the served fraction of the
        #     commanded forward speed, clip(v_x / cmd_vx, 0, 1). Linear, so a
        #     shuffle slower than the command keeps a gradient where the
        #     exp tracking kernel has gone flat.
        #   step_symmetry {scale, sigma_x, sigma_y, sigma_apex, sigma_air}: on
        #     each landing, compare the landing foot's step with the other
        #     foot's last recorded step on four features -- touchdown x and |y|
        #     of the foot site in the base frame, the swing's apex and its air
        #     time -- and pay the mean of exp(-diff^2 / sigma^2) over the four.
        #     Armed once the other foot has a recorded landing (restarts on
        #     every respawn, like the other per-foot gait trackers).
        self.extra = dict(run.get("warp_reward") or {})
        names = list(E._robot_spec.actuated_joints)
        self.elbow_idx = _t([names.index("left_elbow_pitch_joint"), names.index("right_elbow_pitch_joint")],
                            dev, torch.long)
        sides = [("left" if "left" in s else "right") for s in E._robot_spec.foot_sites]
        self.ankle_idx = _t([names.index(f"{s}_ankle_pitch_joint") for s in sides], dev, torch.long)
        self.hip_pitch_idx = _t([names.index(f"{s}_thigh_pitch_joint") for s in sides], dev, torch.long)
        for k, spec in self.extra.items():
            if k not in ("knee_swing", "arm_swing", "elbow_hyperext", "forward_lean", "push_off", "step_symmetry",
                         "step_length", "progress", "landing_tax", "foot_reach", "hip_swing", "swing_back",
                         "swing_forward", "knee_extend", "flight", "heading_hold", "lateral_hold",
                         "march"):
                raise ValueError(f"unknown warp_reward term '{k}'")
            self.scales[k] = float(spec["scale"])
            self.terms.append(k)
        self.knee_qidx = _t(E._foot_ordered_group_qidx("knee"), dev, torch.long)
        self.arm_idx = _t([names.index("left_arm_pitch_joint"), names.index("right_arm_pitch_joint")], dev, torch.long)
        self.thigh_other_idx = _t([names.index("right_thigh_pitch_joint"), names.index("left_thigh_pitch_joint")],
                                  dev, torch.long)

        # obs noise scale vector for the actor list
        comp_size = {"gyro": 3, "gravity": 3, "joint_pos": nu, "joint_vel": nu, "last_action": nu,
                     "command": 3, "phase": 2 * self.n_feet, "heading_err": 1}
        self.state_names = list(cfg.obs.state)
        self.priv_names = list(cfg.obs.privileged)
        noise = cfg.obs_noise
        self.obs_noise = torch.cat([
            torch.full((comp_size[n],), float(noise.get(n, 0.0)), device=dev) for n in self.state_names
        ])
        self.state_size = int(self.obs_noise.numel())

        # -- device model + data ----------------------------------------------
        wp.init()
        self.wdev = wp.get_device(str(dev))
        budget = E._robot_spec.sim_budget
        self.naconmax_per_env = naconmax_per_env or int(budget["naconmax_per_env"])
        self.njmax = njmax or int(budget["njmax"])
        dr_on = bool(run["hydra_config"].get("domain_rand", False)) if dr is None else dr
        if dr_on and self._dr_cfg()["foot_friction"]["enable"]:
            # dr/randomize.py: foot geoms win friction outright. A static
            # field, so it is set on the host model before put_model.
            mjm.geom_priority[np.asarray(E._foot_geom_ids)] = 1
        with wp.ScopedDevice(self.wdev):
            self.m = mjw.put_model(mjm)
            self.d = mjw.make_data(mjm, nworld=N, naconmax=self.naconmax_per_env * N, njmax=self.njmax)
        self._keep = []
        if dr_on:
            self._domain_randomize()

        tt = wp.to_torch
        d = self.d
        self.qpos, self.qvel, self.ctrl = tt(d.qpos), tt(d.qvel), tt(d.ctrl)
        self.qacc, self.qacc_warmstart = tt(d.qacc), tt(d.qacc_warmstart)
        self.sensordata, self.actuator_force = tt(d.sensordata), tt(d.actuator_force)
        self.xpos, self.site_xpos, self.geom_xpos = tt(d.xpos), tt(d.site_xpos), tt(d.geom_xpos)
        self.cdof, self.subtree_com = tt(d.cdof), tt(d.subtree_com)
        self.time = tt(d.time)
        self.nacon = tt(d.nacon)

        self.heading_obs = "heading_err" in (self.state_names + self.priv_names)
        self.deadband = float(cfg.command.get("deadband", SPEED_DEADBAND))
        self._down = torch.tensor([0.0, 0.0, -1.0], device=dev).expand(N, 3).contiguous()
        self._phase_off = torch.tensor(_PHASE_OFFSETS, device=dev)
        pc = cfg.push
        rp, yw = float(pc.get("ang_vel_rp", 0.0)), float(pc.get("ang_vel_yaw", 0.0))
        self._ang_hi = torch.tensor([rp, rp, yw], device=dev)
        self._graph = None
        self._fwd_graph = None
        self.info = {}
        self.first = {}

    # ------------------------------------------------------------------------
    @staticmethod
    def _check_supported(cfg):
        bad = []
        if cfg.get("symmetry") is not None and cfg.symmetry.get("enable", False):
            bad.append("symmetry")
        if cfg.no_progress.enable:
            bad.append("no_progress")
        r = cfg.reward
        for k in ("tracking_relative", "tracking_product", "shaping_tracking_gate"):
            if r.get(k, False):
                bad.append(k)
        if r.get("tracking_far_weight", 0.0):
            bad.append("tracking_far_weight")
        if r.get("orientation_tol_deg", 0.0):
            bad.append("orientation_tol_deg")
        if cfg.get("real_pose_ref", False):
            bad.append("real_pose_ref")
        if cfg.gait.air_time_cap and r.scales.feet_air_time:
            bad.append("gait.air_time_cap")
        for k in ("pure_wz_prob", "pure_vy_prob", "pure_fast_prob", "pure_back_prob"):
            if cfg.command.get(k, 0.0):
                bad.append(f"command.{k}")
        if not (cfg.push.enable and cfg.push.get("interval_steps_range", None)):
            bad.append("push without interval_steps_range")
        if bad:
            raise ValueError(f"WarpJoystick does not port: {bad}")

    def _dr_cfg(self):
        from humanoid_lab.dr.randomize import _DEFAULT_DR

        user = self.run["hydra_config"].get("dr") or {}
        return {k: {**v, **(user.get(k) or {})} for k, v in _DEFAULT_DR.items()}

    def _u(self, shape, lo, hi):
        if not isinstance(lo, (int, float)):
            lo = torch.as_tensor(lo, dtype=torch.float32, device=self.dev)
        if not isinstance(hi, (int, float)):
            hi = torch.as_tensor(hi, dtype=torch.float32, device=self.dev)
        return lo + (hi - lo) * torch.rand(shape, generator=self.gen, device=self.dev)

    def _domain_randomize(self):
        """dr/randomize.py's draws, one model per world, fixed for the run
        (brax applies randomization_fn once at train start)."""
        from humanoid_lab.dr.randomize import _find_base_body_id, _find_floor_geom_id

        c = self._dr_cfg()
        mjm, N, dev = self.mjm, self.num_envs, self.dev
        floor_id = _find_floor_geom_id(mjm, None)
        root_id = _find_base_body_id(mjm)
        foot_ids = _t(self.jenv._foot_geom_ids, dev, torch.long)

        geom_friction = _t(mjm.geom_friction, dev).repeat(N, 1, 1)
        geom_friction[:, floor_id, 0] = self._u((N,), *c["base"]["floor_friction"])
        body_mass = _t(mjm.body_mass, dev).repeat(N, 1)
        base_scale = self._u((N,), *c["base"]["base_mass"])
        link_scale = self._u((N, mjm.nbody), *c["base"]["link_mass"])
        root_mass = body_mass[:, root_id] * base_scale
        body_mass = body_mass * link_scale
        body_mass[:, root_id] = root_mass
        if c["base_mass_add"]["enable"]:
            kg = c["base_mass_add"]["kg"]
            body_mass[:, root_id] += self._u((N,), -kg, kg)
        nu = mjm.nu
        if c["joint_gains"]["enable"]:
            p, kp = c["joint_gains"]["gain_pct"], c["joint_gains"]["kd_pct"]
            gain = self._u((N, nu), 1 - p, 1 + p)
            kd = self._u((N, nu), 1 - kp, 1 + kp)
        else:
            gain = self._u((N, 1), *c["base"]["gain_fallback"]).expand(N, nu)
            kd = self._u((N, 1), *c["base"]["gain_fallback"]).expand(N, nu)
        gainprm = _t(mjm.actuator_gainprm, dev).repeat(N, 1, 1)
        biasprm = _t(mjm.actuator_biasprm, dev).repeat(N, 1, 1)
        gainprm[:, :, 0] *= gain
        biasprm[:, :, 1] *= gain
        biasprm[:, :, 2] *= kd
        motor = self._u((N, nu), *c["motor_strength"]["range"]) if c["motor_strength"]["enable"] else gain
        forcerange = _t(mjm.actuator_forcerange, dev).repeat(N, 1, 1) * motor[:, :, None]
        out = {
            "geom_friction": (geom_friction, wp.vec3),
            "body_mass": (body_mass, wp.float32),
            "actuator_gainprm": (gainprm, mjw_types.vec10f),
            "actuator_biasprm": (biasprm, mjw_types.vec10f),
            "actuator_forcerange": (forcerange, wp.vec2),
        }
        if c["com_offset"]["enable"]:
            xy, z = c["com_offset"]["xy"], c["com_offset"]["z"]
            ipos = _t(mjm.body_ipos, dev).repeat(N, 1, 1)
            ipos[:, root_id] += self._u((N, 3), [-xy, -xy, -z], [xy, xy, z])
            out["body_ipos"] = (ipos, wp.vec3)
        if c["dof"]["enable"]:
            nv = mjm.nv
            out["dof_damping"] = (_t(mjm.dof_damping, dev) * self._u((N, nv), *c["dof"]["damping"]), wp.float32)
            out["dof_armature"] = (_t(mjm.dof_armature, dev) * self._u((N, nv), *c["dof"]["armature"]), wp.float32)
            out["dof_frictionloss"] = (
                _t(mjm.dof_frictionloss, dev) * self._u((N, nv), *c["dof"]["frictionloss"]), wp.float32)
        if c["foot_friction"]["enable"]:
            fs = self._u((N, foot_ids.numel()), *c["foot_friction"]["range"])
            gf = out["geom_friction"][0]
            gf[:, foot_ids, 0] *= fs
        for name, (tensor, dtype) in out.items():
            tensor = tensor.contiguous()
            self._keep.append(tensor)
            setattr(self.m, name, wp.from_torch(tensor, dtype=dtype))
        self.dr_draw = {k: v[0] for k, v in out.items()}

    # -- physics --------------------------------------------------------------
    def _physics(self):
        if self._graph is None:
            with wp.ScopedDevice(self.wdev):
                for _ in range(self.n_substeps):  # warm-up compiles the kernels
                    mjw.step(self.m, self.d)
                with wp.ScopedCapture() as cap:
                    for _ in range(self.n_substeps):
                        mjw.step(self.m, self.d)
                self._graph = cap.graph
            return
        wp.capture_launch(self._graph)

    def _forward(self):
        if self._fwd_graph is None:
            with wp.ScopedDevice(self.wdev):
                mjw.forward(self.m, self.d)
                with wp.ScopedCapture() as cap:
                    mjw.forward(self.m, self.d)
                self._fwd_graph = cap.graph
            return
        wp.capture_launch(self._fwd_graph)

    # -- signal helpers (envs/base.py) ---------------------------------------
    def quat(self):
        return self.sensordata[:, self.s_quat:self.s_quat + 4]

    def gyro(self):
        return self.sensordata[:, self.s_gyro:self.s_gyro + 3]

    def local_linvel(self):
        return self.sensordata[:, self.s_linvel:self.s_linvel + 3]

    def gravity_body(self):
        return quat_rotate_inv(self.quat(), self._down)

    def foot_clearance(self):
        return self.site_xpos[:, self.foot_site_ids, 2] - self.foot_sole_offset

    def foot_contact(self):
        z = self.geom_xpos[:, self.foot_geom_ids, 2]
        per_geom = (z < self.foot_geom_radius + 0.005).float()
        out = torch.zeros(self.num_envs, self.n_feet, device=self.dev)
        out = out.scatter_reduce(1, self.foot_geom_foot_idx.expand(self.num_envs, -1), per_geom,
                                 reduce="amax", include_self=True)
        return out > 0

    def foot_linvel(self):
        """mjx.support.jac(point, body).T @ qvel at each foot site."""
        p = self.site_xpos[:, self.foot_site_ids]  # N,F,3
        off = p - self.subtree_com[:, self.foot_root]  # N,F,3
        cd = self.cdof  # N,nv,6  (ang, lin)
        ang, lin = cd[..., :3], cd[..., 3:]
        jacp = lin[:, None] + torch.cross(ang[:, None].expand(-1, self.n_feet, -1, -1),
                                          off[:, :, None].expand(-1, -1, cd.shape[1], -1), dim=-1)
        jacp = jacp * self.foot_dof_mask[None, :, :, None]
        return (jacp * self.qvel[:, None, :, None]).sum(2)  # N,F,3

    def base_yaw(self):
        q = self.qpos[:, self.base_qadr + 3:self.base_qadr + 7]
        return torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))

    def _heading_err(self, info):
        err = info["yaw_ref"] - self.base_yaw()
        return torch.atan2(torch.sin(err), torch.cos(err)).clamp(-1.0, 1.0)

    def _foot_x_base(self):
        """Foot sites' x in the base frame (forward of the pelvis is +)."""
        rel = self.site_xpos[:, self.foot_site_ids] - self.qpos[:, None, self.base_qadr:self.base_qadr + 3]
        quat = self.quat()[:, None].expand(-1, self.n_feet, -1)
        return quat_rotate_inv(quat, rel)[..., 0]

    def body_y_sep(self, pair):
        delta = self.xpos[:, pair[0]] - self.xpos[:, pair[1]]
        return quat_rotate_inv(self.quat(), delta)[:, 1].abs()

    # -- command / clock --------------------------------------------------------
    def cmd_speed(self, cmd):
        return cmd[:, :2].norm(dim=-1) + YAW_SPEED_WEIGHT * cmd[:, 2].abs()

    def sample_command(self, n):
        c = self.cfg.command
        vel = torch.stack([self._u((n,), *c.vx), self._u((n,), *c.vy), self._u((n,), *c.wz)], -1)
        # run.json `warp_cmd_straight` p: with probability p the draw keeps
        # only vx (vy = wz = 0), so straight walking is a common case rather
        # than a measure-zero one -- where heading drift shows (loco_warp_v29)
        p_straight = float(self.run.get("warp_cmd_straight", 0.0))
        if p_straight:
            straight = torch.rand(n, generator=self.gen, device=self.dev) < p_straight
            vel = torch.where(straight[:, None], vel * torch.tensor([1.0, 0.0, 0.0], device=self.dev), vel)
        slow_p = float(c.get("pure_slow_prob", 0.0))
        if slow_p:
            # a clean slow forward command (vy = wz = 0, vx in slow_vx): the
            # band just above the stand deadband, where a walk starts
            slow = torch.rand(n, generator=self.gen, device=self.dev) < slow_p
            vx = self._u((n,), *c.slow_vx)
            vel = torch.where(slow[:, None], torch.stack([vx, torch.zeros_like(vx), torch.zeros_like(vx)], -1), vel)
        zero = torch.rand(n, generator=self.gen, device=self.dev) < c.zero_prob
        return torch.where(zero[:, None], torch.zeros_like(vel), vel)

    def leg_phases(self, phase):
        ph = phase[:, None] + self._phase_off
        return torch.fmod(ph + math.pi, 2 * math.pi) - math.pi

    def gait_targets(self, phase):
        g = self.cfg.gait
        theta = torch.fmod(self.leg_phases(phase) + 2 * math.pi, 2 * math.pi) / (2 * math.pi)
        swing_frac = 1.0 - g.duty
        in_swing = theta < swing_frac
        return g.swing_height * torch.sin(math.pi * theta / swing_frac) * in_swing

    def phase_dt(self, cmd):
        g = self.cfg.gait
        speed = self.cmd_speed(cmd)
        frac = (speed / self.cmd_vmax).clamp(0.0, 1.0)
        freq = g.freq[0] + (g.freq[1] - g.freq[0]) * frac
        return torch.where(speed > self.deadband, 2 * math.pi * self.dt * freq, torch.zeros_like(freq))

    # -- observations -----------------------------------------------------------
    def catalog(self, info):
        leg = self.leg_phases(info["phase"])
        return {
            "gyro": self.gyro(),
            "gravity": self.gravity_body(),
            "joint_pos": self.qpos[:, self.qadr] - self.default_pose,
            "joint_vel": self.qvel[:, self.vadr],
            "last_action": info["last_action"],
            "linvel": self.local_linvel(),
            "height": self.qpos[:, self.base_qadr + 2:self.base_qadr + 3],
            "actuator_force": self.actuator_force,
            "contacts": self.foot_contact().float(),
            "command": info["command"],
            "heading_err": (self._heading_err(info)[:, None] if "yaw_ref" in info
                            else torch.zeros(self.num_envs, 1, device=self.dev)),
            # frozen clock -> zeros, as envs/joystick.py: no left/right marker
            "phase": (torch.zeros(self.num_envs, 2 * self.n_feet, device=self.dev)
                      if not any(float(f) for f in self.cfg.gait.freq)
                      else torch.cat([torch.cos(leg), torch.sin(leg)], -1)),
        }

    def build_obs(self, info, noise=True):
        cat = self.catalog(info)
        state = torch.cat([cat[n] for n in self.state_names], -1)
        if noise:
            state = state + (torch.rand(state.shape, generator=self.gen, device=self.dev) * 2 - 1) * self.obs_noise
        priv = torch.cat([cat[n] for n in self.priv_names], -1)
        return state, priv

    # -- reset ------------------------------------------------------------------
    def reset(self):
        """Full reset of every world (Joystick.reset + the wrappers' seeds)."""
        N, nu, F, dev = self.num_envs, self.nu, self.n_feet, self.dev
        cmd = self.sample_command(N)
        noise = self._u((N, nu), -1.0, 1.0) * float(self.cfg.get("reset_noise", 0.0))
        qpos = self.reset_qpos.expand(N, -1).clone()
        qpos[:, self.qadr] += noise
        self.qpos.copy_(qpos)
        self.qvel.zero_()
        self.ctrl.copy_(self.neutral_ctrl.expand(N, -1))
        self.qacc_warmstart.zero_()
        self.time.zero_()
        self._forward()
        z = lambda *s: torch.zeros(*s, device=dev)
        lo, hi = (int(v) for v in self.cfg.push.interval_steps_range)
        self.info = {
            # Command ramp (run.json `warp_cmd_ramp` {lin, yaw} in m/s^2, rad/s^2):
            # the served command starts at zero and slews toward the sampled
            # target, so every episode starts from a stand and accelerates,
            # like the battery's walk_ramp.
            "command": torch.zeros_like(cmd) if self.run.get("warp_cmd_ramp") else cmd,
            "cmd_target": cmd,
            "last_action": z(N, nu), "last_last_action": z(N, nu), "last_torque": z(N, nu),
            "feet_air_time": z(N, F), "feet_contact_time": z(N, F), "swing_apex": z(N, F),
            "last_apex": z(N, F), "last_contact": torch.zeros(N, F, dtype=torch.bool, device=dev),
            "phase": z(N), "step_count": torch.zeros(N, dtype=torch.long, device=dev),
            "steps_since_cmd": torch.zeros(N, dtype=torch.long, device=dev),
            "air_dur_ema": z(N, F), "stance_dur_ema": z(N, F),
            "push_countdown": torch.randint(lo, hi + 1, (N,), generator=self.gen, device=dev),
            "steps": torch.zeros(N, dtype=torch.long, device=dev),
            "land_feat": z(N, F, 5), "land_armed": torch.zeros(N, F, dtype=torch.bool, device=dev),
            # per-foot payout fractions of the last landing, for the pair_min
            # option: [foot_reach, step_length, knee_extend touchdown]
            "land_frac": z(N, F, 3),
        }
        self.info["yaw_target"] = self.base_yaw()
        if self.heading_obs:
            self.info["yaw_ref"] = self.base_yaw().clone()
        state, priv = self.build_obs(self.info, noise=False)
        self.first = {
            "yaw": self.base_yaw().clone(),
            "qpos": self.qpos.clone(), "qvel": self.qvel.clone(), "ctrl": self.ctrl.clone(),
            "state": state.clone(), "priv": priv.clone(),
        }
        self.done = torch.zeros(N, dtype=torch.bool, device=dev)
        return state, priv

    # -- step -------------------------------------------------------------------
    def step(self, action):
        """One control step for every world, through the training wrappers.

        Returns (state_obs, priv_obs, reward, done, truncation, terms) where
        the obs are already the auto-reset obs for worlds that finished."""
        info, cfg, N, dev = self.info, self.cfg, self.num_envs, self.dev
        # BraxAutoResetWrapper: reset the episode step counter where the
        # previous step finished.
        info["steps"] = torch.where(self.done, torch.zeros_like(info["steps"]), info["steps"])

        targets = (self.default_pose + action * self.action_scale).clamp(self.ctrl_lo, self.ctrl_hi)

        # pushes (Joystick.step, interval_steps_range branch)
        pc = cfg.push
        lo, hi = (int(v) for v in pc.interval_steps_range)
        push_now = info["push_countdown"] <= 0
        next_cd = torch.randint(lo, hi + 1, (N,), generator=self.gen, device=dev)
        info["push_countdown"] = torch.where(push_now, next_cd, info["push_countdown"] - 1)
        push = self._u((N, 2), -1.0, 1.0)
        push = push / (push.norm(dim=-1, keepdim=True) + 1e-6) * pc.vel
        kick = torch.zeros(N, 6, device=dev)
        kick[:, :2] = push
        if pc.get("vel_z", 0.0):
            kick[:, 2] = self._u((N,), -pc.vel_z, pc.vel_z)
        ang_rp, ang_yaw = pc.get("ang_vel_rp", 0.0), pc.get("ang_vel_yaw", 0.0)
        if ang_rp or ang_yaw:
            kick[:, 3:6] = (torch.rand((N, 3), generator=self.gen, device=dev) * 2 - 1) * self._ang_hi
        bv = self.base_vadr
        self.qvel[:, bv:bv + 6] += kick * push_now[:, None]

        self.ctrl.copy_(targets)
        self._physics()
        if self.heading_obs:
            # follows the actual yaw while a turn is commanded, frozen on straight
            turning = info["command"][:, 2].abs() > 0.05
            info["yaw_ref"] = torch.where(turning, self.base_yaw(), info["yaw_ref"])

        contact = self.foot_contact()
        contact_filt = contact | info["last_contact"]
        first_contact = (info["feet_air_time"] > 0) & contact_filt
        info["swing_apex"] = torch.where(~contact_filt, torch.maximum(info["swing_apex"], self.foot_clearance()),
                                         info["swing_apex"])

        info["yaw_target"] = info["yaw_target"] + info["command"][:, 2] * self.dt
        hh_leak = float((self.extra.get("heading_hold") or {}).get("leak", 0.0))
        if hh_leak:
            # the target relaxes toward the actual heading with rate `leak`
            # (1/s): a steady yaw-rate tracking error e settles at e / leak
            # instead of growing without bound on long turns
            yerr = torch.remainder(self.base_yaw() - info["yaw_target"] + math.pi, 2 * math.pi) - math.pi
            info["yaw_target"] = info["yaw_target"] + hh_leak * self.dt * yerr
        terms, fall = self.compute_rewards(info, action, first_contact, contact)
        info["land_frac"] = torch.where(first_contact[..., None], self._land_frac_now, info["land_frac"])
        if "step_symmetry" in self.extra:
            info["land_feat"] = torch.where(first_contact[..., None], self._land_now, info["land_feat"])
            info["land_armed"] = info["land_armed"] | first_contact

        alpha = cfg.reward.gait_symmetry_alpha
        lift_off = (info["feet_contact_time"] > 0) & ~contact_filt
        info["air_dur_ema"] = torch.where(first_contact, (1 - alpha) * info["air_dur_ema"] + alpha * info["feet_air_time"],
                                          info["air_dur_ema"])
        info["stance_dur_ema"] = torch.where(lift_off, (1 - alpha) * info["stance_dur_ema"] + alpha * info["feet_contact_time"],
                                             info["stance_dur_ema"])
        info["last_apex"] = torch.where(first_contact, info["swing_apex"], info["last_apex"])
        info["swing_apex"] = torch.where(contact_filt, torch.zeros_like(info["swing_apex"]), info["swing_apex"])
        info["feet_air_time"] = torch.where(contact_filt, torch.zeros_like(info["feet_air_time"]), info["feet_air_time"] + self.dt)
        info["feet_contact_time"] = torch.where(contact_filt, info["feet_contact_time"] + self.dt,
                                                torch.zeros_like(info["feet_contact_time"]))
        info["last_contact"] = contact
        info["last_last_action"] = info["last_action"]
        info["last_action"] = action
        info["last_torque"] = self.actuator_force.clone()
        info["step_count"] = info["step_count"] + 1
        ph = info["phase"] + self.phase_dt(info["command"])
        info["phase"] = torch.fmod(ph + math.pi, 2 * math.pi) - math.pi
        info["steps_since_cmd"] = info["steps_since_cmd"] + 1

        resample = info["steps_since_cmd"] >= cfg.command.resample_steps
        new_cmd = self.sample_command(N)
        ramp = self.run.get("warp_cmd_ramp")
        if ramp:
            info["cmd_target"] = torch.where(resample[:, None], new_cmd, info["cmd_target"])
            rate = torch.tensor([ramp["lin"], ramp["lin"], ramp["yaw"]], device=dev) * self.dt
            info["command"] = info["command"] + (info["cmd_target"] - info["command"]).clamp(-rate, rate)
        else:
            info["command"] = torch.where(resample[:, None], new_cmd, info["command"])
        info["steps_since_cmd"] = torch.where(resample, torch.zeros_like(info["steps_since_cmd"]), info["steps_since_cmd"])

        reward = sum(terms[k] * self.scales[k] for k in self.terms)
        reward = (reward * self.dt).clamp(-100.0, 100.0)

        state, priv = self.build_obs(info, noise=True)

        # EpisodeWrapper
        info["steps"] = info["steps"] + 1
        timeout = info["steps"] >= int(self.run["ppo_config"].get("episode_length", cfg.episode_length))
        done = fall | timeout
        truncation = timeout & ~fall

        # BraxAutoResetWrapper(full_reset=False): data and obs from the
        # cached first state; info carries over.
        if True:  # unconditional: a host-side done.any() would sync every step
            dm = done[:, None]
            self.qpos.copy_(torch.where(dm, self.first["qpos"], self.qpos))
            self.qvel.copy_(torch.where(dm, self.first["qvel"], self.qvel))
            self.ctrl.copy_(torch.where(dm, self.first["ctrl"], self.ctrl))
            self.qacc_warmstart.copy_(torch.where(dm, torch.zeros_like(self.qacc_warmstart), self.qacc_warmstart))
            state = torch.where(dm, self.first["state"], state)
            priv = torch.where(dm, self.first["priv"], priv)
            # GaitReseedWrapper
            for k in ("feet_air_time", "feet_contact_time", "swing_apex", "last_apex", "air_dur_ema", "stance_dur_ema"):
                info[k] = torch.where(dm, torch.zeros_like(info[k]), info[k])
            info["last_contact"] = info["last_contact"] & ~dm
            info["yaw_target"] = torch.where(done, self.first["yaw"], info["yaw_target"])
            if self.heading_obs:
                info["yaw_ref"] = torch.where(done, self.first["yaw"], info["yaw_ref"])
            info["land_feat"] = torch.where(dm[..., None], torch.zeros_like(info["land_feat"]), info["land_feat"])
            info["land_armed"] = info["land_armed"] & ~dm
            info["land_frac"] = torch.where(dm[..., None], torch.zeros_like(info["land_frac"]), info["land_frac"])
        self.done = done
        return state, priv, reward, done, truncation, terms

    # -- rewards (Joystick._compute_rewards, ported scales only) --------------
    def compute_rewards(self, info, action, first_contact, contact):
        cfg = self.cfg.reward
        cmd = info["command"]
        linvel, gyro, grav = self.local_linvel(), self.gyro(), self.gravity_body()
        moving = (self.cmd_speed(cmd) > self.deadband).float()
        q = self.qpos[:, self.qadr]
        qd = self.qvel[:, self.vadr]
        tau = self.actuator_force
        base_h = self.qpos[:, self.base_qadr + 2]
        fall = (base_h < self.cfg.fall.min_height) | (grav[:, 2] > self.cfg.fall.max_tilt_gz)
        sq = torch.square
        on = self.scales
        t = {}
        # Speed-scaled lift (run.json `warp_lift` {v_full, min_frac}): the
        # clock's swing height, feet_apex's target and knee_swing's flexion
        # target all shrink with the commanded speed, so a slow command asks
        # for small flat steps instead of a high-knee march in place.
        self._land_frac_now = torch.zeros(self.num_envs, self.n_feet, 3, device=self.dev)
        wl = self.run.get("warp_lift")
        if wl:
            ls = (self.cmd_speed(cmd) / float(wl["v_full"])).clamp(float(wl["min_frac"]), 1.0)
        else:
            ls = torch.ones_like(cmd[:, 0])
        fwd = (cmd[:, 0] > max(0.1, self.deadband)).float()
        if on.get("tracking_lin_vel"):
            t["tracking_lin_vel"] = torch.exp(-sq(cmd[:, :2] - linvel[:, :2]).sum(-1) / cfg.tracking_sigma)
        if on.get("tracking_ang_vel"):
            t["tracking_ang_vel"] = torch.exp(-sq(cmd[:, 2] - gyro[:, 2]) / cfg.tracking_sigma)
        if on.get("lin_vel_z"):
            t["lin_vel_z"] = sq(linvel[:, 2])
        if on.get("ang_vel_xy"):
            t["ang_vel_xy"] = sq(gyro[:, :2]).sum(-1)
        if on.get("orientation"):
            t["orientation"] = sq(grav[:, :2]).sum(-1)
        if on.get("torques"):
            t["torques"] = sq(tau).sum(-1)
        if on.get("torque_rate"):
            t["torque_rate"] = sq(tau - info["last_torque"]).sum(-1)
        if on.get("action_rate"):
            t["action_rate"] = sq(action - info["last_action"]).sum(-1)
        if on.get("action_accel"):
            t["action_accel"] = sq(action - 2 * info["last_action"] + info["last_last_action"]).sum(-1)
        if on.get("energy"):
            t["energy"] = (qd.abs() * tau.abs()).sum(-1)
        if on.get("pose"):
            t["pose"] = (self.pose_weight * sq(q - self.pose_anchor)).sum(-1)
        if on.get("feet_air_time"):
            t["feet_air_time"] = ((info["feet_air_time"] - 0.1) * first_contact).sum(-1) * moving
        if on.get("feet_slip"):
            fv = self.foot_linvel()
            t["feet_slip"] = (sq(fv[..., :2]).sum(-1) * contact).sum(-1) * moving
        if on.get("feet_phase"):
            err = sq(self.foot_clearance() - self.gait_targets(info["phase"]) * ls[:, None]).sum(-1)
            t["feet_phase"] = torch.exp(-err / cfg.phase_sigma) * moving
        if on.get("stand_still"):
            t["stand_still"] = ((q - self.pose_anchor).abs().sum(-1)
                                + cfg.get("stand_still_vel_weight", 0.2) * qd.abs().sum(-1)) * (1 - moving)
        if on.get("termination"):
            t["termination"] = fall.float()
        if on.get("torque_limit"):
            t["torque_limit"] = (tau.abs() - cfg.torque_limit_frac * self.torque_cap).clamp(min=0).sum(-1)
        if on.get("feet_apex"):
            t["feet_apex"] = ((info["swing_apex"] / (cfg.apex_target * ls[:, None])).clamp(0, 1)
                              * first_contact).sum(-1) * moving
        if on.get("feet_landing"):
            fv = self.foot_linvel()
            t["feet_landing"] = (sq(fv[..., 2].clamp(max=0.0))
                                 * (1.0 - self.foot_clearance() / cfg.glide_height).clamp(0, 1)).sum(-1) * moving
        if on.get("pose_l1"):
            t["pose_l1"] = (self.pose_l1_weight * (q - self.pose_anchor).abs()).sum(-1)
        if on.get("joint_pos_limits"):
            t["joint_pos_limits"] = ((self.soft_lo - q).clamp(min=0) + (q - self.soft_hi).clamp(min=0)).sum(-1)
        if on.get("joint_vel"):
            t["joint_vel"] = sq(qd).sum(-1)
        if on.get("joint_acc"):
            t["joint_acc"] = sq(self.qacc[:, self.vadr]).sum(-1)
        if on.get("upward"):
            t["upward"] = -grav[:, 2]

        def band(sep, lo, hi):
            d_min = (sep - lo).clamp(-0.5, 0.0)
            d_max = (sep - hi).clamp(0.0, 0.5)
            return (torch.exp(-d_min.abs() * 100.0) + torch.exp(-d_max.abs() * 100.0)) / 2.0

        if on.get("feet_distance"):
            t["feet_distance"] = band(self.body_y_sep(self.ankle_pair), *cfg.feet_distance_range)
        if on.get("knee_distance"):
            t["knee_distance"] = band(self.body_y_sep(self.knee_pair), *cfg.knee_distance_range)
        if on.get("feet_contact_without_cmd"):
            upright = (-grav[:, 2]).clamp(0.0, 0.7) / 0.7
            t["feet_contact_without_cmd"] = contact.all(-1).float() * upright * (1 - moving)
        if on.get("feet_air_time_biped"):
            in_contact = contact | info["last_contact"]
            mode_t = torch.where(in_contact, info["feet_contact_time"], info["feet_air_time"])
            single = in_contact.sum(-1) == 1
            v = torch.where(single[:, None], mode_t, torch.zeros_like(mode_t)).min(-1).values
            t["feet_air_time_biped"] = v.clamp(max=cfg.get("biped_air_time_threshold", 0.4)) * moving
        if on.get("gait_symmetry_income"):
            floor, cap = cfg.gait_symmetry_floor, cfg.get("gait_symmetry_cap", 1.0)

            def pair(dd):
                armed = (dd[:, 0] > 0) & (dd[:, 1] > 0)
                rel = sq((dd[:, 0] - dd[:, 1]) / torch.clamp(0.5 * (dd[:, 0] + dd[:, 1]), min=floor))
                return rel * armed, armed

            a_sq, a_arm = pair(info["air_dur_ema"])
            s_sq, s_arm = pair(info["stance_dur_ema"])
            single = contact.sum(-1) == 1
            t["gait_symmetry_income"] = ((1.0 - torch.clamp(a_sq + s_sq, max=cap) / cap)
                                         * single * (a_arm | s_arm) * moving)
        if "knee_swing" in self.extra:
            ks = self.extra["knee_swing"]
            anchor = self.default_pose[self.knee_qidx]
            flex = ((q[:, self.knee_qidx] - anchor) / ((float(ks["target"]) - anchor) * ls[:, None])).clamp(0.0, 1.0)
            if "hip_target" in ks:
                hip = ((self.default_pose[self.hip_pitch_idx] - q[:, self.hip_pitch_idx])
                       / (float(ks["hip_target"]) * ls[:, None])).clamp(0.0, 1.0)
                flex = torch.minimum(flex, hip)
            gate = (~contact).float()
            if ks.get("only_behind", False):
                # early swing only: the foot still behind / under the base. Past
                # the base the knee should open (knee_extend), not stay folded.
                gate = gate * (self._foot_x_base() <= float(ks.get("x_split", 0.0))).float()
            t["knee_swing"] = (flex * gate).sum(-1) * moving
        if "knee_extend" in self.extra:
            # late swing: the swing foot is ahead of the base; pay knee extension,
            # 1 at flex_lo or straighter, 0 at flex_hi, so the foot reaches
            # forward by opening the knee rather than by driving the thigh
            ke = self.extra["knee_extend"]
            lo, hi = float(ke["flex_lo"]), float(ke["flex_hi"])
            ext = ((hi - q[:, self.knee_qidx]) / (hi - lo)).clamp(0.0, 1.0)
            late = (~contact) & (self._foot_x_base() > float(ke.get("x_split", 0.0)))
            # optional td_mult: an extra payout of td_mult * ext on the landing
            # step itself, the knee angle the foot actually touches down with
            # only a real step earns it: a swing of at least td_min_air seconds
            # landing ahead of the base. Ungated, the bonus outpaid the landing
            # tax and a foot tapping in place farmed it (loco_warp_v18).
            real_td = (first_contact & (info["feet_air_time"] >= float(ke.get("td_min_air", 0.2)))
                       & (self._foot_x_base() > 0.0))
            td_ext = ext * real_td.float()  # a tap or a step landing behind records 0
            self._land_frac_now[..., 2] = td_ext
            if ke.get("pair_min", False):
                td_ext = torch.minimum(td_ext, info["land_frac"][..., 2].flip(1))
            t["knee_extend"] = ((ext * late.float()).sum(-1)
                                + float(ke.get("td_mult", 0.0)) * (td_ext * first_contact.float()).sum(-1)) * fwd
        if "arm_swing" in self.extra:
            asw = self.extra["arm_swing"]
            dq = q - self.default_pose
            d_arm = dq[:, self.arm_idx]
            err = sq(d_arm - float(asw["k"]) * dq[:, self.thigh_other_idx]).sum(-1)
            if asw.get("k_elbow"):
                flex = -dq[:, self.elbow_idx]  # anchor is straight: flexion lowers q
                err = err + sq(flex - float(asw["k_elbow"]) * (-d_arm).clamp(min=0.0)).sum(-1)
            t["arm_swing"] = torch.exp(-err / float(asw["sigma"])) * moving
        if "elbow_hyperext" in self.extra:
            straight = float(self.extra["elbow_hyperext"]["straight"])
            t["elbow_hyperext"] = (q[:, self.elbow_idx] - straight).clamp(min=0.0).sum(-1)
        fwd = (cmd[:, 0] > max(0.1, self.deadband)).float()
        if "forward_lean" in self.extra:
            fl = self.extra["forward_lean"]
            frac = (cmd[:, 0] / float(fl["vx_max"])).clamp(0.0, 1.0)
            target = torch.sin(math.radians(float(fl["max_deg"])) * frac)
            t["forward_lean"] = torch.exp(-sq(grav[:, 0] - target) / float(fl["sigma"])) * fwd
        if "push_off" in self.extra:
            po = self.extra["push_off"]
            rel = self.site_xpos[:, self.foot_site_ids] - self.qpos[:, None, self.base_qadr:self.base_qadr + 3]
            quat = self.quat()[:, None].expand(-1, self.n_feet, -1)
            behind = quat_rotate_inv(quat, rel)[..., 0] < -float(po["behind"])
            power = tau[:, self.ankle_idx].clamp(min=0.0) * qd[:, self.ankle_idx].clamp(min=0.0)
            t["push_off"] = ((power / float(po["p_ref"])).clamp(0.0, 1.0) * (contact & behind).float()).sum(-1) * fwd
        if "step_length" in self.extra:
            rel = self.site_xpos[:, self.foot_site_ids] - self.qpos[:, None, self.base_qadr:self.base_qadr + 3]
            quat = self.quat()[:, None].expand(-1, self.n_feet, -1)
            x_b = quat_rotate_inv(quat, rel)[..., 0]  # N,F
            ahead = x_b - x_b.flip(1)
            sl = self.extra["step_length"]
            if "target_a" in sl:
                target = float(sl["target_a"]) + float(sl["target_b"]) * cmd[:, 0].clamp(min=0.0)
            else:
                g = self.cfg.gait
                frac = (self.cmd_speed(cmd) / self.cmd_vmax).clamp(0.0, 1.0)
                freq = g.freq[0] + (g.freq[1] - g.freq[0]) * frac
                target = cmd[:, 0] / (2.0 * freq)
            target = target.clamp(min=0.05)
            # power 1 pays the same per second at any cadence for a given speed
            # (twice the landings at half the step); power 2 makes the per-second
            # pay grow with the step itself.
            frac_s = (ahead / target[:, None]).clamp(0.0, 1.0) ** float(sl.get("power", 1.0))
            self._land_frac_now[..., 1] = frac_s
            if sl.get("pair_min", False):
                frac_s = torch.minimum(frac_s, info["land_frac"][..., 1].flip(1))
            t["step_length"] = (frac_s * first_contact.float()).sum(-1) * fwd
        if "foot_reach" in self.extra:
            fr = self.extra["foot_reach"]
            rel = self.site_xpos[:, self.foot_site_ids] - self.qpos[:, None, self.base_qadr:self.base_qadr + 3]
            quat = self.quat()[:, None].expand(-1, self.n_feet, -1)
            x_b = quat_rotate_inv(quat, rel)[..., 0]
            target = (float(fr["target_a"]) + float(fr["target_b"]) * cmd[:, 0].clamp(min=0.0)).clamp(min=0.02)
            x0 = float(fr.get("x_min", 0.0))  # landing here or further back pays 0
            frac_r = ((x_b - x0) / (target[:, None] - x0)).clamp(0.0, 1.0) ** float(fr.get("power", 2.0))
            self._land_frac_now[..., 0] = frac_r
            if fr.get("pair_min", False):
                # pair_min (all three landing terms): a landing pays the lower of
                # its own fraction and the other foot's last one, so one leg's
                # good step cannot carry a shuffling partner (loco_warp_v18-v20)
                frac_r = torch.minimum(frac_r, info["land_frac"][..., 0].flip(1))
            t["foot_reach"] = (frac_r * first_contact.float()).sum(-1) * fwd
        if "hip_swing" in self.extra:
            hs = self.extra["hip_swing"]
            flex_h = (self.default_pose[self.hip_pitch_idx] - q[:, self.hip_pitch_idx]) / (float(hs["target"]) * ls[:, None])
            t["hip_swing"] = (flex_h.clamp(0.0, 1.0) * (~contact).float()).sum(-1) * moving
        if "swing_back" in self.extra:
            v_rel = self.foot_linvel() - self.qvel[:, None, self.base_vadr:self.base_vadr + 3]
            quat = self.quat()[:, None].expand(-1, self.n_feet, -1)
            vx_b = quat_rotate_inv(quat, v_rel)[..., 0]
            t["swing_back"] = (sq((-vx_b).clamp(min=0.0)) * (~contact).float()).sum(-1) * moving
        if "swing_forward" in self.extra:
            # per swing foot, forward speed relative to the base as a fraction of
            # v_ref, clipped to [0, 1]: the reach that carries the foot ahead
            sf = self.extra["swing_forward"]
            v_rel = self.foot_linvel() - self.qvel[:, None, self.base_vadr:self.base_vadr + 3]
            quat = self.quat()[:, None].expand(-1, self.n_feet, -1)
            vx_b = quat_rotate_inv(quat, v_rel)[..., 0]
            t["swing_forward"] = ((vx_b / float(sf["v_ref"])).clamp(0.0, 1.0) * (~contact).float()).sum(-1) * fwd
        if "heading_hold" in self.extra:
            # heading_hold {scale, range}: the base yaw against a target that
            # turns only with the commanded yaw rate, 1 - |err| / range floored
            # at 0. tracking_ang_vel prices the yaw RATE and lets a slow
            # unbidden turn (~3 deg/s, loco_warp_v29) accumulate for free.
            hh = self.extra["heading_hold"]
            if hh.get("ref", False):
                # absolute hold: the error vs the frozen reference, no leak
                err = self._heading_err(info)
            else:
                err = torch.remainder(self.base_yaw() - info["yaw_target"] + math.pi, 2 * math.pi) - math.pi
            t["heading_hold"] = (1.0 - err.abs() / float(hh["range"])).clamp(min=0.0)
        if "march" in self.extra:
            # march {scale, frac}: a touchdown while the body moves slower than
            # frac * the commanded forward speed (frac default 0.5) is stepping
            # in place -- the start of a walk_ramp, where the command is still
            # near zero (loco_warp_v35: 5 touchdowns, -1.5 cm in the first second)
            mf = float(self.extra["march"].get("frac", 0.5))
            slow = (linvel[:, 0] < mf * cmd[:, 0]) & (cmd[:, 0] > 0.05)
            t["march"] = first_contact.float().sum(-1) * slow.float()
        if "lateral_hold" in self.extra:
            # lateral_hold {scale, range}: 1 - |vy - cmd_vy| / range, floored at
            # 0 (base frame). The exp tracking kernel lets a steady sidestep of
            # 0.1 m/s through almost free, which reads as walking diagonally.
            lh = self.extra["lateral_hold"]
            t["lateral_hold"] = (1.0 - (linvel[:, 1] - cmd[:, 1]).abs() / float(lh["range"])).clamp(min=0.0)
        if "flight" in self.extra:
            # both feet off the ground while commanded to move: a walk never
            # has a flight phase; trot (loco_warp_v26) and two-footed hop
            # (loco_warp_v27) do
            t["flight"] = (~contact.any(-1)).float() * moving
        if "landing_tax" in self.extra:
            # one unit per touchdown while moving; with a negative scale, the
            # per-second cost is the step rate, so fewer, longer steps pay
            # Optional {v_ref, max_mult}: below v_ref the per-landing cost grows as
            # v_ref / cmd_vx (capped at max_mult), so the per-second cost tracks
            # 1 / step length at low speed too, where a fixed tax still lets a
            # fast shuffle of tiny steps win.
            lt = self.extra["landing_tax"]
            mult = torch.ones_like(cmd[:, 0])
            if "v_ref" in lt:
                mult = (float(lt["v_ref"]) / self.cmd_speed(cmd).clamp(min=1e-3)).clamp(1.0, float(lt["max_mult"]))
            t["landing_tax"] = first_contact.float().sum(-1) * mult * moving
        if "progress" in self.extra:
            pg = self.extra["progress"]
            if pg.get("two_sided", False):
                # 1 - |v - cmd| / cmd: overshooting the command costs as much as
                # falling short (one-sided, a slow command was walked at up to
                # 1.8x, loco_warp_v31)
                rel = (linvel[:, 0] - cmd[:, 0]).abs() / cmd[:, 0].clamp(min=0.1)
                t["progress"] = (1.0 - rel).clamp(0.0, 1.0) * fwd
            else:
                t["progress"] = (linvel[:, 0] / cmd[:, 0].clamp(min=0.1)).clamp(0.0, 1.0) * fwd
        if "step_symmetry" in self.extra:
            ss = self.extra["step_symmetry"]
            rel = self.site_xpos[:, self.foot_site_ids] - self.qpos[:, None, self.base_qadr:self.base_qadr + 3]
            quat = self.quat()[:, None].expand(-1, self.n_feet, -1)
            rel_b = quat_rotate_inv(quat, rel)
            feats = torch.stack([rel_b[..., 0], rel_b[..., 1].abs(), info["swing_apex"], info["feet_air_time"],
                                 q[:, self.knee_qidx]], -1)
            self._land_now = feats  # step() records it for the feet that landed
            other = info["land_feat"].flip(1)
            # optional sigma_knee adds the touchdown knee angle as a fifth feature
            n_f = 5 if "sigma_knee" in ss else 4
            sig = torch.tensor([ss["sigma_x"], ss["sigma_y"], ss["sigma_apex"], ss["sigma_air"],
                                ss.get("sigma_knee", 1.0)][:n_f], device=self.dev)
            diff = (feats[..., :n_f] - other[..., :n_f]).abs()
            if ss.get("kernel", "exp") == "linear":
                # 1 - |d| / (3 sigma), floored at 0: a constant gradient across the
                # whole band, where exp(-d^2/sigma^2) is flat far from symmetric
                # and a strong limp sees no way back (loco_warp_v20)
                k = (1.0 - diff / (3.0 * sig)).clamp(min=0.0).mean(-1)
            else:
                k = torch.exp(-sq(diff) / sq(sig)).mean(-1)  # N,F
            armed = info["land_armed"].flip(1)
            t["step_symmetry"] = (k * (first_contact & armed).float()).sum(-1) * moving
        return t, fall
