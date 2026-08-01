"""Robot-agnostic MJX env base, shared by every task.

Holds model loading, actuator address tables, IMU/foot helpers and the obs
catalog + include-list mechanism. Every robot-specific detail routes through
a RobotSpec and a resolved actuator preset, so no joint, site or geom name is
hardcoded here. Task envs (envs/joystick.py) subclass this and provide their
own config, reset, step and reward composition.
"""

from __future__ import annotations

import copy

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from brax import math as brax_math
from mujoco import mjx
from mujoco_playground._src import mjx_env

from humanoid_lab.actuators.models import ACTUATOR_MODELS
from humanoid_lab.envs.backend import make_data_fn, resolve_backend
from humanoid_lab.robot.build import build_spec, compile_spec
from humanoid_lab.robot.presets import action_scale as preset_action_scale
from humanoid_lab.robot.presets import load_actuator_preset
from humanoid_lab.robot.spec import RobotSpec, load_robot_spec, validate_against_model


# Largest end-of-settle |qvel| (rad/s, m/s) still read as "came to rest", for
# the real_pose_ref guard. A converged settle is orders of magnitude under it
# (roboto_origin's home keyframe ends at 7e-5); a biped on its way over is
# orders of magnitude above (asimov_v1's knees_bent, 0.15).
_SETTLE_REST_SPEED = 1e-2


def _free_joint_addr(model: mujoco.MjModel) -> tuple[int, int]:
    """(qpos addr, dof addr) of the model's one free joint (the floating base).

    Never assumed to be qpos[0:7]/qvel[0:6]: found by joint type so a future
    robot with a differently-ordered kinematic tree still works.
    """
    free = [i for i in range(model.njnt) if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE]
    if len(free) != 1:
        raise ValueError(f"expected exactly one free joint (floating base), found {len(free)}")
    i = free[0]
    return int(model.jnt_qposadr[i]), int(model.jnt_dofadr[i])


def _foot_geom_to_foot_index(model: mujoco.MjModel, foot_site_ids: np.ndarray, foot_geom_ids: np.ndarray) -> np.ndarray:
    """Map each foot geom to the index of the foot_sites entry that "owns" it.

    A robot's foot_geoms list is every collision primitive on the foot
    (asimov: 5 sole capsules + 5 toe capsules per side, the toes on their own
    passive-joint body), not one geom per foot -- so contact/velocity must be
    aggregated per logical foot by body ancestry: walk each geom's body up
    the kinematic tree until it reaches a foot_sites body (match) or the
    world body (error: robot.yaml's foot_geoms/foot_sites disagree on feet).
    """
    foot_site_body = model.site_bodyid[foot_site_ids]
    foot_geom_body = model.geom_bodyid[foot_geom_ids]

    def foot_of(body_id: int) -> int:
        seen = set()
        b = body_id
        while b not in foot_site_body:
            if b == 0 or b in seen:
                return -1
            seen.add(b)
            b = model.body_parentid[b]
        return int(np.flatnonzero(foot_site_body == b)[0])

    foot_idx = np.array([foot_of(b) for b in foot_geom_body], dtype=np.int32)
    return foot_idx


class HumanoidEnv(mjx_env.MjxEnv):
    """Base class for robot-agnostic MJX humanoid tasks.

    The model is built entirely in memory (build_spec + compile_spec): a
    task env never depends on a previously-run `run.sh build`.
    """

    def __init__(self, robot_dir, preset_name, config, config_overrides=None, actuator_overrides=None):
        super().__init__(config, config_overrides)

        self._robot_spec: RobotSpec = load_robot_spec(robot_dir)
        self._preset = load_actuator_preset(robot_dir, preset_name, actuator_overrides)

        spec = build_spec(robot_dir, preset_name, actuator_overrides)
        self._mj_model = compile_spec(spec)
        self._mj_model.opt.timestep = self.sim_dt
        self._customize_model(self._mj_model)
        validate_against_model(self._robot_spec, self._mj_model)

        sim = self._config.sim
        self._backend = resolve_backend(sim.backend)
        self._mjx_model = mjx.put_model(self._mj_model, impl=self._backend)
        self._make_data_fn = make_data_fn(
            self._backend,
            self._mj_model,
            self._mjx_model,
            sim.naconmax_per_env,
            sim.njmax,
            sim.num_envs,
        )

        m = self._mj_model
        rs = self._robot_spec

        # Reset keyframe: configurable (default "home"), used both for reset
        # qpos and as the PD default_pose anchor below. Validated against
        # robot.yaml's parsed keyframes at construct time so a typo'd config
        # value fails loudly here instead of inside a jitted reset().
        reset_keyframe = self._config.get("reset_keyframe", "home")
        if reset_keyframe not in rs.keyframes:
            raise ValueError(
                f"robot '{rs.name}' has no '{reset_keyframe}' keyframe in robot.yaml "
                f"(config reset_keyframe={reset_keyframe!r}); available: {sorted(rs.keyframes)}"
            )
        self._reset_keyframe = reset_keyframe
        home_qpos_np = np.array(m.key(reset_keyframe).qpos)
        self._home_qpos = jp.array(home_qpos_np)

        # Actuated-joint address tables, canonical order (rs.actuated_joints
        # is the action/obs contract; build_spec injects actuators in this
        # same order, see robot/build.py).
        self._qadr = jp.array([m.joint(n).qposadr[0] for n in rs.actuated_joints])
        self._vadr = jp.array([m.joint(n).dofadr[0] for n in rs.actuated_joints])

        # Default pose: the reset keyframe's actuated-joint values, canonical
        # order (missing joints default to 0.0, per spec.py's Keyframe).
        reset_kf = rs.keyframes[reset_keyframe]
        self._default_pose = jp.array([reset_kf.joints.get(n, 0.0) for n in rs.actuated_joints])

        # Per-joint action scale from the actuator preset (canonical order).
        scales = preset_action_scale(self._preset, rs)
        self._action_scale = jp.array([scales[n] for n in rs.actuated_joints])
        self._actuator_model = ACTUATOR_MODELS[self._preset.model]

        # Per-actuator ctrl clip bounds. Actuators with a real ctrlrange
        # (ideal-torque: ctrllimited=True) clip to it. PD actuators are
        # deliberately ctrllimited=False (see actuators/models.py's
        # PositionPD.inject docstring: "the soft limits ... belong to the RL
        # layer, not the actuator"), so they clip to the joint's own range
        # instead, shrunk by the preset's soft_limit_factor.
        ctrlrange = np.array(m.actuator_ctrlrange)
        ctrllimited = np.array(m.actuator_ctrllimited, dtype=bool)
        joint_range = np.array([m.joint(n).range for n in rs.actuated_joints])
        center = (joint_range[:, 0] + joint_range[:, 1]) / 2.0
        half = (joint_range[:, 1] - joint_range[:, 0]) / 2.0 * self._preset.soft_limit_factor
        # The soft joint limits themselves, always in RADIANS whatever the
        # actuator model's ctrl unit is. _ctrl_lo/_ctrl_hi below are them for
        # a pd preset and the actuator's own force envelope in Nm for an
        # ideal-torque one, so anything that clips an ANGLE (the settled-pose
        # anchor) has to read these two rather than those.
        self._soft_lo = np.asarray(center - half)
        self._soft_hi = np.asarray(center + half)
        lo = np.where(ctrllimited, ctrlrange[:, 0], self._soft_lo)
        hi = np.where(ctrllimited, ctrlrange[:, 1], self._soft_hi)
        self._ctrl_lo = jp.array(lo)
        self._ctrl_hi = jp.array(hi)

        # Foot geom/site id tables from RobotSpec.
        self._foot_site_ids = np.array([m.site(n).id for n in rs.foot_sites])
        self._foot_geom_ids = np.array([m.geom(n).id for n in rs.foot_geoms])
        self._foot_site_body = jp.array(m.site_bodyid[self._foot_site_ids])
        self._foot_geom_radius = jp.array(m.geom_size[self._foot_geom_ids, 0])
        self._n_feet = len(rs.foot_sites)
        foot_idx = _foot_geom_to_foot_index(m, self._foot_site_ids, self._foot_geom_ids)
        if (foot_idx < 0).any():
            bad = [rs.foot_geoms[i] for i in np.flatnonzero(foot_idx < 0)]
            raise ValueError(
                f"foot geom(s) {bad} are not on the same body (or a descendant of it) as "
                "any robot.yaml foot_sites entry"
            )
        self._foot_geom_foot_idx = jp.array(foot_idx)

        # Foot-site height at the reset keyframe pose. foot_sites are named
        # MJCF sites, not the sole surface itself, so they sit a few mm above
        # the geoms that actually touch the ground; a planted foot's raw site
        # z never reaches 0. Computed once here (mj_forward on the CPU model
        # at construct time, plain numpy -- never traced) and subtracted from
        # site z wherever gait clearance is scored.
        #
        # This is the KEYFRAME's site height, not the floor-referenced one.
        # The keyframe deliberately floats the robot a few mm up (5 mm on
        # asimov, 3 mm on roboto), so that margin is baked in and a planted
        # foot reads that much NEGATIVE, not 0 -- see _foot_clearance and
        # docs/lessons/foot-clearance.md. Deliberately not corrected during
        # the port: feet_phase is a pre-port term at scale 1.0, so moving
        # this shifts stock rewards and needs a golden regeneration.
        rest_data = mujoco.MjData(m)
        rest_data.qpos[:] = home_qpos_np
        mujoco.mj_forward(m, rest_data)
        self._foot_site_rest_z = jp.array(rest_data.site_xpos[self._foot_site_ids, 2])

        # Sensor addresses declared by robot.yaml's `sensors` map (gyro,
        # quat, linvel, acc); absent keys fall back to a qpos/qvel-derived
        # computation (see the obs helpers below).
        self._sensor_adr = {key: int(m.sensor(name).adr[0]) for key, name in rs.sensors.items()}

        # Free-joint (floating base) qpos/qvel addresses: fallback obs path
        # plus fall/push logic that needs the base height or planar qvel.
        self._base_qadr, self._base_vadr = _free_joint_addr(m)

        # Reward/reset anchor (see real_pose_ref in the task's default_config).
        # Off, these ARE the keyframe arrays -- the same objects, so the
        # legacy path is bit-exact rather than numerically equal.
        #
        # NOT the ctrl anchor. `_default_pose` above stays what
        # ctrl_from_action centers on and what the joint_pos observation
        # subtracts; only the `pose`/`stand_still` reference and the reset
        # pose move. Keeping the two anchors separate matters because
        # re-centering the action space on a sagged pose would silently
        # change what a zero action commands and what the policy reads back.
        self._settle_ctrl = None
        if self._config.get("real_pose_ref", False):
            settle_ctrl, settled_qpos = self._settle_pose()
            self._check_settled(settled_qpos)
            self._settle_ctrl = settle_ctrl
            self._pose_anchor = jp.array(settled_qpos[np.asarray(self._qadr)])
            self._reset_qpos = jp.array(settled_qpos)
        else:
            self._pose_anchor = self._default_pose
            self._reset_qpos = self._home_qpos

    def _catalog_probe_info(self) -> dict:
        """The `info` entries `_obs_catalog` reads, as zeros.

        Lets a caller build the catalog on fresh data without a rollout;
        only the resulting shapes are meaningful. A task env that adds
        catalog entries fed from `info` extends this too.
        """
        return {"last_action": jp.zeros(self.action_size)}

    # -- settled-pose anchor -------------------------------------------------
    def _settle_pose(self):
        """Settle a quasi-rigid copy of the model and return (ctrl, qpos).

        One settle, because this env has a single standing pose. If a
        commanded stand height ever lands here, the extension is a grid:
        settle a rung per height and interpolate the anchor on the commanded
        height.

        The copy is deliberately not this run's plant. Every actuator becomes
        the same stiff position servo (kp 400, kd 20) with its force cap
        removed, so the pose that comes out is a function of the geometry and
        the soft-limit envelope alone: every gain set and every actuator
        model at a given `soft_limit_factor` anchors on the same one. kp 400
        leaves about 5e-3 rad of residual sag, which is the accepted
        tolerance. implicitfast plus a 5e-4 timestep keep that stiff servo
        integrable whatever the training sim_dt is.

        The end-of-settle velocity lands in `self._settle_qvel` rather than
        in the return value: `_check_settled` needs it to tell a settled pose
        from a snapshot of a fall, and the (ctrl, qpos) pair is what callers
        and tests actually read.

        Plain MuJoCo on the CPU model, run once at construction, never
        traced. About 0.2 s.
        """
        m = copy.deepcopy(self._mj_model)
        # The actuator TYPES are overwritten here as well as the prm arrays,
        # because a torque preset's params are not a servo's.
        # The `ideal_torque` actuator model injects biastype NONE actuators
        # whose ctrl is a torque, and overwriting gainprm/biasprm on one of
        # those leaves `force = 400*ctrl` -- a torque command of 400x the
        # target angle, not a servo. Forcing the TYPES makes the settle
        # identical for every actuator model, which is the entire point of
        # the mechanism. ctrllimited goes with them: a torque preset's
        # ctrlrange is in Nm, and the targets below are radians, clipped to
        # the soft joint limits in their own unit.
        m.actuator_gaintype[:] = mujoco.mjtGain.mjGAIN_FIXED
        m.actuator_biastype[:] = mujoco.mjtBias.mjBIAS_AFFINE
        m.actuator_ctrllimited[:] = 0
        m.actuator_gainprm[:, 0] = 400.0
        m.actuator_biasprm[:, 1] = -400.0
        m.actuator_biasprm[:, 2] = -20.0
        m.actuator_forcerange[:, 0] = -1e6
        m.actuator_forcerange[:, 1] = 1e6
        m.opt.timestep = 5e-4
        m.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST

        # Clip with the preset's SOFT JOINT LIMITS, not the raw ctrlrange and
        # not _ctrl_lo/_ctrl_hi. The targets here are angles: step() clips a
        # pd preset's motor targets to the soft limits, so a pose settled
        # past them is one the policy can never command. The ctrlrange cannot
        # serve as the angle envelope here because the presets are not all
        # position servos. A pd preset's raw ctrlrange is [0, 0]
        # (ctrllimited=False), and clipping to it would command every joint
        # to zero; an ideal_torque preset's _ctrl_hi is
        # its forcerange in Nm, and clipping radians to +-120 does nothing at
        # all. Reading the angle envelope directly keeps the clip in one unit
        # for every actuator model.
        #
        # So the anchor is invariant across actuator MODELS and gain sets at
        # a given soft_limit_factor. It is NOT invariant to the factor, and
        # should not be: a preset that tightens the runtime envelope moves
        # the pose the policy can hold, which is preset policy, not an
        # actuator detail.
        ctrl = np.clip(np.asarray(self._default_pose), self._soft_lo, self._soft_hi)

        data = mujoco.MjData(m)
        mujoco.mj_resetData(m, data)
        data.qpos[:] = np.asarray(self._home_qpos)
        data.ctrl[:] = ctrl
        for _ in range(int(round(2.0 / m.opt.timestep))):
            mujoco.mj_step(m, data)
        self._settle_qvel = data.qvel.copy()
        return ctrl, data.qpos.copy()

    def _check_settled(self, qpos) -> None:
        """Reject a settle that did not end standing still.

        A collapsed settle must not become a silent anchor.

        Two conditions, because a biped needs both. The height check catches
        the robot that fell over (asimov_v1's `home` keyframe stands it on
        straight legs with the CoM about 2 cm behind the heel; held rigid it
        topples backward and comes to rest at 0.111 m). The at-rest check
        catches the one that is still going over at the two-second cut
        (asimov_v1's `knees_bent` is at 0.614 m and 0.15 rad/s there, and on
        the floor by four seconds) -- a height check alone would anchor on a
        snapshot of a fall. A settle that converges leaves five orders of
        magnitude of margin: roboto_origin's `home` ends at 7e-5 rad/s.

        The at-rest condition exists because these robots are bipeds. A
        statically stable quadruped never has to ask whether its keyframe is
        a standing equilibrium; a biped does, and a height check alone cannot
        answer it.
        """
        height = float(qpos[self._base_qadr + 2])
        floor = float(self._config.fall.min_height)
        name = self._robot_spec.name
        if not np.all(np.isfinite(qpos)):
            raise ValueError(
                f"real_pose_ref: the settle for robot '{name}' diverged "
                f"(non-finite qpos, settled height {height})"
            )
        if height < floor:
            raise ValueError(
                f"real_pose_ref: robot '{name}' settled at height {height:.4f} m, below "
                f"fall.min_height {floor} -- the '{self._reset_keyframe}' keyframe does not "
                "hold this robot up, so the settled pose is a fallen one and cannot anchor "
                "the pose rewards"
            )
        speed = float(np.abs(self._settle_qvel).max())
        if speed > _SETTLE_REST_SPEED:
            raise ValueError(
                f"real_pose_ref: robot '{name}' was still moving at {speed:.4f} (max |qvel|) "
                f"when the settle ended at height {height:.4f} m -- the "
                f"'{self._reset_keyframe}' keyframe is not a standing equilibrium, and a "
                "snapshot of a fall cannot anchor the pose rewards"
            )

    def _customize_model(self, m: mujoco.MjModel) -> None:
        """Task-specific tweaks applied before the model is put on device."""

    def _make_data(self):
        """mjx.make_data on the resolved backend, with warp budgets applied."""
        return self._make_data_fn()

    # -- MjxEnv plumbing -------------------------------------------------
    @property
    def xml_path(self) -> str:
        return str(self._robot_spec.model_xml_path)

    @property
    def action_size(self) -> int:
        return self._mj_model.nu

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model

    @property
    def robot_spec(self) -> RobotSpec:
        return self._robot_spec

    # -- obs helpers ----------------------------------------------------------
    def _quat(self, data):
        if "quat" in self._sensor_adr:
            adr = self._sensor_adr["quat"]
            return data.sensordata[adr : adr + 4]
        return data.qpos[self._base_qadr + 3 : self._base_qadr + 7]

    def _gyro(self, data):
        if "gyro" in self._sensor_adr:
            adr = self._sensor_adr["gyro"]
            return data.sensordata[adr : adr + 3]
        # A free joint's angular velocity is already expressed in the body's
        # local frame, matching a body-mounted gyro.
        return data.qvel[self._base_vadr + 3 : self._base_vadr + 6]

    def _gravity_body(self, data):
        return brax_math.rotate(jp.array([0.0, 0.0, -1.0]), brax_math.quat_inv(self._quat(data)))

    def _local_linvel(self, data):
        if "linvel" in self._sensor_adr:
            adr = self._sensor_adr["linvel"]
            return data.sensordata[adr : adr + 3]
        world_linvel = data.qvel[self._base_vadr : self._base_vadr + 3]
        return brax_math.rotate(world_linvel, brax_math.quat_inv(self._quat(data)))

    def _accel(self, data):
        if "acc" in self._sensor_adr:
            adr = self._sensor_adr["acc"]
            return data.sensordata[adr : adr + 3]
        raise KeyError("robot.yaml declares no 'acc' sensor and there is no qpos/qvel fallback for it")

    def _foot_site_pos(self, data):
        return data.site_xpos[self._foot_site_ids]

    def _foot_clearance(self, data):
        """Per-foot site height, referenced to the RESET KEYFRAME's site
        height.

        Not floor-referenced, and not a geom-bottom reading: a
        `geom_z - FOOT_RADIUS` measure would read exactly 0 for a resting
        sphere at any orientation. Here `_foot_site_rest_z` is measured at
        the reset keyframe, whose base height deliberately floats the lowest
        sole a few mm above the floor, so every reading is biased low by that
        margin: measured 4.92 mm on asimov_v1 and 3.11 mm on roboto_origin.

        A planted foot therefore reads about -5 mm (asimov) or -3 mm
        (roboto), never 0, and a swing apex reads low by the same offset --
        which shifts what `feet_phase`'s stance target of 0, `apex_target`
        and `glide_height` mean in physical metres. The numbers and the
        deferred fix are in docs/lessons/foot-clearance.md.
        """
        return self._foot_site_pos(data)[:, 2] - self._foot_site_rest_z

    def _foot_linvel(self, data):
        """Per-foot linear velocity at the foot_sites points (world frame)."""

        def one(point, body_id):
            jacp, _ = mjx.jac(self._mjx_model, data, point, body_id)
            return jacp.T @ data.qvel

        return jax.vmap(one)(self._foot_site_pos(data), self._foot_site_body)

    def _foot_contact(self, data):
        """Per-foot contact bool, aggregated (OR) over every geom on that foot."""
        z = data.geom_xpos[self._foot_geom_ids][:, 2]
        per_geom = (z < self._foot_geom_radius + 0.005).astype(jp.float32)
        return jp.zeros(self._n_feet).at[self._foot_geom_foot_idx].max(per_geom) > 0

    def _noisy(self, rng, clean, scales):
        noise = jax.random.uniform(rng, clean.shape, minval=-1.0, maxval=1.0)
        return clean + noise * scales

    def _obs_catalog(self, data, info):
        """name -> observation vector. Task envs extend with their signals.

        Only gyro/gravity/joint_pos/joint_vel/last_action are signals a real
        robot can expose; the rest (linvel, height, actuator_force, contacts)
        are sim-only and belong on the privileged (critic) list, never on
        the actor's.
        """
        return {
            "gyro": self._gyro(data),
            "gravity": self._gravity_body(data),
            "joint_pos": data.qpos[self._qadr] - self._default_pose,
            "joint_vel": data.qvel[self._vadr],
            "last_action": info["last_action"],
            # Sim-only signals, meant for the privileged critic list:
            "linvel": self._local_linvel(data),
            "height": data.qpos[self._base_qadr + 2 : self._base_qadr + 3],
            "actuator_force": data.actuator_force,
            "contacts": self._foot_contact(data).astype(jp.float32),
        }

    @property
    def actor_obs_names(self):
        """Resolved actor observation list: the task's ordered obs.state,
        filtered by the obs.include whitelist when one is set (sensor-suite
        presets name what the robot HAS; task signals it doesn't list are
        dropped too, so presets must include them explicitly)."""
        include = self._config.obs.get("include", ())
        names = list(self._config.obs.state)
        if include:
            names = [n for n in names if n in include]
        if not names:
            raise ValueError(
                f"obs.include {list(include)} leaves no actor observations "
                f"(task obs.state: {list(self._config.obs.state)})"
            )
        return names

    def _build_obs(self, data, info, rng=None):
        """Observations declared by the env config.

        `obs.state` (actor: sensors the real robot exposes) and
        `obs.privileged` (critic: anything the sim knows) are ordered lists
        of catalog names. Actor noise scales come from `obs_noise` by
        component name (no entry = noise-free).
        """
        catalog = self._obs_catalog(data, info)

        def gather(names):
            missing = [n for n in names if n not in catalog]
            if missing:
                raise KeyError(f"unknown obs component(s) {missing}; available: {sorted(catalog)}")
            return jp.concatenate([catalog[n] for n in names])

        state_names = self.actor_obs_names
        state = gather(state_names)
        if rng is not None:
            noise = self._config.obs_noise
            scales = jp.concatenate([jp.full(catalog[n].shape, noise.get(n, 0.0)) for n in state_names])
            state = self._noisy(rng, state, scales)
        return {
            "state": state,
            "privileged_state": gather(self._config.obs.privileged),
        }

