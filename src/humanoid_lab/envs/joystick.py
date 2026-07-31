"""Joystick velocity-tracking task, robot-agnostic (biped).

Ported from w01-tek's wojtek_rl/env.py (w01-tekJoystick / default_config),
adapted from a 4-leg quadruped to a 2-foot biped: the gait clock is a fixed
antiphase (offsets [0, pi]) two-foot clock instead of w01-tek's walk/trot
blend, there is no commanded stand height (asimov has one static standing
height, not a quadruped's adjustable crouch), and the quadruped-only reward
terms (contact_match, high_step, height_tracking, stand_feet_down) are
dropped -- see rewards/terms.py's module docstring and PLAN.md step 6.

Actor observations use only signals the real robot has (IMU + joint
encoders + own history + the commanded velocity + the env's own gait
clock); everything else (world-frame linvel, base height, contacts,
actuator force) is privileged (critic-only).

Deferred (not yet ported):
- w01-tek's action_delay/latency/encoder-offset/action-filter machinery is
  deliberately NOT ported yet. w01-tek defaulted to a 1-step action delay, so
  this port applies actions one control step earlier than w01-tek did.
  PLAN.md's v1 sysid row (action delay 0-1 control steps) means this
  machinery must return before sim2real-fidelity training.
"""

from __future__ import annotations

import jax
import jax.numpy as jp
import numpy as np
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env

from humanoid_lab.envs.base import HumanoidEnv
from humanoid_lab.rewards import terms

# Per-foot gait clock offset, robot_spec.foot_sites order: antiphase (one
# foot swings while the other stances), the only sensible clock for a biped
# -- w01-tek's quadruped walk/trot blend has no biped analogue.
_PHASE_OFFSETS = (0.0, np.pi)


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        # asimov docs (PLAN.md "Research findings"): 200 Hz physics, 50 Hz
        # policy (decimation 4).
        ctrl_dt=0.02,
        sim_dt=0.005,
        sim=config_dict.create(
            # Physics backend. auto picks warp on a CUDA host and jax
            # elsewhere. naconmax_per_env/njmax carried from wojtek_rl's
            # defaults as a starting point -- untuned for asimov's contact
            # set (biped, capsule feet + toes, more geoms per foot than
            # w01-tek's 4-leg robot); revisit if warp silently drops contacts.
            backend="auto",
            naconmax_per_env=32,
            njmax=320,
            num_envs=1,
        ),
        episode_length=1000,
        # robot.yaml keyframe used for reset qpos and the PD default_pose
        # anchor (see envs/base.py). Validated to exist in the model at
        # construct time.
        reset_keyframe="home",
        # w01-tek-parity reset pose noise: uniform +-reset_noise (rad) added
        # to every actuated joint's qpos at reset, so the policy never sees
        # the exact same start pose twice. 0.0 disables it.
        reset_noise=0.05,
        # Per-joint vector from the actuator preset's action_scale() by
        # default (0.0 sentinel = "use the preset"). A scalar or per-joint
        # list here overrides it uniformly, mirroring w01-tek's action_scale
        # knob.
        action_scale=0.0,
        # asimov docs (PLAN.md): gyro +-0.01, joint_pos +-0.01 rad,
        # joint_vel +-0.1 rad/s. No gravity noise figure was published;
        # unlisted components default to noise-free (see base.py's
        # _build_obs).
        obs_noise=config_dict.create(gyro=0.01, joint_pos=0.01, joint_vel=0.1),
        # Declarative observation spec: ordered lists of catalog names (see
        # HumanoidEnv._obs_catalog + this env's command/phase additions).
        # `state` is the actor (only signals a real robot has -- no
        # world-frame linvel), `privileged` the critic.
        obs=config_dict.create(
            include=(),  # obs presets whitelist actor sensors; () = all
            state=(
                "gyro",
                "gravity",
                "joint_pos",
                "joint_vel",
                "last_action",
                "command",
                "phase",
            ),
            privileged=(
                "gyro",
                "gravity",
                "joint_pos",
                "joint_vel",
                "last_action",
                "command",
                "phase",
                "linvel",
                "height",
                "contacts",
                "actuator_force",
            ),
        ),
        # asimov docs (PLAN.md): command envelope x +-0.8 m/s, y +-0.6 m/s,
        # yaw +-0.6 rad/s.
        command=config_dict.create(
            vx=(-0.8, 0.8),
            vy=(-0.6, 0.6),
            wz=(-0.6, 0.6),
            resample_steps=250,  # 5 s at ctrl_dt=0.02, carried from w01-tek
            # Velocity zeroes with this prob (stand training). Carried
            # from w01-tek's joystick task as a starting value, untuned for
            # a biped.
            zero_prob=0.15,
        ),
        # Carried from w01-tek's joystick task, untuned for asimov's mass/leg
        # length.
        push=config_dict.create(enable=True, interval_steps=200, vel=0.4),
        # asimov stands ~0.72-0.75 m. robot.yaml's home keyframe base_pos z
        # (0.636 m) is not this: it's the floating-base placeholder measured
        # so the feet just touch the floor, not the robot's standing height.
        # max_tilt_gz ported from w01-tek's form (gravity_body z component,
        # near -1 upright, rises toward 0/positive as the robot tips over);
        # -0.4 is w01-tek's own untuned value, carried as a starting point.
        fall=config_dict.create(min_height=0.45, max_tilt_gz=-0.4),
        # Two-foot antiphase gait clock (see _PHASE_OFFSETS). freq/
        # swing_height/duty are untuned starting values: w01-tek's numbers
        # were tuned for a 0.21 m four-bar leg, not a human-scale biped leg,
        # so only the SHAPE (speed-scaled clock frequency, sinusoidal swing
        # profile, fixed stance duty) is carried, not the numbers.
        gait=config_dict.create(
            freq=(1.0, 2.0),  # Hz, speed-scaled between these two bounds
            swing_height=0.08,  # target peak foot clearance, m
            duty=0.5,  # fixed stance fraction (biped: no walk/trot blend)
            air_time_cap=0.0,  # 0 = uncapped
        ),
        reward=config_dict.create(
            tracking_sigma=0.25,
            # Multiplicative velocity tracking (off = legacy additive).
            # Additive tracking pays the easy half of a command: a robot
            # that deadlocks under a pure-spin command still earns the FULL
            # tracking_lin_vel, because its commanded linear velocity is 0
            # and standing still "tracks" it perfectly. w01-tek measured that
            # payout at about 63% of an ideal spin's. With tracking_product
            # both terms are gated by the product of the two kernels: full
            # pay only when the WHOLE command is tracked, and a deadlocked
            # spin earns ~0.
            tracking_product=False,
            phase_sigma=0.002,
            # torque_limit hinge fires above this fraction of each
            # actuator's forcerange cap.
            torque_limit_frac=0.85,
            # UNTUNED starting values, ported from wojtek_rl/env.py's
            # joystick default_config for every term that carries over to a
            # biped (rewards/terms.py's module docstring lists what was
            # dropped). Do not read these as tuned for asimov: they are the
            # PLAN.md-mandated starting point for the first CPU smoke run,
            # nothing more.
            scales=config_dict.create(
                tracking_lin_vel=1.5,
                tracking_ang_vel=0.8,
                lin_vel_z=-2.0,
                ang_vel_xy=-0.05,
                orientation=-5.0,
                torques=-2e-4,
                torque_rate=0.0,
                action_rate=-0.25,
                action_accel=-0.1,
                energy=-2e-3,
                pose=-0.5,
                feet_air_time=2.0,
                feet_slip=-0.25,
                feet_phase=1.0,
                stand_still=-0.5,
                termination=-1.0,
                torque_limit=0.0,
            ),
        ),
    )


class Joystick(HumanoidEnv):
    def __init__(self, robot_dir, preset_name, config=None, config_overrides=None, actuator_overrides=None):
        super().__init__(
            robot_dir, preset_name, config or default_config(), config_overrides, actuator_overrides
        )

        # action_scale override: 0.0 sentinel keeps the preset-derived
        # per-joint vector HumanoidEnv.__init__ already computed; a scalar
        # or per-joint sequence here overrides it uniformly.
        override = self._config.get("action_scale", 0.0)
        if isinstance(override, (tuple, list)):
            self._action_scale = jp.array([float(v) for v in override])
        elif float(override) != 0.0:
            self._action_scale = jp.full(self.action_size, float(override))

        self._torque_cap = jp.array(self._mj_model.actuator_forcerange[:, 1])
        # Pose-deviation weight: uniform. w01-tek's abduction/leg-joint split
        # (full weight on the ab/adduction joint, lighter on the other two)
        # doesn't map cleanly onto asimov's 6-joint leg (hip_pitch, hip_roll,
        # hip_yaw, knee, ankle_pitch, ankle_roll); an explicit deviation,
        # left uniform rather than guessing a split.
        self._pose_weight = jp.ones(self.action_size)

        c = self._config.command
        self._cmd_vmax = self._cmd_speed(jp.array([max(abs(c.vx[0]), abs(c.vx[1])), 0.0, 0.0]))

        # Neutral ctrl (zero action): default_pose for a PD-style actuator
        # model, zero torque for ideal-torque -- whatever the actuator
        # model's own ctrl_from_action says "zero action" means.
        self._neutral_ctrl = self._actuator_model.ctrl_from_action(
            jp.zeros(self.action_size), self._default_pose, self._action_scale
        )

    # -- command / gait clock ------------------------------------------------
    def _sample_command(self, rng):
        r1, r2, r3, r4 = jax.random.split(rng, 4)
        c = self._config.command
        vel = jp.array(
            [
                jax.random.uniform(r1, minval=c.vx[0], maxval=c.vx[1]),
                jax.random.uniform(r2, minval=c.vy[0], maxval=c.vy[1]),
                jax.random.uniform(r3, minval=c.wz[0], maxval=c.wz[1]),
            ]
        )
        zero = jax.random.bernoulli(r4, c.zero_prob)
        return jp.where(zero, jp.zeros(3), vel)

    def _cmd_speed(self, command):
        """Planar speed the gait clock should serve; turning counts too."""
        return jp.linalg.norm(command[:2]) + 0.3 * jp.abs(command[2])

    def _leg_phases(self, info):
        phase = info["phase"] + jp.array(_PHASE_OFFSETS)
        return jp.fmod(phase + jp.pi, 2 * jp.pi) - jp.pi

    def _gait_targets(self, info):
        """(target foot clearance, stance mask) from the duty-aware clock."""
        g = self._config.gait
        duty = g.duty
        theta = jp.fmod(self._leg_phases(info) + 2 * jp.pi, 2 * jp.pi) / (2 * jp.pi)
        swing_frac = 1.0 - duty
        in_swing = theta < swing_frac
        rz = g.swing_height * jp.sin(jp.pi * theta / swing_frac) * in_swing
        return rz, ~in_swing

    def _phase_dt(self, command):
        """Clock increment: speed-scaled; frozen when told to stand."""
        g = self._config.gait
        speed = self._cmd_speed(command)
        frac = jp.clip(speed / self._cmd_vmax, 0.0, 1.0)
        freq = g.freq[0] + (g.freq[1] - g.freq[0]) * frac
        return jp.where(speed > 0.05, 2 * jp.pi * self.dt * freq, 0.0)

    # -- reset / step -------------------------------------------------------
    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, r_cmd, r_pose = jax.random.split(rng, 3)
        command = self._sample_command(r_cmd)

        # w01-tek-parity reset pose noise: uniform +-reset_noise (rad) on
        # every actuated joint's qpos. reset_noise=0.0 degenerates to the
        # old exact-keyframe reset (still draws/consumes r_pose either way,
        # so the rng split discipline doesn't depend on the config value).
        reset_noise = self._config.get("reset_noise", 0.0)
        pose_noise = jax.random.uniform(r_pose, (self.action_size,), minval=-1.0, maxval=1.0)
        qpos = self._home_qpos.at[self._qadr].add(pose_noise * reset_noise)

        data = self._make_data()
        data = data.replace(qpos=qpos, qvel=jp.zeros(self._mj_model.nv), ctrl=self._neutral_ctrl)
        data = mjx.forward(self._mjx_model, data)

        info = {
            "rng": rng,
            "command": command,
            "last_action": jp.zeros(self.action_size),
            "last_last_action": jp.zeros(self.action_size),
            "last_torque": jp.zeros(self.action_size),
            "feet_air_time": jp.zeros(self._n_feet),
            "last_contact": jp.zeros(self._n_feet, dtype=bool),
            "phase": jp.array(0.0),
            "step_count": jp.array(0),
            "steps_since_cmd": jp.array(0),
        }
        # CRITICAL for scan-carry parity: every reward/* key present here
        # must also be present after every step() (see step()'s metric
        # merge below), or brax's training scan chokes on a changing
        # metrics pytree structure across steps.
        metrics = {f"reward/{k}": jp.zeros(()) for k in self._config.reward.scales}
        obs = self._build_obs(data, info)
        return mjx_env.State(data, obs, jp.zeros(()), jp.zeros(()), metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        info = dict(state.info)
        rng, r_noise, r_cmd, r_push = jax.random.split(info["rng"], 4)
        info["rng"] = rng

        motor_targets = jp.clip(
            self._actuator_model.ctrl_from_action(action, self._default_pose, self._action_scale),
            self._ctrl_lo,
            self._ctrl_hi,
        )

        data = state.data
        if self._config.push.enable:
            push_now = (info["step_count"] % self._config.push.interval_steps) == (
                self._config.push.interval_steps - 1
            )
            push = jax.random.uniform(r_push, (2,), minval=-1.0, maxval=1.0)
            push = push / (jp.linalg.norm(push) + 1e-6) * self._config.push.vel
            qvel = data.qvel.at[self._base_vadr : self._base_vadr + 2].add(
                jp.where(push_now, push, jp.zeros(2))
            )
            data = data.replace(qvel=qvel)

        data = mjx_env.step(self._mjx_model, data, motor_targets, self.n_substeps)

        contact = self._foot_contact(data)
        contact_filt = contact | info["last_contact"]
        first_contact = (info["feet_air_time"] > 0) & contact_filt

        rewards, fall = self._compute_rewards(data, info, action, first_contact, contact)

        info["feet_air_time"] = jp.where(contact_filt, 0.0, info["feet_air_time"] + self.dt)
        info["last_contact"] = contact
        info["last_last_action"] = info["last_action"]
        info["last_action"] = action
        info["last_torque"] = data.actuator_force
        info["step_count"] = info["step_count"] + 1

        phase = info["phase"] + self._phase_dt(info["command"])
        info["phase"] = jp.fmod(phase + jp.pi, 2 * jp.pi) - jp.pi

        info["steps_since_cmd"] = info["steps_since_cmd"] + 1
        resample = info["steps_since_cmd"] >= self._config.command.resample_steps
        info["command"] = jp.where(resample, self._sample_command(r_cmd), info["command"])
        info["steps_since_cmd"] = jp.where(resample, 0, info["steps_since_cmd"])

        reward = sum(rewards[k] * self._config.reward.scales[k] for k in rewards)
        reward = jp.clip(reward * self.dt, -100.0, 100.0)

        # Merge over the incoming metrics rather than replacing them: brax's
        # EvalWrapper injects extra keys (e.g. "reward") into state.metrics
        # that must survive every step for scan-carry parity (the pytree
        # structure fed into jax.lax.scan cannot change shape step to step).
        metrics = {
            **state.metrics,
            **{f"reward/{k}": v for k, v in rewards.items()},
        }

        obs = self._build_obs(data, info, r_noise)
        done = fall.astype(jp.float32)
        return mjx_env.State(data, obs, reward, done, metrics, info)

    # -- observations -------------------------------------------------------
    def _obs_catalog(self, data, info):
        catalog = super()._obs_catalog(data, info)
        catalog["command"] = info["command"]
        leg_phase = self._leg_phases(info)
        catalog["phase"] = jp.concatenate([jp.cos(leg_phase), jp.sin(leg_phase)])
        return catalog

    # -- rewards --------------------------------------------------------------
    def _compute_rewards(self, data, info, action, first_contact, contact):
        cmd = info["command"]
        linvel = self._local_linvel(data)
        gyro = self._gyro(data)
        gravity = self._gravity_body(data)
        cfg = self._config.reward
        moving = self._cmd_speed(cmd) > 0.05

        qpos_act = data.qpos[self._qadr]
        qvel_act = data.qvel[self._vadr]

        # foot_sites sit a few mm above the sole (see base.py's
        # _foot_site_rest_z docstring); subtract the construct-time resting
        # height so a planted foot reads ~0 clearance, matching the target
        # clearance's own stance value of 0 (w01-tek's geom-bottom semantic).
        foot_clearance = self._foot_site_pos(data)[:, 2] - self._foot_site_rest_z
        target_clearance, _stance_mask = self._gait_targets(info)
        foot_vel = self._foot_linvel(data)

        base_height = data.qpos[self._base_qadr + 2]
        fall = (base_height < self._config.fall.min_height) | (gravity[2] > self._config.fall.max_tilt_gz)

        k_lin = terms.tracking_lin_vel(cmd[:2], linvel[:2], cfg.tracking_sigma)
        k_ang = terms.tracking_ang_vel(cmd[2], gyro[2], cfg.tracking_sigma)
        if cfg.get("tracking_product", False):
            # Gate each term by the other's kernel: tracking pays only for
            # tracking the whole command (see tracking_product in
            # default_config). The reassignment is simultaneous, so both
            # sides read the pre-product kernels and both come out equal to
            # the same product.
            k_lin, k_ang = k_lin * k_ang, k_ang * k_lin

        rewards = {
            "tracking_lin_vel": k_lin,
            "tracking_ang_vel": k_ang,
            "lin_vel_z": terms.lin_vel_z(linvel[2]),
            "ang_vel_xy": terms.ang_vel_xy(gyro[:2]),
            "orientation": terms.orientation(gravity[:2]),
            "torques": terms.torques(data.actuator_force),
            "torque_rate": terms.torque_rate(data.actuator_force, info["last_torque"]),
            "action_rate": terms.action_rate(action, info["last_action"]),
            "action_accel": terms.action_accel(action, info["last_action"], info["last_last_action"]),
            "energy": terms.energy(qvel_act, data.actuator_force),
            "pose": terms.pose(qpos_act, self._default_pose, self._pose_weight),
            "feet_air_time": terms.feet_air_time(info["feet_air_time"], first_contact, self._config.gait.air_time_cap)
            * moving,
            "feet_slip": terms.feet_slip(foot_vel[:, :2], contact) * moving,
            "feet_phase": terms.feet_phase(foot_clearance, target_clearance, cfg.phase_sigma) * moving,
            "stand_still": terms.stand_still(qpos_act, self._default_pose, qvel_act) * (~moving),
            "termination": terms.termination(fall),
            "torque_limit": terms.torque_limit(data.actuator_force, self._torque_cap, cfg.torque_limit_frac),
        }
        return rewards, fall
