"""Point-goal walking task (task=goal): walk to a target point and stop there.

The RPG-style "click a point, the character walks there" task, trained from
a sparse reward. The policy sees the goal as a planar vector in its own
heading frame; nothing commands a velocity, a gait clock or a foot height.

Reward: `reward.success` is paid once per reached goal, where "reached"
means the base stayed upright within `goal.radius` of the goal for
`goal.dwell_steps` consecutive control steps (the dwell is what stops a
dive at the goal from counting). A reached goal is replaced by a fresh one
drawn around the robot's current position, so one episode serves many
goals. A goal not reached within its time budget is replaced the same way
and pays nothing. A fall ends the episode, which costs every goal the rest
of the episode would have served. Every other scale defaults to 0 and is
there for ablations (`progress` is the dense distance-closing baseline).

Curriculum: each env carries its own `goal_level` in [0, 1] in `info`. The
goal distance is drawn from [0, d_max] with d_max lerped from
`goal.dist_start` to `goal.dist_max` by the level; a reached goal raises it
by `curriculum.up`, a missed goal or a fall lowers it by `curriculum.down`.
`info` survives brax's in-place respawn, so the level does too.
`goal.dist_start` below the radius means the first goals sit at the
robot's feet: holding still upright for the dwell is the first skill the
sparse reward can teach.

Respawn: `BraxAutoResetWrapper(full_reset=False)` restores `data` to the
cached first state but keeps `info`, so the goal (stored in world
coordinates) would point at wherever the dead episode left it. The
GoalReseedWrapper in envs/wrappers.py redraws it on every respawn.
"""

from __future__ import annotations

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env

from humanoid_lab.envs.joystick import Joystick
from humanoid_lab.envs.joystick import default_config as _joystick_default_config
from humanoid_lab.rewards import terms

# reward.scales keys this task computes. Joystick's own scales stay in the
# config (robot overlays write them) but are never read here.
GOAL_REWARD_TERMS = ("success", "progress", "alive", "energy", "action_rate", "termination")


def default_config() -> config_dict.ConfigDict:
    cfg = _joystick_default_config()
    cfg.episode_length = 1000  # 20 s at ctrl_dt 0.02
    cfg.obs.state = ("gyro", "gravity", "joint_pos", "joint_vel", "last_action", "goal")
    cfg.obs.privileged = (
        "gyro", "gravity", "joint_pos", "joint_vel", "last_action", "goal",
        "goal_time", "linvel", "height", "contacts", "actuator_force",
    )
    cfg.goal = config_dict.create(
        radius=0.3,
        dwell_steps=25,  # 0.5 s inside the radius, upright
        dist_start=0.5,  # d_max at level 0
        dist_max=4.0,  # d_max at level 1
        # Per-goal time budget: base + dist / speed seconds.
        time_base=3.0,
        time_speed=0.3,
        # The goal observation is clipped to this norm, so a far goal reads
        # as a direction plus "far".
        obs_clip=5.0,
        curriculum=config_dict.create(enable=True, up=0.05, down=0.05, init=0.0),
    )
    # Goal-task reward. Joystick's reward block stays for the overlays'
    # sake; these scales sit beside it under their own name.
    cfg.goal_reward = config_dict.create(
        scales=config_dict.create(
            success=1.0,
            progress=0.0,  # m of distance closed per step
            alive=0.0,
            energy=0.0,
            action_rate=0.0,
            termination=0.0,
        ),
    )
    return cfg


class Goal(Joystick):
    """Walk to a planar goal and hold there; see the module docstring."""

    def __init__(self, robot_dir, preset_name, config=None, config_overrides=None, actuator_overrides=None):
        super().__init__(
            robot_dir, preset_name, config or default_config(), config_overrides, actuator_overrides
        )
        if self._config.symmetry.enable:
            raise ValueError("task=goal does not support the mirror augmentation yet")

    # -- goal helpers -------------------------------------------------------
    def _base_xy(self, data):
        return data.qpos[self._base_qadr : self._base_qadr + 2]

    def _yaw(self, data):
        q = self._quat(data)
        w, x, y, z = q[0], q[1], q[2], q[3]
        return jp.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    def _goal_body(self, data, goal_xy):
        """Goal relative to the base, in the heading (yaw-only) frame."""
        d = goal_xy - self._base_xy(data)
        yaw = self._yaw(data)
        c, s = jp.cos(yaw), jp.sin(yaw)
        return jp.array([c * d[0] + s * d[1], -s * d[0] + c * d[1]])

    def _dist_cap(self, level):
        g = self._config.goal
        return g.dist_start + level * (g.dist_max - g.dist_start)

    def _sample_goal(self, rng, origin_xy, level):
        """A goal uniform over the disc of radius d_max(level) around origin.
        Returns (goal_xy, time budget in steps)."""
        g = self._config.goal
        r_d, r_a = jax.random.split(rng)
        d = self._dist_cap(level) * jp.sqrt(jax.random.uniform(r_d))
        a = jax.random.uniform(r_a, minval=-jp.pi, maxval=jp.pi)
        goal = origin_xy + d * jp.array([jp.cos(a), jp.sin(a)])
        budget = jp.round((g.time_base + d / g.time_speed) / self.dt).astype(jp.int32)
        return goal, budget

    def respawn_goal(self, info, data, done):
        """Redraw the goal for an env that just respawned (see
        GoalReseedWrapper); `data` is the respawned state."""
        info = dict(info)
        rng, r_goal = jax.random.split(info["rng"])
        goal, budget = self._sample_goal(r_goal, self._base_xy(data), info["goal_level"])
        info["rng"] = jp.where(done, rng, info["rng"])
        info["goal"] = jp.where(done, goal, info["goal"])
        info["goal_budget"] = jp.where(done, budget, info["goal_budget"])
        info["goal_steps"] = jp.where(done, 0, info["goal_steps"])
        info["goal_dwell"] = jp.where(done, 0, info["goal_dwell"])
        info["goal_dist_prev"] = jp.where(
            done, jp.linalg.norm(goal - self._base_xy(data)), info["goal_dist_prev"]
        )
        info["last_action"] = jp.where(done, jp.zeros_like(info["last_action"]), info["last_action"])
        return info

    # -- reset / step -------------------------------------------------------
    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, r_pose, r_goal = jax.random.split(rng, 3)
        reset_noise = self._config.get("reset_noise", 0.0)
        pose_noise = jax.random.uniform(r_pose, (self.action_size,), minval=-1.0, maxval=1.0)
        qpos = self._reset_qpos.at[self._qadr].add(pose_noise * reset_noise)

        data = self._make_data()
        data = data.replace(qpos=qpos, qvel=jp.zeros(self._mj_model.nv), ctrl=self._neutral_ctrl)
        data = mjx.forward(self._mjx_model, data)

        level = jp.array(float(self._config.goal.curriculum.init))
        goal, budget = self._sample_goal(r_goal, self._base_xy(data), level)
        info = {
            "rng": rng,
            "last_action": jp.zeros(self.action_size),
            "goal": goal,
            "goal_budget": budget,
            "goal_steps": jp.array(0, dtype=jp.int32),
            "goal_dwell": jp.array(0, dtype=jp.int32),
            "goal_dist_prev": jp.linalg.norm(goal - self._base_xy(data)),
            "goal_level": level,
            "step_count": jp.array(0),
        }
        metrics = {f"reward/{k}": jp.zeros(()) for k in GOAL_REWARD_TERMS}
        metrics.update(
            {
                "goal/reached": jp.zeros(()),
                "goal/timeout": jp.zeros(()),
                "goal/level_per_step": jp.zeros(()),
                "goal/dist_per_step": jp.zeros(()),
            }
        )
        obs = self._build_obs(data, info)
        return mjx_env.State(data, obs, jp.zeros(()), jp.zeros(()), metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        info = dict(state.info)
        rng, r_noise, r_goal = jax.random.split(info["rng"], 3)
        info["rng"] = rng
        g = self._config.goal

        motor_targets = jp.clip(
            self._actuator_model.ctrl_from_action(action, self._default_pose, self._action_scale),
            self._ctrl_lo,
            self._ctrl_hi,
        )
        data = mjx_env.step(self._mjx_model, state.data, motor_targets, self.n_substeps)

        gravity = self._gravity_body(data)
        base_height = data.qpos[self._base_qadr + 2]
        fall = (base_height < self._config.fall.min_height) | (gravity[2] > self._config.fall.max_tilt_gz)

        dist = jp.linalg.norm(info["goal"] - self._base_xy(data))
        inside = (dist < g.radius) & ~fall
        dwell = jp.where(inside, info["goal_dwell"] + 1, 0)
        reached = dwell >= g.dwell_steps
        goal_steps = info["goal_steps"] + 1
        timeout = (goal_steps >= info["goal_budget"]) & ~reached

        rewards = {
            "success": reached.astype(jp.float32),
            "progress": jp.clip(info["goal_dist_prev"] - dist, -0.05, 0.05),
            "alive": jp.ones(()),
            "energy": terms.energy(data.qvel[self._vadr], data.actuator_force),
            "action_rate": terms.action_rate(action, info["last_action"]),
            "termination": fall.astype(jp.float32),
        }
        scales = self._config.goal_reward.scales
        reward = sum(rewards[k] * scales[k] for k in GOAL_REWARD_TERMS if scales[k])
        reward = jp.asarray(reward, dtype=jp.float32)

        cur = g.curriculum
        level = info["goal_level"]
        if cur.enable:
            level = level + cur.up * reached - cur.down * (timeout | fall)
            level = jp.clip(level, 0.0, 1.0)
        info["goal_level"] = level

        # A reached or timed-out goal is replaced around the current position.
        new_goal = reached | timeout
        goal, budget = self._sample_goal(r_goal, self._base_xy(data), level)
        info["goal"] = jp.where(new_goal, goal, info["goal"])
        info["goal_budget"] = jp.where(new_goal, budget, info["goal_budget"])
        info["goal_steps"] = jp.where(new_goal, 0, goal_steps)
        info["goal_dwell"] = jp.where(new_goal, 0, dwell)
        info["goal_dist_prev"] = jp.linalg.norm(info["goal"] - self._base_xy(data))
        info["last_action"] = action
        info["step_count"] = info["step_count"] + 1

        metrics = {
            **state.metrics,
            **{f"reward/{k}": v for k, v in rewards.items()},
            "goal/reached": reached.astype(jp.float32),
            "goal/timeout": timeout.astype(jp.float32),
            "goal/level_per_step": level,
            "goal/dist_per_step": dist,
        }
        obs = self._build_obs(data, info, r_noise)
        return mjx_env.State(data, obs, reward, fall.astype(jp.float32), metrics, info)

    # -- observations -------------------------------------------------------
    def _obs_catalog(self, data, info):
        catalog = super(Joystick, self)._obs_catalog(data, info)
        goal_body = self._goal_body(data, info["goal"])
        norm = jp.linalg.norm(goal_body)
        clip = self._config.goal.obs_clip
        catalog["goal"] = goal_body * jp.minimum(1.0, clip / jp.maximum(norm, 1e-6))
        catalog["goal_time"] = jp.array(
            [1.0 - info["goal_steps"] / jp.maximum(info["goal_budget"], 1)], dtype=jp.float32
        )
        return catalog

    def _catalog_probe_info(self):
        return {
            "last_action": jp.zeros(self.action_size),
            "goal": jp.zeros(2),
            "goal_steps": jp.array(0),
            "goal_budget": jp.array(1),
        }
