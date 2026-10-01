"""Training-time env wrappers layered on mujoco_playground's.

Two respawn reseeds live here: the no-progress meter's, applied only when
`no_progress.enable` is on, and the per-foot gait trackers', applied only
when a gait-symmetry scale is nonzero. With neither on, `make_wrap_env_fn`
hands back playground's own function unchanged, so such a run takes the
identical training path it always did.
"""

from __future__ import annotations

import jax
import jax.numpy as jp
from mujoco_playground import wrapper as playground_wrapper
from mujoco_playground._src import mjx_env

from humanoid_lab.envs.joystick import gait_dur_ema_on


class ProgressReseedWrapper(playground_wrapper.Wrapper):
    """Restart the no-progress meter whenever an episode respawns.

    `wrap_for_brax_training` ends in `BraxAutoResetWrapper(full_reset=False)`,
    which on done restores `data` and `obs` from the cached first state and
    returns `state.info` untouched ("only data and obs are reset, not the
    environment info" -- its own docstring). `info` therefore survives every
    termination: a fall, a truncation, and the no-progress cut itself.

    That is fatal for the cut specifically. It can only fire once the grace
    window has elapsed, so the respawn arrives already armed, carrying the
    dying episode's sub-threshold `progress_ema` and its large
    `steps_since_cmd`. At `ema_sec=1.0` the meter needs ~50 control steps to
    climb back out while the hazard is live the whole time, so the new
    episode dies inside what should have been its grace window and the run
    burns its samples on a cascade of one-second episodes.

    Reseeding to `_cmd_speed(command)` puts the meter at ratio 1, exactly
    what a command resample does (envs/joystick.py's step). Zeroing
    `steps_since_cmd` restores the grace window on top. Reseeding only the
    EMA and carrying the counter over would re-arm the cut on the respawn's
    first step. The command itself is
    deliberately left alone: the respawn continues serving it, and the meter
    is now measured against it from a fresh start.

    This sits OUTSIDE the vmap that `wrap_for_brax_training` puts on, so
    every info leaf carries a leading env axis. Nothing here is conditional
    on the cut being armed or on why the episode ended -- a respawn is a new
    episode however it was reached.
    """

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        state = self.env.step(state, action)
        done = state.done > 0.0
        info = dict(state.info)
        demand = jax.vmap(self.unwrapped._cmd_speed)(info["command"])
        info["progress_ema"] = jp.where(done, demand, info["progress_ema"])
        info["steps_since_cmd"] = jp.where(done, 0, info["steps_since_cmd"])
        return state.replace(info=info)


# Per-foot gait trackers GaitReseedWrapper restarts on respawn. reset()
# seeds every one of them to zeros (False for last_contact).
GAIT_TRACKER_KEYS = (
    "feet_air_time",
    "feet_contact_time",
    "swing_apex",
    "last_apex",
    "last_contact",
    "air_dur_ema",
    "stance_dur_ema",
)


class GaitReseedWrapper(playground_wrapper.Wrapper):
    """Restart the per-foot gait trackers whenever an episode respawns.

    The same `BraxAutoResetWrapper(full_reset=False)` behavior as above:
    `info` survives the respawn. For the swing/stance duration EMAs that
    gait_symmetry and gait_symmetry_income read, that has two effects.

    - Arming carries over. Both terms stay at 0 until the EMAs hold a
      completed duration, so that an episode's first steps are not priced.
      Carried EMAs are armed from the respawn's first step, and the income
      then pays a lifted foot before the new episode has taken a stride.
    - The dying episode's motion is folded in. A foot still airborne at the
      fall carries its `feet_air_time` into the respawn, and the respawn's
      first landing closes it as a swing into `air_dur_ema`.

    Restarting the EMAs and the mode timers that feed them (feet_air_time,
    feet_contact_time, last_contact, plus swing_apex and last_apex, which
    feed feet_apex and feet_apex_min the same way) to reset()'s values
    makes a respawn a fresh episode for every per-foot gait term.

    Like ProgressReseedWrapper this sits outside the vmap, so every info
    leaf carries a leading env axis and `done` broadcasts over the foot
    axis.
    """

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        state = self.env.step(state, action)
        done = state.done > 0.0
        info = dict(state.info)
        for k in GAIT_TRACKER_KEYS:
            v = info[k]
            mask = jp.reshape(done, done.shape + (1,) * (v.ndim - done.ndim))
            info[k] = jp.where(mask, jp.zeros_like(v), v)
        return state.replace(info=info)


def make_wrap_env_fn(env_config):
    """The `wrap_env_fn` train.py hands brax's ppo.train.

    With the no-progress cut off and both gait-symmetry scales at 0 this IS
    `mujoco_playground.wrapper.wrap_for_brax_training`, the same object, so
    no run that uses neither changes shape.
    """
    layers = []
    no_progress = env_config.get("no_progress")
    if no_progress is not None and no_progress.enable:
        layers.append(ProgressReseedWrapper)
    reward = env_config.get("reward")
    if reward is not None and gait_dur_ema_on(reward.get("scales", {})):
        layers.append(GaitReseedWrapper)
    if not layers:
        return playground_wrapper.wrap_for_brax_training

    def wrap_for_brax_training(env, **kwargs):
        env = playground_wrapper.wrap_for_brax_training(env, **kwargs)
        for layer in layers:
            env = layer(env)
        return env

    return wrap_for_brax_training
