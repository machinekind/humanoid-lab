"""Training-time env wrappers layered on mujoco_playground's.

Three respawn reseeds live here, one per layer of envs/gait_state.py: the
no-progress meter's, applied only when `no_progress.enable` is on, the
per-foot gait trackers', applied only when a gait-symmetry scale is nonzero,
and the heading reference's, applied only when `heading_err` is an
observation. With none on, `make_wrap_env_fn` hands back playground's own
function unchanged, so such a run takes the identical training path it
always did. The clearing itself is `gait_state.clear_done`.
"""

from __future__ import annotations

import jax
from mujoco_playground import wrapper as playground_wrapper
from mujoco_playground._src import mjx_env

from humanoid_lab.envs import gait_state, progress, signals
from humanoid_lab.envs.gait_state import GAIT_TRACKER_KEYS  # noqa: F401  (re-exported)


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
    what a command resample does (step 10 of envs/gait_state.update). Zeroing
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
        info = gait_state.clear_done(
            state.info,
            done,
            ("progress",),
            {"progress_ema": progress.cmd_speed(state.info["command"])},
        )
        return state.replace(info=info)


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
        info = gait_state.clear_done(state.info, done, ("gait",), {})
        return state.replace(info=info)


class HeadingReseedWrapper(playground_wrapper.Wrapper):
    """Restart the heading reference whenever an episode respawns.

    `info['yaw_ref']` is the heading the policy is asked to keep, and
    `heading_err` observes wrap(yaw_ref - yaw). The same
    `BraxAutoResetWrapper(full_reset=False)` behavior as above applies:
    `info` survives the respawn while `data` goes back to the first state.
    A robot that died mid-turn carries a yaw_ref far from the respawn's yaw,
    so the new episode starts with a large heading error that no command
    explains. The policy is then trained to correct a heading offset it
    never earned.

    `state.data` of a done env is already the restored first data, so the
    fresh reference is the yaw of that data. This sits outside the vmap, and
    `_quat` is batch-safe.
    """

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        state = self.env.step(state, action)
        done = state.done > 0.0
        yaw = signals.base_yaw(self.unwrapped._quat(state.data))
        info = gait_state.clear_done(state.info, done, ("heading",), {"yaw_ref": yaw})
        return state.replace(info=info)


_LAYER_WRAPPERS = {
    "progress": ProgressReseedWrapper,
    "gait": GaitReseedWrapper,
    "heading": HeadingReseedWrapper,
}


def make_wrap_env_fn(env_config):
    """The `wrap_env_fn` train.py hands brax's ppo.train.

    With every reseed flag off this IS
    `mujoco_playground.wrapper.wrap_for_brax_training`, the same object, so
    no run that needs none changes shape.
    """
    layers = [_LAYER_WRAPPERS[f] for f in gait_state.reseed_flags(env_config)]
    if not layers:
        return playground_wrapper.wrap_for_brax_training

    def wrap_for_brax_training(env, **kwargs):
        env = playground_wrapper.wrap_for_brax_training(env, **kwargs)
        for layer in layers:
            env = layer(env)
        return env

    return wrap_for_brax_training
