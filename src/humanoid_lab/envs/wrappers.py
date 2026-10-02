"""Training-time env wrappers layered on mujoco_playground's.

One wrapper lives here: the no-progress meter's respawn reseed. It is
applied only when `no_progress.enable` is on. A terrain env (a config with
a `terrain` block) trains under the curriculum auto-reset stack of
envs/terrain_wrapper.py. `make_wrap_env_fn` hands back playground's own
function unchanged when neither applies, so a flat run with the cut off
takes the identical training path it always did.
"""

from __future__ import annotations

import jax
import jax.numpy as jp
from mujoco_playground import wrapper as playground_wrapper
from mujoco_playground._src import mjx_env

from humanoid_lab.envs.terrain_wrapper import wrap_for_terrain_brax_training


class ProgressReseedWrapper(playground_wrapper.Wrapper):
    """Restart the no-progress meter whenever an episode respawns.

    `wrap_for_brax_training` ends in `BraxAutoResetWrapper(full_reset=False)`,
    which on done restores `data` and `obs` from the cached first state and
    returns `state.info` untouched ("only data and obs are reset, not the
    environment info" -- its own docstring). `info` therefore survives every
    termination: a fall, a truncation, and the no-progress cut itself.
    The terrain stack's auto-reset (envs/terrain_wrapper.py) rewrites only
    the curriculum's info keys. The progress keys survive there too. The
    reseed covers both stacks.

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

    This sits OUTSIDE the vmap that either stack puts on, so
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


def make_wrap_env_fn(env_config):
    """The `wrap_env_fn` train.py hands brax's ppo.train.

    A config with a `terrain` block gets the curriculum stack,
    `wrap_for_terrain_brax_training`. Every other config gets
    `mujoco_playground.wrapper.wrap_for_brax_training`. With the
    no-progress cut on, the reseed wraps either stack from outside. A flat
    config with the cut off gets playground's function itself, the same
    object, so no such run changes shape. brax passes `episode_length` to
    the returned function for the training env and the eval env alike.
    """
    terrain = env_config.get("terrain") is not None
    no_progress = env_config.get("no_progress")
    reseed = no_progress is not None and no_progress.enable
    if not terrain and not reseed:
        return playground_wrapper.wrap_for_brax_training
    stack = wrap_for_terrain_brax_training if terrain else playground_wrapper.wrap_for_brax_training
    if not reseed:
        return stack

    def wrap_for_brax_training(env, **kwargs):
        return ProgressReseedWrapper(stack(env, **kwargs))

    return wrap_for_brax_training
