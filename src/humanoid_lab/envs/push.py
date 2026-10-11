"""The random base push: when it fires and what kick it applies.

Shared by both backends: imports only array_api_compat (through
envs/constants.py), numpy and the standard library.

A backend draws the random numbers named in PUSH_DRAWS and passes them in.
`push_schedule` says which worlds are pushed this step. `push_kick` turns the
draws into the velocity added to the base. The JAX adapter in
envs/joystick.py keeps the key scheme: the direction uses the push key itself
and every other draw uses `fold_in(r_push, 0x100 + i)`. A knob that is off
makes no draw, so it does not move any other draw.
"""

from __future__ import annotations

from humanoid_lab.envs.constants import xp_of

# name, fold-in index off the push key (None: the key itself), per-world
# shape, and the config rule that turns the draw on. The JAX adapter folds in
# 0x100 + index, as for COMMAND_DRAWS and DR_DRAWS. Ranges come from
# `push_ranges`.
PUSH_DRAWS = (
    ("dir", None, (2,), "always"),
    ("next_countdown", 1, (), "interval_steps_range"),
    ("vel_z", 2, (), "vel_z"),
    ("ang", 3, (3,), "ang_vel_rp or ang_vel_yaw"),
)


def _ranged(push_cfg):
    return bool(push_cfg.get("interval_steps_range", None))


def push_draw_names(push_cfg) -> tuple:
    """The PUSH_DRAWS names this push config needs, in table order."""
    on = {
        "dir": True,
        "next_countdown": _ranged(push_cfg),
        "vel_z": bool(push_cfg.get("vel_z", 0.0)),
        "ang": bool(push_cfg.get("ang_vel_rp", 0.0) or push_cfg.get("ang_vel_yaw", 0.0)),
    }
    return tuple(name for name, _, _, _ in PUSH_DRAWS if on[name])


def push_ranges(push_cfg) -> dict:
    """Name -> (low, high) for each needed draw.

    dir and vel_z and ang are uniform on [low, high]. For ang, low and high
    are 3-tuples. next_countdown is an integer in [low, high] inclusive.
    """
    out = {"dir": (-1.0, 1.0)}
    if _ranged(push_cfg):
        lo, hi = (int(v) for v in push_cfg.interval_steps_range)
        out["next_countdown"] = (lo, hi)
    vz = push_cfg.get("vel_z", 0.0)
    if vz:
        out["vel_z"] = (-vz, vz)
    # float() keeps a zero knob's low bound at -0.0 even when the config holds
    # the int 0. The JAX draw has always used -bound, which gives -0.0.
    rp = float(push_cfg.get("ang_vel_rp", 0.0))
    yaw = float(push_cfg.get("ang_vel_yaw", 0.0))
    if rp or yaw:
        out["ang"] = ((-rp, -rp, -yaw), (rp, rp, yaw))
    return out


def push_schedule(push_cfg, step_count, countdown=None, next_countdown=None):
    """(push_now, new countdown) for this step.

    With interval_steps_range the push fires when the countdown has reached
    zero, then the countdown restarts at next_countdown. Otherwise it fires on
    the last step of every interval_steps and the countdown is None.
    """
    if _ranged(push_cfg):
        xp = xp_of(countdown, next_countdown)
        push_now = countdown <= 0
        return push_now, xp.where(push_now, next_countdown, countdown - 1)
    interval = int(push_cfg.interval_steps)
    return (step_count % interval) == (interval - 1), None


def push_kick(push_cfg, push_now, draws):
    """The kick of the pushed worlds: {'planar', 'z', 'ang'}.

    planar is the (2,) velocity added to the base x and y. z is the vertical
    velocity, None when vel_z is off. ang is the (3,) angular velocity, None
    when both angular knobs are off. Worlds that are not pushed get zeros.
    """
    xp = xp_of(draws["dir"])
    d = draws["dir"]
    unit = d / (xp.linalg.vector_norm(d, axis=-1, keepdims=True) + 1e-6) * push_cfg.vel
    planar = xp.where(push_now[..., None], unit, xp.zeros_like(unit))
    z = None
    if push_cfg.get("vel_z", 0.0):
        vz = draws["vel_z"]
        z = xp.where(push_now, vz, xp.zeros_like(vz))
    ang = None
    if push_cfg.get("ang_vel_rp", 0.0) or push_cfg.get("ang_vel_yaw", 0.0):
        a = draws["ang"]
        ang = xp.where(push_now[..., None], a, xp.zeros_like(a))
    return {"planar": planar, "z": z, "ang": ang}
