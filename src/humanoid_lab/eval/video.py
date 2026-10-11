"""Roll out a checkpoint under a fixed (or scripted battery) command and
render one scenario to MP4.

Run:
    python -m humanoid_lab.eval.video --run runs/<name> [--scenario walk_ramp] [--steps N]

or via run.sh:
    ./run.sh eval --run runs/<name> --scenario stand --steps 100

`--scenario` (default "walk_ramp") picks one of eval/battery.py's scripted
command trajectories (battery_scenarios), or `long`. `--steps` truncates it
to N control steps. A battery scenario uses the exact command builder the
battery measures, so its video and its battery.json row describe the same
trajectory. `long` is a video-only scenario (see long_scenario). It has no
battery row, so that rule does not hold for it.

`--heading-gain G` closes an outer heading loop on top of the scenario's
command (see heading_correction). The default 0 renders open-loop.

`--checkpoint STEP` renders the checkpoint saved at that step instead of the latest.

`--compare` renders the open-loop rollout next to the heading-closed one in
one clip. It loads the checkpoint once and runs both rollouts at the same
seed, scenario and checkpoint. The gain comes from `--heading-gain`, and
defaults to 1.5 (COMPARE_HEADING_GAIN). Each half is labelled in its top-left
corner. The shorter half repeats its last frame, so the side that keeps
walking stays visible after the other one falls. The default output is
`<scenario>_compare.mp4` in the run dir.

`--plot-torque` appends a normalized-torque strip below the render;
`--plot-joints` appends a joint target-vs-state grid below that; `--joint
NAME` swaps the grid for a single-joint zoom panel and implies
--plot-joints. All three are opt-in (off by default -- plain video output
is the same code path as before); panel rendering lives in eval/plots.py.

`--video-size WxH` sets the rendered frame size (the SceneView widens a
private model copy when the model's offscreen buffer is smaller).
`--overlay-torque` draws eval/overlays.py's per-actuator bar strip into
the frame itself: an instantaneous saturation view, where `--plot-torque`
is the full-episode trace below the frame. Both normalize by each joint's
own actuator cap.

`--push` restores the run's own random pushes for the rollout. They are OFF
by default, matching the battery's measurement convention: a mid-video kick
reads as a policy failure to anyone watching the clip. The default is stated
here (see push_override), not merely inherited from the battery's
measurement env -- this module owns what its own renders show.

MUJOCO_GL: `egl` is set (via setdefault, so an already-exported MUJOCO_GL
wins) only on linux, where headless GPU boxes need it; darwin is left on its
default (CGL), which is what this repo's Mac dev box actually has. Forcing
egl on darwin breaks offscreen rendering there, so this must run before
`mujoco` is imported anywhere in the process.
"""

import os
import sys

if sys.platform == "linux":  # headless GPU boxes; macOS uses its default GL (CGL)
    os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
from pathlib import Path

import jax
import mujoco
import numpy as np

from humanoid_lab.envs import signals
from humanoid_lab.eval import overlays
from humanoid_lab.eval.battery import (
    battery_scenarios,
    load_checkpoint_policy,
)
from humanoid_lab.eval.plots import joint_grid, joint_zoom, torque_strip
from humanoid_lab.eval.render import DEFAULT_SIZE, SceneView, frame_size
from humanoid_lab.eval.writer import write_video

# The outer heading loop never adds more than this to the scenario's wz (rad/s).
HEADING_WZ_CLIP = 0.5


def heading_correction(
    cmd: np.ndarray, yaw: float, yaw_target: float | None, gain: float, dt: float
) -> tuple[np.ndarray, float]:
    """One step of the video's outer heading loop. Returns (cmd, yaw_target).

    The first call (yaw_target None) anchors the target at the measured yaw.
    Later calls advance the target by the scenario's own wz times dt. The
    returned command adds gain times the wrapped heading error to wz,
    clipped to HEADING_WZ_CLIP. The input cmd is left unchanged.

    The error uses signals.wrap_angle, which does not clip.
    signals.heading_err clips to +-1 rad and would change the loop for
    errors above 1 rad.
    """
    yaw_target = yaw if yaw_target is None else yaw_target + float(cmd[2]) * dt
    err = float(signals.wrap_angle(np.asarray(yaw_target - yaw, dtype=np.float64)))
    out = np.array(cmd, dtype=np.float32)
    out[2] += float(np.clip(gain * err, -HEADING_WZ_CLIP, HEADING_WZ_CLIP))
    return out, yaw_target


# The video-only `long` clip: (t0, t1, vx0, vx1, wz) per segment. Times are in
# seconds. vx0 and vx1 are fractions of command.vx's maximum, and vx moves
# linearly from vx0 to vx1 inside the segment. wz is in rad/s, clipped to the
# command box. The sequence is stand, speed up to 0.6 and then 0.9 of the vx
# maximum, slow down, turn left, turn right, walk straight and stop.
LONG_SEGMENTS = (
    (0.0, 1.5, 0.0, 0.0, 0.0),
    (1.5, 5.5, 0.0, 0.6, 0.0),
    (5.5, 10.0, 0.6, 0.6, 0.0),
    (10.0, 13.0, 0.6, 0.9, 0.0),
    (13.0, 17.0, 0.9, 0.9, 0.0),
    (17.0, 20.0, 0.9, 0.3, 0.0),
    (20.0, 25.0, 0.4, 0.4, 0.5),
    (25.0, 30.0, 0.4, 0.4, -0.5),
    (30.0, 34.0, 0.6, 0.6, 0.0),
    (34.0, 37.0, 0.6, 0.0, 0.0),
    (37.0, 40.0, 0.0, 0.0, 0.0),
)


def long_scenario(dt: float, command):
    """The `long` clip as (cmd_at(i) -> np.ndarray([vx, vy, wz]), n_steps).

    Same shape as a battery_scenarios entry. vx scales with command.vx's
    maximum. wz keeps its rad/s value, clipped to command.wz's maximum.
    vy is always 0. The command is zero after the last segment.
    """
    vx_max = float(command["vx"][1])
    wz_max = float(command["wz"][1])
    n_steps = max(1, round(LONG_SEGMENTS[-1][1] / dt))

    def cmd_at(i):
        t = i * dt
        for t0, t1, vx0, vx1, wz in LONG_SEGMENTS:
            if t0 <= t < t1:
                vx = vx_max * (vx0 + (vx1 - vx0) * (t - t0) / (t1 - t0))
                return np.array([vx, 0.0, float(np.clip(wz, -wz_max, wz_max))])
        return np.array([0.0, 0.0, 0.0])

    return cmd_at, n_steps


# Scenarios this module renders that the battery does not measure, as
# name -> builder(dt, command).
VIDEO_ONLY_SCENARIOS = {"long": long_scenario}


def video_scenarios(dt: float, command) -> dict:
    """Every scenario --scenario accepts: the battery's plus the video-only ones."""
    return {
        **battery_scenarios(dt, command),
        **{name: build(dt, command) for name, build in VIDEO_ONLY_SCENARIOS.items()},
    }


def pick_scenario(dt: float, command, name: str):
    """(cmd_at, n_steps) for one scenario name. Raises KeyError listing the names."""
    names = video_scenarios(dt, command)
    if name not in names:
        raise KeyError(f"unknown scenario {name!r}; have {sorted(names)}")
    return names[name]


def push_override(push: bool) -> dict:
    """This module's own push decision, as an env override block.

    Written explicitly in both directions rather than relying on the
    battery's measurement env happening to disable pushes: what a rendered
    clip shows is this module's contract with whoever watches it. `--push`
    sets `enable` only, so the run's own `interval_steps` and `vel` are
    whatever it trained with.
    """
    return {"push": {"enable": bool(push)}}


def _pick_camera(mj_model: mujoco.MjModel, preferred: str | None = None):
    """robot.yaml's `eval_camera` when declared (validated to exist against
    the compiled model at spec load), else a free MjvCamera tracking the
    floating-base body. No camera names are guessed from the model: which
    view a robot's eval videos use is that robot's own call."""
    if preferred is not None:
        return preferred

    free = [
        i for i in range(mj_model.njnt) if mj_model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE
    ]
    body_id = int(mj_model.jnt_bodyid[free[0]]) if free else 0
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = body_id
    cam.distance = 3.0
    cam.azimuth = 90.0
    cam.elevation = -15.0
    return cam


def _composite_plots(
    frames, frame_times, env, joint_names, joint, plot_torque, plot_joints,
    torques, targets, positions,
):
    """vstack render `frames` with the requested plot panel(s), one column
    of matplotlib panels per frame. Returns a new frame list, same length
    and width as `frames`.

    Guards the degenerate no-samples case (e.g. --steps 0: `frames` has
    only the initial entry, `torques`/`targets`/`positions` are empty) by
    skipping panel compositing rather than crashing on an empty-array
    times[0]/times[-1] lookup inside eval/plots.py's builders.
    """
    if not torques:
        print("no control steps captured; skipping plot panels")
        return frames

    # Post-step times, matching when each sample in torques/targets/positions
    # was taken (see render_video's capture site: appended after `state =
    # step(state, act)` for control step i, i.e. at sim time (i + 1) * dt).
    signal_times = (np.arange(len(torques)) + 1) * env.dt
    torques = np.asarray(torques)
    targets = np.asarray(targets)
    positions = np.asarray(positions)
    torque_caps = np.asarray(env.mj_model.actuator_forcerange[:, 1])
    joint_groups = env.robot_spec.joint_groups
    width = frames[0].shape[1]

    panel_frame_ats = []
    if plot_torque:
        panel_frame_ats.append(
            torque_strip(signal_times, torques, torque_caps, joint_names, joint_groups, width=width)
        )
    if plot_joints:
        if joint is not None:
            j = joint_names.index(joint)
            panel_frame_ats.append(
                joint_zoom(signal_times, targets[:, j], positions[:, j], joint, width=width)
            )
        else:
            panel_frame_ats.append(
                joint_grid(signal_times, targets, positions, joint_names, joint_groups, width=width)
            )

    return [
        np.vstack([frame] + [frame_at(t) for frame_at in panel_frame_ats])
        for frame, t in zip(frames, frame_times)
    ]


def _rollout_frames(
    env, reset, step, inf, view, cmd_at, n_steps, *,
    seed, heading_gain, capture, overlay_torque, render_every,
) -> tuple[list, list, list, list, list]:
    """One rollout rendered through `view`.

    Returns (frames, frame_times, torques, targets, positions). The three
    capture lists stay empty unless `capture` is set. heading_gain 0 renders
    the policy open-loop.
    """
    rng = jax.random.PRNGKey(seed)
    state = reset(rng)

    # Plot data capture is opt-in: per-step np.asarray device transfers below
    # aren't free, so plain (no-flag) rendering must not pay them.
    qadr = np.asarray(env._qadr) if capture else None
    torques, targets, positions = [], [], []

    frames = []
    frame_times = []  # sim time of each entry in `frames`, recorded explicitly

    def overlay_force():
        # Per-rendered-frame device transfer, paid only when the overlay is on.
        return np.asarray(state.data.actuator_force) if overlay_torque else None

    # Always capture the initial pose: a scenario that falls at step 0
    # (e.g. an untrained checkpoint on `stand`) must still write a
    # nonzero-length video.
    frames.append(view.frame(state.data.qpos, torque=overlay_force()))
    frame_times.append(0.0)

    yaw_target = None
    for i in range(n_steps):
        cmd = cmd_at(i)
        if heading_gain:
            # Outer heading loop, as a deployment's joystick or planner
            # closes it: the policy observes yaw RATE only, so an unbidden
            # turn accumulates unseen; this adds wz = k * heading error
            # (clipped to 0.5 rad/s) on top of the scenario's command.
            # With heading_err observed, gait_state.update (run by
            # Joystick.step) re-anchors yaw_ref to the measured yaw while
            # |command wz| > progress.TURN_WZ_THRESHOLD (0.05 rad/s). So
            # above about 2 degrees of heading error at gain 1.5, a
            # heading-trained policy's heading_err observation reads near
            # zero.
            # Yaw comes from the IMU quaternion, the same source as the
            # env's yaw and heading_err. It lags qpos by one physics
            # substep (up to 1.5e-3 rad on roboto), so heading-closed
            # clips differ slightly from those rendered from qpos
            # before this change.
            yaw = float(signals.base_yaw(np.asarray(env._quat(state.data))))
            cmd, yaw_target = heading_correction(cmd, yaw, yaw_target, heading_gain, env.dt)
        state.info["command"] = cmd
        rng, act_rng = jax.random.split(rng)
        act, _ = inf(state.obs, act_rng)
        state = step(state, act)

        if capture:
            # This env has no action-delay, latency or action-filter
            # machinery (see envs/joystick.py's module docstring), and
            # step() passes
            # the clipped motor_targets straight to mjx_env.step
            # (joystick.py step(), ~line 290-308) -- so post-step ctrl
            # IS the policy's clipped PD target for this step. No
            # info["motor_targets"] plumbing is needed to recover it.
            torques.append(np.asarray(state.data.actuator_force))
            targets.append(np.asarray(state.data.ctrl))
            positions.append(np.asarray(state.data.qpos)[qadr])

        if bool(state.done):
            print(f"fell at step {i} (heading gain {heading_gain:g})")
            break
        if (i + 1) % render_every == 0:
            frames.append(view.frame(state.data.qpos, torque=overlay_force()))
            frame_times.append((i + 1) * env.dt)

    return frames, frame_times, torques, targets, positions


def _side_by_side(left, right) -> list[np.ndarray]:
    """Join two clips frame by frame, `left` on the left.

    The shorter clip repeats its own last frame up to the longer length.
    So the side that keeps walking stays visible after the other one falls.
    Raises ValueError when the two frame shapes differ.
    """
    if left[0].shape != right[0].shape:
        raise ValueError(
            f"frame shapes differ: left {left[0].shape}, right {right[0].shape}"
        )
    n = max(len(left), len(right))
    left = list(left) + [left[-1]] * (n - len(left))
    right = list(right) + [right[-1]] * (n - len(right))
    return [np.concatenate([a, b], axis=1) for a, b in zip(left, right)]


# --compare's heading gain when --heading-gain is not given.
COMPARE_HEADING_GAIN = 1.5


def render_video(
    run_dir: Path,
    scenario: str = "walk_ramp",
    steps: int | None = None,
    out: Path | None = None,
    seed: int = 0,
    video_size: tuple[int, int] = DEFAULT_SIZE,
    plot_torque: bool = False,
    plot_joints: bool = False,
    joint: str | None = None,
    push: bool = False,
    overlay_torque: bool = False,
    heading_gain: float = 0.0,
    checkpoint: int | None = None,
    compare: bool = False,
) -> Path:
    if compare and not heading_gain:
        raise ValueError("compare needs a nonzero heading_gain: both halves would be the same rollout")

    run, env, ckpt, inf = load_checkpoint_policy(run_dir, push_override(push), step=checkpoint)

    # Actuator column order == robot_spec.actuated_joints order (robot/build.py's
    # injection loop: "for joint_name in robot_spec.actuated_joints", the
    # "canonical order: the action/obs contract"). eval/plots.py's builders
    # assume this order for their joint_names argument.
    joint_names = list(env.robot_spec.actuated_joints)
    # `--joint NAME` alone is enough: one joint's zoom panel is a joint panel.
    plot_joints = plot_joints or joint is not None
    if joint is not None and joint not in joint_names:
        sys.exit(f"unknown joint {joint!r}; valid joints: {joint_names}")

    reset, step = jax.jit(env.reset), jax.jit(env.step)

    cmd_at, n_steps = pick_scenario(env.dt, env._config.command, scenario)
    if steps is not None:
        n_steps = steps

    camera = _pick_camera(env.mj_model, env.robot_spec.eval_camera)
    render_every = max(1, round(1 / (30 * env.dt)))
    fps = 1.0 / (env.dt * render_every)
    capture = plot_torque or plot_joints

    # Both compare halves share one env, one policy and one SceneView. They
    # differ only in the heading gain.
    gains = (0.0, heading_gain) if compare else (heading_gain,)
    clips = []
    with SceneView(env, size=video_size, camera=camera, torque=overlay_torque) as view:
        for gain in gains:
            frames, frame_times, torques, targets, positions = _rollout_frames(
                env, reset, step, inf, view, cmd_at, n_steps,
                seed=seed, heading_gain=gain, capture=capture,
                overlay_torque=overlay_torque, render_every=render_every,
            )
            if capture:
                frames = _composite_plots(
                    frames, frame_times, env, joint_names, joint, plot_torque, plot_joints,
                    torques, targets, positions,
                )
            clips.append(frames)

    if compare:
        left = [overlays.draw_label(f, "open loop") for f in clips[0]]
        right = [overlays.draw_label(f, f"heading loop, gain {heading_gain:g}") for f in clips[1]]
        frames = _side_by_side(left, right)
        out = out or (run_dir / f"{scenario}_compare.mp4")
    else:
        frames = clips[0]
        out = out or (run_dir / f"{scenario}.mp4")

    write_video(out, frames, fps)
    print(f"scenario {scenario}  run {run['run_name']}  checkpoint {ckpt.name}  {len(frames)} frames -> {out}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--scenario", default="walk_ramp")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--video-size", type=frame_size, default=DEFAULT_SIZE, metavar="WxH",
        help="rendered frame size (default 640x480); the stacked --plot panels follow the frame width",
    )
    ap.add_argument(
        "--overlay-torque", action="store_true",
        help="draw the per-actuator torque bars into the frame itself, normalized by each "
        "joint's own cap (an instantaneous view; --plot-torque is the full-episode strip)",
    )
    ap.add_argument("--plot-torque", action="store_true", help="append a normalized-torque strip below the render")
    ap.add_argument("--plot-joints", action="store_true", help="append a joint target-vs-state grid below that")
    ap.add_argument("--joint", default=None, help="single-joint zoom panel instead of the grid (implies --plot-joints)")
    ap.add_argument(
        "--push", action="store_true",
        help="keep the run's own random pushes for this rollout; the default "
        "is push-free, matching the battery's measurement convention (a "
        "mid-video kick reads as a policy failure)",
    )
    ap.add_argument(
        "--heading-gain", type=float, default=None,
        help="close an outer heading loop: add wz = gain * heading error (rad/s, clipped to 0.5) "
        "to the scenario command; 0 (default) renders the policy open-loop. "
        f"With --compare the default is {COMPARE_HEADING_GAIN:g}",
    )
    ap.add_argument(
        "--compare", action="store_true",
        help="render the open-loop rollout next to the heading-closed one, at the same seed, "
        f"scenario and checkpoint. The gain comes from --heading-gain, default {COMPARE_HEADING_GAIN:g}. "
        "The outer loop's wz re-anchors a trained yaw_ref. With heading_err observed, "
        "gait_state.update (run by Joystick.step) resets yaw_ref while |command wz| > 0.05 rad/s, "
        "so the right half's heading_err observation reads near zero",
    )
    ap.add_argument(
        "--checkpoint", type=int, default=None, metavar="STEP",
        help="render the checkpoint saved at this step instead of the latest",
    )
    args = ap.parse_args()

    if not args.compare:
        gain = args.heading_gain or 0.0
    elif args.heading_gain is None:
        gain = COMPARE_HEADING_GAIN
    elif args.heading_gain == 0:
        ap.error("--compare needs a nonzero --heading-gain: both halves would be the same rollout")
    else:
        gain = args.heading_gain

    render_video(
        args.run, args.scenario, args.steps, args.out, args.seed,
        video_size=args.video_size,
        plot_torque=args.plot_torque, plot_joints=args.plot_joints, joint=args.joint,
        push=args.push, overlay_torque=args.overlay_torque, heading_gain=gain,
        checkpoint=args.checkpoint, compare=args.compare,
    )


if __name__ == "__main__":
    main()
