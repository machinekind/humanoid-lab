"""Roll out a checkpoint under a fixed (or scripted battery) command and
render one scenario to MP4.

Run:
    python -m humanoid_lab.eval.video --run runs/<name> [--scenario walk_ramp] [--steps N]

or via run.sh:
    ./run.sh eval --run runs/<name> --scenario stand --steps 100

`--scenario` (default "walk_ramp") picks one of eval/battery.py's scripted
command trajectories (battery_scenarios); `--steps` truncates it to N
control steps -- both use the exact same command builders the battery
measures, so a rendered video and its battery.json row describe the same
trajectory.

`--plot-torque` appends a normalized-torque strip below the render;
`--plot-joints` appends a joint target-vs-state grid below that; `--joint
NAME` swaps the grid for a single-joint zoom panel and implies
--plot-joints. All three are opt-in (off by default -- plain video output
is the same code path as before); panel rendering lives in eval/plots.py.

`--push` restores the run's own random pushes for the rollout. They are OFF
by default, matching the battery's measurement convention: a mid-video kick
reads as a policy failure to anyone watching the clip. The default is stated
here (see env_overrides_for), not merely inherited from the battery's
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
import shutil
from pathlib import Path

import jax
import mujoco
import numpy as np

from humanoid_lab.eval.battery import (
    battery_scenarios,
    load_checkpoint_policy,
    merged_env_overrides,
)
from humanoid_lab.eval.plots import joint_grid, joint_zoom, torque_strip


def push_override(push: bool) -> dict:
    """This module's own push decision, as an env override block.

    Written explicitly in both directions rather than relying on the
    battery's measurement env happening to disable pushes: what a rendered
    clip shows is this module's contract with whoever watches it. `--push`
    sets `enable` only, so the run's own `interval_steps` and `vel` are
    whatever it trained with.
    """
    return {"push": {"enable": bool(push)}}


def env_overrides_for(run: dict, push: bool = False) -> dict:
    """The env overrides a video rollout builds its env with: the battery's
    measurement set, with this module's push decision merged over it."""
    return merged_env_overrides(run, push_override(push))


def resolve_panels(plot_torque: bool, plot_joints: bool, joint: str | None) -> tuple[bool, bool]:
    """(plot_torque, plot_joints) after `--joint`'s implication.

    Asking for one joint's zoom panel is asking for a joint panel, so
    `--joint NAME` alone is enough -- and the implication belongs here rather
    than in main(), so a library caller passing joint= gets it too.
    """
    return bool(plot_torque), bool(plot_joints) or joint is not None


def _pick_camera(mj_model: mujoco.MjModel):
    """Prefer a named MJCF camera already in the model over synthesizing
    one. source/xmls/asimov.xml ships several `mode="track"` cameras
    mounted on the pelvis (front_camera, side_camera, ...) -- use the
    first of a small side-view preference list that exists in this model.
    If none exist (a future robot's XML has no cameras), fall back to a
    free MjvCamera tracking the floating-base body instead of editing the
    XML (out of scope for this module -- generated XML is build output,
    see PLAN.md's ops rules)."""
    for name in ("side_camera", "front_camera", "track"):
        try:
            mj_model.camera(name)
            return name
        except KeyError:
            continue

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


def render_video(
    run_dir: Path,
    scenario: str = "walk_ramp",
    steps: int | None = None,
    out: Path | None = None,
    seed: int = 0,
    width: int = 640,
    height: int = 480,
    plot_torque: bool = False,
    plot_joints: bool = False,
    joint: str | None = None,
    push: bool = False,
) -> Path:
    run, env, _ckpt, inf = load_checkpoint_policy(run_dir, push_override(push))

    # Actuator column order == robot_spec.actuated_joints order (robot/build.py's
    # injection loop: "for joint_name in robot_spec.actuated_joints", the
    # "canonical order: the action/obs contract"). eval/plots.py's builders
    # assume this order for their joint_names argument.
    joint_names = list(env.robot_spec.actuated_joints)
    plot_torque, plot_joints = resolve_panels(plot_torque, plot_joints, joint)
    if joint is not None and joint not in joint_names:
        sys.exit(f"unknown joint {joint!r}; valid joints: {joint_names}")

    reset, step = jax.jit(env.reset), jax.jit(env.step)

    scenarios = battery_scenarios(env.dt)
    if scenario not in scenarios:
        raise KeyError(f"unknown scenario {scenario!r}; have {sorted(scenarios)}")
    cmd_at, n_steps = scenarios[scenario]
    if steps is not None:
        n_steps = steps

    camera = _pick_camera(env.mj_model)
    render_every = max(1, round(1 / (30 * env.dt)))
    fps = 1.0 / (env.dt * render_every)

    rng = jax.random.PRNGKey(seed)
    state = reset(rng)

    # Plot data capture is opt-in: per-step np.asarray device transfers below
    # aren't free, so plain (no-flag) rendering must not pay them.
    capture = plot_torque or plot_joints
    qadr = np.asarray(env._qadr) if capture else None
    torques, targets, positions = [], [], []

    mj_model = env.mj_model
    data = mujoco.MjData(mj_model)
    frames = []
    frame_times = []  # sim time of each entry in `frames`, recorded explicitly
    with mujoco.Renderer(mj_model, height=height, width=width) as renderer:
        # Always capture the initial pose: a scenario that falls at step 0
        # (e.g. an untrained checkpoint on `stand`) must still write a
        # nonzero-length video.
        data.qpos[:] = np.asarray(state.data.qpos)
        mujoco.mj_forward(mj_model, data)
        renderer.update_scene(data, camera=camera)
        frames.append(renderer.render())
        frame_times.append(0.0)

        for i in range(n_steps):
            cmd = cmd_at(i)
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
                print(f"fell at step {i}")
                break
            if (i + 1) % render_every == 0:
                data.qpos[:] = np.asarray(state.data.qpos)
                mujoco.mj_forward(mj_model, data)
                renderer.update_scene(data, camera=camera)
                frames.append(renderer.render())
                frame_times.append((i + 1) * env.dt)

    if plot_torque or plot_joints:
        frames = _composite_plots(
            frames, frame_times, env, joint_names, joint, plot_torque, plot_joints,
            torques, targets, positions,
        )

    out = out or (run_dir / f"{scenario}.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)

    import mediapy

    if shutil.which("ffmpeg") is None:
        # No system ffmpeg (common on a bare Mac); fall back to the binary
        # bundled with imageio-ffmpeg, already a project dependency.
        import imageio_ffmpeg

        mediapy.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())

    mediapy.write_video(str(out), frames, fps=fps)
    print(f"scenario {scenario}  run {run['run_name']}  {len(frames)} frames -> {out}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--scenario", default="walk_ramp")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plot-torque", action="store_true", help="append a normalized-torque strip below the render")
    ap.add_argument("--plot-joints", action="store_true", help="append a joint target-vs-state grid below that")
    ap.add_argument("--joint", default=None, help="single-joint zoom panel instead of the grid (implies --plot-joints)")
    ap.add_argument(
        "--push", action="store_true",
        help="keep the run's own random pushes for this rollout; the default "
        "is push-free, matching the battery's measurement convention (a "
        "mid-video kick reads as a policy failure)",
    )
    args = ap.parse_args()

    render_video(
        args.run, args.scenario, args.steps, args.out, args.seed,
        plot_torque=args.plot_torque, plot_joints=args.plot_joints, joint=args.joint,
        push=args.push,
    )


if __name__ == "__main__":
    main()
