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

MUJOCO_GL: mirrors w01-tek wojtek_rl/eval.py's own handling exactly --
`egl` is set (via setdefault, so an already-exported MUJOCO_GL wins) only
on linux, where headless GPU boxes need it; darwin is left on its default
(CGL), which is what this repo's Mac dev box actually has. Forcing egl on
darwin breaks offscreen rendering there, so this must run before `mujoco`
is imported anywhere in the process.
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

from humanoid_lab.eval.battery import battery_scenarios, load_checkpoint_policy


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


def render_video(
    run_dir: Path,
    scenario: str = "walk_ramp",
    steps: int | None = None,
    out: Path | None = None,
    seed: int = 0,
    width: int = 640,
    height: int = 480,
) -> Path:
    run, env, _ckpt, inf = load_checkpoint_policy(run_dir)
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

    mj_model = env.mj_model
    data = mujoco.MjData(mj_model)
    frames = []
    with mujoco.Renderer(mj_model, height=height, width=width) as renderer:
        # Always capture the initial pose: a scenario that falls at step 0
        # (e.g. an untrained checkpoint on `stand`) must still write a
        # nonzero-length video.
        data.qpos[:] = np.asarray(state.data.qpos)
        mujoco.mj_forward(mj_model, data)
        renderer.update_scene(data, camera=camera)
        frames.append(renderer.render())

        for i in range(n_steps):
            cmd = cmd_at(i)
            state.info["command"] = cmd
            rng, act_rng = jax.random.split(rng)
            act, _ = inf(state.obs, act_rng)
            state = step(state, act)

            if bool(state.done):
                print(f"fell at step {i}")
                break
            if (i + 1) % render_every == 0:
                data.qpos[:] = np.asarray(state.data.qpos)
                mujoco.mj_forward(mj_model, data)
                renderer.update_scene(data, camera=camera)
                frames.append(renderer.render())

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
    args = ap.parse_args()

    render_video(args.run, args.scenario, args.steps, args.out, args.seed)


if __name__ == "__main__":
    main()
