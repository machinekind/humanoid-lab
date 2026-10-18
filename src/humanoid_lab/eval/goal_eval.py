"""Evaluate and render a task=goal checkpoint.

Run:
    python -m humanoid_lab.eval.goal_eval --run runs/<name> [--out DIR]

Two parts, both deterministic (deterministic policy, fixed seeds):

1. A goal battery: for each distance in --dists and each of 8 directions,
   reset, place one goal at that offset and roll until it is reached, the
   robot falls or the time budget runs out. Reports the reach rate, the
   fall rate and the mean time to reach per distance.
2. A tour render: one episode visiting a fixed list of waypoints (a 2 m
   square by default), rendered with the goal drawn as a green post. Writes
   a contact sheet (sheet.jpg), a top-down trajectory plot (path.png), an
   mp4 and metrics.json into --out.

The curriculum is switched off and the episode never truncates here, so the
goal sequence is exactly the one scripted.
"""

import os
import sys

if sys.platform == "linux":
    os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jp
import mujoco
import numpy as np

from humanoid_lab.eval.battery import load_checkpoint_policy

TOUR = [(2.0, 0.0), (2.0, 2.0), (0.0, 2.0), (0.0, 0.0)]


def _set_goal(env, state, goal_xy):
    info = dict(state.info)
    base = np.asarray(state.data.qpos[env._base_qadr : env._base_qadr + 2])
    d = float(np.linalg.norm(np.asarray(goal_xy) - base))
    g = env._config.goal
    info["goal"] = jp.array(goal_xy, dtype=jp.float32)
    info["goal_budget"] = jp.array(int(round((g.time_base + d / g.time_speed) / env.dt)), dtype=jp.int32)
    info["goal_steps"] = jp.array(0, dtype=jp.int32)
    info["goal_dwell"] = jp.array(0, dtype=jp.int32)
    info["goal_dist_prev"] = jp.array(d, dtype=jp.float32)
    state = state.replace(info=info)
    # Rebuild the obs so the first action already sees the new goal.
    obs = env._build_obs(state.data, info)
    return state.replace(obs=obs)


def battery(env, reset, step, inf, dists, n_dirs=8, seed=0):
    rows = []
    for d in dists:
        reached, fell, times = 0, 0, []
        for k in range(n_dirs):
            a = 2 * np.pi * k / n_dirs
            state = reset(jax.random.PRNGKey(seed + k))
            state = _set_goal(env, state, (d * np.cos(a), d * np.sin(a)))
            budget = int(state.info["goal_budget"])
            rng = jax.random.PRNGKey(1000 + k)
            for t in range(budget):
                rng, r = jax.random.split(rng)
                act, _ = inf(state.obs, r)
                state = step(state, act)
                if float(state.metrics["goal/reached"]) > 0:
                    reached += 1
                    times.append((t + 1) * env.dt)
                    break
                if float(state.done) > 0:
                    fell += 1
                    break
        rows.append(
            {
                "dist": d,
                "reach_rate": reached / n_dirs,
                "fall_rate": fell / n_dirs,
                "mean_time_s": float(np.mean(times)) if times else None,
            }
        )
    return rows


def _camera(model):
    free = [i for i in range(model.njnt) if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE]
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = int(model.jnt_bodyid[free[0]])
    cam.distance = 3.5
    cam.azimuth = 135.0
    cam.elevation = -25.0
    return cam


def _add_goal_marker(scene, goal_xy, radius):
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        g,
        mujoco.mjtGeom.mjGEOM_CYLINDER,
        np.array([radius, radius, 0.005]),
        np.array([goal_xy[0], goal_xy[1], 0.005]),
        np.eye(3).flatten(),
        np.array([0.1, 0.9, 0.2, 0.6], dtype=np.float32),
    )
    scene.ngeom += 1
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        g,
        mujoco.mjtGeom.mjGEOM_CYLINDER,
        np.array([0.02, 0.02, 0.5]),
        np.array([goal_xy[0], goal_xy[1], 0.5]),
        np.eye(3).flatten(),
        np.array([0.1, 0.9, 0.2, 1.0], dtype=np.float32),
    )
    scene.ngeom += 1


def tour(env, reset, step, inf, waypoints, out: Path, size=(480, 360), max_s=40.0):
    model = env.mj_model
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, size[0])
    model.vis.global_.offheight = max(model.vis.global_.offheight, size[1])
    renderer = mujoco.Renderer(model, height=size[1], width=size[0])
    data = mujoco.MjData(model)
    cam = _camera(model)

    state = reset(jax.random.PRNGKey(0))
    wp = 0
    state = _set_goal(env, state, waypoints[wp])
    rng = jax.random.PRNGKey(7)
    render_every = max(1, round(1 / (30 * env.dt)))
    frames, path, events = [], [], []
    n = int(max_s / env.dt)
    fell_at = None
    for t in range(n):
        rng, r = jax.random.split(rng)
        act, _ = inf(state.obs, r)
        state = step(state, act)
        qpos = np.asarray(state.data.qpos)
        path.append(qpos[:2].copy())
        if t % render_every == 0:
            data.qpos[:] = qpos
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=cam)
            _add_goal_marker(renderer.scene, waypoints[wp], env._config.goal.radius)
            frames.append(renderer.render().copy())
        if float(state.metrics["goal/reached"]) > 0:
            events.append({"t": round((t + 1) * env.dt, 2), "event": "reached", "waypoint": wp})
            wp += 1
            if wp >= len(waypoints):
                break
            state = _set_goal(env, state, waypoints[wp])
        elif float(state.metrics["goal/timeout"]) > 0:
            events.append({"t": round((t + 1) * env.dt, 2), "event": "timeout", "waypoint": wp})
            wp += 1
            if wp >= len(waypoints):
                break
            state = _set_goal(env, state, waypoints[wp])
        if float(state.done) > 0:
            fell_at = round((t + 1) * env.dt, 2)
            break
    renderer.close()

    path = np.array(path)
    out.mkdir(parents=True, exist_ok=True)
    _contact_sheet(frames, out / "sheet.jpg")
    _path_plot(path, waypoints, events, out / "path.png")
    try:
        from humanoid_lab.eval.writer import write_video

        write_video(out / "tour.mp4", frames, fps=1.0 / (env.dt * render_every))
    except Exception as e:  # noqa: BLE001
        print(f"mp4 skipped: {e}")
    dt_total = len(path) * env.dt
    travelled = float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))) if len(path) > 1 else 0.0
    return {
        "events": events,
        "fell_at": fell_at,
        "duration_s": round(dt_total, 2),
        "path_len_m": round(travelled, 2),
        "mean_speed": round(travelled / max(dt_total, 1e-6), 3),
    }


def _contact_sheet(frames, path, n=12, cols=4):
    from PIL import Image

    if not frames:
        return
    idx = np.linspace(0, len(frames) - 1, min(n, len(frames))).astype(int)
    tiles = [frames[i] for i in idx]
    h, w = tiles[0].shape[:2]
    rows = (len(tiles) + cols - 1) // cols
    sheet = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for k, tile in enumerate(tiles):
        r, c = divmod(k, cols)
        sheet[r * h : (r + 1) * h, c * w : (c + 1) * w] = tile
    Image.fromarray(sheet).save(path, quality=80)


def _path_plot(path, waypoints, events, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
    if len(path):
        ax.plot(path[:, 0], path[:, 1], lw=1.5)
        ax.plot(path[0, 0], path[0, 1], "ko", ms=4)
    for i, (x, y) in enumerate(waypoints):
        ax.add_patch(plt.Circle((x, y), 0.3, color="g", alpha=0.3))
        ax.text(x, y, str(i), ha="center", va="center")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.set_title(" ".join(f"{e['waypoint']}:{e['event'][0]}@{e['t']}" for e in events)[:60], fontsize=7)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, type=Path)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--dists", type=float, nargs="+", default=[0.5, 1.0, 2.0, 3.0])
    p.add_argument("--no-render", action="store_true")
    args = p.parse_args()

    overrides = {
        "goal": {"curriculum": {"enable": False, "up": 0.0, "down": 0.0, "init": 0.0}},
        "episode_length": 100000,
    }
    _run, env, ckpt, inf = load_checkpoint_policy(args.run, overrides)
    reset, step = jax.jit(env.reset), jax.jit(env.step)
    out = args.out or (args.run / "goal_eval")
    result = {"checkpoint": str(ckpt), "battery": battery(env, reset, step, inf, args.dists)}
    if not args.no_render:
        result["tour"] = tour(env, reset, step, inf, TOUR, out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
