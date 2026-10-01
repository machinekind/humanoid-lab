"""CLI: check that a dr.foot_friction draw is the friction the feet walk on.

    ./run.sh check-friction --robot roboto_origin --preset deploy_pd [--task terrain]

MuJoCo combines the friction of two equal-priority geoms with an
element-wise max, so a foot draw below the floor's value would never reach
the contact unless the feet carry contact priority. dr/randomize.py grants
that priority when foot_friction is enabled; this probe measures the result
end to end on whichever backend the box has: it builds the randomized
models, settles the robot onto the floor, and compares the friction inside
each foot-floor contact against that env's drawn value. The same equality
also proves the floor's own friction draw no longer leaks into foot
contacts. Run it on a GPU host to get the warp answer, which is the one a
training run uses.

`--task terrain` runs the same check on a terrain scene. The floor plane
is replaced by `CPU_ARENA` (terrain/config.py), and world i stands on the
flat row's pad i mod 8. A foot contact counts when its other geom is any
ground geom: the heightfield, an arena box or an apron. The flat row is
exactly 0, so each pad start is the keyframe pose moved in x and y.
"""

import argparse
import sys

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from mujoco import mjx

from humanoid_lab import paths
from humanoid_lab.dr.randomize import _find_floor_geom_id, make_domain_randomize
from humanoid_lab.envs.joystick import Joystick
from humanoid_lab.envs.joystick import default_config as joystick_default_config
from humanoid_lab.registry import _apply_overrides, make_env
from humanoid_lab.robot.build import compile_spec
from humanoid_lab.terrain.config import CPU_ARENA, arena_for, params_from_config
from humanoid_lab.terrain.scene import attach_terrain, ground_geom_ids

# Sim steps (not control steps) before the contacts are read: the home
# keyframes start the soles a few millimetres above the floor, so the feet
# need a moment to land and load.
SETTLE_STEPS = 20

TASKS = ("joystick", "terrain")


class _ArenaJoystick(Joystick):
    """Joystick with its floor plane replaced by `CPU_ARENA`.

    The real_pose_ref settle still runs on the robot's own plane."""

    def _customize_spec(self, spec):
        self._flat_model = compile_spec(spec)
        self.arena = arena_for(params_from_config(CPU_ARENA))
        stat = (self._flat_model.stat.extent, self._flat_model.stat.center)
        attach_terrain(spec, self.arena, stat=stat)

    def _settle_model(self):
        return self._flat_model


def _build_env(task: str, robot: str, preset: str, sim: dict):
    if task == "joystick":
        return make_env("joystick", paths.ROBOTS_DIR / robot, preset, env_overrides={"sim": sim})
    cfg = joystick_default_config()
    _apply_overrides(cfg, {"sim": sim})
    return _ArenaJoystick(paths.ROBOTS_DIR / robot, preset, cfg)


def _start_qpos(env, task: str, key_qpos, num_envs: int):
    """(num_envs, nq) start poses: the keyframe, on a flat-row pad for terrain."""
    starts = np.tile(np.asarray(key_qpos), (num_envs, 1))
    if task == "terrain":
        pads = sorted((t for t in env.arena.spec.tiles if t.row == 0), key=lambda t: t.col)
        b = env._base_qadr
        for i in range(num_envs):
            starts[i, b : b + 2] = pads[i % len(pads)].origin[:2]
    return jp.array(starts)


def probe(
    robot: str, preset: str, backend: str, num_envs: int, friction_range, task: str = "joystick"
) -> bool:
    if task not in TASKS:
        raise ValueError(f"task must be one of {TASKS}, got {task!r}")
    env = _build_env(task, robot, preset, {"backend": backend, "num_envs": num_envs})
    m = env.mj_model
    print(f"robot {robot} / preset {preset} / task {task} / backend {env._backend} / envs {num_envs}")

    randomize = make_domain_randomize(
        m,
        env.robot_spec,
        {"foot_friction": {"enable": True, "range": list(friction_range)}},
    )
    keys = jax.random.split(jax.random.PRNGKey(0), num_envs)
    model_v, in_axes = randomize(env.mjx_model, keys)

    foot_ids = [m.geom(name).id for name in env.robot_spec.foot_geoms]
    if task == "joystick":
        ground = {_find_floor_geom_id(m, None)}
    else:
        ground = {int(g) for g in ground_geom_ids(m)}
    priority = np.asarray(model_v.geom_priority)
    if not np.all(priority[foot_ids] == 1):
        print(f"FAIL: foot geom_priority is {priority[foot_ids]}, expected 1")
        return False

    # Held at the reset keyframe's pose: qpos from the keyframe, PD targets
    # at the keyframe's own joint angles (actuator order == actuated_joints
    # order, the action/obs contract).
    key_qpos = jp.array(m.key(env._config.reset_keyframe).qpos)
    ctrl = key_qpos[np.asarray(env._qadr)]
    starts = _start_qpos(env, task, key_qpos, num_envs)

    def run(model, qpos):
        data = env._make_data().replace(qpos=qpos, ctrl=ctrl)
        data = mjx.forward(model, data)

        def body(d, _):
            return mjx.step(model, d), None

        return jax.lax.scan(body, data, None, length=SETTLE_STEPS)[0]

    out = jax.jit(jax.vmap(run, in_axes=(in_axes, 0)))(model_v, starts)
    # Normalize both impls to flat contact rows with an env column. jax
    # keeps a per-env contact struct; DataWarp keeps one global buffer with
    # contact__worldid mapping each slot to its env.
    impl = getattr(out, "_impl", out)
    if hasattr(impl, "contact"):
        c = impl.contact
        n_env, n_con = np.asarray(c.dist).shape
        env_of = np.repeat(np.arange(n_env), n_con)
        geom = np.asarray(c.geom).reshape(-1, 2)
        dist = np.asarray(c.dist).reshape(-1)
        mu = np.asarray(c.friction).reshape(n_env * n_con, -1)[:, 0]
    else:
        env_of = np.asarray(impl.contact__worldid).reshape(-1)
        geom = np.asarray(impl.contact__geom).reshape(-1, 2)
        dist = np.asarray(impl.contact__dist).reshape(-1)
        mu = np.asarray(impl.contact__friction).reshape(len(dist), -1)[:, 0]

    names = list(env.robot_spec.foot_geoms)
    base = np.asarray(env.mjx_model.geom_friction)[foot_ids, 0]
    sampled = np.asarray(model_v.geom_friction)[:, foot_ids, 0]
    # Which foot geoms each env actually has on the ground, and the
    # solver-side friction of each such contact. Not every foot geom
    # touches (toe segments lift), so only the observed ones are compared.
    seen_by_env: list[dict[int, tuple[float, int]]] = [{} for _ in range(num_envs)]
    for k in range(len(dist)):
        g1, g2 = int(geom[k, 0]), int(geom[k, 1])
        if dist[k] >= 0 or not ({g1, g2} & ground):
            continue
        floor, other = (g1, g2) if g1 in ground else (g2, g1)
        if other in foot_ids:
            seen_by_env[int(env_of[k])][foot_ids.index(other)] = (float(mu[k]), floor)
    ok = True
    for e in range(num_envs):
        seen = seen_by_env[e]
        if not seen:
            print(f"env {e}: FAIL, no foot-floor contact after {SETTLE_STEPS} settle steps")
            ok = False
            continue
        for i in sorted(seen):
            value, floor = seen[i]
            drawn = sampled[e, i]
            bad = abs(value - drawn) > 1e-5
            ok = ok and not bad
            on = "" if task == "joystick" else f" on {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, floor)}"
            print(
                f"env {e} {names[i]:<24s} x{drawn / base[i]:5.3f} -> drawn {drawn:6.4f} "
                f"contact {value:6.4f}{on}" + ("   MISMATCH" if bad else "")
            )
    lo, hi = friction_range
    if not np.all((sampled >= base * lo - 1e-6) & (sampled <= base * hi + 1e-6)):
        print(f"FAIL: draws outside the multiplier range {lo}..{hi}")
        ok = False
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--robot", required=True, help="robot dir name under robots/")
    ap.add_argument("--preset", required=True, help="actuator preset name")
    ap.add_argument("--task", choices=TASKS, default="joystick", help="flat floor or the CPU terrain arena")
    ap.add_argument("--backend", choices=["auto", "warp", "jax"], default="auto")
    ap.add_argument("--num-envs", type=int, default=8)
    ap.add_argument(
        "--range",
        type=float,
        nargs=2,
        default=[0.8, 1.2],
        metavar=("LO", "HI"),
        help="dr.foot_friction multiplier range (default: configs/dr/default.yaml's)",
    )
    args = ap.parse_args()
    ok = probe(args.robot, args.preset, args.backend, args.num_envs, args.range, task=args.task)
    print("PROBE PASS" if ok else "PROBE FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
