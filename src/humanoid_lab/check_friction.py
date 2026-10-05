"""CLI: check that a dr.foot_friction draw is the friction the feet walk on.

    ./run.sh check-friction --robot roboto_origin --preset deploy_pd

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
"""

import argparse
import sys

import jax
import jax.numpy as jp
import numpy as np
from mujoco import mjx

from humanoid_lab import paths
from humanoid_lab.dr.randomize import _find_floor_geom_id, make_domain_randomize
from humanoid_lab.registry import make_env

# Sim steps (not control steps) before the contacts are read: the home
# keyframes start the soles a few millimetres above the floor, so the feet
# need a moment to land and load.
SETTLE_STEPS = 20


def probe(robot: str, preset: str, backend: str, num_envs: int, friction_range) -> bool:
    env = make_env(
        "joystick",
        paths.ROBOTS_DIR / robot,
        preset,
        env_overrides={"sim": {"backend": backend, "num_envs": num_envs}},
    )
    m = env.mj_model
    print(f"robot {robot} / preset {preset} / backend {env._backend} / envs {num_envs}")

    randomize = make_domain_randomize(
        m,
        env.robot_spec,
        {"foot_friction": {"enable": True, "range": list(friction_range)}},
    )
    keys = jax.random.split(jax.random.PRNGKey(0), num_envs)
    model_v, in_axes = randomize(env.mjx_model, keys)

    foot_ids = [m.geom(name).id for name in env.robot_spec.foot_geoms]
    floor_id = _find_floor_geom_id(m, None)
    priority = np.asarray(model_v.geom_priority)
    if not np.all(priority[foot_ids] == 1):
        print(f"FAIL: foot geom_priority is {priority[foot_ids]}, expected 1")
        return False

    # Held at the reset keyframe's pose: qpos from the keyframe, PD targets
    # at the keyframe's own joint angles (actuator order == actuated_joints
    # order, the action/obs contract).
    key_qpos = jp.array(m.key(env._config.reset_keyframe).qpos)
    ctrl = key_qpos[np.asarray(env._qadr)]

    def run(model):
        data = env._make_data().replace(qpos=key_qpos, ctrl=ctrl)
        data = mjx.forward(model, data)

        def body(d, _):
            return mjx.step(model, d), None

        return jax.lax.scan(body, data, None, length=SETTLE_STEPS)[0]

    out = jax.jit(jax.vmap(run, in_axes=(in_axes,)))(model_v)
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
    # Which foot geoms each env actually has on the floor, and the
    # solver-side friction of each such contact. Not every foot geom
    # touches (toe segments lift), so only the observed ones are compared.
    seen_by_env: list[dict[int, float]] = [{} for _ in range(num_envs)]
    for k in range(len(dist)):
        g1, g2 = int(geom[k, 0]), int(geom[k, 1])
        if dist[k] >= 0 or floor_id not in (g1, g2):
            continue
        other = g2 if g1 == floor_id else g1
        if other in foot_ids:
            seen_by_env[int(env_of[k])][foot_ids.index(other)] = float(mu[k])
    ok = True
    for e in range(num_envs):
        seen = seen_by_env[e]
        if not seen:
            print(f"env {e}: FAIL, no foot-floor contact after {SETTLE_STEPS} settle steps")
            ok = False
            continue
        for i in sorted(seen):
            drawn = sampled[e, i]
            bad = abs(seen[i] - drawn) > 1e-5
            ok = ok and not bad
            print(
                f"env {e} {names[i]:<24s} x{drawn / base[i]:5.3f} -> drawn {drawn:6.4f} "
                f"contact {seen[i]:6.4f}" + ("   MISMATCH" if bad else "")
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
    ok = probe(args.robot, args.preset, args.backend, args.num_envs, args.range)
    print("PROBE PASS" if ok else "PROBE FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
