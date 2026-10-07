"""Gait measurements of a run's latest checkpoint in the original JAX env
(CPU, pushes off): body pitch while walking forward, touchdown foot speed,
and the per-step values of the smoothness terms.

    python wtrain/measure.py --run runs/loco_warp_v3 --vx 0.6
"""
import argparse
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jp
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--vx", type=float, nargs="+", default=[0.3, 0.6, 0.9])
    ap.add_argument("--steps", type=int, default=300)
    args = ap.parse_args()

    from pathlib import Path

    from humanoid_lab.eval.battery import load_checkpoint_policy

    run, env, ckpt, inf = load_checkpoint_policy(Path(args.run))
    reset, step = jax.jit(env.reset), jax.jit(env.step)
    knee_idx = np.asarray(env._foot_ordered_group_qidx("knee"))
    print(f"checkpoint {ckpt}")
    for vx in args.vx:
        st = reset(jax.random.PRNGKey(0))
        key = jax.random.PRNGKey(1)
        pitch, td_vz, land, arate, aacc, fell = [], [], [], [], [], None
        prev_contact = np.zeros(2, bool)
        prev_fvz = np.zeros(2)
        lands = {0: [], 1: []}  # per foot: (x_rel, |y_rel|, apex, air_time) at touchdown
        n_contact = []  # feet in contact per step, after the settle window
        swing_x = {0: [], 1: []}  # per swing: (min, max) foot x relative to the pelvis, base frame
        cur_x = {0: [], 1: []}
        knee_max = {0: [], 1: []}
        cur_k = {0: 0.0, 1: 0.0}
        x_start = None
        roll, gyr, ys = [], [], []
        apex = np.zeros(2)
        air = np.zeros(2)
        for i in range(args.steps):
            st.info["command"] = jp.array([vx, 0.0, 0.0])
            key, k = jax.random.split(key)
            a, _ = inf(st.obs, k)
            prev_a, prev_prev_a = np.asarray(st.info["last_action"]), np.asarray(st.info["last_last_action"])
            st = step(st, a)
            if bool(st.done):
                fell = i
                break
            if i < 50:  # settle window
                prev_contact = np.asarray(env._foot_contact(st.data))
                continue
            g = np.asarray(env._gravity_body(st.data))
            roll.append(np.degrees(np.arcsin(np.clip(g[1], -1, 1))))
            gyr.append(np.asarray(env._gyro(st.data)))
            ys.append(float(st.data.qpos[1]))
            pitch.append(np.degrees(np.arcsin(np.clip(g[0], -1, 1))))
            contact = np.asarray(env._foot_contact(st.data))
            n_contact.append(int(contact.sum()))
            q4 = np.asarray(st.data.qpos[3:7])
            yaw_now = np.degrees(np.arctan2(2 * (q4[0] * q4[3] + q4[1] * q4[2]), 1 - 2 * (q4[2] ** 2 + q4[3] ** 2)))
            if x_start is None:
                x_start, t_start = float(st.data.qpos[0]), i
                y_start, yaw_start = float(st.data.qpos[1]), yaw_now
            x_now, t_now = float(st.data.qpos[0]), i
            y_now = float(st.data.qpos[1])
            fvz = np.asarray(env._foot_linvel(st.data))[:, 2]
            clear = np.asarray(env._foot_clearance(st.data))
            quat = np.asarray(env._quat(st.data))
            base = np.asarray(st.data.qpos[:3])
            for f in range(2):
                rel_f = np.asarray(st.data.site_xpos[env._foot_site_ids[f]]) - base
                wq, vq = quat[0], quat[1:]
                tq = 2 * np.cross(vq, rel_f)
                xf = float((rel_f - wq * tq + np.cross(vq, tq))[0])
                kq_now = float(np.asarray(st.data.qpos)[np.asarray(env._qadr)[knee_idx[f]]])
                if not contact[f]:
                    cur_x[f].append(xf)
                    cur_k[f] = max(cur_k[f], kq_now)
                elif cur_x[f]:
                    swing_x[f].append((min(cur_x[f]), max(cur_x[f])))
                    knee_max[f].append(cur_k[f])
                    cur_x[f], cur_k[f] = [], 0.0
                if contact[f] and not prev_contact[f]:
                    td_vz.append(-prev_fvz[f])
                    rel = np.asarray(st.data.site_xpos[env._foot_site_ids[f]]) - base
                    w, v = quat[0], quat[1:]
                    t = 2 * np.cross(v, rel)
                    rel_b = rel - w * t + np.cross(v, t)  # inverse rotation into the base frame
                    rel_o = np.asarray(st.data.site_xpos[env._foot_site_ids[1 - f]]) - base
                    t2 = 2 * np.cross(v, rel_o)
                    rel_ob = rel_o - w * t2 + np.cross(v, t2)
                    kq = float(np.asarray(st.data.qpos)[np.asarray(env._qadr)[knee_idx[f]]])
                    lands[f].append((rel_b[0], abs(rel_b[1]), apex[f], air[f], rel_b[0] - rel_ob[0], kq))
                    apex[f], air[f] = 0.0, 0.0
                elif not contact[f]:
                    apex[f] = max(apex[f], clear[f])
                    air[f] += env.dt
            prev_contact, prev_fvz = contact, fvz
            land.append(float(np.sum(np.square(np.minimum(fvz, 0.0))
                                     * np.clip(1.0 - clear / env._config.reward.glide_height, 0.0, 1.0))))
            an = np.asarray(a)
            arate.append(float(np.sum(np.square(an - prev_a))))
            aacc.append(float(np.sum(np.square(an - 2 * prev_a + prev_prev_a))))
        print(f"vx {vx:.2f}: pitch fwd {np.mean(pitch):+5.2f} deg (std {np.std(pitch):.2f})  "
              f"touchdown vz med {np.median(td_vz) if td_vz else float('nan'):.3f} m/s (n={len(td_vz)})  "
              f"feet_landing/step {np.mean(land):.4f}  action_rate/step {np.mean(arate):.3f}  "
              f"action_accel/step {np.mean(aacc):.3f}" + (f"  FELL at {fell}" if fell is not None else ""))
        nc = np.array(n_contact)
        G = np.array(gyr); t = np.arange(len(ys)); yy = np.array(ys)
        yy = yy - np.polyval(np.polyfit(t, yy, 1), t)
        print(f"    STABILITY roll std {np.std(roll):.2f} deg  roll rate rms {np.degrees(np.sqrt(np.mean(G[:, 0] ** 2))):.1f} deg/s  "
              f"pitch rate rms {np.degrees(np.sqrt(np.mean(G[:, 1] ** 2))):.1f} deg/s  yaw rate std {np.degrees(np.std(G[:, 2])):.1f} deg/s  "
              f"pelvis lateral sway std {np.std(yy) * 100:.2f} cm")
        v_real = (x_now - x_start) / max((t_now - t_start) * env.dt, 1e-6)
        dx, dy = x_now - x_start, y_now - y_start
        travel_dir = np.degrees(np.arctan2(dy, dx))
        print(f"    commanded {vx:.2f} m/s, achieved {v_real:.2f} m/s ({v_real / vx * 100:.0f}%)  "
              f"heading drift {yaw_now - yaw_start:+.1f} deg  travel direction {travel_dir:+.1f} deg  "
              f"lateral drift {dy * 100:+.1f} cm over {dx:.2f} m")
        print(f"    flight (no foot down) {np.mean(nc == 0) * 100:4.1f}%  single support {np.mean(nc == 1) * 100:4.1f}%  "
              f"double support {np.mean(nc == 2) * 100:4.1f}%")
        if swing_x[0] and swing_x[1]:
            sx = {f: np.array(swing_x[f][1:] or swing_x[f]) for f in (0, 1)}
            km = {f: np.array(knee_max[f][1:] or knee_max[f]) for f in (0, 1)}
            print(f"    swing: foot furthest back {sx[0][:, 0].mean() * 100:+.1f} | {sx[1][:, 0].mean() * 100:+.1f} cm, "
                  f"furthest forward {sx[0][:, 1].mean() * 100:+.1f} | {sx[1][:, 1].mean() * 100:+.1f} cm, "
                  f"peak knee flex {km[0].mean():.2f} | {km[1].mean():.2f} rad")
        if lands[0] and lands[1]:
            L, R = np.array(lands[0][1:]), np.array(lands[1][1:])  # drop each foot's first landing
            names = ("land x (cm)", "land |y| (cm)", "apex (cm)", "air time (s)", "step len (cm)", "knee at TD (rad)")
            mult = (100, 100, 100, 1, 100, 1)
            print("    left | right   " + "   ".join(
                f"{n} {L[:, i].mean() * s:.2f} | {R[:, i].mean() * s:.2f}" for i, (n, s) in enumerate(zip(names, mult))))


if __name__ == "__main__":
    main()
