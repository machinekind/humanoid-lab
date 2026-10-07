"""Parity of the torch/MJWarp port against the original JAX Joystick env.

Rolls the keeper policy in the JAX env (jax backend, CPU) and, at every
step:
  1. math: copies the JAX post-step mjx.Data into the warp data views and
     the JAX info into torch, then compares every reward term, the fall flag
     and both observation vectors (noise off) against the JAX env's own
     _compute_rewards/_build_obs. Expect float32 round-off.
  2. physics: loads the JAX pre-step state into warp, applies the same motor
     targets, steps MJWarp n_substeps, and compares qpos/qvel to the JAX
     post-step state. MJX-jax and MJWarp are different implementations of
     the same solver, so this is close, not exact.

    python wtrain/parity.py --run runs/roboto-keepers/roboto-locomotion-v1
"""

import argparse
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.dirname(__file__))

import jax
import jax.numpy as jp
import numpy as np
import torch
from ml_collections import config_dict

from wenv import WarpJoystick, load_run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--cmd", type=float, nargs=3, default=[0.5, 0.0, 0.3])
    args = ap.parse_args()

    from humanoid_lab.eval.battery import _find_latest_checkpoint
    from humanoid_lab.policy_io import load_policy

    run = load_run(args.run)
    W = WarpJoystick(run, num_envs=1, dr=False)
    E = W.jenv
    ckpt = _find_latest_checkpoint(run, __import__("pathlib").Path(args.run))
    inf = jax.jit(load_policy(str(ckpt.resolve()), E, config_dict.ConfigDict(run["ppo_config"]), deterministic=True))
    reset, step = jax.jit(E.reset), jax.jit(E.step)
    dev = W.dev

    def T(x, dtype=torch.float32):
        return torch.as_tensor(np.asarray(x), dtype=dtype, device=dev)[None]

    def load_data(d):
        W.qpos.copy_(T(d.qpos)); W.qvel.copy_(T(d.qvel)); W.qacc.copy_(T(d.qacc))
        W.sensordata.copy_(T(d.sensordata)); W.actuator_force.copy_(T(d.actuator_force))
        W.xpos.copy_(T(d.xpos)); W.site_xpos.copy_(T(d.site_xpos)); W.geom_xpos.copy_(T(d.geom_xpos))
        W.cdof.copy_(T(d.cdof)); W.subtree_com.copy_(T(d.subtree_com))

    def load_info(info):
        out = {}
        for k in ("command", "last_action", "last_last_action", "last_torque", "feet_air_time",
                  "feet_contact_time", "swing_apex", "last_apex", "phase", "air_dur_ema", "stance_dur_ema"):
            if k in info:
                out[k] = T(info[k])
        out["last_contact"] = T(info["last_contact"], torch.bool)
        return out

    state = reset(jax.random.PRNGKey(0))
    state.info["command"] = jp.array(args.cmd)
    key = jax.random.PRNGKey(1)
    worst_term, worst_obs, worst_q, worst_v = {}, [0.0, 0.0], [], []
    for i in range(args.steps):
        key, k = jax.random.split(key)
        action, _ = inf(state.obs, k)
        post = step(state, action)
        a_t = T(action)

        # 1. math on the JAX post-step data
        load_data(post.data)
        info_pre = load_info(state.info)
        contact = W.foot_contact()
        contact_filt = contact | info_pre["last_contact"]
        first_contact = (info_pre["feet_air_time"] > 0) & contact_filt
        info_pre["swing_apex"] = torch.where(~contact_filt, torch.maximum(info_pre["swing_apex"], W.foot_clearance()),
                                             info_pre["swing_apex"])
        terms, fall = W.compute_rewards(info_pre, a_t, first_contact, contact)

        jc = E._foot_contact(post.data)
        jcf = jc | state.info["last_contact"]
        jfc = (state.info["feet_air_time"] > 0) & jcf
        jinfo = dict(state.info)
        jinfo["swing_apex"] = jp.where(~jcf, jp.maximum(jinfo["swing_apex"], E._foot_clearance(post.data)),
                                       jinfo["swing_apex"])
        jterms, jfall = E._compute_rewards(post.data, jinfo, action, jfc, jc)
        assert bool(fall[0]) == bool(jfall), (i, "fall")
        assert np.array_equal(contact[0].cpu().numpy(), np.asarray(jc)), (i, "contact")
        for kname, v in terms.items():
            ref = float(jterms[kname])
            err = abs(float(v[0]) - ref) / max(1.0, abs(ref))
            worst_term[kname] = max(worst_term.get(kname, 0.0), err)

        s, p = W.build_obs(load_info(post.info) | {"phase": T(post.info["phase"])}, noise=False)
        jobs = E._build_obs(post.data, post.info)
        worst_obs[0] = max(worst_obs[0], float(np.abs(s[0].cpu().numpy() - np.asarray(jobs["state"])).max()))
        worst_obs[1] = max(worst_obs[1], float(np.abs(p[0].cpu().numpy() - np.asarray(jobs["privileged_state"])).max()))

        # 2. one control step of physics from the JAX pre-step state
        W.qpos.copy_(T(state.data.qpos)); W.qvel.copy_(T(state.data.qvel))
        W.qacc_warmstart.copy_(T(state.data.qacc_warmstart))
        targets = (W.default_pose + a_t * W.action_scale).clamp(W.ctrl_lo, W.ctrl_hi)
        W.ctrl.copy_(targets)
        W._physics()
        torch.cuda.synchronize()
        worst_q.append(float(np.abs(W.qpos[0].cpu().numpy() - np.asarray(post.data.qpos)).max()))
        worst_v.append(float(np.abs(W.qvel[0].cpu().numpy() - np.asarray(post.data.qvel)).max()))

        state = post
        state.info["command"] = jp.array(args.cmd)
        if bool(post.done):
            print(f"JAX env fell at step {i}")
            break

    print("reward terms, worst relative error over", i + 1, "steps:")
    for kname, e in sorted(worst_term.items(), key=lambda x: -x[1]):
        print(f"  {kname:26s} {e:.2e}")
    print(f"obs state max abs err {worst_obs[0]:.2e}, privileged {worst_obs[1]:.2e}")
    print(f"one-step physics vs MJX-jax: qpos max {max(worst_q):.2e} (median {np.median(worst_q):.2e}), "
          f"qvel max {max(worst_v):.2e} (median {np.median(worst_v):.2e})")


if __name__ == "__main__":
    main()
