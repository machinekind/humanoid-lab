"""Brax-PPO-equivalent trainer in torch on the MJWarp port, warm-started
from a Brax checkpoint.

Mirrors brax.training.agents.ppo.train as train.py configures it (values
from the source run's run.json ppo_config): tanh-normal policy with
softplus scale + 1e-3, silu MLPs, Welford observation normalizer updated on
each batch before SGD, GAE inside the loss per minibatch, clip 0.3,
vf coefficient 0.5, normalized advantages, Adam with global-norm clipping,
`batch_size * num_minibatches // num_envs` unrolls of `unroll_length` per
training step, and a full env reset every training epoch
(num_resets_per_eval).

Checkpoints are written in Brax's own format (orbax params + a copied
ppo_network_config.json, written last), so the repo's battery, eval video
and export read them unchanged. A checkpoint lands every --ckpt-minutes of
wall time.

    python wtrain/train.py --src runs/roboto-keepers/roboto-locomotion-v1 \
        --name loco_warp_v1 --steps 3e8
"""

import argparse
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from wenv import WarpJoystick, load_run

REPO = Path(__file__).resolve().parents[1]
LOG2 = math.log(2.0)


# -- networks -----------------------------------------------------------------
def mlp(sizes):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.SiLU())
    return nn.Sequential(*layers)


class Normalizer:
    """brax.training.acme.running_statistics, Welford mode, float32 stats
    with a float64 count."""

    def __init__(self, mean, summed_variance, count, dev):
        self.mean = torch.as_tensor(mean, dtype=torch.float32, device=dev).clone()
        self.svar = torch.as_tensor(summed_variance, dtype=torch.float32, device=dev).clone()
        self.count = float(count)
        self.std = self._std()

    def _std(self):
        return torch.sqrt(self.svar.clamp(min=0) / np.float32(self.count)).clamp(1e-6, 1e6)

    def update(self, x):
        x = x.reshape(-1, x.shape[-1])
        self.count += x.shape[0]
        cf = np.float32(self.count)
        diff_old = x - self.mean
        self.mean = self.mean + diff_old.sum(0) / cf
        self.svar = self.svar + (diff_old * (x - self.mean)).sum(0)
        self.std = self._std()

    def __call__(self, x, mean=None, std=None):
        return (x - (self.mean if mean is None else mean)) / (self.std if std is None else std)


def tanh_log_det(x):
    """brax TanhBijector.forward_log_det_jacobian."""
    return 2.0 * (LOG2 - x - F.softplus(-2.0 * x))


def dist_params(logits):
    loc, scale = logits.chunk(2, dim=-1)
    return loc, F.softplus(scale) + 1e-3


def log_prob(loc, scale, raw):
    lp = -0.5 * ((raw - loc) / scale) ** 2 - torch.log(scale) - 0.5 * math.log(2 * math.pi)
    return (lp - tanh_log_det(raw)).sum(-1)


def entropy(loc, scale, gen):
    ent = 0.5 + 0.5 * math.log(2 * math.pi) + torch.log(scale)
    sample = loc + scale * torch.randn(loc.shape, generator=gen, device=loc.device)
    return (ent + tanh_log_det(sample)).sum(-1)


# -- Brax checkpoint I/O --------------------------------------------------------
def load_brax(ckpt_dir):
    from humanoid_lab.policy_io import load_params

    return load_params(str(Path(ckpt_dir).resolve()))


def flax_to_torch(params, net: nn.Sequential):
    lin = [m for m in net if isinstance(m, nn.Linear)]
    p = params["params"]
    for i, layer in enumerate(lin):
        w = torch.as_tensor(np.asarray(p[f"hidden_{i}"]["kernel"]).T)
        if w.shape != layer.weight.shape:
            # the env grew an observation (e.g. heading_err, appended last): the
            # new input columns start at zero, so the policy begins unchanged
            assert w.shape[0] == layer.weight.shape[0] and w.shape[1] < layer.weight.shape[1], (w.shape, layer.weight.shape)
            layer.weight.data.zero_()
            layer.weight.data[:, : w.shape[1]] = w
        else:
            layer.weight.data.copy_(w)
        layer.bias.data.copy_(torch.as_tensor(np.asarray(p[f"hidden_{i}"]["bias"])))


def torch_to_flax(net: nn.Sequential):
    lin = [m for m in net if isinstance(m, nn.Linear)]
    return {"params": {f"hidden_{i}": {"kernel": layer.weight.detach().cpu().numpy().T.copy(),
                                       "bias": layer.bias.detach().cpu().numpy().copy()}
                       for i, layer in enumerate(lin)}}


def save_brax(template, norm_s, norm_p, policy, value, ckpt_root: Path, step: int, net_cfg: Path):
    import jax
    import jax.numpy as jnp
    import orbax.checkpoint as ocp
    from flax.training import orbax_utils

    rs = template[0]
    count = int(norm_s.count)
    cnt = type(rs.count)(hi=jnp.uint32(count >> 32), lo=jnp.uint32(count & 0xFFFFFFFF))
    j = lambda t: jnp.asarray(t.detach().cpu().numpy())
    new_rs = rs.replace(
        count=cnt,
        mean={"privileged_state": j(norm_p.mean), "state": j(norm_s.mean)},
        std={"privileged_state": j(norm_p.std), "state": j(norm_s.std)},
        summed_variance={"privileged_state": j(norm_p.svar), "state": j(norm_s.svar)},
    )
    params = [new_rs, jax.tree_util.tree_map(jnp.asarray, torch_to_flax(policy)),
              jax.tree_util.tree_map(jnp.asarray, torch_to_flax(value))]
    path = ckpt_root / f"{step:012d}"
    path.mkdir(parents=True, exist_ok=True)
    ocp.PyTreeCheckpointer().save(str(path.resolve()), params, force=True,
                                  save_args=orbax_utils.save_args_from_target(params))
    shutil.copy(net_cfg, path / "ppo_network_config.json")  # last: marks a complete checkpoint
    return path


# -- training -------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="run dir holding run.json + checkpoints/<step>")
    ap.add_argument("--name", required=True, help="new run dir under runs/")
    ap.add_argument("--steps", type=float, default=3e8)
    ap.add_argument("--num-envs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt-minutes", type=float, default=10.0)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--sym-coef", type=float, default=0.0,
                    help="weight of the mirror-symmetry loss mean((mu(M s) - M mu(s))^2) on the policy mean")
    ap.add_argument("--critic-warmup-iters", type=int, default=0,
                    help="iterations that train the value net only (policy frozen)")
    ap.add_argument("--max-minutes", type=float, default=None)
    ap.add_argument("--src-ckpt", default=None, help="checkpoint step dir name in --src (default: the latest)")
    ap.add_argument("--resume", action="store_true", help="continue runs/<name> from its trainer_state.pt")
    ap.add_argument("--reset-keyframe", default=None, help="task.env.reset_keyframe (also the ctrl/obs anchor)")
    ap.add_argument("--act-scale", action="append", default=[], metavar="GROUP=RAD",
                    help="per-joint-group action_scale_rad override, repeatable")
    ap.add_argument("--warp-reward", default=None, help="JSON for run.json's warp_reward block (see wenv.py)")
    ap.add_argument("--pose-l1", action="append", default=[], metavar="GROUP=W",
                    help="task.env.reward.pose_l1_weights override, repeatable")
    ap.add_argument("--scale", action="append", default=[], metavar="TERM=S",
                    help="task.env.reward.scales override, repeatable")
    ap.add_argument("--warp-cmd-ramp", default=None, help='JSON for run.json warp_cmd_ramp, e.g. {"lin": 0.25, "yaw": 0.6}')
    ap.add_argument("--cmd-straight", type=float, default=None, help="probability of a vx-only command (vy = wz = 0)")
    ap.add_argument("--warp-lift", default=None, help='JSON for run.json warp_lift, e.g. {"v_full": 0.5, "min_frac": 0.25}')
    ap.add_argument("--env", action="append", default=[], metavar="DOTTED.KEY=JSON",
                    help="any task.env override, e.g. gait.freq=[0.8,1.4]; repeatable")
    args = ap.parse_args()

    src = REPO / args.src
    run = load_run(src)
    # Recipe changes go into the run dict itself, so the new run.json carries
    # them and eval/video/battery rebuild the same env.
    hyd = run["hydra_config"]
    if args.reset_keyframe:
        hyd["task"]["env"]["reset_keyframe"] = args.reset_keyframe
    if args.act_scale:
        ov = hyd["actuators"].get("overrides") or {}
        groups = ov.setdefault("groups", {})
        for kv in args.act_scale:
            g, v = kv.split("=")
            groups.setdefault(g, {})["action_scale_rad"] = float(v)
        hyd["actuators"]["overrides"] = ov
    if args.warp_reward:
        run["warp_reward"] = json.loads(args.warp_reward)
    if args.warp_lift:
        run["warp_lift"] = json.loads(args.warp_lift)
    if args.warp_cmd_ramp:
        run["warp_cmd_ramp"] = json.loads(args.warp_cmd_ramp)
    if args.cmd_straight is not None:
        run["warp_cmd_straight"] = args.cmd_straight
    for kv in args.pose_l1:
        g, v = kv.split("=")
        hyd["task"]["env"].setdefault("reward", {}).setdefault("pose_l1_weights", {})[g] = float(v)
    for kv in args.scale:
        g, v = kv.split("=")
        hyd["task"]["env"].setdefault("reward", {}).setdefault("scales", {})[g] = float(v)
    for kv in args.env:
        path, v = kv.split("=", 1)
        node = hyd["task"]["env"]
        *parents, leaf = path.split(".")
        for p_ in parents:
            node = node.setdefault(p_, {})
        node[leaf] = json.loads(v)
    pc = dict(run["ppo_config"])
    N = args.num_envs or int(pc["num_envs"])
    T = int(pc["unroll_length"])
    n_mb = int(pc["num_minibatches"])
    mb = int(pc["batch_size"])
    n_unrolls = max(1, mb * n_mb // N)
    epochs = int(pc["num_updates_per_batch"])
    gamma, lam = float(pc["discounting"]), float(pc["gae_lambda"])
    ent_cost = float(pc["entropy_cost"])
    lr = args.lr or float(pc["learning_rate"])
    clip_eps, vf_coef = float(pc.get("clipping_epsilon", 0.3)), float(pc.get("vf_loss_coefficient", 0.5))
    max_grad = float(pc.get("max_grad_norm") or 0.0)
    steps_per_iter = N * T * n_unrolls
    resets_per_eval = max(int(pc.get("num_resets_per_eval", 0)), 1)
    iters_per_reset = max(1, round(float(pc["num_timesteps"]) / (max(int(pc["num_evals"]) - 1, 1)
                                                                  * steps_per_iter * resets_per_eval)))

    torch.manual_seed(args.seed)
    env = WarpJoystick(run, num_envs=N, seed=args.seed)
    dev = env.dev
    gen = torch.Generator(device=dev)
    gen.manual_seed(args.seed + 1)

    ckpts = sorted((p for p in (src / "checkpoints").iterdir() if p.name.isdigit()), key=lambda p: int(p.name))
    src_ckpt = ckpts[-1] if not args.src_ckpt else src / "checkpoints" / args.src_ckpt
    params = load_brax(src_ckpt)
    rs = params[0]
    count = int(rs.count.hi) * 2 ** 32 + int(rs.count.lo)
    norm_s = Normalizer(rs.mean["state"], rs.summed_variance["state"], count, dev)
    norm_p = Normalizer(rs.mean["privileged_state"], rs.summed_variance["privileged_state"], count, dev)
    net_cfg = json.loads((src_ckpt / "ppo_network_config.json").read_text())["network_factory_kwargs"]
    if net_cfg["activation"] != "silu" or net_cfg["distribution_type"] != "tanh_normal":
        raise ValueError(f"unsupported network config {net_cfg}")
    A = env.nu
    def grow(nrm, size):
        k = size - nrm.mean.numel()
        if k > 0:  # new observation dims: mean 0, std 0.2 until the running stats take over
            nrm.mean = torch.cat([nrm.mean, torch.zeros(k, device=dev)])
            nrm.svar = torch.cat([nrm.svar, torch.full((k,), 0.04 * nrm.count, device=dev)])
            nrm.std = nrm._std()

    _s0, _p0 = env.reset()
    priv_size = _p0.shape[-1]
    grow(norm_s, env.state_size)
    grow(norm_p, priv_size)
    policy = mlp([env.state_size, *net_cfg["policy_hidden_layer_sizes"], 2 * A]).to(dev)
    value = mlp([norm_p.mean.numel(), *net_cfg["value_hidden_layer_sizes"], 1]).to(dev)
    flax_to_torch(params[1], policy)
    flax_to_torch(params[2], value)
    opt = torch.optim.Adam(list(policy.parameters()) + list(value.parameters()), lr=lr, eps=1e-8)
    if args.sym_coef:
        # Left/right equivariance of the policy itself (docs/plans/roboto-first-run.md,
        # "Next" item 1): env-side mirroring only randomizes chirality, a limp
        # still earns as much as an even gait; this loss makes the actor's
        # mean for the mirrored state the mirror of its mean for the state.
        from humanoid_lab.envs import symmetry

        rs = env.jenv.robot_spec
        _joints, _feet = list(rs.actuated_joints), list(rs.foot_sites)
        a_perm, a_sign = symmetry.joint_mirror(_joints, rs.name)
        _base = [n for n in env.state_names if n != "heading_err"]
        s_perm, s_sign = symmetry.obs_mirror(_base, _joints, _feet, rs.name)
        if "heading_err" in env.state_names:
            assert env.state_names[-1] == "heading_err"
            # a yaw error flips sign under the left/right mirror
            s_perm = np.concatenate([s_perm, [len(s_perm)]]).astype(int)
            s_sign = np.concatenate([s_sign, [-1.0]])
        a_perm, a_sign = torch.as_tensor(a_perm, device=dev), torch.as_tensor(a_sign, dtype=torch.float32, device=dev)
        s_perm, s_sign = torch.as_tensor(s_perm, device=dev), torch.as_tensor(s_sign, dtype=torch.float32, device=dev)
        print(f"symmetry loss on, coef {args.sym_coef}", flush=True)

    # new run dir with a run.json the repo's eval tools can rebuild the env from
    out = REPO / "runs" / args.name
    ckpt_root = out / "checkpoints"
    ckpt_root.mkdir(parents=True, exist_ok=True)
    new_run = dict(run)
    new_run.update(run_name=args.name, checkpoint_dir=str(ckpt_root.resolve()), num_timesteps=int(args.steps),
                   warm_start={"src": str(src), "checkpoint": src_ckpt.name}, trainer="wtrain/train.py (torch+MJWarp)")
    new_run["ppo_config"] = {**pc, "num_envs": N, "num_timesteps": int(args.steps), "learning_rate": lr}
    (out / "run.json").write_text(json.dumps(new_run, indent=1))
    (out / "trainer.pid").write_text(str(os.getpid()))  # the real process, not the venv launcher
    log_f = open(out / "train_log.jsonl", "a")
    print(f"src {src_ckpt}  ->  {out}")
    print(f"N={N} T={T} unrolls={n_unrolls} minibatches={n_mb}x{mb} epochs={epochs} lr={lr} "
          f"steps/iter={steps_per_iter:,} full reset every {iters_per_reset} iters", flush=True)

    S, P = env.reset()
    ep_ret = torch.zeros(N, device=dev)
    ep_len = torch.zeros(N, device=dev)
    def zacc():
        return ({k: torch.zeros((), device=dev) for k in ("ret", "len", "n", "falls", "rew", "steps")},
                {k: torch.zeros((), device=dev) for k in env.terms})

    acc, acc_terms = zacc()
    step_total, it = 0, 0
    state_file = out / "trainer_state.pt"
    if args.resume and state_file.exists():
        # Exact continuation of this run: nets, Adam moments, normalizer
        # (float64 count) and the step/iteration counters.
        st = torch.load(state_file, map_location=dev, weights_only=False)
        policy.load_state_dict(st["policy"])
        value.load_state_dict(st["value"])
        opt.load_state_dict(st["opt"])
        for nrm, k in ((norm_s, "norm_s"), (norm_p, "norm_p")):
            nrm.mean, nrm.svar, nrm.count = st[k]["mean"], st[k]["svar"], st[k]["count"]
            nrm.std = nrm._std()
        step_total, it = st["step"], st["it"]
        print(f"resumed {state_file} at step {step_total:,} it {it}", flush=True)
    t_start = t_last_ckpt = time.time()
    t_last_log, steps_last_log = t_start, step_total

    def ckpt(step):
        p = save_brax(params, norm_s, norm_p, policy, value, ckpt_root, step, src_ckpt / "ppo_network_config.json")
        torch.save({"policy": policy.state_dict(), "value": value.state_dict(), "opt": opt.state_dict(),
                    "norm_s": {"mean": norm_s.mean, "svar": norm_s.svar, "count": norm_s.count},
                    "norm_p": {"mean": norm_p.mean, "svar": norm_p.svar, "count": norm_p.count},
                    "step": step, "it": it}, state_file.with_suffix(".tmp"))
        os.replace(state_file.with_suffix(".tmp"), state_file)
        print(f"checkpoint {p}", flush=True)

    if step_total == 0:
        ckpt(0)  # the warm-start policy itself: the first video is the baseline

    while step_total < args.steps:
        # ---- collect ----------------------------------------------------------
        buf_s = torch.empty(n_unrolls, T, N, env.state_size, device=dev)
        buf_p = torch.empty(n_unrolls, T, N, P.shape[-1], device=dev)
        buf_raw = torch.empty(n_unrolls, T, N, A, device=dev)
        buf_lp = torch.empty(n_unrolls, T, N, device=dev)
        buf_r = torch.empty(n_unrolls, T, N, device=dev)
        buf_d = torch.empty(n_unrolls, T, N, device=dev)
        buf_tr = torch.empty(n_unrolls, T, N, device=dev)
        last_s = torch.empty(n_unrolls, N, env.state_size, device=dev)
        last_p = torch.empty(n_unrolls, N, P.shape[-1], device=dev)
        mean_s, std_s = norm_s.mean.clone(), norm_s.std.clone()
        with torch.no_grad():
            for u in range(n_unrolls):
                for t in range(T):
                    loc, scale = dist_params(policy(norm_s(S, mean_s, std_s)))
                    raw = loc + scale * torch.randn(loc.shape, generator=gen, device=dev)
                    buf_s[u, t], buf_p[u, t], buf_raw[u, t] = S, P, raw
                    buf_lp[u, t] = log_prob(loc, scale, raw)
                    S, P, r, d, tr, terms = env.step(torch.tanh(raw))
                    buf_r[u, t], buf_d[u, t], buf_tr[u, t] = r, d.float(), tr.float()
                    ep_ret += r
                    ep_len += 1
                    df = d.float()
                    acc["ret"] += (ep_ret * df).sum()
                    acc["len"] += (ep_len * df).sum()
                    acc["n"] += df.sum()
                    acc["falls"] += (d & ~tr).float().sum()
                    acc["rew"] += r.mean()
                    acc["steps"] += 1
                    ep_ret *= 1 - df
                    ep_len *= 1 - df
                    for k in env.terms:
                        acc_terms[k] += terms[k].mean()
                last_s[u], last_p[u] = S, P
        step_total += steps_per_iter
        it += 1

        # ---- update -----------------------------------------------------------
        # [n_unrolls*N trajectories, T]
        def traj(x):
            return x.transpose(1, 2).reshape(n_unrolls * N, T, *x.shape[3:])

        Ds, Dp, Draw, Dlp, Dr, Dd, Dtr = map(traj, (buf_s, buf_p, buf_raw, buf_lp, buf_r, buf_d, buf_tr))
        Ls, Lp = last_s.reshape(-1, env.state_size), last_p.reshape(-1, P.shape[-1])
        norm_s.update(buf_s)
        norm_p.update(buf_p)
        train_policy = it > args.critic_warmup_iters
        stats = {k: torch.zeros((), device=dev) for k in ("policy_loss", "v_loss", "entropy", "approx_kl", "clipfrac", "sym_loss")}
        nb = 0
        B = n_unrolls * N
        for _ in range(epochs):
            perm = torch.randperm(B, generator=gen, device=dev)
            for k in range(n_mb):
                idx = perm[k * (B // n_mb):(k + 1) * (B // n_mb)]
                s, p, raw, lp_old = Ds[idx], Dp[idx], Draw[idx], Dlp[idx]
                rew, dn, trn = Dr[idx], Dd[idx], Dtr[idx]
                loc, scale = dist_params(policy(norm_s(s)))
                v = value(norm_p(p)).squeeze(-1)  # b,T
                v_boot = value(norm_p(Lp[idx])).squeeze(-1)  # b
                # compute_gae, time-major semantics on [b, T]
                with torch.no_grad():
                    term = dn * (1 - trn)
                    tmask = 1 - trn
                    v_next = torch.cat([v[:, 1:], v_boot[:, None]], 1)
                    deltas = (rew + gamma * (1 - term) * v_next - v) * tmask
                    gacc = torch.zeros_like(v_boot)
                    vs_minus_v = torch.empty_like(v)
                    for tt in range(T - 1, -1, -1):
                        gacc = deltas[:, tt] + gamma * (1 - term[:, tt]) * tmask[:, tt] * lam * gacc
                        vs_minus_v[:, tt] = gacc
                    vs = vs_minus_v + v
                    vs_next = torch.cat([vs[:, 1:], v_boot[:, None]], 1)
                    adv = (rew + gamma * (1 - term) * vs_next - v) * tmask
                    adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
                lp_new = log_prob(loc, scale, raw)
                ratio = torch.exp(lp_new - lp_old)
                pl = -torch.min(ratio * adv, ratio.clamp(1 - clip_eps, 1 + clip_eps) * adv).mean()
                vl = ((vs - v) ** 2).mean() * 0.5 * vf_coef
                ent = entropy(loc, scale, gen).mean()
                loss = vl + ((pl - ent_cost * ent) if train_policy else 0.0)
                if args.sym_coef and train_policy:
                    loc_m, _ = dist_params(policy(norm_s(s[..., s_perm] * s_sign)))
                    sym = ((loc_m - loc[..., a_perm] * a_sign) ** 2).mean()
                    loss = loss + args.sym_coef * sym
                    stats["sym_loss"] += sym.detach()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                if not train_policy:
                    for prm in policy.parameters():
                        prm.grad = None
                if max_grad:
                    torch.nn.utils.clip_grad_norm_(list(policy.parameters()) + list(value.parameters()), max_grad)
                opt.step()
                with torch.no_grad():
                    stats["policy_loss"] += pl.detach()
                    stats["v_loss"] += vl.detach()
                    stats["entropy"] += ent.detach()
                    lr_ = (lp_new - lp_old).detach().clamp(-20, 20)
                    stats["approx_kl"] += ((lr_.exp() - 1) - lr_).mean()
                    stats["clipfrac"] += ((ratio.detach() - 1).abs() > clip_eps).float().mean()
                nb += 1

        if it % iters_per_reset == 0:
            S, P = env.reset()
            ep_ret.zero_()
            ep_len.zero_()

        # ---- log / checkpoint ---------------------------------------------------
        now = time.time()
        if it % 5 == 0 or it == 1:
            sps = (step_total - steps_last_log) / (now - t_last_log)
            t_last_log, steps_last_log = now, step_total
            a = {k: float(v) for k, v in acc.items()}
            n_steps = max(a["steps"], 1.0)
            rec = {
                "it": it, "steps": step_total, "sps": round(sps), "minutes": round((now - t_start) / 60, 2), "commit_mb": round(__import__("psutil").Process().memory_info().private / 2**20),
                # mean reward per control step x 1000: comparable to brax's
                # eval/episode_reward for a policy that does not fall
                "reward_x1000": round(a["rew"] / n_steps * 1000, 3),
                "falls_per_1M": round(a["falls"] / (n_steps * N) * 1e6, 2),
                "done_ep_len": round(a["len"] / a["n"], 1) if a["n"] else None,
                **{k: round(float(v) / nb, 5) for k, v in stats.items()},
                "terms": {k: round(float(v) / n_steps, 5) for k, v in acc_terms.items()},
            }
            log_f.write(json.dumps(rec) + "\n")
            log_f.flush()
            print(f"it {it:5d}  steps {step_total:>12,}  reward/1000 steps {rec['reward_x1000']:7.2f}  "
                  f"falls/1M {rec['falls_per_1M']:7.1f}  v_loss {rec['v_loss']:.4f}  kl {rec['approx_kl']:.4f}  "
                  + (f"sym {rec['sym_loss']:.4f}  " if args.sym_coef else "") +
                  f"ent {rec['entropy']:.2f}  {sps:,.0f} steps/s  mem {rec['commit_mb']} MB{'' if train_policy else '  [critic warm-up]'}",
                  flush=True)
            acc, acc_terms = zacc()
        if now - t_last_ckpt >= args.ckpt_minutes * 60:
            ckpt(step_total)
            t_last_ckpt = now
        if args.max_minutes and now - t_start >= args.max_minutes * 60:
            break

    ckpt(step_total)
    (out / "DONE").write_text(str(step_total))


if __name__ == "__main__":
    main()
