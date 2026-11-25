"""The Torch PPO trainer: the config ledger and the training loop.

The loop takes the PPO params exactly as the Brax trainer builds them
(ppo_config.compose_ppo_params). The ledger below marks every key as
implemented, ignored with a recorded reason, or refused when it is set to a
value the Torch loop does not implement. check_config raises one ValueError
that lists every refusal.

Trainer is the PPO loop. It takes any env with the protocol documented on
torch_warp.env.WarpJoystick, so env construction stays outside the loop.
The caller passes the mirror maps for the symmetry loss.

main is the training run, `python -m humanoid_lab.torch_warp.ppo
<hydra overrides>` on configs/torch_warp.yaml. It trains from scratch, warm
starts from a Brax checkpoint with restore=, or resumes a Torch run with
torch_warp.resume=true. Everything goes under runs/<run_name>: run.json
(written at start with null results and again at the end), Brax
checkpoints, trainer_state.pt, train_log.jsonl, trainer.pid and DONE.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import math
import os
import shutil
import time
from collections.abc import Callable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

from humanoid_lab.torch_warp import policy_io

IMPLEMENTED = "implemented"
IGNORED = "ignored"
REFUSED = "refused"


@dataclasses.dataclass(frozen=True)
class Entry:
    """One ledger entry.

    allowed is None when every value is accepted. Otherwise a value it
    rejects is a refusal, and reason says why.
    """

    status: str
    reason: str = ""
    allowed: Callable[[Any], bool] | None = None


def _impl(reason="", allowed=None):
    return Entry(IMPLEMENTED, reason, allowed)


def _ignored(reason):
    return Entry(IGNORED, reason)


def _refused_unless(allowed, reason):
    return Entry(REFUSED, reason, allowed)


def _is_none(v):
    return v is None


def _is_false(v):
    return not v


def _never(_v):
    return False


def _activation_name(v):
    return getattr(v, "__name__", v)


_ADAPTIVE_KL_ONLY = "only the ADAPTIVE_KL learning-rate schedule reads it, and that schedule is refused"
_NO_WRAPPERS = (
    "the Torch env always applies the episode counter and auto-reset, and DR comes from the dr block"
)
_NO_EVAL = "there is no eval loop: the trainer reports the training reward, and final_reward is null"
_OWN_LOGS = "the Torch trainer logs to train_log.jsonl and wandb"
_OWN_CHECKPOINTS = "checkpoints go to runs/<run_name>/checkpoints, and a warm start uses restore="

# Every parameter of brax's ppo.train and every key compose_ppo_params
# produces.
PPO_LEDGER: dict[str, Entry] = {
    "num_timesteps": _impl(),
    "num_envs": _impl(),
    "episode_length": _impl(),
    "learning_rate": _impl(),
    "entropy_cost": _impl(),
    "discounting": _impl(),
    "unroll_length": _impl(),
    "batch_size": _impl(),
    "num_minibatches": _impl(),
    "num_updates_per_batch": _impl(),
    "num_resets_per_eval": _impl(),
    "num_evals": _impl("feeds iters_per_reset"),
    "clipping_epsilon": _impl(),
    "gae_lambda": _impl(),
    "max_grad_norm": _impl("None means no clip"),
    "vf_loss_coefficient": _impl(),
    "network_factory": _impl("checked key by key against NETWORK_LEDGER"),
    "seed": _impl(),
    "sym_coef": _impl("repo-owned key: the mirror-symmetry loss weight"),
    "normalize_observations": _impl(
        "the Torch trainer always normalizes observations, so it must be true", lambda v: v is True
    ),
    "normalize_advantage": _impl(
        "the Torch trainer always normalizes advantages, so it must be true", lambda v: v is True
    ),
    "action_repeat": _refused_unless(lambda v: v == 1, "action_repeat must be 1"),
    "reward_scaling": _refused_unless(lambda v: float(v) == 1.0, "reward_scaling must be 1.0"),
    "clipping_epsilon_value": _refused_unless(_is_none, "the clipped value loss is not implemented"),
    "normalize_observations_mode": _refused_unless(
        lambda v: v == "welford", "only the welford normalizer is implemented"
    ),
    "normalize_observations_std_eps": _refused_unless(
        lambda v: float(v) == 0.0, "the normalizer implements std_eps 0.0 only"
    ),
    "normalize_until_count": _refused_unless(_is_none, "the normalizer never stops updating"),
    "bootstrap_on_timeout": _refused_unless(_is_false, "bootstrap_on_timeout is not implemented"),
    "use_distributional_critic": _refused_unless(_is_false, "the distributional critic is not implemented"),
    "learning_rate_schedule": _refused_unless(
        _is_none, "only a constant learning rate is implemented (ADAPTIVE_KL is refused)"
    ),
    "vision": _refused_unless(_is_false, "vision observations are not implemented"),
    "augment_pixels": _refused_unless(_is_false, "vision observations are not implemented"),
    "restore_value_fn": _refused_unless(
        lambda v: v is True, "a warm start always loads the value network"
    ),
    "max_devices_per_host": _refused_unless(_is_none, "the Torch trainer uses one device per process"),
    "desired_kl": _ignored(_ADAPTIVE_KL_ONLY),
    "learning_rate_schedule_min_lr": _ignored(_ADAPTIVE_KL_ONLY),
    "learning_rate_schedule_max_lr": _ignored(_ADAPTIVE_KL_ONLY),
    "wrap_env": _ignored(_NO_WRAPPERS),
    "wrap_env_fn": _ignored(_NO_WRAPPERS),
    "randomization_fn": _ignored(_NO_WRAPPERS),
    "use_pmap_on_reset": _ignored("the Torch trainer runs one device and has no pmap"),
    "num_eval_envs": _ignored(_NO_EVAL),
    "deterministic_eval": _ignored(_NO_EVAL),
    "run_evals": _ignored(_NO_EVAL),
    "eval_env": _ignored(_NO_EVAL),
    "log_training_metrics": _ignored(_OWN_LOGS),
    "training_metrics_steps": _ignored(_OWN_LOGS),
    "progress_fn": _ignored(_OWN_LOGS),
    "policy_params_fn": _ignored(_OWN_LOGS),
    "save_checkpoint_path": _ignored(_OWN_CHECKPOINTS),
    "restore_checkpoint_path": _ignored(_OWN_CHECKPOINTS),
    "restore_params": _ignored(_OWN_CHECKPOINTS),
    "environment": _ignored("the trainer builds the env from the task config"),
}

_PLAIN_MLP = "brax's tanh_normal policy is a plain MLP that ignores it"
_SET_BY_TRAINER = "the trainer sets it from the env"

# Every keyword of brax's make_ppo_networks.
NETWORK_LEDGER: dict[str, Entry] = {
    "policy_hidden_layer_sizes": _impl(),
    "value_hidden_layer_sizes": _impl(),
    "policy_obs_key": _impl("the actor reads the 'state' observation", lambda v: v == "state"),
    "value_obs_key": _impl(
        "the critic reads the 'privileged_state' observation", lambda v: v == "privileged_state"
    ),
    # A Hydra override can only give a string, and make_ppo_networks needs a
    # callable. So only brax's own default function passes.
    "activation": _refused_unless(
        lambda v: callable(v) and _activation_name(v) in policy_io.ACTIVATIONS,
        "activation must stay at brax's default (swish), the only one the Torch networks implement",
    ),
    "distribution_type": _refused_unless(
        lambda v: v == policy_io.DISTRIBUTION, "the Torch policy implements tanh_normal only"
    ),
    "layer_norm": _refused_unless(_is_false, "layer norm is not implemented"),
    "policy_network_kernel_init_fn": _refused_unless(_is_none, "custom kernel init is not implemented"),
    "value_network_kernel_init_fn": _refused_unless(_is_none, "custom kernel init is not implemented"),
    "mean_kernel_init_fn": _refused_unless(_is_none, "custom kernel init is not implemented"),
    "policy_network_kernel_init_kwargs": _refused_unless(_is_none, "custom kernel init is not implemented"),
    "value_network_kernel_init_kwargs": _refused_unless(_is_none, "custom kernel init is not implemented"),
    "mean_kernel_init_kwargs": _refused_unless(_is_none, "custom kernel init is not implemented"),
    "use_distributional_critic": _refused_unless(_is_false, "the distributional critic is not implemented"),
    "observation_size": _refused_unless(_never, _SET_BY_TRAINER),
    "action_size": _refused_unless(_never, _SET_BY_TRAINER),
    "preprocess_observations_fn": _refused_unless(_never, _SET_BY_TRAINER),
    "noise_std_type": _ignored(_PLAIN_MLP),
    "init_noise_std": _ignored(_PLAIN_MLP),
    "state_dependent_std": _ignored(_PLAIN_MLP),
    "mean_clip_scale": _ignored(_PLAIN_MLP),
    "num_quantiles": _ignored("only the distributional critic reads it"),
}


def _sim_backend(task):
    env = (task or {}).get("env") or {}
    return (env.get("sim") or {}).get("backend")


# Every top-level key of the composed torch_warp config.
TOP_LEVEL_LEDGER: dict[str, Entry] = {
    "run_name": _impl(),
    "seed": _impl(),
    "smoke": _impl(),
    "restore": _impl(),
    "domain_rand": _impl(),
    "dr": _impl(),
    "contact_preflight": _impl(),
    "wandb": _impl(),
    "robot": _impl(),
    "actuators": _impl(),
    "network": _impl("reaches the trainer through ppo.network_factory"),
    "ppo": _impl("checked key by key against PPO_LEDGER"),
    "torch_warp": _impl(),
    "bench": _impl(),
    "task": _impl(
        "task.env.sim.backend=jax: the Torch trainer steps MJWarp physics",
        lambda v: _sim_backend(v) != "jax",
    ),
    "early_stop": _refused_unless(
        lambda v: not (v or {}).get("enable"), "early_stop.enable=true needs an eval reward"
    ),
}


def _plain(x):
    """A DictConfig or ConfigDict as plain Python containers."""
    from ml_collections import config_dict
    from omegaconf import DictConfig, ListConfig, OmegaConf

    if isinstance(x, (DictConfig, ListConfig)):
        return OmegaConf.to_container(x, resolve=True)
    if isinstance(x, config_dict.ConfigDict):
        return x.to_dict()
    return x


def _walk(values: Mapping, ledger: Mapping[str, Entry], prefix: str, ignored: dict, refusals: list):
    for key, value in values.items():
        name = f"{prefix}{key}"
        entry = ledger.get(key)
        if entry is None:
            refusals.append(f"{name}: unknown key, the Torch trainer does not implement it")
        elif entry.status == IGNORED:
            ignored[name] = entry.reason
        elif entry.allowed is not None and not entry.allowed(value):
            refusals.append(f"{name}={_activation_name(value)!r}: {entry.reason}")


def loop_sizes(ppo_params) -> dict:
    """n_unrolls, steps_per_iter and iters_per_reset of brax's ppo.train.

    iters_per_reset is brax's num_training_steps_per_epoch. Raises
    ValueError when the batch does not split into whole unrolls, which brax
    asserts.
    """
    n = int(ppo_params["num_envs"])
    b, m = int(ppo_params["batch_size"]), int(ppo_params["num_minibatches"])
    if (b * m) % n != 0:
        raise ValueError(
            f"batch_size * num_minibatches = {b} * {m} must be a multiple of num_envs = {n}, as brax asserts"
        )
    n_unrolls = b * m // n
    steps_per_iter = n * int(ppo_params["unroll_length"]) * n_unrolls
    epochs = max(int(ppo_params["num_evals"]) - 1, 1) * max(int(ppo_params["num_resets_per_eval"]), 1)
    iters_per_reset = math.ceil(float(ppo_params["num_timesteps"]) / (epochs * steps_per_iter))
    return {"n_unrolls": n_unrolls, "steps_per_iter": steps_per_iter, "iters_per_reset": iters_per_reset}


def check_config(hydra, ppo_params) -> dict:
    """Check the composed config and PPO params against the ledgers.

    hydra is the composed config (a DictConfig or its plain dict) and
    ppo_params the result of compose_ppo_params. Returns the ignored keys
    that are set, as {key: reason}, for the trainer block. Raises one
    ValueError listing every refusal.
    """
    hydra, params = _plain(hydra), dict(_plain(ppo_params))
    ignored, refusals = {}, []
    _walk(hydra, TOP_LEVEL_LEDGER, "", ignored, refusals)
    factory = params.get("network_factory")
    _walk(params, PPO_LEDGER, "ppo.", ignored, refusals)
    if isinstance(factory, Mapping):
        _walk(factory, NETWORK_LEDGER, "ppo.network_factory.", ignored, refusals)
    try:
        loop_sizes(params)
    except (KeyError, ValueError) as e:
        refusals.append(f"ppo: {e}")
    if refusals:
        raise ValueError(
            "the Torch trainer refuses this config (train it with ./run.sh train, or change these keys):\n  "
            + "\n  ".join(refusals)
        )
    return ignored


# -- the tanh-normal policy head ----------------------------------------------
_LOG2 = math.log(2.0)


def tanh_log_det(x):
    """brax TanhBijector.forward_log_det_jacobian."""
    return 2.0 * (_LOG2 - x - F.softplus(-2.0 * x))


def dist_params(logits):
    loc, scale = logits.chunk(2, dim=-1)
    return loc, F.softplus(scale) + 1e-3


def log_prob(loc, scale, raw):
    lp = -0.5 * ((raw - loc) / scale) ** 2 - torch.log(scale) - 0.5 * math.log(2 * math.pi)
    return (lp - tanh_log_det(raw)).sum(-1)


def entropy(loc, scale, gen):
    """brax's sampled tanh-normal entropy. It draws one sample from gen."""
    ent = 0.5 + 0.5 * math.log(2 * math.pi) + torch.log(scale)
    sample = loc + scale * torch.randn(loc.shape, generator=gen, device=loc.device)
    return (ent + tanh_log_det(sample)).sum(-1)


LOSS_KEYS = ("policy_loss", "v_loss", "entropy", "approx_kl", "clipfrac", "sym_loss")
_ACC_KEYS = ("ret", "len", "n", "falls", "rew", "steps")


class Trainer:
    """Brax's PPO update on Torch tensors.

    env has the protocol on torch_warp.env.WarpJoystick. ppo_params holds
    the keys PPO_LEDGER implements. mirror_maps is (obs_perm, obs_sign,
    act_perm, act_sign) from symmetry.policy_mirror_maps, needed when
    sym_coef > 0. The networks start at torch's default init: call
    load_params before the first iteration.

    One generator, seeded with seed + 1, draws the action samples, the
    minibatch permutations and the entropy samples, in that order.
    """

    def __init__(self, env, ppo_params, *, sym_coef=0.0, mirror_maps=None, critic_warmup_iters=0, seed=0):
        ppo_params = dict(_plain(ppo_params))
        self.env = env
        self.device = dev = torch.device(env.device)
        sizes = loop_sizes(ppo_params)
        if int(ppo_params["num_envs"]) != int(env.num_envs):
            raise ValueError(f"ppo.num_envs is {ppo_params['num_envs']} and the env has {env.num_envs} worlds")
        self.N = int(env.num_envs)
        self.T = int(ppo_params["unroll_length"])
        self.n_minibatches = int(ppo_params["num_minibatches"])
        self.epochs = int(ppo_params["num_updates_per_batch"])
        self.n_unrolls = sizes["n_unrolls"]
        self.steps_per_iter = sizes["steps_per_iter"]
        self.iters_per_reset = sizes["iters_per_reset"]
        self.resets = int(ppo_params["num_resets_per_eval"]) > 0
        self.gamma = float(ppo_params["discounting"])
        self.lam = float(ppo_params["gae_lambda"])
        self.ent_cost = float(ppo_params["entropy_cost"])
        self.lr = float(ppo_params["learning_rate"])
        # brax ppo.train's defaults for the keys the Go1 config leaves out.
        self.clip_eps = float(ppo_params.get("clipping_epsilon", 0.3))
        self.vf_coef = float(ppo_params.get("vf_loss_coefficient", 0.5))
        max_grad = ppo_params.get("max_grad_norm")
        self.max_grad_norm = None if max_grad is None else float(max_grad)
        self.sym_coef = float(sym_coef)
        self.critic_warmup_iters = int(critic_warmup_iters)

        net = ppo_params["network_factory"]
        S, P, A = int(env.actor_obs_size), int(env.privileged_obs_size), int(env.action_size)
        self.S, self.P, self.A = S, P, A
        self.policy = policy_io.mlp([S, *net["policy_hidden_layer_sizes"], 2 * A]).to(dev)
        self.value = policy_io.mlp([P, *net["value_hidden_layer_sizes"], 1]).to(dev)
        self.norm_s = policy_io.Normalizer.zeros(S, dev)
        self.norm_p = policy_io.Normalizer.zeros(P, dev)
        self.opt = torch.optim.Adam(self._params(), lr=self.lr, eps=1e-8)
        self.gen = torch.Generator(device=dev)
        self.gen.manual_seed(int(seed) + 1)

        self.mirror = None
        if self.sym_coef > 0:
            if mirror_maps is None:
                raise ValueError("sym_coef > 0 needs mirror_maps from symmetry.policy_mirror_maps")
            obs_perm, obs_sign, act_perm, act_sign = mirror_maps
            self.mirror = (
                torch.as_tensor(obs_perm, dtype=torch.int64, device=dev),
                torch.as_tensor(obs_sign, dtype=torch.float32, device=dev),
                torch.as_tensor(act_perm, dtype=torch.int64, device=dev),
                torch.as_tensor(act_sign, dtype=torch.float32, device=dev),
            )

        self.step = 0
        self.it = 0
        self._obs = None
        self.ep_ret = torch.zeros(self.N, device=dev)
        self.ep_len = torch.zeros(self.N, device=dev)
        self._zero_acc()
        self._losses = {k: torch.zeros((), device=dev) for k in LOSS_KEYS}

    def _params(self):
        return list(self.policy.parameters()) + list(self.value.parameters())

    def _zero_acc(self):
        dev = self.device
        self.acc = {k: torch.zeros((), device=dev) for k in _ACC_KEYS}
        self.acc_terms = {k: torch.zeros((), device=dev) for k in self.env.term_names}

    def load_params(self, params, grow_state=0, grow_priv=0):
        """Load a Brax (normalizer, policy, value) tuple, from a checkpoint or init_from_brax."""
        self.norm_s, self.norm_p = policy_io.from_brax(params, self.policy, self.value, grow_state, grow_priv)

    def reset(self):
        """A full env reset. The episode accumulators keep their sums."""
        self._obs = self.env.reset()
        self.ep_ret.zero_()
        self.ep_len.zero_()

    @property
    def train_policy(self) -> bool:
        return self.it > self.critic_warmup_iters

    # -- one iteration ----------------------------------------------------------
    def _collect(self):
        env, dev, gen = self.env, self.device, self.gen
        U, T, N = self.n_unrolls, self.T, self.N
        buf_s = torch.empty(U, T, N, self.S, device=dev)
        buf_p = torch.empty(U, T, N, self.P, device=dev)
        buf_raw = torch.empty(U, T, N, self.A, device=dev)
        buf_lp = torch.empty(U, T, N, device=dev)
        buf_r = torch.empty(U, T, N, device=dev)
        buf_d = torch.empty(U, T, N, device=dev)
        buf_tr = torch.empty(U, T, N, device=dev)
        last_p = torch.empty(U, N, self.P, device=dev)
        acc, acc_terms = self.acc, self.acc_terms
        S, P = self._obs
        with torch.no_grad():
            for u in range(U):
                for t in range(T):
                    loc, scale = dist_params(self.policy(self.norm_s(S)))
                    raw = loc + scale * torch.randn(loc.shape, generator=gen, device=dev)
                    buf_s[u, t], buf_p[u, t], buf_raw[u, t] = S, P, raw
                    buf_lp[u, t] = log_prob(loc, scale, raw)
                    S, P, r, d, tr, terms = env.step(torch.tanh(raw))
                    df = d.float()
                    buf_r[u, t], buf_d[u, t], buf_tr[u, t] = r, df, tr.float()
                    self.ep_ret += r
                    self.ep_len += 1
                    acc["ret"] += (self.ep_ret * df).sum()
                    acc["len"] += (self.ep_len * df).sum()
                    acc["n"] += df.sum()
                    acc["falls"] += (d & ~tr).float().sum()
                    acc["rew"] += r.mean()
                    acc["steps"] += 1
                    self.ep_ret *= 1 - df
                    self.ep_len *= 1 - df
                    for k in acc_terms:
                        acc_terms[k] += terms[k].mean()
                last_p[u] = P
        self._obs = (S, P)
        return buf_s, buf_p, buf_raw, buf_lp, buf_r, buf_d, buf_tr, last_p

    def _gae(self, v, v_boot, rew, dn, trn):
        """brax compute_gae, time-major semantics on [b, T]. Returns (vs, advantages)."""
        gamma, lam, T = self.gamma, self.lam, self.T
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
        return vs, adv

    def iteration(self) -> dict:
        """Collect n_unrolls x T steps, then run the PPO update.

        Returns this iteration's mean losses as 0-d device tensors. Nothing
        here moves a value to the host.
        """
        if self._obs is None:
            self.reset()
        buf_s, buf_p, buf_raw, buf_lp, buf_r, buf_d, buf_tr, last_p = self._collect()
        self.step += self.steps_per_iter
        self.it += 1

        U, T, N = self.n_unrolls, self.T, self.N
        B = U * N

        def traj(x):  # [U, T, N, ...] -> [U*N trajectories, T, ...]
            return x.transpose(1, 2).reshape(B, T, *x.shape[3:])

        Ds, Dp, Draw, Dlp, Dr, Dd, Dtr = map(traj, (buf_s, buf_p, buf_raw, buf_lp, buf_r, buf_d, buf_tr))
        Lp = last_p.reshape(B, self.P)
        # brax updates the normalizer with the batch before the SGD epochs.
        self.norm_s.update(buf_s)
        self.norm_p.update(buf_p)

        train_policy = self.train_policy
        stats = {k: torch.zeros((), device=self.device) for k in LOSS_KEYS}
        mb = B // self.n_minibatches
        nb = 0
        for _ in range(self.epochs):
            perm = torch.randperm(B, generator=self.gen, device=self.device)
            for k in range(self.n_minibatches):
                idx = perm[k * mb : (k + 1) * mb]
                s, p, raw, lp_old = Ds[idx], Dp[idx], Draw[idx], Dlp[idx]
                loc, scale = dist_params(self.policy(self.norm_s(s)))
                v = self.value(self.norm_p(p)).squeeze(-1)  # [b, T]
                v_boot = self.value(self.norm_p(Lp[idx])).squeeze(-1)  # [b]
                with torch.no_grad():
                    vs, adv = self._gae(v, v_boot, Dr[idx], Dd[idx], Dtr[idx])
                    adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
                lp_new = log_prob(loc, scale, raw)
                ratio = torch.exp(lp_new - lp_old)
                pl = -torch.min(ratio * adv, ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * adv).mean()
                vl = ((vs - v) ** 2).mean() * 0.5 * self.vf_coef
                ent = entropy(loc, scale, self.gen).mean()
                loss = vl + ((pl - self.ent_cost * ent) if train_policy else 0.0)
                if self.mirror is not None and train_policy:
                    obs_perm, obs_sign, act_perm, act_sign = self.mirror
                    loc_m = dist_params(self.policy(self.norm_s(s[..., obs_perm] * obs_sign)))[0]
                    sym = ((loc_m - loc[..., act_perm] * act_sign) ** 2).mean()
                    loss = loss + self.sym_coef * sym
                    stats["sym_loss"] += sym.detach()
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                if not train_policy:
                    for prm in self.policy.parameters():
                        prm.grad = None
                if self.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self._params(), self.max_grad_norm)
                self.opt.step()
                with torch.no_grad():
                    stats["policy_loss"] += pl.detach()
                    stats["v_loss"] += vl.detach()
                    stats["entropy"] += ent.detach()
                    log_ratio = (lp_new - lp_old).detach().clamp(-20, 20)
                    stats["approx_kl"] += ((log_ratio.exp() - 1) - log_ratio).mean()
                    stats["clipfrac"] += ((ratio.detach() - 1).abs() > self.clip_eps).float().mean()
                nb += 1

        # brax resets the envs once per training epoch, and only when
        # num_resets_per_eval > 0.
        if self.resets and self.it % self.iters_per_reset == 0:
            self.reset()

        self._losses = {k: v / nb for k, v in stats.items()}
        return dict(self._losses)

    def read_stats(self) -> dict:
        """Host values of the last losses and the episode stats since the last read.

        This is the only place values move to the host. It zeroes the
        episode accumulators.
        """
        a = {k: float(v) for k, v in self.acc.items()}
        n_steps = max(a["steps"], 1.0)
        rec = {
            # Mean reward per control step x 1000: comparable to brax's
            # eval/episode_reward for a policy that does not fall.
            "reward_x1000": a["rew"] / n_steps * 1000.0,
            "falls_per_1M": a["falls"] / (n_steps * self.N) * 1e6,
            "done_ep_len": a["len"] / a["n"] if a["n"] else None,
            **{k: float(v) for k, v in self._losses.items()},
            "terms": {k: float(v) / n_steps for k, v in self.acc_terms.items()},
        }
        self._zero_acc()
        return rec

    # -- state ------------------------------------------------------------------
    def state_dict(self) -> dict:
        return {
            "policy": self.policy.state_dict(),
            "value": self.value.state_dict(),
            "opt": self.opt.state_dict(),
            "norm_s": self.norm_s.state_dict(),
            "norm_p": self.norm_p.state_dict(),
            "step": int(self.step),
            "it": int(self.it),
            "gen": self.gen.get_state(),
        }

    def load_state_dict(self, state) -> None:
        self.policy.load_state_dict(state["policy"])
        self.value.load_state_dict(state["value"])
        self.opt.load_state_dict(state["opt"])
        self.norm_s.load_state_dict(state["norm_s"])
        self.norm_p.load_state_dict(state["norm_p"])
        self.step = int(state["step"])
        self.it = int(state["it"])
        self.gen.set_state(state["gen"].cpu())


# -- the training run -----------------------------------------------------------
# Composed keys a resume may change. torch_warp.resume always differs.
RESUME_MAY_CHANGE = frozenset(
    {"ppo.num_timesteps", "torch_warp.max_minutes", "torch_warp.ckpt_minutes", "torch_warp.resume"}
)
TRAINER_NAME = "torch_warp_ppo"
STATE_FILE = "trainer_state.pt"
LOG_FILE = "train_log.jsonl"
LOG_EVERY = 5

# Steps of each probe phase: zero actions, then uniform random actions. It
# mirrors train.PREFLIGHT_STEPS. humanoid_lab.train is not imported, because
# it sets XLA environment variables at import.
PROBE_STEPS = 100


def _leaves(tree, prefix: str = "") -> dict:
    """Every leaf of a nested dict as {dotted key: value}. Lists are leaves."""
    out = {}
    if isinstance(tree, Mapping) and tree:
        for key, value in tree.items():
            out.update(_leaves(value, f"{prefix}{key}."))
    else:
        out[prefix[:-1]] = list(tree) if isinstance(tree, (list, tuple)) else tree
    return out


def _resume_view(hydra: dict, run_name: str) -> dict:
    from humanoid_lab import run_record

    view = copy.deepcopy(hydra)
    if view.get("restore"):
        view["restore"] = run_record.portable_path(view["restore"])
    # A run started without run_name was named by the trainer, and its
    # resume names it explicitly.
    view["run_name"] = view.get("run_name") or run_name
    return view


def resume_record(run_dir: Path, hydra: dict) -> dict:
    """The run.json of the Torch run to resume, after every resume check.

    Raises when the run has no run.json or trainer_state.pt, when another
    trainer wrote it, or when the composed config differs from its
    hydra_config outside RESUME_MAY_CHANGE.
    """
    from humanoid_lab import run_record

    for name in ("run.json", STATE_FILE):
        if not (run_dir / name).exists():
            raise FileNotFoundError(f"torch_warp.resume=true: {run_dir / name} not found")
    run = run_record.load_run(run_dir)
    trainer = run.get("trainer")
    if not (isinstance(trainer, Mapping) and trainer.get("name") == TRAINER_NAME):
        name = trainer.get("name") if isinstance(trainer, Mapping) else trainer
        raise ValueError(
            f"torch_warp.resume=true: {run_dir} was written by {name!r}, not by {TRAINER_NAME}. "
            f"Only a {TRAINER_NAME} run can be resumed. To continue this run, warm-start from its "
            f"last checkpoint with restore={run_dir.name}/checkpoints/<step> and a new run_name."
        )
    old = _leaves(_resume_view(run["hydra_config"], run["run_name"]))
    new = _leaves(_resume_view(hydra, run["run_name"]))
    changed = sorted(k for k in old.keys() | new.keys() if old.get(k) != new.get(k) or (k in old) != (k in new))
    changed = [k for k in changed if k not in RESUME_MAY_CHANGE]
    if changed:
        raise ValueError(
            f"torch_warp.resume=true: the composed config differs from {run_dir / 'run.json'} at "
            + ", ".join(f"{k} ({old.get(k)!r} -> {new.get(k)!r})" for k in changed)
            + f". A resume may change only {sorted(RESUME_MAY_CHANGE)}."
        )
    return run


def probe_contacts(hydra: dict, device, episode_length: int, seed: int) -> tuple[int, int]:
    """(nacon peak, nefc peak) of a 1-world MJWarp probe.

    PROBE_STEPS zero actions, then PROBE_STEPS uniform(-1, 1) actions from a
    torch generator seeded with seed. DR is off in the probe.
    """
    from humanoid_lab.torch_warp.env import WarpJoystick

    probe = {**hydra, "domain_rand": False, "dr": {}}
    env = WarpJoystick(probe, 1, episode_length, device=device, seed=seed)
    gen = torch.Generator(device=env.device)
    gen.manual_seed(int(seed))
    env.reset()
    nacon_peak, nefc_peak = 0, 0
    zeros = torch.zeros(1, env.action_size, device=env.device)
    for i in range(2 * PROBE_STEPS):
        if i < PROBE_STEPS:
            action = zeros
        else:
            action = torch.rand(1, env.action_size, generator=gen, device=env.device) * 2.0 - 1.0
        env.step(action)
        nacon, nefc = env.physics.peaks()
        nacon_peak, nefc_peak = max(nacon_peak, nacon), max(nefc_peak, nefc)
    return nacon_peak, nefc_peak


def memory_mb(env) -> dict:
    """Process and device memory in MB.

    rss always. private only where psutil has it (Windows). On CUDA, torch's
    peak allocation and warp's mempool high-water mark.
    """
    import psutil

    mb = 2**20
    info = psutil.Process().memory_info()
    out = {"rss": round(info.rss / mb)}
    if hasattr(info, "private"):
        out["private"] = round(info.private / mb)
    device = torch.device(env.device)
    if device.type == "cuda":
        import warp as wp

        index = device.index if device.index is not None else torch.cuda.current_device()
        out["torch_max_allocated"] = round(torch.cuda.max_memory_allocated(device) / mb)
        out["warp_mempool_high"] = round(wp.get_mempool_used_mem_high(wp.get_device(f"cuda:{index}")) / mb)
    return out


def _wandb_group(cfg) -> str | None:
    # train._wandb_group's rule. train is not imported (see PROBE_STEPS).
    group = cfg.wandb.get("group")
    if group:
        return str(group)
    from humanoid_lab.ppo_config import experiment_name

    return experiment_name()


def _wandb_flat(rec: dict) -> dict:
    out = {}
    for key, value in rec.items():
        if isinstance(value, Mapping):
            out.update({f"{key}/{k}": v for k, v in value.items()})
        elif value is not None:
            out[key] = value
    return out


def _round(v, digits):
    return None if v is None else round(v, digits)


@hydra.main(version_base=None, config_path="../../../configs", config_name="torch_warp")
def main(cfg: DictConfig) -> None:
    from humanoid_lab import paths, run_record, sim_budget
    from humanoid_lab import policy_io as brax_policy_io
    from humanoid_lab.envs import symmetry
    from humanoid_lab.ppo_config import (
        brax_train_kwargs,
        compose_ppo_params,
        resolve_episode_length,
    )
    from humanoid_lab.torch_warp.env import WarpJoystick, env_config_of

    # -- config ------------------------------------------------------------------
    hydra_cfg = OmegaConf.to_container(cfg, resolve=True)
    ppo_params = compose_ppo_params(cfg)
    ignored = check_config(hydra_cfg, ppo_params.to_dict())
    _training, net_kwargs, repo = brax_train_kwargs(ppo_params)
    net_kwargs = net_kwargs.to_dict() if hasattr(net_kwargs, "to_dict") else dict(net_kwargs)
    sym_coef = float(repo.get("sym_coef", 0.0))
    resolve_episode_length(ppo_params, cfg, env_config_of(hydra_cfg).episode_length)
    episode_length = int(ppo_params.episode_length)
    num_timesteps = int(ppo_params.num_timesteps)
    tw = cfg.torch_warp
    for key, reason in ignored.items():
        print(f"ignored {key}: {reason}")

    # -- run dir -----------------------------------------------------------------
    task = cfg.task.name
    run_name = cfg.run_name or (
        ("smoke_" if cfg.smoke else "") + f"{task}_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    run_dir = paths.RUNS_DIR / run_name
    ckpt_root = run_dir / "checkpoints"
    state_file = run_dir / STATE_FILE
    previous = None
    if tw.resume:
        previous = resume_record(run_dir, hydra_cfg)
        (run_dir / "DONE").unlink(missing_ok=True)
    elif (run_dir / "run.json").exists():
        raise FileExistsError(
            f"{run_dir / 'run.json'} exists. Continue it with torch_warp.resume=true, or pick another run_name."
        )
    ckpt_root.mkdir(parents=True, exist_ok=True)

    # -- env and loop ------------------------------------------------------------
    env = WarpJoystick(hydra_cfg, int(ppo_params.num_envs), episode_length, device=tw.device, seed=cfg.seed)
    print(f"actor obs ({len(env.actor_obs_names)} components): {env.actor_obs_names}")
    mirror_maps = None
    if sym_coef > 0:
        maps = symmetry.policy_mirror_maps(env.robot_spec, env.actor_obs_names)
        mirror_maps = tuple(torch.as_tensor(np.asarray(m), device=env.device) for m in maps)
        print(f"symmetry loss on, coef {sym_coef}")
    trainer = Trainer(
        env,
        ppo_params,
        sym_coef=sym_coef,
        mirror_maps=mirror_maps,
        critic_warmup_iters=int(tw.critic_warmup_iters),
        seed=int(cfg.seed),
    )

    # -- initial weights ---------------------------------------------------------
    warm_start = None
    if previous is not None:
        state = policy_io.load_trainer_state(state_file, env.device)
        trainer.load_state_dict(state)
        # torch.Generator.set_state takes a CPU tensor.
        draws = {k: v.cpu() for k, v in state["draws"]["draws"].items()}
        env.load_state_dict({"draws": draws})
        warm_start = previous.get("warm_start")
        # The env restarts with env.reset(), so a resume is not bit-exact.
        print(f"resumed {state_file} at step {trainer.step:,} it {trainer.it}")
    elif cfg.restore:
        path = run_record.resolve_path(cfg.restore)
        policy_io.check_network_kwargs(path, net_kwargs)
        warm_start = run_record.warm_start_block(cfg.restore, hydra_cfg, strict=True)
        print(f"warm start from {warm_start['checkpoint']}, config diff against {warm_start['source_run']}:")
        for key, (old, new) in warm_start["config_diff"].items():
            print(f"  {key}: {old!r} -> {new!r}")
        source = run_record.load_run(Path(path).resolve().parent.parent)
        old_state, old_priv = policy_io.source_obs_layout(source)
        grow_s = policy_io.check_layout(old_state, env.actor_obs_names, env.sizes, "obs.state")
        grow_p = policy_io.check_layout(old_priv, env.priv_obs_names, env.sizes, "obs.privileged")
        params = brax_policy_io.load_params(path)
        for key, names in ((policy_io.STATE, old_state), (policy_io.PRIVILEGED, old_priv)):
            width = int(np.shape(params[0].mean[key])[-1])
            expected = sum(env.sizes[n] for n in names)
            if width != expected:
                raise ValueError(
                    f"{path}: the {key} normalizer has {width} columns, and the source run's "
                    f"observation list {names} has width {expected}"
                )
        trainer.load_params(params, grow_s, grow_p)
    else:
        trainer.load_params(
            policy_io.init_from_brax(cfg.seed, env.actor_obs_size, env.privileged_obs_size, env.action_size, net_kwargs)
        )

    # -- contacts ----------------------------------------------------------------
    nacon_peak = nefc_peak = None
    if not cfg.smoke and cfg.get("contact_preflight", True):
        nacon_peak, nefc_peak = probe_contacts(hydra_cfg, tw.device, episode_length, int(cfg.seed))
    consts = env.consts_np
    contacts = sim_budget.budget_report(
        "warp", nacon_peak, nefc_peak, consts.naconmax_per_env, consts.njmax, env.num_envs
    )
    # Recorded, never refused on, as in the Brax trainer.
    print(f"contacts: {contacts}")
    if contacts["overflow"] or contacts["rows_overflow"]:
        print(
            "WARNING: the probe already reached a warp buffer ceiling. Warp drops the overflow "
            "silently. Raise task.env.sim.naconmax_per_env / njmax (see ./run.sh check-contacts)."
        )

    # -- run.json at start -------------------------------------------------------
    resumes = list(((previous or {}).get("trainer") or {}).get("resumes") or [])
    if previous is not None:
        resumes.append({"step": int(trainer.step), "time": datetime.now().isoformat(timespec="seconds")})
    budget_peaks = {"nacon_pool_max": None, "nefc_max": None}
    trainer_info = run_record.trainer_block(
        TRAINER_NAME,
        "warp",
        env.device,
        torch_warp=dict(hydra_cfg["torch_warp"]),
        constants_env_overrides=dict(env.constants_env_overrides),
        ignored=ignored,
        budget_peaks=budget_peaks,
        train_reward_last=None,
        resumes=resumes,
    )
    gains = run_record.actuator_gains(env, hydra_cfg)

    def record(results):
        return run_record.build_run_record(
            env,
            hydra_cfg,
            ppo_params,
            run_name=run_name,
            checkpoint_dir=ckpt_root,
            contacts=contacts,
            env_config=env.env_config,
            actuator_gains=gains,
            trainer=trainer_info,
            results=results,
            warm_start=warm_start,
        )

    # jobs/video_watch.sh starts rendering once run.json exists.
    run_record.write_run(run_dir, record({"early_stopped": False, "stopped_at_steps": None, "final_reward": None}))
    (run_dir / "trainer.pid").write_text(str(os.getpid()), encoding="utf-8")  # the real process

    wb = None
    if cfg.wandb.enable:
        try:
            import wandb

            wb = wandb.init(
                project=cfg.wandb.project,
                name=run_name,
                group=_wandb_group(cfg),
                config={"hydra": hydra_cfg, "ppo": ppo_params.to_dict(), "env": env.env_config},
                resume="allow" if previous is not None else None,
            )
        except Exception as e:  # noqa: BLE001
            print(f"wandb disabled: {e}")

    # -- checkpoints -------------------------------------------------------------
    def checkpoint(step):
        # save_brax refuses an existing step dir (see policy_io.save_brax).
        # brax writes the network config json last, so a step dir without it
        # is a save that was cut off. No reader opens such a dir, so it is
        # removed and saved again.
        step_dir = ckpt_root / f"{int(step):012d}"
        if step_dir.exists() and not (step_dir / policy_io.NETWORK_CONFIG).exists():
            shutil.rmtree(step_dir)
        if not step_dir.exists():
            path = policy_io.save_brax(
                ckpt_root,
                step,
                trainer.norm_s,
                trainer.norm_p,
                trainer.policy,
                trainer.value,
                net_kwargs,
                env.action_size,
            )
            print(f"checkpoint {path}")
        policy_io.save_trainer_state(state_file, {**trainer.state_dict(), "draws": env.state_dict()})

    if previous is None:
        checkpoint(0)  # the starting policy: the first video is the baseline
    else:
        # A kill during a save at step X leaves the trainer state at an
        # earlier step and the X dir without its network config. The resumed
        # run may never save at X again, so every such dir is removed here.
        for step_dir in sorted(ckpt_root.iterdir()):
            if step_dir.is_dir() and step_dir.name.isdigit() and not (step_dir / policy_io.NETWORK_CONFIG).exists():
                shutil.rmtree(step_dir)
                print(f"removed the cut-off checkpoint at step {int(step_dir.name):,}")

    print(
        f"N={env.num_envs} unroll={trainer.T} unrolls={trainer.n_unrolls} minibatches={trainer.n_minibatches} "
        f"epochs={trainer.epochs} lr={trainer.lr} steps/iter={trainer.steps_per_iter:,} -> {run_dir}"
    )

    # -- loop --------------------------------------------------------------------
    t_start = t_last_ckpt = t_last_log = time.time()
    steps_last_log = trainer.step
    first_it = trainer.it + 1
    early_stopped = False
    reward_last = None
    with open(run_dir / LOG_FILE, "a", encoding="utf-8") as log_f:
        try:
            while trainer.step < num_timesteps:
                trainer.iteration()
                now = time.time()
                # The first iteration of this process (it 1 on a fresh run)
                # and every LOG_EVERY iterations.
                if trainer.it == first_it or trainer.it % LOG_EVERY == 0:
                    sps = (trainer.step - steps_last_log) / max(now - t_last_log, 1e-9)
                    t_last_log, steps_last_log = now, trainer.step
                    stats = trainer.read_stats()
                    nacon, nefc = env.physics.peaks()
                    for key, value in (("nacon_pool_max", nacon), ("nefc_max", nefc)):
                        budget_peaks[key] = value if budget_peaks[key] is None else max(budget_peaks[key], value)
                    reward_last = stats["reward_x1000"] / 1000.0
                    rec = {
                        "it": trainer.it,
                        "steps": trainer.step,
                        "sps": round(sps),
                        "minutes": round((now - t_start) / 60, 2),
                        "reward_x1000": round(stats["reward_x1000"], 3),
                        "falls_per_1M": round(stats["falls_per_1M"], 2),
                        "done_ep_len": _round(stats["done_ep_len"], 1),
                        "losses": {k: round(stats[k], 5) for k in LOSS_KEYS},
                        "terms": {k: round(v, 5) for k, v in stats["terms"].items()},
                        "memory": memory_mb(env),
                        "budget": {"nacon_pool_max": nacon, "nefc_max": nefc},
                    }
                    log_f.write(json.dumps(rec) + "\n")
                    log_f.flush()
                    if wb is not None:
                        wb.log(_wandb_flat(rec), step=trainer.step)
                    losses = rec["losses"]
                    print(
                        f"it {trainer.it:5d}  steps {trainer.step:>12,}  reward/1000 steps {rec['reward_x1000']:7.2f}  "
                        f"falls/1M {rec['falls_per_1M']:7.1f}  v_loss {losses['v_loss']:.4f}  "
                        f"kl {losses['approx_kl']:.4f}  "
                        + (f"sym {losses['sym_loss']:.4f}  " if sym_coef > 0 else "")
                        + f"ent {losses['entropy']:.2f}  {sps:,.0f} steps/s  rss {rec['memory']['rss']} MB"
                        + ("" if trainer.train_policy else "  [critic warm-up]"),
                        flush=True,
                    )
                if tw.max_minutes is not None and now - t_start >= float(tw.max_minutes) * 60:
                    checkpoint(trainer.step)
                    early_stopped = True
                    print(f"max_minutes {tw.max_minutes} reached at step {trainer.step:,}")
                    break
                if now - t_last_ckpt >= float(tw.ckpt_minutes) * 60:
                    checkpoint(trainer.step)
                    t_last_ckpt = now
        except KeyboardInterrupt:
            # Ctrl+C, on Windows too. The final checkpoint below saves the run.
            early_stopped = True
            print(f"interrupted at step {trainer.step:,}")

    # -- end ---------------------------------------------------------------------
    checkpoint(trainer.step)
    trainer_info["budget_peaks"] = budget_peaks
    trainer_info["train_reward_last"] = reward_last
    run_record.write_run(
        run_dir,
        record({"early_stopped": early_stopped, "stopped_at_steps": int(trainer.step), "final_reward": None}),
    )
    (run_dir / "DONE").write_text(str(trainer.step), encoding="utf-8")
    print(f"done -> {run_dir}")
    if wb is not None:
        wb.finish()


if __name__ == "__main__":
    main()
