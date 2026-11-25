"""Torch networks, the observation normalizer and Brax checkpoint conversion.

The Torch trainer reads and writes the Brax PPO checkpoint format. A
checkpoint it writes loads through humanoid_lab.policy_io, the battery and
the exporter like one from the Brax trainer. The networks are brax's MLPs
and the normalizer is brax's running_statistics in Welford mode.

brax, jax and humanoid_lab.registry are imported inside the functions that
need them, so the training loop pays for them only at load and save.
"""

from __future__ import annotations

import functools
import json
import os
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
from torch import nn

STATE = "state"
PRIVILEGED = "privileged_state"
NETWORK_CONFIG = "ppo_network_config.json"

# The Torch networks implement only these two.
ACTIVATIONS = ("silu", "swish")
DISTRIBUTION = "tanh_normal"

# A grown normalizer column starts with std 0.2.
GROWN_VAR = 0.04


# -- networks -----------------------------------------------------------------
def mlp(sizes) -> nn.Sequential:
    """brax's MLP with activate_final False: Linear layers with SiLU between them."""
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.SiLU())
    return nn.Sequential(*layers)


def _linears(net: nn.Sequential):
    return [m for m in net if isinstance(m, nn.Linear)]


class Normalizer:
    """brax running_statistics in Welford mode with std_eps 0.

    mean and svar are float32 tensors. count is a Python int, so it holds
    brax's UInt64 count exactly. At count 0 the std is 1, as brax's
    init_state gives.
    """

    def __init__(self, mean, svar, count: int = 0, device=None):
        self.mean = torch.as_tensor(np.array(mean, np.float32), device=device)
        self.svar = torch.as_tensor(np.array(svar, np.float32), device=device)
        self.count = int(count)
        self._refresh()

    @classmethod
    def zeros(cls, size: int, device=None) -> Normalizer:
        z = np.zeros(size, np.float32)
        return cls(z, z, 0, device)

    @property
    def size(self) -> int:
        return int(self.mean.shape[0])

    def count_f32(self) -> np.float32:
        """brax's UInt64-to-float32 rule: hi * 2**32 + lo, in float32."""
        c = self.count
        return np.float32(np.float32(c >> 32) * np.float32(2.0**32) + np.float32(c & 0xFFFFFFFF))

    def _refresh(self):
        if self.count == 0:
            self.std = torch.ones_like(self.mean)
        else:
            self.std = torch.sqrt(self.svar.clamp(min=0) / self.count_f32()).clamp(1e-6, 1e6)

    def update(self, x):
        x = x.reshape(-1, self.size)
        self.count += int(x.shape[0])
        cf = self.count_f32()
        diff = x - self.mean
        self.mean = self.mean + diff.sum(0) / cf
        self.svar = self.svar + (diff * (x - self.mean)).sum(0)
        self._refresh()

    def __call__(self, x, mean=None, std=None):
        return (x - (self.mean if mean is None else mean)) / (self.std if std is None else std)

    def grow(self, k: int):
        """Append k columns with mean 0 and svar 0.04 * count, so their std starts at 0.2.

        At count 0 the new columns have std 1 like the rest.
        """
        if k <= 0:
            return
        dev = self.mean.device
        self.mean = torch.cat([self.mean, torch.zeros(k, dtype=torch.float32, device=dev)])
        new_svar = torch.full((k,), float(np.float32(GROWN_VAR) * self.count_f32()), dtype=torch.float32, device=dev)
        self.svar = torch.cat([self.svar, new_svar])
        self._refresh()

    def to(self, device) -> Normalizer:
        self.mean = self.mean.to(device)
        self.svar = self.svar.to(device)
        self._refresh()
        return self

    def state_dict(self) -> dict:
        return {"mean": self.mean.detach().clone(), "svar": self.svar.detach().clone(), "count": self.count}

    def load_state_dict(self, state: Mapping):
        dev = self.mean.device
        self.mean = torch.as_tensor(state["mean"], dtype=torch.float32, device=dev).clone()
        self.svar = torch.as_tensor(state["svar"], dtype=torch.float32, device=dev).clone()
        self.count = int(state["count"])
        self._refresh()


# -- Brax conversion ----------------------------------------------------------
def _count(rs) -> int:
    return int(np.asarray(rs.count.hi)) * 2**32 + int(np.asarray(rs.count.lo))


def _load_linears(flax_params, net: nn.Sequential, label: str):
    layers = flax_params["params"]
    linears = _linears(net)
    if len(layers) != len(linears):
        raise ValueError(f"{label}: the checkpoint has {len(layers)} layers, the Torch network {len(linears)}")
    for i, layer in enumerate(linears):
        w = torch.from_numpy(np.array(layers[f"hidden_{i}"]["kernel"], np.float32).T.copy())
        b = torch.from_numpy(np.array(layers[f"hidden_{i}"]["bias"], np.float32))
        with torch.no_grad():
            if w.shape == layer.weight.shape:
                layer.weight.copy_(w)
            elif i == 0 and w.shape[0] == layer.weight.shape[0] and w.shape[1] < layer.weight.shape[1]:
                # Input columns beyond the checkpoint's width start at zero,
                # so the policy begins unchanged.
                layer.weight.zero_()
                layer.weight[:, : w.shape[1]] = w
            else:
                raise ValueError(
                    f"{label} layer {i}: checkpoint kernel {tuple(w.shape)} does not fit "
                    f"the Torch layer {tuple(layer.weight.shape)}"
                )
            layer.bias.copy_(b)


def from_brax(params, policy: nn.Sequential, value: nn.Sequential, grow_state: int = 0, grow_priv: int = 0):
    """Load a Brax (normalizer, policy, value) tuple into Torch networks.

    params comes from humanoid_lab.policy_io.load_params or init_from_brax.
    Returns (norm_s, norm_p), each grown by grow_state and grow_priv columns.
    """
    rs = params[0]
    count = _count(rs)
    _load_linears(params[1], policy, "policy")
    _load_linears(params[2], value, "value")
    device = _linears(policy)[0].weight.device
    norm_s = Normalizer(rs.mean[STATE], rs.summed_variance[STATE], count, device)
    norm_p = Normalizer(rs.mean[PRIVILEGED], rs.summed_variance[PRIVILEGED], count, device)
    norm_s.grow(grow_state)
    norm_p.grow(grow_priv)
    return norm_s, norm_p


def to_brax(net: nn.Sequential) -> dict:
    """Torch Linear layers as a flax params dict of float32 numpy arrays."""
    return {
        "params": {
            f"hidden_{i}": {
                "kernel": layer.weight.detach().cpu().numpy().astype(np.float32).T.copy(),
                "bias": layer.bias.detach().cpu().numpy().astype(np.float32).copy(),
            }
            for i, layer in enumerate(_linears(net))
        }
    }


def _obs_size(state_size: int, priv_size: int) -> dict:
    return {STATE: (int(state_size),), PRIVILEGED: (int(priv_size),)}


def init_from_brax(seed: int, state_size: int, priv_size: int, action_size: int, network_factory_kwargs):
    """The same weights Brax starts from at that seed.

    Builds the network with make_ppo_networks on CPU JAX and draws the keys
    as brax ppo.train does. The normalizer is running_statistics.init_state.
    Returns a (normalizer, policy, value) tuple shaped like load_params.
    """
    import jax
    import jax.numpy as jnp
    from brax.training.acme import running_statistics, specs
    from brax.training.agents.ppo import networks as ppo_networks

    obs_size = _obs_size(state_size, priv_size)
    with jax.default_device(jax.devices("cpu")[0]):
        net = ppo_networks.make_ppo_networks(
            obs_size,
            int(action_size),
            preprocess_observations_fn=running_statistics.normalize,
            **dict(network_factory_kwargs),
        )
        key = jax.random.PRNGKey(int(seed))
        global_key, _ = jax.random.split(key)
        key_policy, key_value = jax.random.split(global_key)
        policy = net.policy_network.init(key_policy)
        value = net.value_network.init(key_value)
        normalizer = running_statistics.init_state(
            {k: specs.Array(v, jnp.dtype("float32")) for k, v in obs_size.items()}
        )
    return normalizer, jax.device_get(policy), jax.device_get(value)


# -- layout checks ------------------------------------------------------------
def source_obs_layout(run) -> tuple[list[str], list[str]]:
    """The (state, privileged) obs name lists of a source run. Builds no model."""
    from humanoid_lab import registry

    cfg = registry.env_config(run["task"], run["hydra_config"]["task"]["env"])
    return list(cfg.obs.state), list(cfg.obs.privileged)


def check_layout(old_names, new_names, sizes, label: str) -> int:
    """The width of the names appended after old_names.

    new_names may only extend old_names at the end. A mid-list insertion
    would silently misalign weights and normalizer columns, so it raises
    ValueError, as do a removal, a reorder and a repeated name.
    """
    old, new = list(old_names), list(new_names)
    if len(set(new)) != len(new) or new[: len(old)] != old:
        raise ValueError(
            f"{label}: the new observation list must be the source list plus trailing names. "
            f"source {old}, new {new}"
        )
    return int(sum(sizes[n] for n in new[len(old) :]))


def _plain(v):
    if callable(v):
        return getattr(v, "__name__", v)
    if isinstance(v, Mapping):
        return {k: _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    return v


def check_network_kwargs(ckpt_dir, composed_kwargs):
    """Refuse a checkpoint whose network differs from the composed config.

    Every composed key must equal the checkpoint's value. The checkpoint's
    activation must be silu and its distribution tanh_normal, the only ones
    the Torch networks implement. Raises ValueError listing every mismatch.
    """
    path = Path(ckpt_dir) / NETWORK_CONFIG
    saved = json.loads(path.read_text(encoding="utf-8"))["network_factory_kwargs"]
    problems = []
    for key, value in dict(composed_kwargs).items():
        if key not in saved:
            problems.append(f"{key}: composed {_plain(value)!r}, checkpoint has no such key")
        elif _plain(value) != _plain(saved[key]):
            problems.append(f"{key}: composed {_plain(value)!r}, checkpoint {_plain(saved[key])!r}")
    if "activation" in saved and saved["activation"] not in ACTIVATIONS:
        problems.append(f"activation: checkpoint {saved['activation']!r}, the Torch trainer supports {ACTIVATIONS}")
    if "distribution_type" in saved and saved["distribution_type"] != DISTRIBUTION:
        problems.append(
            f"distribution_type: checkpoint {saved['distribution_type']!r}, the Torch trainer supports {DISTRIBUTION!r}"
        )
    if problems:
        raise ValueError(f"network config mismatch with {path}:\n  " + "\n  ".join(problems))


# -- checkpoint writes --------------------------------------------------------
def save_brax(ckpt_root, step: int, norm_s, norm_p, policy, value, network_factory_kwargs, action_size: int) -> Path:
    """Write a Brax PPO checkpoint at ckpt_root/<step:012d> with brax's own writer.

    brax writes ppo_network_config.json last, so its presence marks a
    complete checkpoint. The JSON carries the true observation_size.
    """
    from brax.training import types as brax_types
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import checkpoint as ppo_checkpoint
    from brax.training.agents.ppo import networks as ppo_networks

    ckpt_root = Path(ckpt_root).resolve()
    path = ckpt_root / f"{int(step):012d}"
    if path.exists():
        # brax's orbax save with force=True would delete it first, which
        # fails on Windows while a reader holds one of its files.
        raise FileExistsError(f"checkpoint {path} already exists")
    if norm_s.count != norm_p.count:
        raise ValueError(f"the two normalizers disagree on count: {norm_s.count} and {norm_p.count}")

    def arr(t):
        return t.detach().cpu().numpy().astype(np.float32)

    count = norm_s.count
    rs = running_statistics.RunningStatisticsState(
        count=brax_types.UInt64(hi=np.uint32(count >> 32), lo=np.uint32(count & 0xFFFFFFFF)),
        mean={STATE: arr(norm_s.mean), PRIVILEGED: arr(norm_p.mean)},
        summed_variance={STATE: arr(norm_s.svar), PRIVILEGED: arr(norm_p.svar)},
        std={STATE: arr(norm_s.std), PRIVILEGED: arr(norm_p.std)},
    )
    config = ppo_checkpoint.network_config(
        observation_size=_obs_size(norm_s.size, norm_p.size),
        action_size=int(action_size),
        normalize_observations=True,
        network_factory=functools.partial(ppo_networks.make_ppo_networks, **dict(network_factory_kwargs)),
    )
    ppo_checkpoint.save(ckpt_root, int(step), (rs, to_brax(policy), to_brax(value)), config)
    return path


def save_trainer_state(path, state):
    """torch.save to a .tmp file, then os.replace, so a reader never sees half a file."""
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_trainer_state(path, device):
    return torch.load(Path(path), map_location=device, weights_only=True)
