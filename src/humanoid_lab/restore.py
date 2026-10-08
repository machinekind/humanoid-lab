"""Warm start across critic layouts.

brax restores a checkpoint's normalizer and value network wholesale. A
checkpoint whose critic list differs from the env's therefore fails with a
shape error inside the first jitted training step. The terrain critic
appends `height_scan_clean` to the flat one: roboto_origin's critic is 111
wide on task=joystick and 209 on task=terrain. A checkpoint moves between
the two tasks in either direction.

The env's observation has two keys. obs.state fills `state` and
obs.privileged fills `privileged_state`. brax's normalizer holds
statistics for both, whichever network reads them. The policy reads the
key its network config names as `policy_obs_key`, and the value network
reads `value_obs_key`. Both runs' keys come from their network configs.

`plan_restore` compares the checkpoint's source run with the env and the
run's network keys. It picks `as_is` (brax's own restore) or `adapt`.
`adapt_params` rebuilds the `privileged_state` columns by component name.
The policy is never adapted: a change to the list it reads refuses.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
from brax.training import types as brax_types

from humanoid_lab.envs import height_scan

AS_IS = "as_is"
ADAPT = "adapt"
# The observation keys obs.state and obs.privileged fill.
STATE_KEY = "state"
PRIVILEGED_KEY = "privileged_state"
# The network keys train.py's build_ppo_params sets. A source run that
# records no keys used these.
POLICY_OBS_KEY = STATE_KEY
VALUE_OBS_KEY = PRIVILEGED_KEY
# The only normalizer mode a restore keeps. brax's checkpoint does not save
# RunningStatisticsState.mode, a non-pytree field, so its load rebuilds
# every normalizer in this mode.
WELFORD = "welford"
# (mean, std) written into an added critic component's normalizer columns.
# A component without an entry gets DEFAULT_PRIOR.
PRIORS = {height_scan.NAME: (height_scan.PRIOR_MEAN, height_scan.PRIOR_STD)}
DEFAULT_PRIOR = (0.0, 1.0)


@dataclasses.dataclass(frozen=True)
class RestorePlan:
    path: str
    action: str  # AS_IS | ADAPT
    source_run: str | None
    source_task: str | None
    added: tuple[str, ...]  # critic components only the env lists
    removed: tuple[str, ...]  # critic components only the source run lists
    source_arena: dict | None  # {fingerprint, generator_version} of the source run
    # (mean, std) of each added component's normalizer columns.
    priors: dict[str, tuple[float, float]] = dataclasses.field(default_factory=dict)
    value_obs_key: str = VALUE_OBS_KEY  # the key the value network reads
    restore_value: bool = True  # whether brax restores the value network

    def record(self) -> dict:
        return dataclasses.asdict(self)


def source_record(ckpt_dir) -> dict | None:
    """The source run's run.json, two levels above a checkpoint step
    directory (<run>/checkpoints/<step>), or None."""
    path = Path(ckpt_dir).resolve().parent.parent / "run.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def _checkpoint_config(ckpt_dir) -> dict:
    """The network config brax saved next to the params, or {}."""
    path = Path(ckpt_dir) / "ppo_network_config.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text())


def source_obs_keys(ckpt_dir, src: dict | None) -> tuple[str, str]:
    """(policy_obs_key, value_obs_key) of the run that wrote `ckpt_dir`.

    Read from the checkpoint's own network config, then from run.json's
    ppo_config. A key neither records is train.py's default."""
    kwargs = _checkpoint_config(ckpt_dir).get("network_factory_kwargs") or {}
    factory = ((src or {}).get("ppo_config") or {}).get("network_factory") or {}

    def key(name, default):
        return kwargs.get(name) or factory.get(name) or default

    return key("policy_obs_key", POLICY_OBS_KEY), key("value_obs_key", VALUE_OBS_KEY)


def _widths(params) -> tuple[int, int]:
    """(state, privileged_state) widths of a checkpoint's normalizer."""
    mean = params[0].mean
    return int(np.shape(mean[STATE_KEY])[-1]), int(np.shape(mean[PRIVILEGED_KEY])[-1])


def _shared_in_order(src: tuple[str, ...], dst: tuple[str, ...]) -> bool:
    """Whether the components both lists name appear in the same relative
    order in each."""
    shared = set(src) & set(dst)
    return [n for n in src if n in shared] == [n for n in dst if n in shared]


def _check_keys(path, src_keys, dst_keys, restore_value: bool) -> None:
    """Refuse a policy that reads another key, and a restored value
    network that reads another key."""
    (src_policy, src_value), (dst_policy, dst_value) = src_keys, dst_keys
    if src_policy != dst_policy:
        raise ValueError(
            f"checkpoint {path} has a policy that reads obs {src_policy!r}, and this run's "
            f"reads {dst_policy!r}. The policy is never adapted."
        )
    if restore_value and src_value != dst_value:
        raise ValueError(
            f"checkpoint {path} has a value network that reads obs {src_value!r}, and this "
            f"run's reads {dst_value!r}. restore_value=false starts a fresh critic instead."
        )


def plan_restore(ckpt_dir, params, env, *, policy_obs_key: str = POLICY_OBS_KEY,
                 value_obs_key: str = VALUE_OBS_KEY, restore_value: bool = True) -> RestorePlan:
    """How to restore `params`, loaded from `ckpt_dir`, into `env`.

    The keyword arguments are this run's network keys and its
    restore_value. The source's policy must read the same key, and so must
    its value network when restore_value is on.

    Without a source run.json the columns cannot be named, so equal widths
    restore as they are and anything else refuses. With one, a different
    robot or action size refuses, and so does a source normalizer mode
    other than Welford, or a different obs.state list. Equal obs.privileged
    lists restore as they are. Lists whose shared components keep their
    relative order adapt, unless the policy reads `privileged_state`.
    Anything else refuses. The checkpoint's widths must match the source
    run's lists."""
    cfg = env._config
    sizes = env.obs_component_sizes()
    dst_actor, dst_critic = tuple(cfg.obs.state), tuple(cfg.obs.privileged)
    want = (sum(sizes[n] for n in dst_actor), sum(sizes[n] for n in dst_critic))
    have = _widths(params)
    src = source_record(ckpt_dir)
    path = str(ckpt_dir)
    _check_keys(path, source_obs_keys(ckpt_dir, src), (policy_obs_key, value_obs_key), restore_value)
    network = {"value_obs_key": value_obs_key, "restore_value": restore_value}
    if src is None:
        if have != want:
            raise ValueError(
                f"checkpoint {path} has state/privileged_state widths {have}, and this env has "
                f"{want}. Its run.json is missing, so its columns cannot be mapped without the "
                "source run's obs lists."
            )
        return RestorePlan(path, AS_IS, None, None, (), (), None, **network)

    robot = (src.get("hydra_config") or {}).get("robot", {}).get("name")
    if robot is not None and robot != env.robot_spec.name:
        raise ValueError(
            f"checkpoint {path} is from robot {robot!r}, and this env is {env.robot_spec.name!r}"
        )
    action_size = _checkpoint_config(ckpt_dir).get("action_size")
    if action_size is not None and int(action_size) != env.action_size:
        raise ValueError(
            f"checkpoint {path} has action size {action_size}, and this env has {env.action_size}"
        )
    mode = (src.get("ppo_config") or {}).get("normalize_observations_mode") or WELFORD
    if mode != WELFORD:
        raise ValueError(
            f"checkpoint {path} was trained with normalize_observations_mode {mode!r}. brax's "
            "checkpoint does not save the normalizer's mode, and its load rebuilds a Welford "
            "normalizer, which would misread the saved statistics."
        )
    obs = src["env_config"]["obs"]
    src_actor, src_critic = tuple(obs["state"]), tuple(obs["privileged"])
    if src_actor != dst_actor:
        raise ValueError(
            f"checkpoint {path} has the actor list {list(src_actor)}, and this env has "
            f"{list(dst_actor)}. The actor is never adapted."
        )
    src_want = (sum(sizes[n] for n in src_actor), sum(sizes[n] for n in src_critic))
    if have != src_want:
        raise ValueError(
            f"checkpoint {path} has state/privileged_state widths {have}, but its run.json "
            f"lists components {src_want} wide"
        )
    arena = src.get("arena")
    source_arena = None if not arena else {
        "fingerprint": arena.get("fingerprint"),
        "generator_version": arena.get("generator_version"),
    }
    run = src.get("run_name")
    task = src.get("task")
    if src_critic == dst_critic:
        return RestorePlan(path, AS_IS, run, task, (), (), source_arena, **network)
    if policy_obs_key == PRIVILEGED_KEY:
        raise ValueError(
            f"checkpoint {path} has the critic list {list(src_critic)}, and this env has "
            f"{list(dst_critic)}. The policy reads obs 'privileged_state', and the policy is "
            "never adapted."
        )
    if not _shared_in_order(src_critic, dst_critic):
        raise ValueError(
            f"checkpoint {path} has the critic list {list(src_critic)}, and this env has "
            f"{list(dst_critic)}. Their shared components are in a different order."
        )
    added = tuple(n for n in dst_critic if n not in src_critic)
    removed = tuple(n for n in src_critic if n not in dst_critic)
    priors = {n: PRIORS.get(n, DEFAULT_PRIOR) for n in added}
    return RestorePlan(path, ADAPT, run, task, added, removed, source_arena, priors, **network)


def _count(count) -> float:
    if isinstance(count, brax_types.UInt64):
        return float(np.asarray(count.hi, np.float64)) * 2.0**32 + float(np.asarray(count.lo, np.float64))
    return float(np.asarray(count))


def _columns(names, sizes) -> dict[str, slice]:
    out, start = {}, 0
    for n in names:
        out[n] = slice(start, start + sizes[n])
        start += sizes[n]
    return out


def adapt_params(params, plan: RestorePlan, sizes: dict, src_priv, dst_priv):
    """`params` with the critic's columns rebuilt for `dst_priv`.

    The normalizer's `privileged_state` statistics are rebuilt by
    component, whichever network reads them:
    - A shared component copies its columns: mean, std, summed variance.
    - An added component gets its prior from plan.priors: a mean m and a
      std s. The prior is a fixed value per component (PRIORS), not a
      statistic of this run's data. A fresh brax normalizer would instead
      take its first batch's mean and std.
    - A removed component's columns are dropped.

    brax's Welford update reads std as sqrt(summed_variance / count +
    std_eps). One count serves every column, and it carries over from the
    source. It counts every sample the source's normalizer has seen. A
    warm-started source's count includes the runs it started from. The
    prior is therefore written as summed_variance = count * s^2. It weighs
    as much as every sample the source saw. After n more samples the prior
    keeps about the weight count / (count + n) in the column's mean and
    variance. restore_value=false keeps this normalizer too.

    The value network's first layer is rebuilt by component when
    plan.restore_value is on and the value network reads
    `privileged_state`. An added component's rows are zero, so at step 0
    the critic's output is the source critic's. A removed component's rows
    are dropped. The output then differs from the source's: it is an
    approximate start. A value network that reads `state` keeps its
    params, since the obs.state lists are equal. With restore_value off
    brax discards them.

    The count, std_eps, the `state` statistics and the policy are the
    source's. brax's checkpoint load always returns a Welford normalizer."""
    if plan.action != ADAPT:
        return params
    normalizer, policy, value = params
    src_cols = _columns(tuple(src_priv), sizes)
    dst_cols = _columns(tuple(dst_priv), sizes)
    width = sum(sizes[n] for n in dst_priv)
    count = _count(normalizer.count)
    std_eps = float(np.asarray(normalizer.std_eps))
    priors = {n: plan.priors.get(n, DEFAULT_PRIOR) for n in dst_cols if n not in src_cols}

    def rebuild(src, fill):
        """`src` (rows, ...) remapped to the dst components. An added
        component's rows hold fill[name]."""
        src = np.asarray(src)
        out = np.zeros((width, *src.shape[1:]), src.dtype)
        for name, cols in dst_cols.items():
            out[cols] = src[src_cols[name]] if name in src_cols else fill[name]
        return out

    critic = {
        "mean": rebuild(normalizer.mean[PRIVILEGED_KEY], {n: m for n, (m, _) in priors.items()}),
        "std": rebuild(
            normalizer.std[PRIVILEGED_KEY], {n: np.sqrt(s * s + std_eps) for n, (_, s) in priors.items()}
        ),
        "summed_variance": rebuild(
            normalizer.summed_variance[PRIVILEGED_KEY], {n: count * s * s for n, (_, s) in priors.items()}
        ),
    }
    normalizer = normalizer.replace(
        **{k: {**getattr(normalizer, k), PRIVILEGED_KEY: v} for k, v in critic.items()}
    )
    if plan.restore_value and plan.value_obs_key == PRIVILEGED_KEY:
        kernel = np.asarray(value["params"]["hidden_0"]["kernel"])
        src_width = sum(sizes[n] for n in src_priv)
        if kernel.shape[0] != src_width:
            raise ValueError(
                f"the value network reads obs 'privileged_state', and its first layer takes "
                f"{kernel.shape[0]} inputs, but the source critic list is {src_width} wide"
            )
        first = {**value["params"]["hidden_0"], "kernel": rebuild(kernel, dict.fromkeys(priors, 0.0))}
        value = {**value, "params": {**value["params"], "hidden_0": first}}
    return normalizer, policy, value


def restore_arguments(ckpt_dir, env, *, normalize_mode: str = WELFORD,
                      policy_obs_key: str = POLICY_OBS_KEY, value_obs_key: str = VALUE_OBS_KEY,
                      restore_value: bool = True) -> tuple[RestorePlan, dict]:
    """(plan, the ppo.train keyword arguments that carry it out).

    The keyword arguments are this run's: its normalizer mode, its network
    keys and its restore_value. A mode other than Welford refuses before
    anything loads. brax's checkpoint load rebuilds every normalizer in
    Welford mode, on the as_is path too.

    `as_is` passes the checkpoint path, as brax restores it. `adapt`
    passes the adapted params and no path."""
    from humanoid_lab.policy_io import load_params

    if normalize_mode != WELFORD:
        raise ValueError(
            f"ppo.normalize_observations_mode={normalize_mode!r} cannot restore a checkpoint. "
            "brax's checkpoint does not save the normalizer's mode, and its load rebuilds a "
            "Welford normalizer."
        )
    params = load_params(ckpt_dir)
    plan = plan_restore(ckpt_dir, params, env, policy_obs_key=policy_obs_key,
                        value_obs_key=value_obs_key, restore_value=restore_value)
    if plan.action == AS_IS:
        return plan, {"restore_checkpoint_path": str(ckpt_dir)}
    src = source_record(ckpt_dir)
    adapted = adapt_params(
        params, plan, env.obs_component_sizes(), src["env_config"]["obs"]["privileged"],
        env._config.obs.privileged,
    )
    return plan, {"restore_checkpoint_path": None, "restore_params": adapted}
