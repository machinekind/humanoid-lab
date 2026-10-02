"""restore.py: the warm start across critic layouts, on synthetic params.

The normalizer is brax's running_statistics state. The value network is a
two-layer tree shaped as brax builds it. The env is a stub carrying the
four things the planner reads: obs lists, component sizes, robot name and
action size. Nothing here builds a model.
"""

from __future__ import annotations

import json
import types

import jax
import jax.numpy as jp
import numpy as np
import pytest
from brax.training import types as brax_types
from brax.training.acme import running_statistics, specs
from ml_collections import config_dict

from humanoid_lab import restore
from humanoid_lab.envs import height_scan

SCAN = height_scan.NAME
SIZES = {"gyro": 3, "joint_pos": 4, "linvel": 3, "height": 1, SCAN: 6, "extra": 2}
ACTOR = ("gyro", "joint_pos")
FLAT = ("gyro", "joint_pos", "linvel", "height")
WITH_SCAN = (*FLAT, SCAN)
HIDDEN = 5
ROBOT = "roboto_origin"
ACTIONS = 4
# The normalizer count of a source trained 4.03e8 env steps from scratch.
LONG_COUNT = 403_046_400
PRIOR = (height_scan.PRIOR_MEAN, height_scan.PRIOR_STD)


def width(names):
    return sum(SIZES[n] for n in names)


def stub_env(critic=WITH_SCAN, actor=ACTOR, robot=ROBOT, actions=ACTIONS):
    cfg = config_dict.ConfigDict({"obs": {"state": list(actor), "privileged": list(critic)}})
    return types.SimpleNamespace(
        _config=cfg,
        obs_component_sizes=lambda: dict(SIZES),
        robot_spec=types.SimpleNamespace(name=robot),
        action_size=actions,
    )


def make_params(critic=FLAT, actor=ACTOR, seed=0, count=None, value_width=None):
    """(normalizer, policy, value) with real statistics. `count`, an int,
    replaces the sample count and scales summed_variance with it. It is
    split into two 32-bit words, as brax stores it. The value network's
    first layer takes `value_width` inputs, the critic's width by
    default."""
    k_obs, k_val = jax.random.split(jax.random.PRNGKey(seed))
    obs_spec = {
        "state": specs.Array((width(actor),), jp.float32),
        "privileged_state": specs.Array((width(critic),), jp.float32),
    }
    norm = running_statistics.init_state(obs_spec)
    batch = {
        k: 2.0 + 3.0 * jax.random.normal(jax.random.fold_in(k_obs, i), (64, *s.shape))
        for i, (k, s) in enumerate(obs_spec.items())
    }
    norm = running_statistics.update(norm, batch)
    if count is not None:
        scale = count / 64.0
        norm = norm.replace(
            count=brax_types.UInt64(hi=count >> 32, lo=count & 0xFFFFFFFF),
            summed_variance=jax.tree.map(lambda v: v * scale, norm.summed_variance),
        )
    k1, k2 = jax.random.split(k_val)
    value = {
        "params": {
            "hidden_0": {
                "kernel": jax.random.normal(k1, (value_width or width(critic), HIDDEN)),
                "bias": jp.zeros(HIDDEN),
            },
            "hidden_1": {"kernel": jax.random.normal(k2, (HIDDEN, 1)), "bias": jp.zeros(1)},
        }
    }
    policy = {"params": {"hidden_0": {"kernel": jp.ones((width(actor), 2 * ACTIONS))}}}
    return norm, policy, value


def write_checkpoint(tmp_path, critic=FLAT, actor=ACTOR, robot=ROBOT, actions=ACTIONS, run_json=True,
                     network=None, ppo_config=None):
    """A run directory layout around a checkpoint step directory, with the
    source run's run.json when asked. `network` goes into the checkpoint's
    network config as its factory kwargs, `ppo_config` into run.json.
    Returns the step directory."""
    step = tmp_path / "src_run" / "checkpoints" / "000000001000"
    step.mkdir(parents=True)
    config = {"action_size": actions}
    if network is not None:
        config["network_factory_kwargs"] = network
    (step / "ppo_network_config.json").write_text(json.dumps(config))
    if run_json:
        record = {
            "run_name": "src_run",
            "task": "joystick" if SCAN not in critic else "terrain",
            "env_config": {"obs": {"state": list(actor), "privileged": list(critic)}},
            "hydra_config": {"robot": {"name": robot, "dir": f"robots/{robot}"}},
            "arena": None if SCAN not in critic else {"fingerprint": "f00d", "generator_version": 1},
        }
        if ppo_config is not None:
            record["ppo_config"] = ppo_config
        (tmp_path / "src_run" / "run.json").write_text(json.dumps(record))
    return step


def value_out(params, obs):
    """The value MLP on normalized `obs`, as brax applies it."""
    norm, _, value = params
    x = (obs - norm.mean["privileged_state"]) / norm.std["privileged_state"]
    v = value["params"]
    h = jax.nn.swish(x @ v["hidden_0"]["kernel"] + v["hidden_0"]["bias"])
    return h @ v["hidden_1"]["kernel"] + v["hidden_1"]["bias"]


def scan_batch(n, mean, std, critic=WITH_SCAN, key=1):
    """A batch whose critic columns are drawn from N(mean, std), as brax
    feeds the normalizer: `state` and `privileged_state`."""
    noise = jax.random.normal(jax.random.PRNGKey(key), (n, width(critic)))
    return {"state": jp.zeros((n, width(ACTOR))), "privileged_state": mean + std * noise}


def test_equal_lists_restore_as_is(tmp_path):
    ckpt = write_checkpoint(tmp_path, critic=WITH_SCAN)
    plan = restore.plan_restore(ckpt, make_params(critic=WITH_SCAN), stub_env())
    assert plan.action == restore.AS_IS
    assert (plan.source_run, plan.source_task) == ("src_run", "terrain")
    assert plan.added == plan.removed == ()
    assert plan.source_arena == {"fingerprint": "f00d", "generator_version": 1}
    assert (plan.priors, plan.value_obs_key, plan.restore_value) == ({}, "privileged_state", True)
    params = make_params(critic=WITH_SCAN)
    assert restore.adapt_params(params, plan, SIZES, WITH_SCAN, WITH_SCAN) is params


def test_an_added_trailing_component_pads_zero_rows_and_the_prior(tmp_path):
    ckpt = write_checkpoint(tmp_path, critic=FLAT)
    params = make_params(critic=FLAT)
    plan = restore.plan_restore(ckpt, params, stub_env(critic=WITH_SCAN))
    assert plan.action == restore.ADAPT
    assert (plan.added, plan.removed, plan.source_arena) == ((SCAN,), (), None)
    assert plan.priors == {SCAN: PRIOR}
    assert json.loads(json.dumps(plan.record()))["priors"] == {SCAN: list(PRIOR)}
    norm, policy, value = restore.adapt_params(params, plan, SIZES, FLAT, WITH_SCAN)
    src_norm, src_policy, src_value = params
    n = width(FLAT)
    count = int(src_norm.count.hi) * 2**32 + int(src_norm.count.lo)

    mean, std = PRIOR
    for field, prior in (("mean", mean), ("std", std), ("summed_variance", count * std * std)):
        new = np.asarray(getattr(norm, field)["privileged_state"])
        assert new.shape == (width(WITH_SCAN),)
        np.testing.assert_array_equal(new[:n], getattr(src_norm, field)["privileged_state"])
        np.testing.assert_allclose(new[n:], prior, rtol=1e-6)
        np.testing.assert_array_equal(getattr(norm, field)["state"], getattr(src_norm, field)["state"])
    assert norm.count == src_norm.count and norm.mode == src_norm.mode and norm.std_eps == src_norm.std_eps
    assert policy is src_policy

    kernel = np.asarray(value["params"]["hidden_0"]["kernel"])
    assert kernel.shape == (width(WITH_SCAN), HIDDEN)
    np.testing.assert_array_equal(kernel[:n], src_value["params"]["hidden_0"]["kernel"])
    np.testing.assert_array_equal(kernel[n:], 0.0)
    np.testing.assert_array_equal(value["params"]["hidden_1"]["kernel"], src_value["params"]["hidden_1"]["kernel"])

    # The critic's output on the shared columns is the source critic's,
    # whatever the scan reads.
    obs = jax.random.normal(jax.random.PRNGKey(5), (8, width(WITH_SCAN))) * 4.0
    np.testing.assert_allclose(
        value_out((norm, policy, value), obs), value_out(params, obs[:, :n]), rtol=1e-6, atol=1e-6
    )


@pytest.mark.parametrize("count", [LONG_COUNT, 2**32 + 5], ids=["4.03e8", "past_2^32"])
def test_an_added_component_keeps_its_prior_through_an_update(tmp_path, count):
    """One brax update, on data narrower than the prior, leaves the columns
    at the prior. The source count is 4.03e8 env steps, or 2^32 + 5. The
    second count uses both of brax's 32-bit count words. A zero
    summed_variance would have collapsed the std."""
    ckpt = write_checkpoint(tmp_path, critic=FLAT)
    params = make_params(critic=FLAT, count=count)
    plan = restore.plan_restore(ckpt, params, stub_env(critic=WITH_SCAN))
    norm, _, _ = restore.adapt_params(params, plan, SIZES, FLAT, WITH_SCAN)
    added = slice(width(FLAT), width(WITH_SCAN))
    sv = np.asarray(norm.summed_variance["privileged_state"])[added]
    np.testing.assert_allclose(sv, count * PRIOR[1] ** 2, rtol=1e-6)
    # brax puts the restored params on the device before the first update.
    norm = jax.tree.map(jp.asarray, norm)
    # A first rollout from pad spawns on rows 0 to 4 reads the scan at about
    # this mean and std.
    batch = scan_batch(4096, mean=0.0, std=0.026)
    updated = running_statistics.update(norm, batch)
    std = np.asarray(updated.std["privileged_state"])[added]
    mean = np.asarray(updated.mean["privileged_state"])[added]
    np.testing.assert_allclose(std, PRIOR[1], rtol=1e-3)
    np.testing.assert_allclose(mean, PRIOR[0], atol=1e-4)

    sv = norm.summed_variance["privileged_state"].at[added].set(0.0)
    collapsed = norm.replace(summed_variance={**norm.summed_variance, "privileged_state": sv})
    low = np.asarray(running_statistics.update(collapsed, batch).std["privileged_state"])
    assert low[added].max() < 0.01 * PRIOR[1]


def test_a_component_without_a_prior_gets_mean_0_and_std_1(tmp_path):
    """The prior's weight is the source's count, so std 1 holds through an
    update on data spread 0.1."""
    ckpt = write_checkpoint(tmp_path, critic=FLAT)
    params = make_params(critic=FLAT, count=LONG_COUNT)
    critic = (*FLAT, "extra")
    plan = restore.plan_restore(ckpt, params, stub_env(critic=critic))
    assert plan.priors == {"extra": restore.DEFAULT_PRIOR} == {"extra": (0.0, 1.0)}
    norm, _, _ = restore.adapt_params(params, plan, SIZES, FLAT, critic)
    added = slice(width(FLAT), width(critic))
    np.testing.assert_array_equal(np.asarray(norm.mean["privileged_state"])[added], 0.0)
    np.testing.assert_array_equal(np.asarray(norm.std["privileged_state"])[added], 1.0)
    np.testing.assert_allclose(np.asarray(norm.summed_variance["privileged_state"])[added], LONG_COUNT)
    updated = running_statistics.update(jax.tree.map(jp.asarray, norm), scan_batch(256, 0.0, 0.1, critic))
    np.testing.assert_allclose(np.asarray(updated.std["privileged_state"])[added], 1.0, atol=1e-3)


def test_a_removed_component_is_sliced(tmp_path):
    ckpt = write_checkpoint(tmp_path, critic=WITH_SCAN)
    params = make_params(critic=WITH_SCAN)
    critic = ("gyro", "joint_pos", "height", SCAN)  # linvel removed from the middle
    plan = restore.plan_restore(ckpt, params, stub_env(critic=critic))
    assert (plan.action, plan.added, plan.removed, plan.priors) == (restore.ADAPT, (), ("linvel",), {})
    norm, _, value = restore.adapt_params(params, plan, SIZES, WITH_SCAN, critic)
    keep = np.r_[0:7, 10:17]  # gyro, joint_pos | height, scan
    src_mean = np.asarray(params[0].mean["privileged_state"])
    np.testing.assert_array_equal(norm.mean["privileged_state"], src_mean[keep])
    np.testing.assert_array_equal(
        value["params"]["hidden_0"]["kernel"], np.asarray(params[2]["params"]["hidden_0"]["kernel"])[keep]
    )


def test_a_reordered_or_actor_mismatch_is_refused_naming_both_lists(tmp_path):
    ckpt = write_checkpoint(tmp_path, critic=FLAT)
    params = make_params(critic=FLAT)
    reordered = ("joint_pos", "gyro", "linvel", "height", SCAN)
    with pytest.raises(ValueError, match=r"critic list \['gyro', 'joint_pos'.*\['joint_pos', 'gyro'"):
        restore.plan_restore(ckpt, params, stub_env(critic=reordered))
    with pytest.raises(ValueError, match=r"actor list \['gyro', 'joint_pos'\].*\['joint_pos', 'gyro'\]"):
        restore.plan_restore(ckpt, params, stub_env(actor=("joint_pos", "gyro")))
    with pytest.raises(ValueError, match="from robot 'roboto_origin', and this env is 'asimov_v1'"):
        restore.plan_restore(ckpt, params, stub_env(robot="asimov_v1"))
    with pytest.raises(ValueError, match="action size 4, and this env has 6"):
        restore.plan_restore(ckpt, params, stub_env(actions=6))
    # A checkpoint that disagrees with its own run.json.
    with pytest.raises(ValueError, match="run.json lists"):
        restore.plan_restore(ckpt, make_params(critic=WITH_SCAN), stub_env())


def test_no_source_record_with_equal_widths_is_as_is_and_otherwise_refused(tmp_path):
    ckpt = write_checkpoint(tmp_path, critic=FLAT, run_json=False)
    assert restore.source_record(ckpt) is None
    plan = restore.plan_restore(ckpt, make_params(critic=FLAT), stub_env(critic=FLAT))
    assert (plan.action, plan.source_run) == (restore.AS_IS, None)
    with pytest.raises(ValueError, match="cannot be mapped without the source run's obs lists"):
        restore.plan_restore(ckpt, make_params(critic=FLAT), stub_env(critic=WITH_SCAN))


# -- normalizer mode ---------------------------------------------------------------


def test_an_ema_source_run_is_refused(tmp_path):
    """brax's load rebuilds every normalizer as Welford, and an EMA run's
    summed_variance holds a variance."""
    ckpt = write_checkpoint(tmp_path, critic=FLAT, ppo_config={"normalize_observations_mode": "ema"})
    for critic in (FLAT, WITH_SCAN):
        with pytest.raises(ValueError, match="normalize_observations_mode 'ema'.*Welford"):
            restore.plan_restore(ckpt, make_params(critic=FLAT), stub_env(critic=critic))


def test_an_ema_run_refuses_before_loading(tmp_path):
    missing = tmp_path / "no_such_run" / "checkpoints" / "000000001000"
    with pytest.raises(ValueError, match="normalize_observations_mode='ema' cannot restore"):
        restore.restore_arguments(missing, stub_env(), normalize_mode="ema")


# -- network observation keys ------------------------------------------------------


SYMMETRIC = {"policy_obs_key": "state", "value_obs_key": "state"}


def test_source_keys_come_from_the_checkpoint_then_run_json(tmp_path):
    ckpt = write_checkpoint(tmp_path / "a", network=SYMMETRIC,
                            ppo_config={"network_factory": {"value_obs_key": "privileged_state"}})
    assert restore.source_obs_keys(ckpt, restore.source_record(ckpt)) == ("state", "state")
    ckpt = write_checkpoint(tmp_path / "b", ppo_config={"network_factory": SYMMETRIC})
    assert restore.source_obs_keys(ckpt, restore.source_record(ckpt)) == ("state", "state")
    ckpt = write_checkpoint(tmp_path / "c")
    assert restore.source_obs_keys(ckpt, restore.source_record(ckpt)) == ("state", "privileged_state")


def test_a_symmetric_critic_adapts_its_normalizer_and_keeps_its_kernel(tmp_path):
    """Both runs' value networks read `state`. The critic list gains the
    scan, so the normalizer's privileged columns adapt. The value network's
    input is the unchanged actor list, so its params pass through."""
    ckpt = write_checkpoint(tmp_path, critic=FLAT, network=SYMMETRIC)
    params = make_params(critic=FLAT, value_width=width(ACTOR))
    plan = restore.plan_restore(ckpt, params, stub_env(critic=WITH_SCAN), value_obs_key="state")
    assert (plan.action, plan.added, plan.value_obs_key) == (restore.ADAPT, (SCAN,), "state")
    norm, _, value = restore.adapt_params(params, plan, SIZES, FLAT, WITH_SCAN)
    assert np.shape(norm.mean["privileged_state"]) == (width(WITH_SCAN),)
    np.testing.assert_allclose(np.asarray(norm.std["privileged_state"])[width(FLAT):], PRIOR[1], rtol=1e-6)
    assert value is params[2]


@pytest.mark.parametrize("critic", [FLAT, WITH_SCAN], ids=["as_is", "adapt"])
def test_value_keys_that_differ_refuse_unless_the_critic_starts_fresh(tmp_path, critic):
    """A source critic that read `state` cannot restore into one that reads
    `privileged_state`, on either path. With restore_value=false brax
    discards the value params, so the plan goes ahead."""
    ckpt = write_checkpoint(tmp_path, critic=FLAT, network=SYMMETRIC)
    params = make_params(critic=FLAT, value_width=width(ACTOR))
    env = stub_env(critic=critic)
    with pytest.raises(ValueError, match="reads obs 'state', and this run's reads 'privileged_state'.*"
                                         "restore_value=false"):
        restore.plan_restore(ckpt, params, env)
    plan = restore.plan_restore(ckpt, params, env, restore_value=False)
    assert plan.restore_value is False
    _, _, value = restore.adapt_params(params, plan, SIZES, FLAT, critic)
    assert value is params[2]


def test_a_fresh_critic_skips_the_kernel(tmp_path):
    """restore_value=false with a value kernel the source critic list does
    not match: the normalizer adapts and the kernel is not read."""
    ckpt = write_checkpoint(tmp_path, critic=FLAT)
    params = make_params(critic=FLAT, value_width=3)
    env = stub_env(critic=WITH_SCAN)
    plan = restore.plan_restore(ckpt, params, env)
    with pytest.raises(ValueError, match="reads obs 'privileged_state', and its first layer takes 3 inputs"):
        restore.adapt_params(params, plan, SIZES, FLAT, WITH_SCAN)
    plan = restore.plan_restore(ckpt, params, env, restore_value=False)
    norm, _, value = restore.adapt_params(params, plan, SIZES, FLAT, WITH_SCAN)
    assert np.shape(norm.mean["privileged_state"]) == (width(WITH_SCAN),)
    assert value is params[2]


def test_a_policy_on_another_key_is_refused(tmp_path):
    """The policy is never adapted. A source policy on another key refuses.
    So does a policy that reads `privileged_state` when that list changes.
    With the list unchanged it restores as it is."""
    privileged = {"policy_obs_key": "privileged_state", "value_obs_key": "privileged_state"}
    ckpt = write_checkpoint(tmp_path / "a", critic=FLAT, network=privileged)
    with pytest.raises(ValueError, match="policy that reads obs 'privileged_state', and this run's reads 'state'"):
        restore.plan_restore(ckpt, make_params(critic=FLAT), stub_env(critic=FLAT))
    keys = {"policy_obs_key": "privileged_state"}
    with pytest.raises(ValueError, match="The policy reads obs 'privileged_state'"):
        restore.plan_restore(ckpt, make_params(critic=FLAT), stub_env(critic=WITH_SCAN), **keys)
    plan = restore.plan_restore(ckpt, make_params(critic=FLAT), stub_env(critic=FLAT), **keys)
    assert plan.action == restore.AS_IS
