import copy
import inspect
import json
import os
import stat
import subprocess
import types
from importlib import metadata

import jax
import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from ml_collections import config_dict
from omegaconf import OmegaConf

from humanoid_lab import paths, run_record, tasks
from humanoid_lab.registry import EnvArgs, env_args_from_config
from humanoid_lab.train import (
    apply_xla_defaults,
    build_ppo_params,
    check_terrain_training,
    curriculum_log,
    progress_line,
    restore_options,
    terrain_suffix,
)


def test_defaults_come_from_go1_tuning():
    p = build_ppo_params([], smoke=False)
    assert p.num_timesteps >= 100_000_000
    assert p.network_factory.policy_obs_key == "state"
    assert p.network_factory.value_obs_key == "privileged_state"


def test_overrides_apply_with_type_coercion():
    p = build_ppo_params(["learning_rate=1e-4", "num_envs=512"], smoke=False)
    assert p.learning_rate == 1e-4
    assert p.num_envs == 512


def test_smoke_is_tiny():
    p = build_ppo_params([], smoke=True)
    assert p.num_timesteps <= 200_000
    assert p.num_envs <= 64


def test_gae_lambda_exists_and_takes_overrides():
    p = build_ppo_params({}, smoke=False)
    assert p.gae_lambda == 0.95
    p = build_ppo_params({"gae_lambda": 0.9}, smoke=False)
    assert p.gae_lambda == 0.9


# -- terrain wiring --------------------------------------------------------------


def _compose(overrides):
    with initialize_config_dir(version_base=None, config_dir=str(paths.CONFIGS_DIR)):
        return compose(config_name="config", overrides=overrides)


def test_xla_defaults_apply_to_terrain_only_and_use_the_name_jaxlib_reads():
    environ = {}
    assert apply_xla_defaults("joystick", environ) == {} and environ == {}
    assert apply_xla_defaults("terrain", environ) == {"XLA_PYTHON_CLIENT_PREALLOCATE": "false"}
    assert environ == {"XLA_PYTHON_CLIENT_PREALLOCATE": "false"}
    # A launcher's own value wins.
    environ = {"XLA_PYTHON_CLIENT_PREALLOCATE": "true"}
    assert apply_xla_defaults("terrain", environ) == {"XLA_PYTHON_CLIENT_PREALLOCATE": "true"}
    # jaxlib reads this name and no shorter one.
    import jaxlib.xla_client

    source = inspect.getsource(jaxlib.xla_client)
    for name in tasks.TERRAIN_XLA_DEFAULTS:
        assert f"'{name}'" in source
    assert "'XLA_PYTHON_CLIENT_PREALLOC'" not in source


def _ppo(**values):
    return config_dict.ConfigDict(values)


def test_terrain_refuses_periodic_resets():
    off = {"enable": False}
    check_terrain_training("joystick", _ppo(num_resets_per_eval=10), off)
    check_terrain_training("terrain", _ppo(num_resets_per_eval=0), off)
    with pytest.raises(ValueError, match="num_resets_per_eval=0, got 10"):
        check_terrain_training("terrain", _ppo(num_resets_per_eval=10), off)


def test_terrain_refuses_early_stop():
    check_terrain_training("joystick", _ppo(num_resets_per_eval=0), {"enable": True})
    with pytest.raises(ValueError, match="early_stop.enable"):
        check_terrain_training("terrain", _ppo(num_resets_per_eval=0), {"enable": True})


def test_terrain_yaml_sets_resets_to_zero():
    """The Go1 config the trainer starts from resets every env ten times per
    eval. The terrain task sets 0, and the composed run passes the guard."""
    assert build_ppo_params([], smoke=False).num_resets_per_eval == 10
    cfg = _compose(["task=terrain"])
    assert cfg.task.ppo.num_resets_per_eval == 0
    assert cfg.task.ppo.log_training_metrics is True
    p = build_ppo_params({}, smoke=False)
    p.update(OmegaConf.to_container(cfg.task.ppo))
    check_terrain_training(cfg.task.name, p, cfg.early_stop)


def test_restore_value_defaults_to_brax_default():
    from brax.training.agents.ppo import train as ppo

    default = inspect.signature(ppo.train).parameters["restore_value_fn"].default
    assert _compose([]).restore_value is default is True


def test_restore_options_read_this_runs_mode_and_network_keys():
    """The restore planner gets the normalizer mode and the obs keys the
    run's network config holds, after its overrides."""
    p = build_ppo_params({}, smoke=False)
    assert restore_options(p, True) == {
        "normalize_mode": "welford", "policy_obs_key": "state",
        "value_obs_key": "privileged_state", "restore_value": True,
    }
    p = build_ppo_params(
        {"normalize_observations_mode": "ema", "network_factory": {"value_obs_key": "state"}}, smoke=False
    )
    assert restore_options(p, False) == {
        "normalize_mode": "ema", "policy_obs_key": "state", "value_obs_key": "state",
        "restore_value": False,
    }


def test_env_args_from_config():
    """The make_env arguments, in make_env's order, from a composed config.
    The overrides are copies a caller may edit."""
    cfg = _compose(["experiment=terrain_cpu", "+actuators.overrides.groups.knee.kp=99"])
    container = OmegaConf.to_container(cfg, resolve=True)
    args = env_args_from_config(container)
    assert isinstance(args, EnvArgs)
    assert args == (
        "terrain",
        paths.REPO_ROOT / "robots/roboto_origin",
        cfg.actuators.name,
        OmegaConf.to_container(cfg.task.env, resolve=True),
        {"groups": {"knee": {"kp": 99}}},
    )
    before = copy.deepcopy(container)
    args.env_overrides.setdefault("sim", {})["num_envs"] = 7
    args.actuator_overrides["groups"]["knee"]["kp"] = 1
    assert container == before


def levels(level, free=1.0, **extra):
    """Training metrics that report a mean free level `level` over a
    window in which a fraction `free` of the episodes was free."""
    return {run_record.LEVEL_FREE_METRIC: level * free, run_record.FREE_METRIC: free, **extra}


def test_progress_suffix_reads_the_terrain_level():
    assert terrain_suffix({"eval/episode_reward": 3.0}) == ""
    assert terrain_suffix(levels(0.4567)) == "  terrain_lvl 0.46"
    assert terrain_suffix(levels(1.5, free=0.5)) == "  terrain_lvl 1.50"
    line = progress_line(4000, levels(1.0), 250.0)
    assert line.startswith("steps        4,000  reward      nan")
    assert line.endswith("250 steps/s  terrain_lvl 1.00")
    opt = {"training/learning_rate": 3e-4, "training/kl_mean": 0.0123}
    assert progress_line(4000, levels(1.0, **opt), 250.0).endswith(
        "250 steps/s  lr 0.0003  kl 0.0123  terrain_lvl 1.00"
    )
    assert progress_line(4000, opt, 250.0).endswith("250 steps/s  lr 0.0003  kl 0.0123")


# The metrics run_record reads, by the env's names. brax's logger adds the
# `episode/` prefix.
READ = tuple(
    m.removeprefix("episode/")
    for m in (run_record.LEVEL_FREE_METRIC, run_record.FREE_METRIC, run_record.PROMOTED_METRIC,
              run_record.DEMOTED_METRIC)
)


def log_episodes(episodes):
    """What brax's episode logger hands the progress callback for these
    ended episodes: (length, level, free, promoted, demoted) each. The
    env's per-step metrics hold their per-step value times the length."""
    from brax.training.logger import EpisodeMetricsLogger

    from humanoid_lab.envs.terrain_joystick import CURRICULUM_METRICS, TERRAIN_METRICS

    length, level, free, promoted, demoted = (np.array(c, float) for c in zip(*episodes))
    sums = {k: np.zeros(len(episodes)) for k in TERRAIN_METRICS + CURRICULUM_METRICS}
    sums.update(zip(READ, (level * free * length, free * length, promoted, demoted)))
    # brax's logger needs `length`. The env does not declare it.
    sums.update({"length": length, "terrain/level_per_step": level * length})
    got = {}
    log = EpisodeMetricsLogger(steps_between_logging=1, progress_fn=lambda n, m: got.update(m))
    log.update_episode_metrics(sums, np.ones(len(episodes)), {})
    return got


def test_run_record_reads_the_metrics_brax_logs_for_the_terrain_env():
    """The env's own metric names, through brax's episode logger, reach
    run.json's curriculum block, the progress line and wandb. The logger
    adds the `episode/` prefix and divides a `_per_step` sum by the episode
    length. Two pinned episodes on levels 1 and 0 drop out: the free ones
    sit on levels 2 and 4, and one of the two promoted. The training-mix
    level counts all four. Every name run_record reads is one the env
    declares."""
    from humanoid_lab.envs.terrain_joystick import CURRICULUM_METRICS, TERRAIN_METRICS

    assert set(READ) <= set(TERRAIN_METRICS + CURRICULUM_METRICS)
    got = log_episodes([(400, 2, 1, 1, 0), (1000, 4, 1, 0, 1), (1000, 1, 0, 0, 0), (60, 0, 0, 0, 0)])
    assert {run_record.LEVEL_FREE_METRIC, run_record.FREE_METRIC, run_record.PROMOTED_METRIC,
            run_record.DEMOTED_METRIC} <= set(got)
    assert got["episode/terrain/level_per_step"] == pytest.approx(1.75)
    trace = run_record.CurriculumTrace()
    assert trace.update(10, got)
    record = trace.record()
    assert record["level_last"] == pytest.approx(3.0)
    assert record["promoted_last"] == record["demoted_last"] == pytest.approx(0.5)
    assert terrain_suffix(got) == "  terrain_lvl 3.00"
    assert curriculum_log(got) == pytest.approx(
        {"curriculum/level": 3.0, "curriculum/promoted": 0.5, "curriculum/demoted": 0.5}
    )


def test_a_window_of_pinned_episodes_records_no_level():
    """Every episode in the window pinned: no level, no rate, and the trace
    keeps what it had."""
    got = log_episodes([(1000, 1, 0, 0, 0), (1000, 0, 0, 0, 0)])
    assert run_record.curriculum_rates(got) is None
    assert terrain_suffix(got) == "" and curriculum_log(got) == {}
    trace = run_record.CurriculumTrace()
    assert trace.update(10, levels(2.0))
    assert not trace.update(20, got)
    assert (trace.record()["level_last"], trace.record()["level_last_steps"]) == (2.0, 10)


def test_run_record_keys(tmp_path, monkeypatch):
    """run.json is written before training with the final fields null, its
    progress block is rewritten at each eval, and the end rewrites the
    record with the status, wall time, steps/s and curriculum."""
    clock = iter([100.0, 110.0, 140.0])
    monkeypatch.setattr(run_record.time, "monotonic", lambda: next(clock))
    path = tmp_path / "run.json"
    fields = {"run_name": "r", "task": "terrain", "num_timesteps": 2000, "env_config": {"a": 1}}
    record = run_record.RunRecord(path, fields, provenance={"git_commit": "abc"}, arena={"n_boxes": 33},
                                  restore=None)
    initial = json.loads(path.read_text())
    assert initial == {
        **fields,
        "status": "running",
        "early_stopped": None,
        "stopped_at_steps": None,
        "final_reward": None,
        "provenance": {"git_commit": "abc"},
        "arena": {"n_boxes": 33},
        "restore": None,
        "progress": None,
        "wall_s": None,
        "steps_per_s": None,
        "curriculum": None,
    }
    trace = run_record.CurriculumTrace()
    assert trace.record() is None
    assert not trace.update(500, {"eval/episode_reward": 1.0})
    trace.update(1000, levels(0.5, **{run_record.PROMOTED_METRIC: 0.25}))
    trace.update(1500, levels(0.25))
    record.start_clock()
    record.progress(1000, 3.5, trace.record())
    assert json.loads(path.read_text())["progress"] == {
        "steps": 1000,
        "wall_s": 10.0,
        "steps_per_s": 100.0,
        "eval_reward": 3.5,
        "curriculum": trace.record(),
    }
    record.finish(status=run_record.FINISHED, early_stopped=False, stopped_at_steps=2000,
                  final_reward=4.0, steps=2000, curriculum=trace.record())
    final = json.loads(path.read_text())
    assert set(final) == set(initial)
    assert {k: final[k] for k in ("status", "early_stopped", "stopped_at_steps", "final_reward")} == {
        "status": "finished", "early_stopped": False, "stopped_at_steps": 2000, "final_reward": 4.0,
    }
    assert final["wall_s"] == 40.0 and final["steps_per_s"] == 50.0
    assert final["curriculum"] == {
        "level_last": 0.25, "level_last_steps": 1500, "level_max": 0.5, "level_max_steps": 1000,
        "promoted_last": 0.25, "demoted_last": None,
    }
    assert sorted(p.name for p in tmp_path.iterdir()) == ["run.json"]


def test_run_json_mode_follows_the_umask(tmp_path):
    """run.json gets the mode a plain write gives it, on create and on
    overwrite. Its group can read it under umask 022."""
    old = os.umask(0o022)
    try:
        plain = tmp_path / "plain.json"
        plain.write_text("{}")
        path = tmp_path / "run.json"
        for status in ("running", "finished"):
            run_record.write_json_atomic(path, {"status": status})
            assert stat.S_IMODE(path.stat().st_mode) == stat.S_IMODE(plain.stat().st_mode) == 0o644
            assert json.loads(path.read_text()) == {"status": status}
    finally:
        os.umask(old)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["plain.json", "run.json"]


def test_every_distribution_resolves():
    """Each DISTRIBUTIONS name is an installed distribution. A wrong name
    would leave its version null in every run.json."""
    v = run_record.versions()
    assert set(v) == set(run_record.DISTRIBUTIONS)
    assert [k for k, x in v.items() if x is None] == []


def test_provenance_reads_commit_versions_and_device(monkeypatch):
    """A distribution that is not installed reads None. The stub lacks
    warp-lang."""
    present = {"jax": "0.9.2", "jaxlib": "0.9.2", "mujoco": "3.10.0", "mujoco-mjx": "3.10.0",
               "brax": "0.14.2", "playground": "0.2.0"}

    def version(dist):
        if dist not in present:
            raise metadata.PackageNotFoundError(dist)
        return present[dist]

    def run(cmd, **kwargs):
        out = {"rev-parse": "0123abc\n", "status": " M src/x.py\n"}[cmd[3]]
        return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr="")

    monkeypatch.setattr(run_record.metadata, "version", version)
    monkeypatch.setattr(run_record.subprocess, "run", run)
    monkeypatch.setattr(jax, "devices", lambda: [types.SimpleNamespace(device_kind="NVIDIA H100")] * 2)
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    p = run_record.provenance(paths.REPO_ROOT, wandb_run_id="w1", started_at="2026-10-02T00:00:00+00:00")
    assert p == {
        "git_commit": "0123abc",
        "git_dirty": True,
        "versions": {
            "jax": "0.9.2", "jaxlib": "0.9.2", "mujoco": "3.10.0", "mujoco_mjx": "3.10.0",
            "warp_lang": None, "brax": "0.14.2", "playground": "0.2.0",
        },
        "device": {"backend": "gpu", "kind": "NVIDIA H100", "count": 2},
        "wandb_run_id": "w1",
        "started_at": "2026-10-02T00:00:00+00:00",
    }

    def no_git(cmd, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(run_record.subprocess, "run", no_git)
    p = run_record.provenance(paths.REPO_ROOT)
    assert (p["git_commit"], p["git_dirty"]) == (None, None)
    assert p["started_at"].endswith("+00:00")
