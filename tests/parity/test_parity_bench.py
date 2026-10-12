"""torch_warp.bench.main on the CPU warp device: the JSON it writes.

The bench is tiny: N=4, unroll 5, batch 4, 2 minibatches, one timed
iteration and five timed physics steps.
"""

from __future__ import annotations

import json

import pytest
from hydra import compose, initialize_config_dir

from humanoid_lab import paths, run_record

torch = pytest.importorskip("torch")

from humanoid_lab.torch_warp import bench

TOP_KEYS = {
    "created",
    "device",
    "git_sha",
    "hardware",
    "versions",
    "config",
    "warmup_s",
    "throughput",
    "budgets",
    "memory",
    "ignored",
}
HARDWARE_KEYS = {"gpu_name", "gpu_capability", "cuda_driver", "warp_cuda_toolkit", "torch_cuda", "cpu", "os"}
CONFIG_KEYS = {
    "robot",
    "actuators",
    "experiment",
    "seed",
    "num_envs",
    "unroll_length",
    "batch_size",
    "num_minibatches",
    "num_updates_per_batch",
    "episode_length",
    "steps_per_iter",
}
WARMUP_KEYS = {"construct", "reset", "first_step", "first_iteration"}
THROUGHPUT_KEYS = {"physics_steps", "iters", "physics_steps_per_s", "env_steps_per_s", "ppo_env_steps_per_s"}
BUDGET_KEYS = {
    "backend",
    "nacon_max",
    "naconmax_per_env",
    "num_envs",
    "pool",
    "overflow",
    "nefc_max",
    "njmax",
    "rows_overflow",
    "nacon_pool_max",
    "pool_overflow",
}


@pytest.fixture(scope="module")
def record(tmp_path_factory):
    out = tmp_path_factory.mktemp("bench") / "bench.json"
    overrides = [
        "torch_warp.device=cpu",
        "wandb.enable=false",
        "smoke=true",
        "network=small",
        "++ppo.num_envs=4",
        "++ppo.batch_size=4",
        "++ppo.num_minibatches=2",
        "++ppo.unroll_length=5",
        "bench.iters=1",
        "bench.physics_steps=5",
        f"bench.out={out.as_posix()}",
    ]
    with initialize_config_dir(version_base=None, config_dir=str(paths.CONFIGS_DIR)):
        cfg = compose(config_name="torch_warp", overrides=overrides)
    bench.main(cfg)
    assert not out.with_name(out.name + ".tmp").exists()
    return cfg, json.loads(out.read_text(encoding="utf-8"))


def test_bench_writes_the_documented_json(record):
    cfg, rec = record
    assert set(rec) == TOP_KEYS
    assert set(rec["hardware"]) == HARDWARE_KEYS
    assert set(rec["config"]) == CONFIG_KEYS
    assert set(rec["warmup_s"]) == WARMUP_KEYS
    assert set(rec["throughput"]) == THROUGHPUT_KEYS
    assert set(rec["budgets"]) == BUDGET_KEYS
    assert set(bench.CUDA_FIELDS) <= HARDWARE_KEYS

    tp = rec["throughput"]
    for key in ("physics_steps_per_s", "env_steps_per_s", "ppo_env_steps_per_s"):
        assert tp[key] > 0, key
    assert (tp["physics_steps"], tp["iters"]) == (5, 1)
    assert all(v > 0 for v in rec["warmup_s"].values())

    assert rec["device"] == "cpu"
    for key in bench.CUDA_FIELDS:
        assert rec["hardware"][key] is None, key
    assert rec["hardware"]["cpu"] and rec["hardware"]["os"]

    assert rec["config"]["seed"] == cfg.seed
    assert rec["config"]["num_envs"] == 4
    assert rec["config"]["robot"] == cfg.robot.name
    assert rec["config"]["experiment"] is None
    assert rec["config"]["steps_per_iter"] == 4 * 5 * 2

    assert rec["versions"] == run_record.versions()
    assert rec["git_sha"] is None or len(rec["git_sha"]) == 40
    budgets = rec["budgets"]
    assert budgets["backend"] == "warp" and budgets["num_envs"] == 4
    assert budgets["nacon_max"] is None
    assert budgets["pool"] == budgets["naconmax_per_env"] * 4
    assert budgets["nacon_pool_max"] > 0 and budgets["nefc_max"] > 0
    assert rec["memory"]["rss"] > 0
