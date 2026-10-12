"""What a Brax run reports and writes.

This runs train.main once on CPU with tiny PPO settings and ppo.sym_coef
on, then checks the eval lines and run.json. paths.RUNS_DIR points at a tmp
dir, so nothing lands under runs/.
"""

import contextlib
import io
import json
import math
import re
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from humanoid_lab import paths, run_record, train

OVERRIDES = [
    "smoke=true",
    "wandb.enable=false",
    "run_name=brax_record",
    "ppo.sym_coef=0.5",
    "network=small",
    "++ppo.num_envs=8",
    "++ppo.batch_size=8",
    "++ppo.num_minibatches=2",
    "++ppo.unroll_length=5",
    "++ppo.num_updates_per_batch=1",
    "++ppo.num_evals=2",
    "++ppo.num_eval_envs=8",
    "++ppo.episode_length=20",
    # One eval after init, and 8 envs x 5 steps x 2 minibatches = 80 env
    # steps per training step, so 160 gives two training steps.
    "++ppo.num_timesteps=160",
]


@pytest.fixture(scope="module")
def smoke_run(tmp_path_factory):
    runs_dir = tmp_path_factory.mktemp("runs")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(paths, "RUNS_DIR", runs_dir)
        with initialize_config_dir(version_base=None, config_dir=str(paths.CONFIGS_DIR)):
            cfg = compose(config_name="config", overrides=OVERRIDES)
        out = io.StringIO()
        # hydra.main passes a given cfg straight to the function.
        with contextlib.redirect_stdout(out):
            train.main(cfg)
    return runs_dir / "brax_record", out.getvalue()


def test_a_nonzero_sym_coef_reports_the_sym_loss(smoke_run):
    _, stdout = smoke_run
    values = [float(v) for v in re.findall(r"^steps .*\bsym (\S+)", stdout, flags=re.MULTILINE)]
    assert values, stdout
    # A fresh policy is not mirror symmetric, so its mirror loss is positive.
    assert all(math.isfinite(v) and v > 0 for v in values), values


def test_run_json_records_the_sym_coef(smoke_run):
    run_dir, _ = smoke_run
    record = json.loads((run_dir / "run.json").read_text())
    assert record["ppo_config"]["sym_coef"] == 0.5
    assert record["hydra_config"]["ppo"]["sym_coef"] == 0.5


def test_run_json_comes_from_the_builder(smoke_run):
    run_dir, _ = smoke_run
    record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert set(record) == set(run_record.RECORD_KEYS)
    trainer = record["trainer"]
    assert trainer["name"] == "brax_ppo"
    assert trainer["physics"] == "jax"
    assert set(trainer["versions"]) == set(run_record.VERSION_PACKAGES)
    assert record["warm_start"] is None
    # tmp is outside the project, so the path is absolute, in POSIX form.
    ckpt = record["checkpoint_dir"]
    assert "\\" not in ckpt
    assert ckpt == Path(ckpt).as_posix()
    assert Path(ckpt).is_absolute()
    assert Path(run_record.load_run(run_dir)["checkpoint_dir"]).exists()
    assert record["contacts"]["backend"] == "jax"
    assert isinstance(record["final_reward"], float)
    assert not (run_dir / "run.json.tmp").exists()
