"""torch_warp.ppo.main end to end on the CPU warp device: run outputs,
warm start, resume, max_minutes and run.json through run_record.

The runs are tiny: N=4, unroll 5, batch 4, 2 minibatches, so one iteration
is 40 steps, and the scratch run is 2 iterations. paths.RUNS_DIR points at
a module tmp dir. The tests share the runs, and the resume test runs last
because it extends the scratch run.
"""

from __future__ import annotations

import dataclasses
import json
import math

import jax
import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from humanoid_lab import paths, registry, run_record

torch = pytest.importorskip("torch")

from humanoid_lab.torch_warp import ppo as warp_ppo
from humanoid_lab.torch_warp.env import WarpJoystick

STEPS_PER_ITER = 4 * 5 * 2
BASE = [
    # A nonzero seed, so the step-0 check pins the run seed.
    "seed=3",
    "torch_warp.device=cpu",
    "wandb.enable=false",
    "smoke=true",
    "network=small",
    "++ppo.num_envs=4",
    "++ppo.batch_size=4",
    "++ppo.num_minibatches=2",
    "++ppo.unroll_length=5",
    "++ppo.num_updates_per_batch=1",
    "++ppo.num_evals=2",
    "++ppo.episode_length=20",
]
LOG_KEYS = {
    "it",
    "steps",
    "sps",
    "minutes",
    "reward_x1000",
    "falls_per_1M",
    "done_ep_len",
    "losses",
    "terms",
    "memory",
    "budget",
}


def _compose(*overrides, config_name="torch_warp"):
    with initialize_config_dir(version_base=None, config_dir=str(paths.CONFIGS_DIR)):
        return compose(config_name=config_name, overrides=list(overrides))


def _train(*overrides, steps=80):
    warp_ppo.main(_compose(*BASE, f"++ppo.num_timesteps={steps}", *overrides))


def _run_json(run_dir):
    return json.loads((run_dir / "run.json").read_text(encoding="utf-8"))


def _log(run_dir):
    return [json.loads(line) for line in (run_dir / "train_log.jsonl").read_text(encoding="utf-8").splitlines()]


def _steps(run_dir):
    return sorted(int(p.name) for p in (run_dir / "checkpoints").iterdir() if p.is_dir())


def _assert_leaves_equal(got, want):
    # The checkpoint loads as a list, init_from_brax returns a tuple.
    got_leaves, got_tree = jax.tree.flatten(list(got))
    want_leaves, want_tree = jax.tree.flatten(list(want))
    assert got_tree == want_tree
    for g, w in zip(got_leaves, want_leaves):
        assert np.array_equal(np.asarray(g), np.asarray(w))


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    root = tmp_path_factory.mktemp("runs")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(paths, "RUNS_DIR", root)
        _train("run_name=scratch")
        yield root


@pytest.fixture(scope="module")
def scratch(runs):
    return runs / "scratch"


def test_from_scratch_writes_a_brax_checkpoint_and_the_record(scratch):
    run = _run_json(scratch)
    assert tuple(run) == run_record.RECORD_KEYS
    assert run["trainer"]["name"] == "torch_warp_ppo"
    assert run["trainer"]["physics"] == "warp"
    assert run["env_config"]["sim"]["backend"] == "warp"
    assert run["env_config"]["sim"]["num_envs"] == 4
    contacts = run["contacts"]
    assert contacts["backend"] == "warp"
    assert contacts["num_envs"] == 4
    # smoke=true skips the probe.
    assert contacts["nacon_max"] is None
    assert contacts["nefc_max"] is None
    assert run["final_reward"] is None
    assert run["stopped_at_steps"] == 2 * STEPS_PER_ITER
    assert run["early_stopped"] is False
    assert run["warm_start"] is None
    assert "sym_coef" in run["ppo_config"]
    assert run["trainer"]["resumes"] == []
    assert set(run["trainer"]["budget_peaks"]) == {"nacon_pool_max", "nefc_max"}
    assert run["trainer"]["train_reward_last"] is not None

    steps = _steps(scratch)
    assert steps[0] == 0
    assert steps[-1] == 2 * STEPS_PER_ITER
    for step in (0, steps[-1]):
        assert (scratch / "checkpoints" / f"{step:012d}" / "ppo_network_config.json").exists()
    assert (scratch / "DONE").read_text(encoding="utf-8") == str(steps[-1])
    assert int((scratch / "trainer.pid").read_text(encoding="utf-8")) > 0
    assert (scratch / "trainer_state.pt").exists()

    # Owner decision 2: the step-0 networks are brax's own init at the run seed.
    from humanoid_lab import policy_io as brax_policy_io
    from humanoid_lab.ppo_config import brax_train_kwargs, compose_ppo_params
    from humanoid_lab.torch_warp import policy_io

    cfg = _compose(*BASE, "++ppo.num_timesteps=80", "run_name=scratch")
    _training, net_kwargs, _repo = brax_train_kwargs(compose_ppo_params(cfg))
    wenv = WarpJoystick(run["hydra_config"], 4, 20, device="cpu")
    want = policy_io.init_from_brax(
        run["hydra_config"]["seed"], wenv.actor_obs_size, wenv.privileged_obs_size, wenv.action_size, dict(net_kwargs)
    )
    got = brax_policy_io.load_params(scratch / "checkpoints" / f"{0:012d}")
    _assert_leaves_equal(got[1:], want[1:])

    log = _log(scratch)
    assert log
    for rec in log:
        assert set(rec) == LOG_KEYS
        assert set(rec["losses"]) == set(warp_ppo.LOSS_KEYS)
        assert "rss" in rec["memory"]
        assert set(rec["budget"]) == {"nacon_pool_max", "nefc_max"}
        assert rec["terms"] and all(isinstance(v, float) for v in rec["terms"].values())
    assert log[0]["it"] == 1


def test_battery_loads_the_torch_checkpoint(scratch):
    from humanoid_lab.eval import battery

    run, env, ckpt, inf = battery.load_checkpoint_policy(scratch)
    assert int(ckpt.name) == 2 * STEPS_PER_ITER
    state = jax.jit(env.reset)(jax.random.PRNGKey(0))
    action, _ = inf(state.obs, jax.random.PRNGKey(1))
    assert np.all(np.isfinite(np.asarray(action)))

    hydra = run["hydra_config"]
    jenv = registry.make_env(
        hydra["task"]["name"],
        paths.REPO_ROOT / hydra["robot"]["dir"],
        hydra["actuators"]["name"],
        {**(hydra["task"]["env"] or {}), "sim": {"backend": "jax"}},
        hydra["actuators"].get("overrides") or {},
    )
    built = jenv.config.to_dict()
    recorded = json.loads(json.dumps(run["env_config"]))
    built = json.loads(json.dumps(built))
    built.pop("sim")
    recorded.pop("sim")
    assert built == recorded


def test_a_warm_start_records_its_source(runs, scratch):
    last = scratch / "checkpoints" / f"{_steps(scratch)[-1]:012d}"
    _train(f"restore={last}", "++ppo.learning_rate=1e-4", "run_name=warm")
    run = _run_json(runs / "warm")
    ws = run["warm_start"]
    assert ws["checkpoint"] == run_record.portable_path(last)
    assert ws["source_run"] == run_record.portable_path(scratch)
    source = _run_json(scratch)
    assert ws["source_results"] == {k: source[k] for k in run_record.SOURCE_RESULT_KEYS}
    assert "ppo.learning_rate" in ws["config_diff"]
    assert ws["config_diff"]["ppo.learning_rate"][1] == 1e-4
    assert _steps(runs / "warm")[0] == 0

    # The warm run starts from the source weights and normalizer as they are.
    from humanoid_lab import policy_io as brax_policy_io

    got = brax_policy_io.load_params(runs / "warm" / "checkpoints" / f"{0:012d}")
    want = brax_policy_io.load_params(last)
    _assert_leaves_equal(got, want)


def test_max_minutes_stops_after_a_checkpoint(runs):
    _train("torch_warp.max_minutes=1e-9", "run_name=short")
    run = _run_json(runs / "short")
    assert run["early_stopped"] is True
    assert run["stopped_at_steps"] == STEPS_PER_ITER
    assert _steps(runs / "short") == [0, STEPS_PER_ITER]
    assert (runs / "short" / "DONE").exists()


def test_a_cut_off_step_dir_is_saved_again(tmp_path, monkeypatch):
    from humanoid_lab import policy_io as brax_policy_io

    monkeypatch.setattr(paths, "RUNS_DIR", tmp_path)
    step_dir = tmp_path / "partial" / "checkpoints" / f"{STEPS_PER_ITER:012d}"
    step_dir.mkdir(parents=True)
    (step_dir / "junk").write_text("cut off", encoding="utf-8")
    _train("torch_warp.max_minutes=1e-9", "run_name=partial")
    assert (step_dir / "ppo_network_config.json").exists()
    assert not (step_dir / "junk").exists()
    brax_policy_io.load_params(step_dir)


def test_a_resume_removes_a_cut_off_step_dir(tmp_path, monkeypatch):
    # The resumed run never saves at the cut-off step.
    monkeypatch.setattr(paths, "RUNS_DIR", tmp_path)
    _train("torch_warp.max_minutes=1e-9", "run_name=cut")
    run_dir = tmp_path / "cut"
    cut = run_dir / "checkpoints" / f"{5 * STEPS_PER_ITER:012d}"
    cut.mkdir()
    (cut / "junk").write_text("cut off", encoding="utf-8")
    _train("run_name=cut", "torch_warp.resume=true", steps=2 * STEPS_PER_ITER)
    assert not cut.exists()
    assert _steps(run_dir) == [0, STEPS_PER_ITER, 2 * STEPS_PER_ITER]


def test_a_new_run_refuses_an_existing_run_dir(runs):
    with pytest.raises(FileExistsError, match="torch_warp.resume=true"):
        _train("run_name=scratch")


# A string trainer marks a legacy Torch run (run_record.is_legacy).
@pytest.mark.parametrize("trainer", ["train.py (torch+MJWarp)", {"name": "brax_ppo"}])
def test_only_a_torch_run_can_be_resumed(scratch, tmp_path, trainer):
    run = _run_json(scratch)
    run["trainer"] = trainer
    run_dir = tmp_path / "scratch"
    run_dir.mkdir()
    run_record.write_run(run_dir, run)
    (run_dir / warp_ppo.STATE_FILE).write_bytes(b"")
    with pytest.raises(ValueError, match="restore="):
        warp_ppo.resume_record(run_dir, run["hydra_config"])


def test_a_brax_record_builds_an_equal_warp_joystick(tmp_path):
    from humanoid_lab.ppo_config import compose_ppo_params

    cfg = _compose("smoke=true", "wandb.enable=false", config_name="config")
    hydra = OmegaConf.to_container(cfg, resolve=True)
    env = registry.make_env(
        cfg.task.name,
        paths.REPO_ROOT / cfg.robot.dir,
        cfg.actuators.name,
        hydra["task"]["env"] or {},
        hydra["actuators"].get("overrides") or {},
    )
    record = run_record.build_run_record(
        env,
        hydra,
        compose_ppo_params(cfg),
        run_name="brax",
        checkpoint_dir=tmp_path / "checkpoints",
        contacts={},
        env_config=env.config.to_dict(),
        actuator_gains=run_record.actuator_gains(env, hydra),
        trainer=run_record.trainer_block("brax_ppo", "jax", "cpu"),
        results={"early_stopped": False, "stopped_at_steps": 0, "final_reward": 0.0},
        warm_start=None,
    )
    run_record.write_run(tmp_path, record)
    run = run_record.load_run(tmp_path)

    wenv = WarpJoystick(run["hydra_config"], 2, 20, device="cpu")
    want = env.resolved_constants()
    for f in dataclasses.fields(want):
        a, b = getattr(wenv.consts_np, f.name), getattr(want, f.name)
        if isinstance(b, np.ndarray):
            np.testing.assert_array_equal(np.asarray(a), b, err_msg=f.name)
            assert np.asarray(a).dtype == b.dtype, f.name
        elif isinstance(b, float) and math.isnan(b):
            assert math.isnan(a), f.name
        else:
            assert a == b, f.name


def test_resume_continues_the_run(scratch):
    step0 = scratch / "checkpoints" / f"{0:012d}"
    mtime = step0.stat().st_mtime_ns
    log_before = _log(scratch)
    last_before = _steps(scratch)[-1]

    _train("run_name=scratch", "torch_warp.resume=true", steps=160)
    run = _run_json(scratch)
    assert len(_log(scratch)) > len(log_before)
    assert len(run["trainer"]["resumes"]) == 1
    assert run["trainer"]["resumes"][0]["step"] == last_before
    assert step0.stat().st_mtime_ns == mtime
    assert _steps(scratch)[-1] > last_before
    assert run["stopped_at_steps"] == 160
    assert (scratch / "DONE").read_text(encoding="utf-8") == "160"

    with pytest.raises(ValueError, match="ppo.learning_rate"):
        _train("run_name=scratch", "torch_warp.resume=true", "++ppo.learning_rate=1e-4", steps=240)


# Every main() test above runs with smoke=true, which skips the probe.
@pytest.mark.parametrize("overrides", [[], ["domain_rand=true", "dr.foot_friction.enable=true"]])
def test_the_contact_probe_measures_peaks(overrides):
    from humanoid_lab import sim_budget
    from humanoid_lab.torch_warp import env as warp_env

    # probe_contacts imports WarpJoystick from torch_warp.env at call time.
    seen, probes = {}, []

    class Recording(WarpJoystick):
        def __init__(self, hydra, *args, **kwargs):
            seen.update(hydra)
            super().__init__(hydra, *args, **kwargs)
            probes.append(self)

    hydra = OmegaConf.to_container(_compose("torch_warp.device=cpu", *overrides), resolve=True)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(warp_env, "WarpJoystick", Recording)
        nacon, nefc = warp_ppo.probe_contacts(hydra, "cpu", 50, 0)
    assert seen["domain_rand"] is False
    assert seen["dr"] == {}
    assert len(probes) == 1 and probes[0].domain_rand is False
    assert type(nacon) is int and nacon > 0
    assert type(nefc) is int and nefc > 0
    consts = WarpJoystick({**hydra, "domain_rand": False, "dr": {}}, 1, 50, device="cpu").consts_np
    report = sim_budget.budget_report("warp", nacon, nefc, consts.naconmax_per_env, consts.njmax, 1)
    assert report["overflow"] is False
    assert report["rows_overflow"] is False
