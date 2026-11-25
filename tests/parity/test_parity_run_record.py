"""torch_warp.ppo.main end to end on the CPU warp device: run outputs,
warm start, resume, max_minutes and run.json through run_record.

The runs are tiny: N=4, unroll 5, batch 4, 2 minibatches, so one iteration
is 40 steps, and the scratch run is 2 iterations. paths.RUNS_DIR points at
a module tmp dir. The tests share the runs and never change them, so they
pass in any order. The resume test extends a run of its own.
"""

from __future__ import annotations

import dataclasses
import functools
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
    assert set(run) == set(run_record.RECORD_KEYS)
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
    assert (scratch / "trainer_state.pt").exists()
    assert (scratch / "trainer.pid").exists()

    log = _log(scratch)
    assert log
    for rec in log:
        assert LOG_KEYS <= set(rec)
        assert set(warp_ppo.LOSS_KEYS) <= set(rec["losses"])
        assert "rss" in rec["memory"]
        assert set(rec["budget"]) == {"nacon_pool_max", "nefc_max"}
        assert rec["terms"] and all(isinstance(v, float) for v in rec["terms"].values())
    assert log[0]["it"] == 1


def _brax_init(cfg):
    """The (normalizer, policy, value) brax ppo.train returns for cfg at num_timesteps=0.

    The call is built the way train.main builds it. At num_timesteps=0 brax
    returns its initial state without a training step.
    """
    from brax.training.agents.ppo import networks as ppo_networks
    from brax.training.agents.ppo import train as ppo

    from humanoid_lab.dr.randomize import make_domain_randomize
    from humanoid_lab.envs.wrappers import make_wrap_env_fn
    from humanoid_lab.ppo_config import (
        brax_train_kwargs,
        compose_ppo_params,
        resolve_episode_length,
    )

    hydra = OmegaConf.to_container(cfg, resolve=True)
    ppo_params = compose_ppo_params(cfg)
    env_overrides = dict(hydra["task"]["env"] or {})
    # The torch_warp config runs on warp. brax needs the JAX backend. The sizes and seed are the same.
    env_overrides["sim"] = {
        **(env_overrides.get("sim") or {}),
        "backend": "jax",
        "num_envs": int(max(ppo_params.num_envs, ppo_params.get("num_eval_envs", 0))),
    }
    env = registry.make_env(
        cfg.task.name,
        paths.REPO_ROOT / cfg.robot.dir,
        cfg.actuators.name,
        env_overrides,
        hydra["actuators"].get("overrides") or {},
    )
    resolve_episode_length(ppo_params, cfg, env._config.episode_length)
    training_params, net_kwargs, _repo = brax_train_kwargs(ppo_params)
    if cfg.domain_rand:
        training_params["randomization_fn"] = make_domain_randomize(env.mj_model, env.robot_spec, hydra["dr"])
    training_params["num_timesteps"] = 0
    _make_policy, params, _metrics = ppo.train(
        environment=env,
        **training_params,
        network_factory=functools.partial(ppo_networks.make_ppo_networks, **net_kwargs),
        seed=cfg.seed,
        wrap_env_fn=make_wrap_env_fn(env._config),
    )
    return params


def test_a_scratch_run_starts_from_the_weights_brax_starts_from(scratch):
    """Owner decision 2: the step-0 networks and normalizer are brax ppo.train's own init at the run seed."""
    from humanoid_lab import policy_io as brax_policy_io

    want = _brax_init(_compose(*BASE, "++ppo.num_timesteps=80", "run_name=scratch"))
    got = brax_policy_io.load_params(scratch / "checkpoints" / f"{0:012d}")
    _assert_leaves_equal(got[1:], want[1:])
    got_norm, want_norm = got[0], want[0]
    assert int(got_norm.count.hi) == int(want_norm.count.hi) == 0
    assert int(got_norm.count.lo) == int(want_norm.count.lo) == 0
    for field in ("mean", "summed_variance", "std"):
        _assert_leaves_equal([getattr(got_norm, field)], [getattr(want_norm, field)])


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


@pytest.fixture(scope="module")
def resumed(runs):
    """A run trained like scratch, then resumed to 160 steps. Returns its dir and the facts before the resume."""
    run_dir = runs / "resumed"
    _train("run_name=resumed")
    before = {
        "step0_mtime": (run_dir / "checkpoints" / f"{0:012d}").stat().st_mtime_ns,
        "log_len": len(_log(run_dir)),
        "last_step": _steps(run_dir)[-1],
    }
    _train("run_name=resumed", "torch_warp.resume=true", steps=160)
    return run_dir, before


def test_resume_continues_the_run(resumed):
    run_dir, before = resumed
    run = _run_json(run_dir)
    assert len(_log(run_dir)) > before["log_len"]
    assert len(run["trainer"]["resumes"]) == 1
    assert run["trainer"]["resumes"][0]["step"] == before["last_step"]
    assert (run_dir / "checkpoints" / f"{0:012d}").stat().st_mtime_ns == before["step0_mtime"]
    assert _steps(run_dir)[-1] > before["last_step"]
    assert run["stopped_at_steps"] == 160
    assert (run_dir / "DONE").read_text(encoding="utf-8") == "160"

    with pytest.raises(ValueError, match="ppo.learning_rate"):
        _train("run_name=resumed", "torch_warp.resume=true", "++ppo.learning_rate=1e-4", steps=240)


# Every main() test above runs with smoke=true, which skips the probe.
def test_the_contact_probe_measures_peaks_with_dr_off():
    hydra = OmegaConf.to_container(_compose("torch_warp.device=cpu"), resolve=True)
    nacon, nefc = warp_ppo.probe_contacts(hydra, "cpu", 50, 0)
    assert type(nacon) is int and nacon > 0
    assert type(nefc) is int and nefc > 0
    # The probe turns DR off. With foot friction DR on, the same steps reach other peaks.
    dr = OmegaConf.to_container(
        _compose("torch_warp.device=cpu", "domain_rand=true", "dr.foot_friction.enable=true"), resolve=True
    )
    assert warp_ppo.probe_contacts(dr, "cpu", 50, 0) == (nacon, nefc)
