"""run_record: load_run, build_run_record and write_run, model-free.

Every test writes a run.json under tmp_path. Nothing builds a model.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from ml_collections import config_dict

from humanoid_lab import deploy_contract, paths, registry, run_record
from humanoid_lab.envs import joystick

TRAINER = "wtrain/train.py (torch+MJWarp)"


def _brax_run(env=None):
    return {
        "run_name": "fixture",
        "task": "joystick",
        "checkpoint_dir": "runs/fixture/checkpoints",
        "env_config": {"gait": {"freq": [1.0, 1.5]}},
        "ppo_config": {"normalize_observations": True},
        "hydra_config": {
            "robot": {"dir": "robots/roboto_origin"},
            "actuators": {"name": "deploy_pd"},
            "task": {"name": "joystick", "env": copy.deepcopy(env) if env is not None else {}},
        },
    }


def _legacy_run(env=None, **legacy):
    run = _brax_run(env)
    run["trainer"] = TRAINER
    run.update(legacy)
    return run


def _write(tmp_path, run):
    (tmp_path / "run.json").write_text(json.dumps(run), encoding="utf-8")
    return tmp_path


def _load(tmp_path, run):
    return run_record.load_run(_write(tmp_path, run))


def _task_env(run):
    return run["hydra_config"]["task"]["env"]


def _resolved(run):
    """`run` as load_run returns it: the relative checkpoint_dir resolved."""
    return {**run, "checkpoint_dir": str(paths.PROJECT_DIR / "runs/fixture/checkpoints")}


# -- Brax runs and the required keys ------------------------------------------


def test_a_brax_run_loads_unchanged(tmp_path):
    run = _brax_run({"command": {"deadband": 0.1}})
    assert _load(tmp_path, run) == _resolved(run)


def test_a_brax_run_with_a_frozen_clock_loads(tmp_path):
    """A Brax run with freq [0, 0] trained on cos/sin, the default."""
    run = _brax_run({"gait": {"freq": [0.0, 0.0]}})
    assert _load(tmp_path, run) == _resolved(run)


def test_a_missing_run_json_is_a_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError, match="completed run.sh train/smoke output dir"):
        run_record.load_run(tmp_path)


@pytest.mark.parametrize("key", run_record.REQUIRED_KEYS)
def test_a_missing_top_level_key_is_refused(tmp_path, key):
    run = _brax_run()
    del run[key]
    with pytest.raises(ValueError, match=rf"missing required key\(s\) \['{key}'\]"):
        _load(tmp_path, run)


@pytest.mark.parametrize("section,key", [("robot", "dir"), ("actuators", "name")])
def test_a_missing_robot_dir_or_preset_is_refused(tmp_path, section, key):
    run = _brax_run()
    del run["hydra_config"][section][key]
    with pytest.raises(ValueError, match=rf"hydra_config\.{section}\.{key} missing"):
        _load(tmp_path, run)


def test_a_missing_checkpoint_dir_on_disk_is_not_checked(tmp_path):
    run = _brax_run()
    run["checkpoint_dir"] = str(tmp_path / "nowhere")
    assert _load(tmp_path, run)["checkpoint_dir"] == run["checkpoint_dir"]


# -- legacy detection ----------------------------------------------------------


def test_a_string_trainer_alone_marks_a_legacy_run(tmp_path):
    run = _legacy_run()
    loaded = _load(tmp_path, run)
    assert run_record.is_legacy(run)
    # env_config is replaced by the (empty) migrated task.env.
    assert loaded["env_config"] == {}


def test_a_warp_key_alone_marks_a_legacy_run(tmp_path):
    run = _brax_run()
    run["warp_cmd_straight"] = 0.3
    assert run_record.is_legacy(run)
    assert _task_env(_load(tmp_path, run))["command"]["straight_prob"] == 0.3


def test_a_dict_trainer_is_not_legacy():
    run = _brax_run()
    run["trainer"] = {"name": "brax"}
    assert not run_record.is_legacy(run)


# -- migration -------------------------------------------------------------------

STEP_LENGTH_LINEAR = {"scale": 2.0, "target_a": 0.1, "target_b": 0.3, "power": 1.5, "pair_min": True}
FOOT_REACH = {"scale": 1.0, "target_a": 0.05, "target_b": 0.1, "power": 2.0, "x_min": -0.02, "pair_min": True}
KNEE_EXTEND = {
    "scale": 0.5,
    "flex_lo": 0.1,
    "flex_hi": 0.4,
    "x_split": 0.0,
    "td_min_air": 0.2,
    "td_mult": 1.0,
    "pair_min": False,
}
FORWARD_LEAN = {"scale": 0.3, "max_deg": 6.0, "vx_max": 1.0, "sigma": 0.01}


def _full_legacy_run():
    return _legacy_run(
        env={"gait": {"freq": [1.0, 1.5]}, "command": {"deadband": 0.1}},
        warp_reward={
            "step_length": STEP_LENGTH_LINEAR,
            "foot_reach": FOOT_REACH,
            "knee_extend": KNEE_EXTEND,
            "forward_lean": FORWARD_LEAN,
        },
        warp_cmd_ramp={"lin": 0.25, "yaw": 0.6},
        warp_cmd_straight=0.2,
        warp_lift={"v_full": 0.5, "min_frac": 0.25},
    )


@pytest.mark.parametrize(
    "term,spec",
    [
        ("step_length", STEP_LENGTH_LINEAR),
        ("foot_reach", FOOT_REACH),
        ("knee_extend", KNEE_EXTEND),
        ("forward_lean", FORWARD_LEAN),
    ],
)
def test_each_kept_term_migrates_its_scale_and_parameters(tmp_path, term, spec):
    env = _task_env(_load(tmp_path, _legacy_run(warp_reward={term: spec})))
    params = {k: v for k, v in spec.items() if k != "scale"}
    if term == "step_length":
        params["target"] = "linear"
    assert env["reward"]["scales"] == {term: spec["scale"]}
    assert env["reward"][term] == params


def test_step_length_without_target_a_migrates_to_the_clock_target(tmp_path):
    run = _legacy_run(warp_reward={"step_length": {"scale": 1.0, "power": 2.0}})
    env = _task_env(_load(tmp_path, run))
    assert env["reward"]["step_length"] == {"power": 2.0, "target": "clock"}


def test_ramp_straight_and_lift_migrate(tmp_path):
    run = _legacy_run(
        warp_cmd_ramp={"lin": 0.25, "yaw": 0.6},
        warp_cmd_straight=0.2,
        warp_lift={"v_full": 0.5, "min_frac": 0.25},
    )
    env = _task_env(_load(tmp_path, run))
    assert env["command"] == {"ramp": {"lin": 0.25, "yaw": 0.6}, "straight_prob": 0.2}
    assert env["gait"] == {"lift": {"v_full": 0.5, "min_frac": 0.25}}


def test_the_migration_keeps_the_existing_task_env_keys(tmp_path):
    env = _task_env(_load(tmp_path, _full_legacy_run()))
    assert env["gait"]["freq"] == [1.0, 1.5]
    assert env["command"]["deadband"] == 0.1


def test_the_warp_keys_are_gone_and_env_config_is_the_migrated_task_env(tmp_path):
    loaded = _load(tmp_path, _full_legacy_run())
    assert not [k for k in loaded if k.startswith("warp_")]
    assert loaded["env_config"] == _task_env(loaded)
    assert loaded["env_config"] is not _task_env(loaded)
    assert loaded["trainer"] == TRAINER


def test_the_migrated_task_env_builds_a_config_and_passes_the_ledger(tmp_path):
    loaded = _load(tmp_path, _full_legacy_run())
    cfg = joystick.default_config()
    registry._apply_overrides(cfg, _task_env(loaded))
    deploy_contract.check_config_covered(loaded["env_config"])

    assert cfg.reward.scales.step_length == 2.0
    assert cfg.reward.step_length.target == "linear"
    assert cfg.reward.foot_reach.x_min == -0.02
    assert cfg.reward.knee_extend.td_mult == 1.0
    assert cfg.reward.forward_lean.sigma == 0.01
    assert cfg.command.ramp.lin == 0.25
    assert cfg.command.straight_prob == 0.2
    assert cfg.gait.lift.v_full == 0.5
    joystick.check_kept_terms(cfg.reward, cfg.gait, 2)
    joystick.check_command_ramp(cfg.command)


def test_a_matching_value_already_in_task_env_is_kept(tmp_path):
    run = _legacy_run(env={"command": {"straight_prob": 0.2}}, warp_cmd_straight=0.2)
    assert _task_env(_load(tmp_path, run))["command"]["straight_prob"] == 0.2


def test_falsy_legacy_knobs_are_off_and_dropped(tmp_path):
    loaded = _load(tmp_path, _legacy_run(warp_cmd_ramp=None, warp_cmd_straight=0.0, warp_lift={}))
    assert _task_env(loaded) == {}
    assert not [k for k in loaded if k.startswith("warp_")]


# -- refusals -------------------------------------------------------------------


def test_deleted_terms_are_refused_by_name(tmp_path):
    run = _legacy_run(
        warp_reward={
            "step_length": STEP_LENGTH_LINEAR,
            "march": {"scale": -1.0},
            "arm_swing": {"scale": 0.5, "k": 1.0, "sigma": 0.1},
        }
    )
    with pytest.raises(ValueError, match=r"\['arm_swing', 'march'\]\. PR #11 removed"):
        _load(tmp_path, run)


def test_an_unknown_parameter_is_refused(tmp_path):
    run = _legacy_run(warp_reward={"foot_reach": {**FOOT_REACH, "reach_bonus": 1.0}})
    with pytest.raises(ValueError, match=r"warp_reward\.foot_reach .*\['reach_bonus'\]"):
        _load(tmp_path, run)


def test_an_unknown_warp_key_is_refused(tmp_path):
    run = _legacy_run(warp_heading={"leak": 0.1})
    with pytest.raises(ValueError, match="warp_heading"):
        _load(tmp_path, run)


def test_a_conflicting_task_env_value_is_refused(tmp_path):
    run = _legacy_run(env={"command": {"straight_prob": 0.1}}, warp_cmd_straight=0.2)
    with pytest.raises(ValueError, match=r"task\.env\.command\.straight_prob = 0\.1 conflicts"):
        _load(tmp_path, run)


def test_a_conflicting_reward_scale_is_refused(tmp_path):
    run = _legacy_run(
        env={"reward": {"scales": {"forward_lean": 1.0}}}, warp_reward={"forward_lean": FORWARD_LEAN}
    )
    with pytest.raises(ValueError, match=r"reward\.scales\.forward_lean"):
        _load(tmp_path, run)


def test_a_legacy_frozen_clock_without_frozen_phase_obs_is_refused(tmp_path):
    """The plan's case: the stale env_config says the clock runs, task.env says frozen."""
    run = _legacy_run(env={"gait": {"freq": [0, 0]}})
    run["env_config"] = {"gait": {"freq": [1.0, 1.5]}}
    with pytest.raises(ValueError, match="task.env.gait.frozen_phase_obs") as err:
        _load(tmp_path, run)
    assert "'cos_sin'" in str(err.value) and "'zeros'" in str(err.value)


def test_a_legacy_frozen_clock_with_frozen_phase_obs_loads(tmp_path):
    run = _legacy_run(env={"gait": {"freq": [0, 0], "frozen_phase_obs": "zeros"}})
    run["env_config"] = {"gait": {"freq": [1.0, 1.5]}}
    loaded = _load(tmp_path, run)
    assert loaded["env_config"]["gait"] == {"freq": [0, 0], "frozen_phase_obs": "zeros"}


def test_a_legacy_running_clock_in_task_env_loads_whatever_env_config_says(tmp_path):
    run = _legacy_run(env={"gait": {"freq": [1.0, 1.5]}})
    run["env_config"] = {"gait": {"freq": [0.0, 0.0]}}
    assert _load(tmp_path, run)["env_config"]["gait"]["freq"] == [1.0, 1.5]


# -- the readers -----------------------------------------------------------------


def test_sizing_collect_still_needs_the_task_name(tmp_path):
    from humanoid_lab.sizing import collect

    run = _brax_run()
    del run["hydra_config"]["task"]["name"]
    _write(tmp_path, run)
    with pytest.raises(ValueError, match=r"hydra_config\.task\.name missing"):
        collect.collect(tmp_path, episodes=1, steps=1, seed=0)


def test_the_sizing_report_reads_a_legacy_run_through_load_run(tmp_path):
    from humanoid_lab.robot.spec import load_robot_spec
    from humanoid_lab.sizing import report

    joints = load_robot_spec(paths.REPO_ROOT / "robots/roboto_origin").actuated_joints
    n = len(joints)
    np.savez(
        tmp_path / "sizing_data.npz",
        tau=np.ones((2, n)),
        omega=np.ones((2, n)),
        joint_names=np.array(joints),
        effort_limit=np.ones(n),
        velocity_limit=np.ones(n),
    )
    _write(tmp_path, _full_legacy_run())
    md, _ = report.build_report(tmp_path, motors_name=None)
    assert "fixture" in md and "deploy_pd" in md

    # The migration runs, so a legacy run with a deleted term is refused.
    _write(tmp_path, _legacy_run(warp_reward={"march": {"scale": -1.0}}))
    with pytest.raises(ValueError, match=r"\['march'\]\. PR #11 removed"):
        report.build_report(tmp_path, motors_name=None)


# -- build_run_record and write_run ---------------------------------------------

JOINTS = ["hip", "knee", "ankle"]


def _env():
    return SimpleNamespace(
        mj_model=SimpleNamespace(
            actuator_gainprm=np.array([[40.0, 0, 0], [60.0, 0, 0], [20.0, 0, 0]]),
            actuator_biasprm=np.array([[0, -40.0, -1.0], [0, -60.0, -2.0], [0, -20.0, -0.5]]),
        ),
        robot_spec=SimpleNamespace(actuated_joints=JOINTS),
    )


def _hydra(restore=None, **env):
    return {
        "run_name": "built",
        "seed": 0,
        "restore": restore,
        "robot": {"dir": "robots/roboto_origin"},
        "actuators": {"name": "deploy_pd", "overrides": None},
        "task": {"name": "joystick", "env": env},
        "ppo": {"sym_coef": 0.5},
    }


def _ppo_params():
    return config_dict.ConfigDict(
        {"num_timesteps": 1000, "num_envs": 8, "sym_coef": 0.5, "network_factory": {"policy_obs_key": "state"}}
    )


def _record(hydra=None, checkpoint_dir=None, results=None, warm_start=None):
    env = _env()
    hydra = hydra or _hydra()
    return run_record.build_run_record(
        env,
        hydra,
        _ppo_params(),
        run_name="built",
        checkpoint_dir=checkpoint_dir or paths.PROJECT_DIR / "runs" / "built" / "checkpoints",
        contacts={"backend": "jax"},
        env_config={"episode_length": 1000},
        actuator_gains=run_record.actuator_gains(env, hydra),
        trainer=run_record.trainer_block("brax_ppo", "jax", "cpu"),
        results=results or {"early_stopped": False, "stopped_at_steps": 1000, "final_reward": 1.5},
        warm_start=warm_start,
    )


def test_the_record_has_origin_mains_keys_plus_trainer_and_warm_start():
    record = _record()
    assert tuple(record) == run_record.RECORD_KEYS
    assert run_record.RECORD_KEYS[: len(run_record.ORIGIN_MAIN_KEYS)] == run_record.ORIGIN_MAIN_KEYS
    assert not [k for k in record if k.startswith("warp_")]
    assert record["ppo_config"]["sym_coef"] == 0.5
    assert record["task"] == "joystick"
    assert record["num_timesteps"] == 1000
    assert record["actuators"] == {"name": "deploy_pd", "overrides": None}
    assert record["actuator_gains"]["joints"] == JOINTS
    assert record["actuator_gains"]["kp"] == [40.0, 60.0, 20.0]
    assert record["actuator_gains"]["kd"] == [1.0, 2.0, 0.5]
    assert record["actuator_gains"]["preset"] == "deploy_pd"
    assert record["trainer"]["name"] == "brax_ppo"
    assert record["trainer"]["device"] == "cpu"
    assert record["warm_start"] is None


def test_null_results_are_allowed():
    record = _record(results={"early_stopped": None, "stopped_at_steps": None, "final_reward": None})
    assert (record["early_stopped"], record["stopped_at_steps"], record["final_reward"]) == (None,) * 3


def test_paths_are_relative_posix_inside_the_project_and_absolute_outside(tmp_path):
    inside = paths.PROJECT_DIR / "runs" / "x" / "checkpoints"
    assert run_record.portable_path(inside) == "runs/x/checkpoints"
    # A relative input is relative to the project dir, not the cwd.
    assert run_record.portable_path(Path("runs") / "x" / ".." / "y") == "runs/y"
    outside = tmp_path / "ckpts"
    assert run_record.portable_path(outside) == outside.resolve().as_posix()
    assert Path(run_record.portable_path(outside)).is_absolute()
    assert _record(checkpoint_dir=inside)["checkpoint_dir"] == "runs/x/checkpoints"


def test_load_run_resolves_a_relative_checkpoint_dir(tmp_path):
    run = _brax_run()
    loaded = _load(tmp_path, run)
    assert loaded["checkpoint_dir"] == str(paths.PROJECT_DIR / "runs/fixture/checkpoints")
    # The file on disk is not rewritten.
    on_disk = json.loads((tmp_path / "run.json").read_text(encoding="utf-8"))
    assert on_disk["checkpoint_dir"] == "runs/fixture/checkpoints"
    absolute = str(tmp_path / "abs" / "checkpoints")
    run["checkpoint_dir"] = absolute
    assert _load(tmp_path, run)["checkpoint_dir"] == absolute
    # A null path stays null.
    run["checkpoint_dir"] = None
    assert _load(tmp_path, run)["checkpoint_dir"] is None
    assert run_record.resolve_path(None) is None


def test_restore_is_stored_portably():
    restore = str(paths.PROJECT_DIR / "runs" / "src" / "checkpoints" / "000000000100")
    hydra = _hydra(restore=restore)
    record = _record(hydra=hydra)
    assert record["hydra_config"]["restore"] == "runs/src/checkpoints/000000000100"
    # The caller's dict is not changed.
    assert hydra["restore"] == restore
    assert _record()["hydra_config"]["restore"] is None


def test_config_diff_lists_changed_added_and_removed_leaves():
    old = _hydra(gait={"freq": [1.0, 1.5]}, reward={"scales": {"a": 1.0, "gone": 2.0}})
    new = _hydra(gait={"freq": (1.0, 2.0)}, reward={"scales": {"a": 1.0, "new": 3.0}})
    new["actuators"] = {"name": "deploy_pd", "overrides": {"kp": 50}}
    new["ppo"] = {"sym_coef": 0.5}
    new["seed"] = 7  # outside the compared sections
    diff = run_record.config_diff(old, new)
    assert diff == {
        # overrides was a null leaf on the old side and is a block on the new.
        "actuators.overrides": [None, None],
        "actuators.overrides.kp": [None, 50],
        "task.env.gait.freq": [[1.0, 1.5], [1.0, 2.0]],
        "task.env.reward.scales.gone": [2.0, None],
        "task.env.reward.scales.new": [None, 3.0],
    }
    assert list(diff) == sorted(diff)
    assert run_record.config_diff(old, copy.deepcopy(old)) == {}
    # Equal lists and tuples are no change.
    assert run_record.config_diff(_hydra(gait={"freq": [1.0]}), _hydra(gait={"freq": (1.0,)})) == {}


def _source_run(tmp_path):
    run_dir = tmp_path / "src_run"
    ckpt = run_dir / "checkpoints" / "000000000100"
    ckpt.mkdir(parents=True)
    run = _brax_run({"gait": {"freq": [1.0, 1.5]}})
    run.update(final_reward=12.5, stopped_at_steps=100, early_stopped=False, num_timesteps=200)
    _write(run_dir, run)
    return run_dir, ckpt


def test_warm_start_block_reads_the_source_run(tmp_path):
    run_dir, ckpt = _source_run(tmp_path)
    composed = _hydra(gait={"freq": [1.0, 2.0]})
    for strict in (True, False):
        block = run_record.warm_start_block(ckpt, composed, strict=strict)
        assert set(block) == {"checkpoint", "source_run", "source_results", "config_diff"}
        assert block["checkpoint"] == ckpt.resolve().as_posix()
        assert block["source_run"] == run_dir.resolve().as_posix()
        assert block["source_results"] == {
            "final_reward": 12.5,
            "stopped_at_steps": 100,
            "early_stopped": False,
            "num_timesteps": 200,
        }
        assert block["config_diff"]["task.env.gait.freq"] == [[1.0, 1.5], [1.0, 2.0]]


def test_warm_start_block_without_a_source_run(tmp_path):
    no_run = tmp_path / "gone" / "checkpoints" / "000000000100"
    no_run.mkdir(parents=True)
    not_a_step = tmp_path / "loose_ckpt"
    not_a_step.mkdir()
    for ckpt in (no_run, not_a_step):
        with pytest.raises(ValueError, match=str(ckpt.name)):
            run_record.warm_start_block(ckpt, _hydra(), strict=True)
        block = run_record.warm_start_block(ckpt, _hydra(), strict=False)
        assert block == {
            "checkpoint": ckpt.resolve().as_posix(),
            "source_run": None,
            "source_results": None,
            "config_diff": None,
        }


def test_write_run_replaces_the_file_atomically(tmp_path):
    (tmp_path / "run.json").write_text("old", encoding="utf-8")
    record = _record(results={"early_stopped": False, "stopped_at_steps": 0, "final_reward": float("nan")})
    out = run_record.write_run(tmp_path, record)
    assert out == tmp_path / "run.json"
    assert [p.name for p in tmp_path.iterdir()] == ["run.json"]
    back = json.loads(out.read_text(encoding="utf-8"))
    assert math.isnan(back["final_reward"])
    assert list(back) == list(run_record.RECORD_KEYS)
