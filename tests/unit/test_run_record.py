"""run_record.load_run, model-free.

Every test writes a run.json under tmp_path. Nothing builds a model.
"""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from humanoid_lab import deploy_contract, registry, run_record
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


# -- Brax runs and the required keys ------------------------------------------


def test_a_brax_run_loads_unchanged(tmp_path):
    run = _brax_run({"command": {"deadband": 0.1}})
    assert _load(tmp_path, run) == run


def test_a_brax_run_with_a_frozen_clock_loads(tmp_path):
    """A Brax run with freq [0, 0] trained on cos/sin, the default."""
    run = _brax_run({"gait": {"freq": [0.0, 0.0]}})
    assert _load(tmp_path, run) == run


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


def test_step_length_with_target_a_migrates_to_the_linear_target(tmp_path):
    run = _legacy_run(warp_reward={"step_length": {"scale": 1.0, "target_a": 0.1, "target_b": 0.2}})
    env = _task_env(_load(tmp_path, run))
    assert env["reward"]["step_length"]["target"] == "linear"


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


def test_migrate_legacy_does_not_mutate_its_input():
    run = _full_legacy_run()
    before = copy.deepcopy(run)
    run_record.migrate_legacy(run)
    assert run == before


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


def test_the_frozen_phase_obs_values_match_the_env():
    assert run_record.FROZEN_PHASE_OBS == joystick.FROZEN_PHASE_OBS


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


def test_the_deleted_term_list_has_fourteen_names():
    assert len(set(run_record.DELETED_TERMS)) == 14
    assert not set(run_record.DELETED_TERMS) & set(run_record.KEPT_TERM_PARAMS)


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


def test_the_sizing_report_reads_through_load_run(tmp_path):
    from humanoid_lab.sizing import report

    names = np.array(["a"])
    np.savez(
        tmp_path / "sizing_data.npz",
        tau=np.zeros((1, 1)),
        omega=np.zeros((1, 1)),
        joint_names=names,
        effort_limit=np.ones(1),
        velocity_limit=np.ones(1),
    )
    run = _brax_run()
    del run["hydra_config"]["robot"]["dir"]
    _write(tmp_path, run)
    with pytest.raises(ValueError, match=r"hydra_config\.robot\.dir missing"):
        report.build_report(tmp_path)
