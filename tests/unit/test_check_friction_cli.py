"""check_friction's CLI wiring: the flags that reach `probe`, the exit
status, and the overrides `_build_env` hands to `make_env`.

Model-free. `probe` and `make_env` are replaced by recorders, so nothing
builds an env. The exit status, not the printed text, is the CLI's
contract: 0 on PASS, 1 on FAIL. The probe itself is
tests/integration/test_check_friction.py.
"""

from __future__ import annotations

import sys

import pytest
from ml_collections import config_dict

from humanoid_lab import check_friction, paths, sim_budget
from humanoid_lab.robot.spec import load_robot_spec
from humanoid_lab.terrain.config import require_terrain_budgets

ROBOT = "roboto_origin"
ROBOT_ARGS = ["--robot", ROBOT, "--preset", "deploy_pd"]


@pytest.fixture
def stub_probe(monkeypatch):
    """Replace probe with a recorder. Returns the dict its arguments land
    in. Set seen["result"] to choose what the probe returns."""
    seen = {"result": True}

    def fake_probe(robot, preset, backend, num_envs, friction_range, task="joystick"):
        seen["call"] = {
            "robot": robot,
            "preset": preset,
            "backend": backend,
            "num_envs": num_envs,
            "friction_range": list(friction_range),
            "task": task,
        }
        return seen["result"]

    monkeypatch.setattr(check_friction, "probe", fake_probe)
    return seen


def _main(argv, monkeypatch) -> int:
    monkeypatch.setattr(sys, "argv", ["check-friction", *argv])
    with pytest.raises(SystemExit) as exit_info:
        check_friction.main()
    return exit_info.value.code


def test_the_task_flag_reaches_probe(monkeypatch, stub_probe):
    assert _main([*ROBOT_ARGS, "--task", "terrain"], monkeypatch) == 0
    assert stub_probe["call"] == {
        "robot": "roboto_origin",
        "preset": "deploy_pd",
        "backend": "auto",
        "num_envs": 8,
        "friction_range": [0.8, 1.2],
        "task": "terrain",
    }


def test_the_task_defaults_to_the_flat_path(monkeypatch, stub_probe):
    assert _main(ROBOT_ARGS, monkeypatch) == 0
    assert stub_probe["call"]["task"] == "joystick"


def test_an_unknown_task_is_a_usage_error(monkeypatch, stub_probe):
    assert _main([*ROBOT_ARGS, "--task", "stairs"], monkeypatch) == 2
    assert "call" not in stub_probe


def test_a_failed_probe_exits_nonzero(monkeypatch, stub_probe):
    stub_probe["result"] = False
    assert _main([*ROBOT_ARGS, "--task", "terrain"], monkeypatch) == 1


@pytest.fixture
def env_recorder(monkeypatch):
    """Replace make_env with a recorder. Returns the dict its arguments
    land in."""
    seen = {}

    def record_env(task, robot_dir, preset, env_overrides=None, actuator_overrides=None):
        seen.update(task=task, robot_dir=robot_dir, preset=preset, env_overrides=env_overrides)
        return object()

    monkeypatch.setattr(check_friction, "make_env", record_env)
    return seen


@pytest.mark.parametrize("backend", ["warp", "auto"])
def test_terrain_takes_the_flat_budget_the_warp_refusal_needs(backend, env_recorder):
    """The terrain task refuses warp without explicit naconmax_per_env and
    njmax. Every probe world stands on a flat-row pad, so `_build_env`
    passes robot.yaml's flat-floor budget in. "auto" is warp on a CUDA
    host. The caller's keys stay as given."""
    sim = {"backend": backend, "num_envs": 2}
    check_friction._build_env("terrain", ROBOT, "deploy_pd", sim)
    assert env_recorder["task"] == "terrain"
    got = env_recorder["env_overrides"]["sim"]
    budget = load_robot_spec(paths.ROBOTS_DIR / ROBOT).sim_budget
    want = {"naconmax_per_env": budget["naconmax_per_env"], "njmax": budget["njmax"], **sim}
    assert got == want
    require_terrain_budgets(
        "warp", config_dict.ConfigDict(got), ccd_slot_bytes=sim_budget.ccd_slot_bytes(35, box_box=True)
    )


def test_the_flat_path_gets_the_sim_block_unchanged(env_recorder):
    sim = {"backend": "warp", "num_envs": 2}
    check_friction._build_env("joystick", ROBOT, "deploy_pd", sim)
    assert env_recorder["task"] == "joystick"
    assert env_recorder["env_overrides"] == {"sim": sim}
