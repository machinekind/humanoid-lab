"""check_friction's CLI wiring: the flags that reach `probe`, and the exit
status.

Model-free. `probe` is replaced by a recorder, so main() runs without
building an env. The exit status, not the printed text, is the CLI's
contract: 0 on PASS, 1 on FAIL. The probe itself is
tests/integration/test_check_friction.py.
"""

from __future__ import annotations

import sys

import pytest

from humanoid_lab import check_friction

ROBOT_ARGS = ["--robot", "roboto_origin", "--preset", "deploy_pd"]


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
