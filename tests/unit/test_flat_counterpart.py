"""The counterpart tasks a run is measured on, and the arena check.

A terrain run's env config is joystick's plus one `terrain` block, so its
flat counterpart is joystick with that block dropped, and a joystick run's
terrain counterpart is the terrain task with the block's defaults. The
measurement env's arguments are pure functions of run.json, so the whole
matrix is checked here without building an env.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

from humanoid_lab import registry, tasks
from humanoid_lab.eval.battery import check_run_arena, measurement_env_args
from humanoid_lab.terrain import GENERATOR_VERSION, fingerprint
from humanoid_lab.terrain.config import CPU_ARENA, arena_for, params_from_config

CRITIC = ["gyro", "gravity", "height", "height_scan_clean"]
# Fresh copies, so no test here can write through to CPU_ARENA or CRITIC.
TERRAIN_ENV = {
    "terrain": {"arena": copy.deepcopy(CPU_ARENA), "spawn": {"mode": "feature"}},
    "obs": {"privileged": list(CRITIC)},
    "episode_length": 50,
}
JOYSTICK_ENV = {"obs": {"privileged": CRITIC[:3]}, "episode_length": 50}
# What _measurement_env_overrides adds to every run's overrides.
MEASUREMENT_BLOCKS = {"push", "command", "no_progress"}


def run_for(task: str, env: dict) -> dict:
    """The run.json fields the measurement env reads."""
    return {"task": task, "hydra_config": {"task": {"name": task, "env": copy.deepcopy(env)}}}


# -- counterparts ----------------------------------------------------------------


def test_flat_counterpart_drops_only_the_terrain_block():
    task, overrides = registry.flat_counterpart("terrain", TERRAIN_ENV)
    assert task == "joystick"
    assert overrides == {k: v for k, v in TERRAIN_ENV.items() if k != "terrain"}
    assert "terrain" in TERRAIN_ENV


@pytest.mark.parametrize("task", ["joystick", "sizing"])
def test_flat_counterpart_is_identity_for_joystick_and_sizing(task):
    assert registry.flat_counterpart(task, JOYSTICK_ENV) == (task, JOYSTICK_ENV)
    assert registry.flat_counterpart(task, None) == (task, {})


@pytest.mark.parametrize("task, env", [("joystick", JOYSTICK_ENV), ("terrain", TERRAIN_ENV)])
def test_terrain_counterpart_accepts_joystick_and_terrain_runs(task, env):
    assert registry.terrain_counterpart(task, env) == ("terrain", env)


def test_sizing_has_no_terrain_counterpart():
    with pytest.raises(ValueError, match="task 'sizing' has no terrain counterpart"):
        registry.terrain_counterpart("sizing", {})


@pytest.mark.parametrize("task, env", [("terrain", TERRAIN_ENV), ("joystick", JOYSTICK_ENV)])
def test_the_counterparts_return_copies(task, env):
    """No block of a counterpart's overrides is the run's own. Writing to
    one leaves the run's overrides as they were."""
    env = copy.deepcopy(env)
    before = copy.deepcopy(env)
    _, flat = registry.flat_counterpart(task, env)
    _, terrain = registry.terrain_counterpart(task, env)
    for got in (flat, terrain):
        assert got is not env
        assert got["obs"] is not env["obs"]
        assert got["obs"]["privileged"] is not env["obs"]["privileged"]
    if "terrain" in env:
        assert terrain["terrain"] is not env["terrain"]
        assert terrain["terrain"]["arena"] is not env["terrain"]["arena"]

    flat["obs"]["privileged"].append("linvel")
    terrain["obs"]["privileged"].append("linvel")
    terrain.setdefault("terrain", {}).setdefault("arena", {})["border"] = 9.0
    assert env == before


def test_the_counterpart_tasks_are_registered():
    for terrain_task, flat_task in tasks.FLAT_COUNTERPART.items():
        assert terrain_task in tasks.TERRAIN_TASKS
        assert flat_task in registry.TASKS and terrain_task in registry.TASKS
    assert tasks.TERRAIN_TASK in tasks.TERRAIN_TASKS
    assert set(tasks.TERRAIN_SOURCE_TASKS) <= set(registry.TASKS)


# -- measurement_env_args --------------------------------------------------------


@pytest.mark.parametrize(
    "task, env, flat, want_task, want_terrain",
    [
        ("terrain", TERRAIN_ENV, True, "joystick", None),
        ("terrain", TERRAIN_ENV, False, "terrain", TERRAIN_ENV["terrain"]),
        ("joystick", JOYSTICK_ENV, True, "joystick", None),
        ("joystick", JOYSTICK_ENV, False, "terrain", None),
        ("sizing", JOYSTICK_ENV, True, "sizing", None),
    ],
)
def test_measurement_env_args_matrix(task, env, flat, want_task, want_terrain):
    got_task, overrides = measurement_env_args(run_for(task, env), flat=flat)
    assert got_task == want_task
    assert overrides.get("terrain") == want_terrain
    # The run's own overrides survive, and the measurement blocks are added.
    assert overrides["obs"] == env["obs"]
    assert overrides["episode_length"] == 50
    assert MEASUREMENT_BLOCKS <= set(overrides)
    assert overrides["push"]["enable"] is False
    assert overrides["no_progress"]["enable"] is False


def test_a_terrain_block_in_the_extra_overrides_refuses_the_flat_rebuild():
    extra = {"terrain": {"arena": CPU_ARENA}}
    with pytest.raises(ValueError, match="flat=False"):
        measurement_env_args(run_for("terrain", TERRAIN_ENV), extra, flat=True)
    with pytest.raises(ValueError, match="flat=False"):
        measurement_env_args(run_for("joystick", JOYSTICK_ENV), extra)


def test_sizing_has_no_terrain_measurement():
    with pytest.raises(ValueError, match="no terrain counterpart"):
        measurement_env_args(run_for("sizing", JOYSTICK_ENV), flat=False)


def test_an_extra_terrain_sub_block_replaces_the_runs_whole():
    """The merge is one level deep: `terrain.arena` in `extra` replaces the
    run's arena block whole, and the run's other terrain blocks stay."""
    arena = {"difficulties": [0.2], "flat_row": False}
    _, overrides = measurement_env_args(
        run_for("terrain", TERRAIN_ENV), {"terrain": {"arena": arena}}, flat=False
    )
    assert overrides["terrain"] == {"arena": arena, "spawn": {"mode": "feature"}}


def test_the_flat_rebuild_keeps_the_extra_overrides():
    _, overrides = measurement_env_args(
        run_for("terrain", TERRAIN_ENV), {"sim": {"backend": "auto", "num_envs": 1}}
    )
    assert overrides["sim"] == {"backend": "auto", "num_envs": 1}
    assert "terrain" not in overrides


# -- check_run_arena -------------------------------------------------------------


@pytest.fixture(scope="module")
def cpu_env():
    """An object carrying the CPU arena where the check reads it."""
    return SimpleNamespace(_arena=arena_for(params_from_config(CPU_ARENA)))


def recorded(env, **changes) -> dict:
    """A run.json arena record of `env`'s arena, with `changes`."""
    arena = {"generator_version": GENERATOR_VERSION, "fingerprint": fingerprint(env._arena)}
    return {"arena": {**arena, **changes}}


def test_the_runs_own_arena_passes(cpu_env):
    check_run_arena(cpu_env, recorded(cpu_env))


@pytest.mark.parametrize("arena", [None, {}])
def test_a_run_without_an_arena_record_passes(cpu_env, arena):
    check_run_arena(cpu_env, {"arena": arena})
    check_run_arena(cpu_env, {})


def test_a_different_fingerprint_is_refused_naming_both(cpu_env):
    stale = "0" * 64
    with pytest.raises(ValueError, match="terrain.arena") as excinfo:
        check_run_arena(cpu_env, recorded(cpu_env, fingerprint=stale))
    assert stale in str(excinfo.value)
    assert fingerprint(cpu_env._arena) in str(excinfo.value)


def test_a_different_generator_version_is_refused(cpu_env):
    with pytest.raises(ValueError, match=f"generator version {GENERATOR_VERSION + 1}"):
        check_run_arena(cpu_env, recorded(cpu_env, generator_version=GENERATOR_VERSION + 1))
