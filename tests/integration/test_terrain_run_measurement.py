"""A terrain run measured, exported and sized: the flat rebuild and the
terrain rebuild of a run directory.

roboto_origin under deploy_pd. The terrain run trains on the CPU arena with
the height scan on its critic, as configs/task/terrain.yaml lists it. Each
run directory holds a random-init checkpoint written by brax's own
checkpoint code (run_fixtures.write_random_run) and the run.json train.py
writes, arena record included.
"""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jp
import mujoco
import numpy as np
import pytest
from run_fixtures import write_random_run

from humanoid_lab import deploy_contract as dc
from humanoid_lab import paths
from humanoid_lab.envs import height_scan
from humanoid_lab.envs.joystick import Joystick
from humanoid_lab.envs.joystick import default_config as joystick_default_config
from humanoid_lab.envs.terrain_joystick import TerrainJoystick
from humanoid_lab.eval import battery
from humanoid_lab.export import policy as export
from humanoid_lab.registry import make_env
from humanoid_lab.sizing import collect
from humanoid_lab.terrain import fingerprint, scene
from humanoid_lab.terrain.config import CPU_ARENA

ROBOT_DIR = paths.ROBOTS_DIR / "roboto_origin"
PRESET = "deploy_pd"
FLAT_CRITIC = list(joystick_default_config().obs.privileged)
TERRAIN_OVERRIDES = {
    "terrain": {"arena": CPU_ARENA},
    "obs": {"privileged": [*FLAT_CRITIC, height_scan.NAME]},
    # roboto_origin's overlay box. The default back_vx lies outside it.
    "command": {"vx": [-0.6, 1.0]},
}
CPU = {"terrain": {"arena": CPU_ARENA}}


@pytest.fixture(scope="module")
def terrain_env():
    return make_env("terrain", ROBOT_DIR, PRESET, TERRAIN_OVERRIDES)


@pytest.fixture(scope="module")
def flat_env():
    return make_env("joystick", ROBOT_DIR, PRESET, {})


@pytest.fixture(scope="module")
def terrain_run(terrain_env, tmp_path_factory) -> Path:
    run_dir = tmp_path_factory.mktemp("terrain_run")
    write_random_run(terrain_env, run_dir, "terrain", TERRAIN_OVERRIDES, preset=PRESET)
    return run_dir


@pytest.fixture(scope="module")
def joystick_run(flat_env, tmp_path_factory) -> Path:
    run_dir = tmp_path_factory.mktemp("joystick_run")
    write_random_run(flat_env, run_dir, "joystick", {}, preset=PRESET)
    return run_dir


@pytest.fixture(scope="module")
def flat_loaded(terrain_run):
    """The terrain run through the measurement loader's default."""
    return battery.load_checkpoint_policy(terrain_run)


@pytest.fixture(scope="module")
def terrain_export(terrain_run):
    return export.load_for_export(terrain_run)


def run_json(run_dir: Path) -> dict:
    return json.loads((run_dir / "run.json").read_text())


def acts(env, inference) -> np.ndarray:
    """One action of `inference` on zero observations of `env`'s widths."""
    obs = {key: jp.zeros(shape) for key, shape in env.observation_size.items()}
    action, _ = inference(obs, jax.random.PRNGKey(0))
    return np.asarray(action)


def test_the_fixture_run_records_its_arena(terrain_run, terrain_env, flat_env):
    arena = run_json(terrain_run)["arena"]
    assert arena["fingerprint"] == fingerprint(terrain_env._arena)
    critic = terrain_env.observation_size["privileged_state"][0]
    assert critic == flat_env.observation_size["privileged_state"][0] + height_scan.SIZE


def test_a_terrain_run_loads_as_the_flat_joystick(flat_loaded, terrain_env):
    run, env, _ckpt, inference = flat_loaded
    assert run["task"] == "terrain"
    assert type(env) is Joystick
    assert not scene.is_terrain_model(env.mj_model)
    assert np.any(env.mj_model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE)
    # Both widths are training's: the flat floor serves the critic's scan.
    assert env.observation_size == terrain_env.observation_size
    assert env.actor_obs_names == terrain_env.actor_obs_names
    assert list(env._config.obs.privileged) == TERRAIN_OVERRIDES["obs"]["privileged"]
    action = acts(env, inference)
    assert action.shape == (env.action_size,) and np.all(np.isfinite(action))



def test_a_terrain_run_gets_flat_courses(terrain_run):
    """The courses runner measures a terrain run on its flat rebuild, and
    the document names the task it measured."""
    from humanoid_lab.eval.courses import runner

    doc = runner.run_courses(terrain_run, seeds=1, only=["spin_left"], budget_cap=20)
    assert doc["ground_class"] == "flat"
    assert (doc["trained_task"], doc["measured_task"]) == ("terrain", "joystick")
    (seed,) = doc["courses"]["spin_left"]["per_seed"]
    assert 0 < seed["steps"] <= 20

def test_a_terrain_run_exports_and_validates(
    terrain_export, terrain_run, joystick_run, flat_env, tmp_path
):
    loaded = terrain_export
    assert type(loaded.env) is Joystick
    assert loaded.meta["task"] == "terrain"
    joystick_meta = dc.build_contract(flat_env, run_json(joystick_run), "ckpt")
    assert loaded.meta["obs_layout"] == joystick_meta["obs_layout"]
    assert loaded.meta["obs_size"] == joystick_meta["obs_size"]

    forward = export.validate_numpy_vs_brax(
        loaded.weights, loaded.meta, loaded.inference, loaded.privileged_size
    )
    assert forward < export.TOLERANCE
    out_dir = export.write_validated(
        loaded.env, loaded.meta, loaded.weights, tmp_path / "deploy",
        loaded.inference, loaded.privileged_size,
    )
    runtime_error = export.validate_runtime_vs_env(
        loaded.env, out_dir, loaded.inference, loaded.privileged_size
    )
    assert runtime_error < export.TOLERANCE
    meta = json.loads((out_dir / "policy_meta.json").read_text())
    assert meta["task"] == "terrain" and meta["run_name"] == terrain_run.name


def test_the_contract_checks_a_terrain_runs_command_bias(terrain_export):
    """An armed back draw leaves roboto_origin's vx box. The flat env has
    no bias, so the contract reads the run's own."""
    loaded = terrain_export
    run = json.loads(json.dumps(loaded.run))
    run["env_config"]["terrain"]["command_bias"].update(enable=True, pure_back_prob=0.1)
    with pytest.raises(ValueError, match="terrain.command_bias"):
        dc.build_contract(loaded.env, run, "ckpt")


def test_flat_false_builds_the_terrain_env_for_terrain_and_joystick_runs(
    terrain_run, joystick_run, terrain_env, flat_env
):
    _, env, _, inference = battery.load_checkpoint_policy(terrain_run, flat=False)
    assert type(env) is TerrainJoystick
    assert fingerprint(env._arena) == run_json(terrain_run)["arena"]["fingerprint"]
    assert env.observation_size == terrain_env.observation_size
    assert np.all(np.isfinite(acts(env, inference)))

    # A joystick run records no arena, so the caller names one. The run
    # keeps its flat critic.
    _, env, _, inference = battery.load_checkpoint_policy(joystick_run, CPU, flat=False)
    assert type(env) is TerrainJoystick
    assert fingerprint(env._arena) == fingerprint(terrain_env._arena)
    assert env.observation_size == flat_env.observation_size
    assert np.all(np.isfinite(acts(env, inference)))


def test_a_stale_arena_is_refused(terrain_run, tmp_path):
    """run.json's fingerprint stands in for an arena a different generator
    built. Another terrain sub-block keeps the run's arena and still gets
    the check. An explicit terrain.arena is the caller's choice and passes."""
    run = run_json(terrain_run)
    built = run["arena"]["fingerprint"]
    run["arena"]["fingerprint"] = "0" * 64
    stale = tmp_path / "stale_run"
    stale.mkdir()
    (stale / "run.json").write_text(json.dumps(run))

    with pytest.raises(ValueError, match="terrain.arena") as excinfo:
        battery.load_checkpoint_policy(stale, flat=False)
    assert "0" * 64 in str(excinfo.value) and built in str(excinfo.value)

    spawn = {"terrain": {"spawn": {"yaw": False}}}
    with pytest.raises(ValueError, match="terrain.arena") as excinfo:
        battery.load_checkpoint_policy(stale, spawn, flat=False)
    assert "0" * 64 in str(excinfo.value)

    _, env, _, _ = battery.load_checkpoint_policy(stale, CPU, flat=False)
    assert fingerprint(env._arena) == built


def test_sizing_rebuilds_a_terrain_run_flat(terrain_run, terrain_env, capsys):
    env, *_ = collect.make_env_for_run(collect._load_run(terrain_run))
    assert "sizing on the flat scene" in capsys.readouterr().out
    assert type(env) is Joystick
    assert not scene.is_terrain_model(env.mj_model)
    assert env.observation_size == terrain_env.observation_size
