"""The terrain task's config: the arena block and its conversion to
`ArenaParams`, the arena cache, the task registration and yaml, the warp
budget refusal and the command bias check. Model-free."""

from __future__ import annotations

import pytest
from hydra import compose, initialize_config_dir
from ml_collections import config_dict
from omegaconf import OmegaConf

from humanoid_lab import paths, terrain
from humanoid_lab.envs.joystick import check_pure_draw_ranges
from humanoid_lab.envs.joystick import default_config as joystick_default_config
from humanoid_lab.envs.terrain_joystick import (
    TerrainJoystick,
    bias_command_config,
)
from humanoid_lab.envs.terrain_joystick import default_config as terrain_default_config
from humanoid_lab.registry import TASKS, _apply_overrides
from humanoid_lab.terrain.config import (
    CPU_ARENA,
    arena_for,
    config_from_params,
    params_from_config,
    require_terrain_budgets,
)


def test_arena_config_round_trips_to_the_default_params():
    block = config_from_params(terrain.ArenaParams())
    assert set(block["type_caps"]) == set(terrain.TYPES)
    assert all(cap == 1.0 for cap in block["type_caps"].values())
    assert params_from_config(block) == terrain.ArenaParams()


def test_a_cap_of_one_is_dropped_and_another_cap_reaches_the_params():
    block = config_from_params(terrain.ArenaParams())
    block["type_caps"]["pyramid_stairs"] = 0.7
    params = params_from_config(block)
    assert params.type_caps == (("pyramid_stairs", 0.7),)
    assert params.cap("pyramid_stairs") == 0.7
    assert params.cap("wave") == 1.0


def test_an_unknown_arena_key_is_refused():
    block = config_from_params(terrain.ArenaParams())
    with pytest.raises(TypeError, match="no_such_key"):
        params_from_config({**block, "no_such_key": 1})
    with pytest.raises(ValueError, match="unknown terrain types"):
        params_from_config({"type_caps": {"lava": 0.5}})
    # The env config block has a fixed key set, so an override naming an
    # unknown key or terrain type fails at the merge.
    with pytest.raises(AttributeError, match="no_such_key"):
        _apply_overrides(terrain_default_config(), {"terrain": {"arena": {"no_such_key": 1}}})
    with pytest.raises(AttributeError, match="lava"):
        _apply_overrides(
            terrain_default_config(), {"terrain": {"arena": {"type_caps": {"lava": 0.5}}}}
        )


def test_arena_overrides_reach_the_params():
    """The yaml shapes of the CPU arena and a tread range merge onto the
    block and build the params they name."""
    cfg = terrain_default_config()
    _apply_overrides(cfg, {"terrain": {"arena": {**CPU_ARENA, "stair_tread": [0.3, 0.4]}}})
    params = params_from_config(cfg.terrain.arena.to_dict())
    assert params == terrain.ArenaParams(**CPU_ARENA, stair_tread=(0.3, 0.4))


def test_the_cpu_arena_is_small():
    arena = arena_for(params_from_config(CPU_ARENA))
    s = arena.spec
    assert len(arena.boxes) == 33
    assert all(b.yaw == 0.0 for b in arena.boxes)
    assert (s.hfield.nrow, s.hfield.ncol) == (221, 821)
    flat = [t for t in s.tiles if t.row == 0]
    assert len(flat) == len(terrain.TYPES)
    half = s.params.tile_size / 2
    assert min(t.origin[0] for t in flat) - half == -16.0
    assert max(t.origin[0] for t in flat) + half == 16.0
    assert {t.origin[1] for t in flat} == {-2.0}


def test_arena_for_caches_a_read_only_arena():
    params = params_from_config(CPU_ARENA)
    arena = arena_for(params)
    assert arena_for(params_from_config(CPU_ARENA)) is arena
    assert not arena.lookup.flags.writeable
    assert not arena.hfield_data.flags.writeable


def test_terrain_default_config_is_joystick_plus_one_block():
    """A terrain run turns flat by dropping `terrain`, and any flat config
    measures on terrain by adding it."""
    terrain_cfg = terrain_default_config().to_dict()
    flat = joystick_default_config().to_dict()
    block = terrain_cfg.pop("terrain")
    assert terrain_cfg == flat
    assert set(block) == {
        "arena",
        "spawn",
        "curriculum",
        "base_contact",
        "no_progress",
        "command_bias",
        "jax_contacts",
    }
    assert params_from_config(block["arena"]) == terrain.ArenaParams()
    assert block["spawn"]["mode"] == "pad"
    assert block["command_bias"]["enable"] is False
    assert block["command_bias"]["pure_back_prob"] == 0.0


def test_terrain_is_registered():
    cls, default_config = TASKS["terrain"]
    assert cls is TerrainJoystick
    assert default_config is terrain_default_config


def _sim(**kw):
    sim = joystick_default_config().sim
    for key, value in kw.items():
        sim[key] = value
    return sim


def test_require_terrain_budgets(capsys):
    with pytest.raises(ValueError, match="flat-floor measurement.*check-terrain"):
        require_terrain_budgets("warp", _sim(num_envs=8), ccd_slot_bytes=5480)
    with pytest.raises(ValueError, match=r"naconmax_per_env and task\.env\.sim\.njmax are unset"):
        require_terrain_budgets("warp", _sim(num_envs=8), ccd_slot_bytes=5480)
    with pytest.raises(ValueError, match=r"task\.env\.sim\.njmax is unset"):
        require_terrain_budgets("warp", _sim(naconmax_per_env=128), ccd_slot_bytes=5480)

    sim = _sim(naconmax_per_env=128, njmax=1024, num_envs=8192)
    scratch = require_terrain_budgets("warp", sim, ccd_slot_bytes=5480)
    assert scratch == {
        "bytes_per_slot": 5480,
        "slots": 128 * 8192,
        "bytes": 128 * 8192 * 5480,
        "source": "naconmax pool",
    }
    assert "1048576 slots x 5480 B = 5.75 GB" in capsys.readouterr().out

    sim.naccdmax_per_env = 32
    scratch = require_terrain_budgets("warp", sim, ccd_slot_bytes=5480)
    assert scratch["slots"] == 32 * 8192
    assert scratch["source"] == "sim.naccdmax_per_env"
    capsys.readouterr()

    # jax has no fixed buffers: nothing to refuse or project.
    assert require_terrain_budgets("jax", _sim(), ccd_slot_bytes=5480) is None
    assert capsys.readouterr().out == ""


def _compose(overrides):
    with initialize_config_dir(version_base=None, config_dir=str(paths.CONFIGS_DIR)):
        return compose(config_name="config", overrides=overrides)


@pytest.mark.parametrize("robot", ["roboto_origin", "asimov_v1"])
def test_terrain_task_composes_as_joystick_plus_its_ppo_keys(robot):
    """terrain.yaml is joystick.yaml plus the critic's height scan and the
    PPO keys, and the robot overlays still patch it."""
    terrain_cfg = _compose([f"robot={robot}", "task=terrain"])
    flat = _compose([f"robot={robot}", "task=joystick"])
    assert terrain_cfg.task.name == "terrain"
    env = OmegaConf.to_container(terrain_cfg.task.env)
    flat_env = OmegaConf.to_container(flat.task.env)
    assert env["obs"].pop("privileged") == [*flat_env["obs"].pop("privileged"), "height_scan_clean"]
    assert env == flat_env
    assert OmegaConf.to_container(terrain_cfg.task.ppo) == {
        **OmegaConf.to_container(flat.task.ppo),
        "num_resets_per_eval": 0,
        "log_training_metrics": True,
        "training_metrics_steps": 10_000_000,
    }
    env_cfg = terrain_default_config()
    _apply_overrides(env_cfg, OmegaConf.to_container(terrain_cfg.task.env, resolve=True))
    assert tuple(env_cfg.command.vx) == tuple(flat.task.env.command.vx)


@pytest.mark.parametrize("robot", ["roboto_origin", "asimov_v1"])
def test_terrain_yaml_critic_list_is_joystick_plus_the_scan(robot):
    """The critic reads joystick's list, then the height scan. The actor
    list is joystick's."""
    terrain_cfg = _compose([f"robot={robot}", "task=terrain"])
    obs = terrain_cfg.task.env.obs
    joystick_obs = joystick_default_config().obs
    assert list(obs.privileged) == [*joystick_obs.privileged, "height_scan_clean"]
    assert list(obs.state) == list(joystick_obs.state)
    env_cfg = terrain_default_config()
    _apply_overrides(env_cfg, OmegaConf.to_container(terrain_cfg.task.env, resolve=True))
    assert env_cfg.obs.privileged[-1] == "height_scan_clean"


def test_a_bias_cannot_arm_an_out_of_box_draw():
    """roboto_origin's vx box is [-0.6, 1.0], and back_vx (-0.8, -0.2) lies
    outside it. The bias is checked like the base command config."""
    command = joystick_default_config().command
    command.vx = (-0.6, 1.0)
    bias = terrain_default_config().terrain.command_bias
    check_pure_draw_ranges(bias_command_config(command, bias))

    bias.pure_back_prob = 0.1
    biased = bias_command_config(command, bias)
    assert isinstance(biased, config_dict.ConfigDict)
    assert biased.pure_back_prob == 0.1 and biased.vx == (-0.6, 1.0)
    with pytest.raises(ValueError, match="pure_back_prob"):
        check_pure_draw_ranges(biased)
