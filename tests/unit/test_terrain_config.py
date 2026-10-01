"""The arena config block, its conversion to `ArenaParams`, and the arena cache."""

from __future__ import annotations

import pytest

from humanoid_lab import terrain
from humanoid_lab.terrain.config import (
    CPU_ARENA,
    arena_for,
    config_from_params,
    params_from_config,
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
