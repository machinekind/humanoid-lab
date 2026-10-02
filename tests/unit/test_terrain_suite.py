"""The terrain scan suite: rows, cells and their names, bars, the eval
arena and its fingerprint, r_out, the course, deadlines and the suite
lookup.

Model-free. The suite is numpy and the arena generator, and the eval arena
is generated once for the module. The checks against the robot's own
config read its composed Hydra config and the env's code defaults.
"""

from __future__ import annotations

import inspect
import math
from dataclasses import fields, replace

import numpy as np
import pytest
from hydra import compose, initialize_config_dir

from humanoid_lab import paths, terrain
from humanoid_lab.eval import terrain_suite as ts
from humanoid_lab.terrain.config import CPU_ARENA, arena_for, params_from_config

SUITE = ts.ROBOTO_SUITE

# The eval arena's sha256, by suite version. A generator or params change
# that moves it needs a new suite version: old scans no longer compare.
PINNED_FINGERPRINTS = {
    1: "7691ccb7a24c5ac0a09b492aec0bd206eaf5d601cfeee153d294aa0c5504ba8f",
}

# Suite fields the protocol pin leaves out. The robot and version are its
# identity. The fingerprint and VALUES pin the arena and the cells. The warp
# budgets decide only whether a scan is valid, not its numbers.
UNPINNED = {
    "robot",
    "version",
    "arena",
    "fingerprint",
    "cells",
    "naconmax_per_env",
    "njmax",
    "naccdmax_per_env",
}

# The protocol constants and the gated cells' bars, by suite version. A
# change to any of them moves a cell's numbers and needs a new version.
PINNED_PROTOCOL = {
    1: {
        "speeds": (0.3, 0.6),
        "headings": 8,
        "offsets": (-0.09, -0.03, 0.03, 0.09),
        "draws": 2,
        "footprint_reach": 0.15,
        "settle_steps": 50,
        "budget_slack": 1.6,
        "saturation_frac": 0.95,
        "base_contact_tol": 0.01,
    },
}
PINNED_BARS = {
    1: {
        "rough_uniform_1.2cm": 0.95,
        "rough_uniform_1.9cm": 0.80,
        "rough_uniform_2.6cm": 0.60,
        "pyramid_slope_4deg": 0.95,
        "pyramid_slope_8deg": 0.80,
        "pyramid_slope_12deg": 0.60,
        "inverted_pyramid_slope_4deg": 0.95,
        "inverted_pyramid_slope_8deg": 0.80,
        "inverted_pyramid_slope_12deg": 0.60,
        "wave_2cm": 0.95,
        "wave_3cm": 0.80,
        "wave_4cm": 0.60,
    },
}

# Every cell's realized value, by type, one per row 0.2 ... 1.2. The names
# are built from these, so this table pins every name.
VALUES = {
    "rough_uniform": (1.2, 1.9, 2.6, 3.3, 4.0, 4.7),
    "pyramid_slope": (4.0, 8.0, 12.0, 16.0, 20.1, 24.1),
    "inverted_pyramid_slope": (4.0, 8.0, 12.0, 16.0, 20.1, 24.1),
    "pyramid_stairs": (4.6, 7.2, 9.8, 12.4, 15.0, 17.6),
    "inverted_pyramid_stairs": (4.6, 7.2, 9.8, 12.4, 15.0, 17.6),
    "discrete_obstacles": (3.0, 5.0, 7.0, 9.0, 11.0, 13.0),
    "random_grid": (3.0, 5.0, 7.0, 9.0, 11.0, 13.0),
    "wave": (2.0, 3.0, 4.0, 5.0, 6.0, 7.0),
}

# A cell's value is rounded to 0.1 of its unit, so the realized dimension
# lies within half of that of the value.
ROUNDING = 0.05


@pytest.fixture(scope="module")
def arena():
    return ts.eval_arena(SUITE)


@pytest.fixture(scope="module")
def ctrl_dt():
    from humanoid_lab.envs.joystick import default_config

    return default_config().ctrl_dt


def _compose(overrides):
    with initialize_config_dir(version_base=None, config_dir=str(paths.CONFIGS_DIR)):
        return compose(config_name="config", overrides=overrides)


def _tile_nodes(arena, tile):
    """Node coordinates and (ground, lookup) heights over one tile,
    perimeter included. Ground is the heightfield without the boxes."""
    xs, ys = terrain.grid_axes(arena.spec)
    half = arena.spec.params.tile_size / 2
    cx, cy, _ = tile.origin
    cols = np.flatnonzero(np.abs(xs - cx) <= half + 1e-9)
    rows = np.flatnonzero(np.abs(ys - cy) <= half + 1e-9)
    hf = arena.spec.hfield
    ground = (
        hf.pos_z + arena.hfield_data[np.ix_(rows, cols)].astype(float) * hf.elevation_z
    )
    lookup = arena.lookup[np.ix_(rows, cols)].astype(float)
    gx, gy = np.meshgrid(xs[cols], ys[rows])
    return gx, gy, ground, lookup


def _tile_boxes(arena, tile):
    half = arena.spec.params.tile_size / 2
    cx, cy, _ = tile.origin
    return [
        b
        for b in arena.boxes
        if abs(b.pos[0] - cx) < half and abs(b.pos[1] - cy) < half
    ]


def _drawn_heights(arena, tile):
    """Each box's top above the highest ground node under its footprint:
    the height the generator drew for it."""
    gx, gy, ground, _ = _tile_nodes(arena, tile)
    out = []
    for b in _tile_boxes(arena, tile):
        c, s = math.cos(b.yaw), math.sin(b.yaw)
        dx, dy = gx - b.pos[0], gy - b.pos[1]
        inside = (np.abs(dx * c + dy * s) <= b.half[0] + 1e-9) & (
            np.abs(dy * c - dx * s) <= b.half[1] + 1e-9
        )
        out.append(b.pos[2] + b.half[2] - ground[inside].max())
    return np.array(out)


def test_rows_are_sorted_and_unique(arena):
    assert list(ts.DIFFICULTIES) == sorted(set(ts.DIFFICULTIES))
    assert SUITE.arena.difficulties == ts.DIFFICULTIES
    # One cell per tile, each (row, type) once, rows in arena order.
    assert len(SUITE.cells) == len(arena.spec.tiles) == 48
    assert len({(c.row, c.terrain_type) for c in SUITE.cells}) == 48
    assert [c.row for c in SUITE.cells] == sorted(c.row for c in SUITE.cells)
    for cell in SUITE.cells:
        assert cell.difficulty == ts.DIFFICULTIES[cell.row]
        assert ts.cell_tile(arena.spec, cell).difficulty == cell.difficulty


def test_cell_names_are_unique_and_stable():
    names = [c.name for c in SUITE.cells]
    assert len(set(names)) == len(names)
    expected = {
        f"{t}_{v:g}{'deg' if 'slope' in t else 'cm'}"
        for t, values in VALUES.items()
        for v in values
    }
    assert set(names) == expected
    for cell in SUITE.cells:
        assert cell.value == VALUES[cell.terrain_type][cell.row]
    assert "pyramid_stairs_9.8cm" in names
    assert "inverted_pyramid_slope_20.1deg" in names
    assert "discrete_obstacles_13cm" in names


def test_realized_dimensions_match_the_names(arena):
    """Each tile's geometry, read back from the generated arena, has the
    dimension its cell is named by."""
    p = arena.spec.params
    for cell in SUITE.cells:
        tile = ts.cell_tile(arena.spec, cell)
        if cell.terrain_type in terrain.STAIR_TYPES:
            # The ground, the box tops and the pad are the flight's levels,
            # one riser apart.
            tops = {round(b.pos[2] + b.half[2], 9) for b in _tile_boxes(arena, tile)}
            levels = np.array(sorted(tops | {0.0, round(tile.pad_height, 9)}))
            risers = np.diff(levels)
            assert len(risers) == tile.n_steps, cell.name
            assert np.ptp(risers) < 1e-9, cell.name
            assert round(risers[0] * 100, 1) == cell.value, cell.name
        elif "slope" in cell.terrain_type:
            run = p.tile_size / 2 - p.slope_platform_half
            angle = math.degrees(math.atan(abs(tile.pad_height) / run))
            assert round(angle, 1) == cell.value, cell.name
        elif cell.terrain_type in ("rough_uniform", "wave"):
            # The relief peaks at its amplitude. Rough noise is drawn, so it
            # peaks below it.
            amplitude = cell.value / 100
            peak = np.abs(_tile_nodes(arena, tile)[3]).max()
            assert peak <= amplitude + ROUNDING / 100, cell.name
            assert peak >= 0.9 * (amplitude - ROUNDING / 100), cell.name
        else:
            # Each box is drawn at a fraction of the named height.
            height = cell.value / 100
            lo = (
                p.discrete_height_fraction[0]
                if cell.terrain_type == "discrete_obstacles"
                else p.grid_height_fraction[0]
            )
            drawn = _drawn_heights(arena, tile)
            assert len(drawn) > 0, cell.name
            assert drawn.max() <= height + ROUNDING / 100, cell.name
            assert drawn.max() >= 0.8 * height, cell.name
            assert drawn.min() >= lo * (height - ROUNDING / 100) - 1e-6, cell.name


def test_eval_arena_fingerprint_is_pinned(arena):
    assert SUITE.arena == ts.eval_arena_params()
    assert terrain.fingerprint(arena) == SUITE.fingerprint
    assert SUITE.fingerprint == PINNED_FINGERPRINTS[SUITE.version]
    assert len(arena.boxes) == 1135
    assert arena.lookup.shape == (701, 901)
    assert ts.eval_arena(SUITE) is arena


def test_protocol_is_pinned_by_version():
    """Every Suite field outside UNPINNED is pinned, so a new protocol
    field fails here until it is pinned too."""
    pinned = PINNED_PROTOCOL[SUITE.version]
    assert set(pinned) == {f.name for f in fields(ts.Suite)} - UNPINNED
    assert {k: getattr(SUITE, k) for k in pinned} == pinned
    # Every other cell is tracked.
    bars = {c.name: c.bar for c in SUITE.cells if not c.tracked}
    assert bars == PINNED_BARS[SUITE.version]


def test_r_out_crosses_every_stair_flight_and_keeps_the_feet_on_the_tile(arena):
    p = arena.spec.params
    reach = SUITE.footprint_reach
    for cell in SUITE.cells:
        tile = ts.cell_tile(arena.spec, cell)
        r = ts.r_out(tile.feature_radius, p.tile_size, reach)
        # The feet of a standing robot at r stay on the tile.
        assert r + reach <= p.tile_size / 2 + 1e-12, cell.name
        if cell.terrain_type in terrain.STAIR_TYPES:
            flight = terrain.stair_flight_half(tile.stair_tread, tile.n_steps, p)
            assert tile.feature_radius == flight
            # Every foot point lies past the flight's outer edge.
            assert r - reach >= flight - 1e-12, cell.name
            assert r == pytest.approx(1.75)
        else:
            # The features run on to 1.9 m. The tile edge stops the run
            # short of them.
            assert r == pytest.approx(p.tile_size / 2 - reach) == pytest.approx(1.85)
            assert r < tile.feature_radius


def test_offsets_keep_the_footprint_on_the_pad(arena):
    p = arena.spec.params
    reach = max(abs(o) for o in SUITE.offsets) + SUITE.footprint_reach
    # A cell inside the pad, where the lookup reads it flat.
    assert reach <= p.pad_radius - p.cell_size
    radii = np.linspace(0.0, reach, 25)
    angles = np.linspace(0.0, 2 * math.pi, 96, endpoint=False)
    rr, aa = np.meshgrid(radii, angles)
    for cell in SUITE.cells:
        tile = ts.cell_tile(arena.spec, cell)
        assert tile.pad_radius == p.pad_radius
        cx, cy, _ = tile.origin
        z = terrain.lookup_height(arena, cx + rr * np.cos(aa), cy + rr * np.sin(aa))
        np.testing.assert_allclose(z, tile.pad_height, atol=1e-6, err_msg=cell.name)
    # The guard reads both sides: a far start behind or ahead is refused.
    for offsets in [(-0.25, 0.03), (-0.03, 0.25)]:
        with pytest.raises(ValueError, match="flat pad"):
            replace(SUITE, offsets=offsets)


def test_speeds_sit_inside_the_robot_command_box():
    cfg = _compose(["robot=roboto_origin", "task=terrain"])
    lo, hi = cfg.task.env.command.vx
    for speed in SUITE.speeds:
        assert lo < 0 < speed <= hi
    with pytest.raises(ValueError, match="positive"):
        replace(SUITE, speeds=(0.3, -0.3))


def test_the_settle_is_one_second_at_the_control_step(ctrl_dt):
    assert SUITE.settle_steps * ctrl_dt == pytest.approx(1.0)


def test_deadlines_cover_each_run_at_the_same_speed_fraction(arena, ctrl_dt):
    """A diagonal run walks sqrt(2) further to the same Chebyshev radius.
    Each run's deadline is sized on its own distance, so every run must
    hold the same fraction of its commanded speed to finish."""
    p = arena.spec.params
    radii = {
        ts.r_out(
            ts.cell_tile(arena.spec, c).feature_radius,
            p.tile_size,
            SUITE.footprint_reach,
        )
        for c in SUITE.cells
    }
    assert sorted(radii) == pytest.approx([1.75, 1.85])
    runs = ts.course(SUITE)
    for speed in SUITE.speeds:
        for r in radii:
            deadlines = ts.run_deadlines(SUITE, r, speed, ctrl_dt)
            assert len(deadlines) == len(runs)
            for run, deadline in zip(runs, deadlines):
                distance = ts.run_distance(r, run)
                assert isinstance(deadline, int)
                assert deadline == ts.episode_budget(speed, ctrl_dt, distance, SUITE)
                # What the commanded speed covers by the deadline: the slack
                # times the run's distance, rounded up to a whole step.
                walked = (deadline - SUITE.settle_steps) * ctrl_dt * speed
                assert walked / distance >= SUITE.budget_slack - 1e-9, run
                assert (
                    walked / distance < SUITE.budget_slack + speed * ctrl_dt / distance
                ), run
            # Every axis run gets fewer steps than every diagonal run.
            axis = [d for d, run in zip(deadlines, runs) if run.heading_index % 2 == 0]
            diagonal = [
                d for d, run in zip(deadlines, runs) if run.heading_index % 2 == 1
            ]
            assert max(axis) < min(diagonal)
    # A faster command gets fewer steps.
    for r in radii:
        slow, fast = (max(ts.run_deadlines(SUITE, r, s, ctrl_dt)) for s in SUITE.speeds)
        assert slow > fast


def test_heading_stretch():
    assert ts.heading_stretch(0.0) == pytest.approx(1.0)
    assert ts.heading_stretch(math.pi / 2) == pytest.approx(1.0)
    assert ts.heading_stretch(math.pi / 4) == pytest.approx(math.sqrt(2))
    assert ts.heading_stretch(-3 * math.pi / 4) == pytest.approx(math.sqrt(2))
    for run in ts.course(SUITE):
        stretch = ts.heading_stretch(run.yaw)
        assert stretch == pytest.approx(
            1.0 if run.heading_index % 2 == 0 else math.sqrt(2)
        )
        # Walking `stretch` metres along the heading moves 1 m in Chebyshev.
        u = np.array([math.cos(run.yaw), math.sin(run.yaw)])
        assert np.abs(stretch * u).max() == pytest.approx(1.0)
        # The run's distance from its start ends at Chebyshev r_out.
        for r in (1.75, 1.85):
            end = (run.offset + ts.run_distance(r, run)) * u
            assert np.abs(end).max() == pytest.approx(r), run
    # A start ahead along the heading walks less.
    behind, ahead = (
        r
        for r in ts.course(SUITE)
        if r.draw == 0 and r.heading_index == 1 and abs(r.offset) == 0.09
    )
    assert behind.offset < 0 < ahead.offset
    gap = ts.run_distance(1.75, behind) - ts.run_distance(1.75, ahead)
    assert gap == pytest.approx(ahead.offset - behind.offset)


def test_bar_counts_and_provenance():
    n = SUITE.runs_per_cell
    gated = [c for c in SUITE.cells if not c.tracked]
    assert len(gated) == 12
    assert {c.terrain_type for c in gated} == set(ts.GATED_TYPES)
    expected = {
        0.2: (61, "provisional"),
        0.4: (52, "provisional"),
        0.6: (39, "provisional"),
    }
    for cell in gated:
        assert ts.threshold(cell, n) == expected[cell.difficulty], cell.name
    for cell in SUITE.cells:
        if cell.tracked:
            assert ts.threshold(cell, n) == (None, "tracked"), cell.name
    # 0.55 * 100 is 55.00000000000001 in floats. The count stays 55.
    assert ts.threshold(replace(gated[0], bar=0.55), 100) == (55, "provisional")


def test_stairs_obstacles_and_rubble_are_tracked():
    stepped = (*terrain.STAIR_TYPES, "discrete_obstacles", "random_grid")
    for cell in SUITE.cells:
        if cell.terrain_type in stepped or cell.difficulty > 0.6:
            assert cell.tracked, cell.name
    assert not set(stepped) & set(ts.GATED_TYPES)


def test_course_is_fixed_and_has_64_runs():
    runs = ts.course(SUITE)
    assert runs == ts.course(SUITE)
    assert len(runs) == SUITE.runs_per_cell == 64
    assert [r.index for r in runs] == list(range(64))
    # Draw outermost, then heading, then offset.
    assert [r.draw for r in runs] == [0] * 32 + [1] * 32
    assert [r.heading_index for r in runs[:8]] == [0, 0, 0, 0, 1, 1, 1, 1]
    assert [r.offset for r in runs[:4]] == list(SUITE.offsets)
    assert {r.yaw for r in runs} == {2 * math.pi * h / 8 for h in range(8)}
    # The second draw repeats the first draw's starts.
    first, second = runs[:32], runs[32:]
    assert [(r.yaw, r.offset) for r in first] == [(r.yaw, r.offset) for r in second]
    # Heading h faces yaw 2 pi h / headings on any heading count.
    small = replace(SUITE, headings=2, offsets=(0.0,), draws=3)
    for suite in (SUITE, small):
        for run in ts.course(suite):
            assert run.yaw == 2 * math.pi * run.heading_index / suite.headings, run
    small_runs = ts.course(small)
    assert len(small_runs) == small.runs_per_cell == 6
    assert [r.heading_index for r in small_runs] == [0, 1] * 3
    assert {r.yaw for r in small_runs} == {0.0, math.pi}


def test_zero_speed_has_no_budget(ctrl_dt):
    for speed in (0.0, -0.3):
        with pytest.raises(ValueError, match="no step budget"):
            ts.episode_budget(speed, ctrl_dt, 1.75, SUITE)


def test_an_unknown_robot_has_no_suite():
    assert ts.suite_for("roboto_origin") is SUITE
    with pytest.raises(ValueError, match="no terrain suite for asimov_v1"):
        ts.suite_for("asimov_v1")


def test_full_suite_is_3072_worlds_per_speed():
    """The full suite is 48 cells of 64 runs, 3072 worlds per speed."""
    assert len(SUITE.cells) == 48
    assert len(SUITE.cells) * len(ts.course(SUITE)) == 3072


def test_cells_on_an_arena_with_a_flat_row_start_at_row_1():
    params = params_from_config(CPU_ARENA)
    cells = ts.build_cells(params)
    assert [c.terrain_type for c in cells] == list(terrain.TYPES)
    assert {c.row for c in cells} == {1}
    assert {c.difficulty for c in cells} == {0.5}
    # 0.5 is not a ladder row.
    assert all(c.tracked for c in cells)
    assert "pyramid_stairs_8.5cm" in {c.name for c in cells}
    spec = arena_for(params).spec
    for cell in cells:
        assert ts.cell_tile(spec, cell).row == 1
    with pytest.raises(ValueError, match="explicit row difficulties"):
        ts.build_cells(terrain.ArenaParams())


def test_cells_on_a_capped_arena_carry_the_realized_difficulty():
    """A type cap scales the row's difficulty. The cell's difficulty, value,
    name and bar all follow the capped tile the generator builds."""
    caps = (("pyramid_stairs", 0.4), ("pyramid_slope", 0.4))
    params = replace(params_from_config(CPU_ARENA), type_caps=caps)
    cells = {c.terrain_type: c for c in ts.build_cells(params)}
    stairs, slope = cells["pyramid_stairs"], cells["pyramid_slope"]
    # The row is 0.5, and 0.5 x 0.4 is 0.2.
    assert stairs.difficulty == slope.difficulty == 0.2
    assert stairs.name == "pyramid_stairs_4.6cm"
    assert slope.name == "pyramid_slope_4deg"
    # The bar follows the realized difficulty. Stairs carry none.
    assert slope.bar == ts.LADDER[0.2] == 0.95
    assert stairs.tracked
    for t, cell in cells.items():
        if t not in dict(caps):
            assert cell.difficulty == 0.5, t
    spec = arena_for(params).spec
    for cell in cells.values():
        assert ts.cell_tile(spec, cell).difficulty == cell.difficulty, cell.name


def test_a_cell_from_another_arena_is_refused(arena):
    cell = SUITE.cells[0]
    with pytest.raises(ValueError, match="another arena"):
        ts.cell_tile(arena.spec, replace(cell, difficulty=0.3))
    with pytest.raises(ValueError, match="no rough_uniform tile on row 9"):
        ts.cell_tile(arena.spec, replace(cell, row=9))


def test_shared_constants_match_their_sources():
    """The suite's saturation fraction is the flat battery's, and its base
    contact tolerance is the terrain task's default."""
    from humanoid_lab.envs.terrain_joystick import default_config
    from humanoid_lab.eval import battery

    frac = (
        inspect.signature(battery.torque_saturation_fraction).parameters["frac"].default
    )
    assert SUITE.saturation_frac == frac
    assert SUITE.base_contact_tol == default_config().terrain.base_contact.tol
