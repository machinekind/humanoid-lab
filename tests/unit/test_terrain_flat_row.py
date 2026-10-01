"""The prepended flat row: its geometry, and that it is off unless asked for.

A difficulty-0 row is not flat -- its stairs still have a 2 cm riser -- so
flat ground is a row of its own rather than the ramps turned down.
"""

from __future__ import annotations

import numpy as np
import pytest

from humanoid_lab import terrain


@pytest.fixture(scope="module")
def plain():
    return terrain.generate()


@pytest.fixture(scope="module")
def flat():
    return terrain.generate(flat_row=True)


def _row_cells(arena):
    """Node rows one arena row spans."""
    p = arena.spec.params
    return round(p.tile_size / p.cell_size)


def _rows(arena):
    """{row: {type: (col, difficulty, pad_height, feature_radius)}}."""
    out: dict[int, dict] = {}
    for t in arena.spec.tiles:
        out.setdefault(t.row, {})[t.terrain_type] = (
            t.col,
            t.difficulty,
            t.pad_height,
            t.feature_radius,
        )
    return out


def _box_params(arena):
    return np.array([b.pos + b.half + (b.yaw,) for b in arena.boxes])


def test_the_flat_row_is_flat_ground(flat):
    s = flat.spec
    p = s.params
    border = round(p.border / p.cell_size)
    # Every node of the row, its shared edge with row 1 included.
    nodes = flat.lookup[border : border + _row_cells(flat) + 1]
    assert np.all(nodes == 0.0)
    for t in (t for t in s.tiles if t.row == 0):
        assert t.pad_height == 0.0
        assert t.difficulty == 0.0
        assert t.feature_radius == 0.0
        assert t.n_steps == 0


def test_the_flat_row_holds_every_type(flat):
    """One tile of every type, like any row, so every (row, type) pair
    names a tile."""
    row = sorted(t.terrain_type for t in flat.spec.tiles if t.row == 0)
    assert row == sorted(terrain.TYPES)


def test_no_box_reaches_over_the_flat_row(flat):
    s = flat.spec
    top = s.y_min + s.params.border + s.params.tile_size
    for b in flat.boxes:
        hx, hy, _ = b.half
        reach = hx * abs(np.sin(b.yaw)) + hy * abs(np.cos(b.yaw))
        assert b.pos[1] - reach > top


def test_the_terrain_rows_move_up_one_tile_unchanged(plain, flat):
    cells = _row_cells(plain)
    assert flat.spec.n_rows == plain.spec.n_rows + 1
    assert flat.spec.hfield.nrow == plain.spec.hfield.nrow + cells
    assert flat.spec.hfield.ncol == plain.spec.hfield.ncol
    assert len(flat.spec.tiles) == len(plain.spec.tiles) + len(terrain.TYPES)
    # Every terrain row keeps its column order, difficulty and pads ...
    before, after = _rows(plain), _rows(flat)
    for row in range(plain.spec.n_rows):
        assert after[row + 1] == before[row]
    # ... and its surface, one tile up the node grid ...
    assert np.array_equal(flat.hfield_data[cells:], plain.hfield_data)
    assert np.array_equal(flat.lookup[cells:], plain.lookup)
    # ... and its boxes. The arena stays centred on the origin, so it grows
    # half a tile at each end and the terrain moves half a tile in world y.
    shifted = _box_params(plain)
    shifted[:, 1] += plain.spec.params.tile_size / 2
    np.testing.assert_allclose(_box_params(flat), shifted, rtol=0, atol=1e-12)


def test_the_flat_row_is_absent_when_off(plain):
    p = plain.spec.params
    assert plain.spec.n_rows == p.n_rows
    assert len(plain.spec.tiles) == p.n_rows * len(terrain.TYPES)
    # No extra tile of node rows.
    border = round(p.border / p.cell_size)
    assert plain.spec.hfield.nrow == p.n_rows * _row_cells(plain) + 2 * border + 1
    assert terrain.spec_to_dict(plain.spec)["params"]["flat_row"] is False
    # Row 0 is the easiest terrain, features and all.
    row0 = [t for t in plain.spec.tiles if t.row == 0]
    assert all(t.feature_radius > 0 for t in row0)
    stairs = next(t for t in row0 if t.terrain_type == "pyramid_stairs")
    assert stairs.pad_height == pytest.approx(stairs.n_steps * 0.02)
