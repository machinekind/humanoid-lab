"""model_patches.geoms `split`: the schema, and the cell layout it expands to.

Model-free. Parsing reads robot.yaml only, and split_box_cells is plain
arithmetic on the parsed entry. What the cells do once they are in a compiled
model (contact attributes, inertia, Roboto Origin's own boxes) is
tests/integration/test_geom_split_build.py's job.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from humanoid_lab.robot.build import SPLIT_CELL_SHRINK, split_box_cells
from humanoid_lab.robot.spec import ModelPatchGeom, load_robot_spec

_ROBOT_YAML = """
name: split_test
model_xml: does/not/matter.xml
actuated_joints: []
joint_groups: {{}}
passive_joints: {{}}
foot_sites: []
foot_geoms: []
model_patches:
  geoms:
    g1: {geom}
"""


def _load_geom(tmp_path, geom_yaml: str) -> ModelPatchGeom:
    (tmp_path / "robot.yaml").write_text(_ROBOT_YAML.format(geom=geom_yaml))
    return load_robot_spec(tmp_path).model_patches.geoms["g1"]


def _box(**overrides) -> ModelPatchGeom:
    fields = dict(body="b", type="box", size=(0.09, 0.12, 0.06), pos=None, split=(2, 3, 2))
    fields.update(overrides)
    return ModelPatchGeom(**fields)


# -- schema ------------------------------------------------------------------


def test_split_parses_to_an_int_triple(tmp_path):
    geom = _load_geom(
        tmp_path, "{body: b, type: box, size: [0.1, 0.1, 0.1], pos: [0, 0, 0], split: [2, 3, 1]}"
    )
    assert geom.split == (2, 3, 1)
    assert all(type(n) is int for n in geom.split)


@pytest.mark.parametrize("split", [[1, 1, 1], [2, 2, 1], [10, 2, 1]])
def test_a_split_on_two_axes_or_none_is_accepted(tmp_path, split):
    geom = _load_geom(tmp_path, f"{{body: b, type: box, size: [0.1, 0.1, 0.1], split: {split}}}")
    assert geom.split == tuple(split)


def test_split_defaults_to_none(tmp_path):
    geom = _load_geom(tmp_path, "{body: b, type: box, size: [0.1, 0.1, 0.1]}")
    assert geom.split is None


@pytest.mark.parametrize(
    ("geom_yaml", "match"),
    [
        pytest.param(
            "{body: b, type: capsule, size: [0.05], fromto: [0, 0, 0, 0, 0, 1], split: [1, 1, 2]}",
            "type box only",
            id="capsule",
        ),
        pytest.param(
            "{body: b, type: sphere, size: [0.05], split: [2, 2, 2]}",
            "type box only",
            id="sphere",
        ),
        pytest.param(
            "{body: b, type: box, size: [0.1, 0.1], fromto: [0, 0, 0, 0, 0, 1], split: [2, 2, 2]}",
            "fromto",
            id="with-fromto",
        ),
        pytest.param(
            "{body: b, type: box, size: [0.1, 0.1], split: [2, 2, 2]}",
            "3-element box size",
            id="short-size",
        ),
        pytest.param(
            "{body: b, type: box, size: [0.1, 0.1, 0.1], split: [2, 2]}",
            "three integers",
            id="two-entries",
        ),
        pytest.param(
            "{body: b, type: box, size: [0.1, 0.1, 0.1], split: [2, 2, 2, 2]}",
            "three integers",
            id="four-entries",
        ),
        pytest.param(
            "{body: b, type: box, size: [0.1, 0.1, 0.1], split: [2, 2.0, 2]}",
            "three integers",
            id="float-entry",
        ),
        pytest.param(
            "{body: b, type: box, size: [0.1, 0.1, 0.1], split: [2, true, 2]}",
            "three integers",
            id="bool-entry",
        ),
        pytest.param(
            "{body: b, type: box, size: [0.1, 0.1, 0.1], split: 2}",
            "three integers",
            id="scalar",
        ),
        pytest.param(
            "{body: b, type: box, size: [0.1, 0.1, 0.1], split: [2, 0, 2]}",
            "between 1 and 10",
            id="zero",
        ),
        pytest.param(
            "{body: b, type: box, size: [0.1, 0.1, 0.1], split: [-1, 2, 2]}",
            "between 1 and 10",
            id="negative",
        ),
        pytest.param(
            "{body: b, type: box, size: [0.1, 0.1, 0.1], split: [11, 1, 1]}",
            "between 1 and 10",
            id="two-digit-index",
        ),
        pytest.param(
            "{body: b, type: box, size: [0.1, 0.1, 0.1], split: [2, 1, 1]}",
            "one axis only",
            id="x-only-even",
        ),
        pytest.param(
            "{body: b, type: box, size: [0.1, 0.1, 0.1], split: [1, 1, 2]}",
            "one axis only",
            id="z-only-even",
        ),
        pytest.param(
            "{body: b, type: box, size: [0.1, 0.1, 0.1], split: [3, 1, 1]}",
            "one axis only",
            id="x-only-odd-slabs",
        ),
    ],
)
def test_bad_split_raises_a_named_error(tmp_path, geom_yaml, match):
    with pytest.raises(ValueError, match=match):
        _load_geom(tmp_path, geom_yaml)


# -- cell layout -------------------------------------------------------------


def test_cells_are_the_even_parity_half_of_the_grid():
    cells = split_box_cells("box", _box(split=(2, 3, 2)))
    names = [name for name, _, _ in cells]
    expected = [
        f"box_{i}{j}{k}"
        for i, j, k in itertools.product(range(2), range(3), range(2))
        if (i + j + k) % 2 == 0
    ]
    assert names == expected
    assert len(names) == 6


def test_the_largest_split_still_names_every_cell_apart():
    """10 per axis keeps every index one digit, so no two cell names collide."""
    cells = split_box_cells("box", _box(size=(0.1, 0.1, 0.1), split=(10, 10, 10)))
    assert len({name for name, _, _ in cells}) == len(cells) == 500
    assert cells[-1][0] == "box_998"


def test_an_all_odd_grid_keeps_the_extra_corner_cell():
    assert len(split_box_cells("box", _box(split=(3, 3, 3)))) == 14
    assert [n for n, _, _ in split_box_cells("box", _box(split=(1, 1, 1)))] == ["box_000"]


def test_cell_half_size_is_the_grid_cell_less_the_shrink():
    hx, hy, hz = 0.09, 0.12, 0.06
    expected = (hx / 2 - SPLIT_CELL_SHRINK, hy / 3 - SPLIT_CELL_SHRINK, hz / 2 - SPLIT_CELL_SHRINK)
    for _, _, half in split_box_cells("box", _box(size=(hx, hy, hz), split=(2, 3, 2))):
        assert half == pytest.approx(expected)


def test_cells_tile_the_box_frame_and_follow_its_rotation():
    """Cell centres are the box pos plus the box rotation applied to the
    in-box offset. A box turned 90 degrees about z puts the cell that sits
    at -x in the box frame at -y in the body frame."""
    pos = (0.3, -0.2, 0.1)
    plain = split_box_cells("box", _box(pos=pos))
    s = np.sqrt(0.5)
    turned = split_box_cells("box", _box(pos=pos, quat=(s, 0.0, 0.0, s)))

    rz90 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    for (name_a, centre_a, half_a), (name_b, centre_b, half_b) in zip(plain, turned):
        assert name_a == name_b
        assert half_a == half_b
        offset = np.subtract(centre_a, pos)
        assert np.allclose(np.subtract(centre_b, pos), rz90 @ offset)

    # box_000 is the (-x, -y, -z) corner cell of a 2x3x2 grid over 0.09 x 0.12 x 0.06.
    assert plain[0][1] == pytest.approx((0.3 - 0.045, -0.2 - 0.08, 0.1 - 0.03))
    # A box with no pos is centred on the body origin.
    assert split_box_cells("box", _box())[0][1] == pytest.approx((-0.045, -0.08, -0.03))


def test_an_unnormalised_quat_lays_out_like_its_unit_quat():
    unit = split_box_cells("box", _box(quat=(np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5))))
    scaled = split_box_cells("box", _box(quat=(2.0, 0.0, 0.0, 2.0)))
    for (_, a, _), (_, b, _) in zip(unit, scaled):
        assert np.allclose(a, b)


def test_a_one_axis_even_split_built_directly_misses_a_face():
    """[2, 1, 1] keeps only the i = 0 layer, so nothing reaches +x."""
    with pytest.raises(ValueError, match="x face of the box with no cell"):
        split_box_cells("box", _box(split=(2, 1, 1)))


def test_a_one_axis_odd_split_built_directly_leaves_a_gap():
    """[1, 1, 3] reaches both z faces but keeps no k = 1 layer between them."""
    with pytest.raises(ValueError, match=r"z layer\(s\) \[1\] with no cell"):
        split_box_cells("box", _box(split=(1, 1, 3)))


def test_a_split_finer_than_the_shrink_raises():
    with pytest.raises(ValueError, match="use fewer cells"):
        split_box_cells("box", _box(size=(0.002, 0.1, 0.1), split=(4, 1, 1)))
