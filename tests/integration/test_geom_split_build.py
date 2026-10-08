"""model_patches.geoms `split`, read back off compiled models.

Roboto Origin's base and torso boxes are split into 3D chessboard cells so no
face of either one can overlap more heightfield prisms than the 50 hits
MJWarp collects per geom-heightfield pair. The cells are read off the
compiled model. They have to sit inside the box, touch no other cell, reach
each of its six faces, and have no edge longer than 10 cm. Against the same
robot.yaml with the `split` keys removed, they have to replace the box one
for one in the collision geom count, keep its contact attributes, body and
orientation, and leave every inertial quantity where it was.

The synthetic fixture covers what Roboto's boxes cannot: a rotated box,
contact attributes that all differ from mujoco's defaults and come partly
from the body's childclass, a body whose mass would come from its geoms, and
a compiler inertiafromgeom="true" that takes every body's mass from its
geoms.
"""

from __future__ import annotations

import itertools

import mujoco
import numpy as np
import pytest
import yaml

from humanoid_lab import paths
from humanoid_lab.robot.build import (
    SPLIT_CELL_SHRINK,
    build_spec,
    compile_spec,
    split_box_cells,
)
from humanoid_lab.robot.spec import load_robot_spec

ROBOTO_DIR = paths.ROBOTS_DIR / "roboto_origin"
ROBOTO_PRESET = "deploy_pd"

# A face at most this wide overlaps at most 4 x 4 cells of a 4 cm
# heightfield, 32 prisms, under the 50-hit cap.
MAX_CELL_EDGE_M = 0.10

# The per-geom attributes a contact is built from, plus the render group.
CONTACT_ATTRS = (
    "contype",
    "conaffinity",
    "condim",
    "priority",
    "friction",
    "solmix",
    "solref",
    "solimp",
    "margin",
    "gap",
    "group",
)


def _collision_geoms(model: mujoco.MjModel) -> list[str]:
    """Names of the robot's collision geoms: collidable, and not the ground plane."""
    return [
        model.geom(i).name
        for i in range(model.ngeom)
        if (model.geom_contype[i] or model.geom_conaffinity[i])
        and model.geom_type[i] != mujoco.mjtGeom.mjGEOM_PLANE
    ]


@pytest.fixture(scope="module")
def roboto(tmp_path_factory):
    """(split model, unsplit model, the split boxes' patch entries)."""
    split = compile_spec(build_spec(ROBOTO_DIR, ROBOTO_PRESET))

    # The same robot with the split keys removed: robot.yaml rewritten,
    # source and presets linked rather than copied.
    unsplit_dir = tmp_path_factory.mktemp("roboto_unsplit")
    (unsplit_dir / "source").symlink_to(ROBOTO_DIR / "source")
    (unsplit_dir / "actuators").symlink_to(ROBOTO_DIR / "actuators")
    raw = yaml.safe_load((ROBOTO_DIR / "robot.yaml").read_text())
    for geom in raw["model_patches"]["geoms"].values():
        geom.pop("split", None)
    (unsplit_dir / "robot.yaml").write_text(yaml.safe_dump(raw, sort_keys=False))
    unsplit = compile_spec(build_spec(unsplit_dir, ROBOTO_PRESET))

    patches = load_robot_spec(ROBOTO_DIR).model_patches.geoms
    boxes = {name: g for name, g in patches.items() if g.split is not None}
    return split, unsplit, boxes


def _box_frame_cells(model: mujoco.MjModel, name: str, box) -> list[tuple]:
    """(cell name, centre in the box frame, half-sizes) of each cell of `box`, read off `model`."""
    rot = np.zeros(9)
    quat = np.asarray(box.quat) / np.linalg.norm(box.quat)
    mujoco.mju_quat2Mat(rot, quat)
    rot = rot.reshape(3, 3)
    out = []
    for cell, _, _ in split_box_cells(name, box):
        geom = model.geom(cell)
        out.append((cell, rot.T @ (geom.pos - np.asarray(box.pos)), np.asarray(geom.size)))
    return out


# -- Roboto Origin ------------------------------------------------------------


def test_roboto_splits_both_boxes_into_15_cells(roboto):
    split, unsplit, boxes = roboto
    assert boxes.keys() == {"base_collision", "torso_collision"}
    assert boxes["base_collision"].split == (2, 3, 2)
    assert boxes["torso_collision"].split == (2, 3, 3)

    expected = {
        "base_collision": ["000", "011", "020", "101", "110", "121"],
        "torso_collision": ["000", "002", "011", "020", "022", "101", "110", "112", "121"],
    }
    collision = _collision_geoms(split)
    for name, suffixes in expected.items():
        cells = [n for n in collision if n.startswith(f"{name}_")]
        assert cells == [f"{name}_{s}" for s in suffixes]
        assert name not in collision
    assert len(_collision_geoms(unsplit)) == 16
    assert len(collision) == 16 - 2 + 6 + 9


@pytest.mark.parametrize("name", ["base_collision", "torso_collision"])
def test_every_cell_sits_inside_its_box(roboto, name):
    split, _, boxes = roboto
    half_box = np.asarray(boxes[name].size)
    for cell, centre, half in _box_frame_cells(split, name, boxes[name]):
        assert np.all(np.abs(centre) + half <= half_box + 1e-9), cell


@pytest.mark.parametrize("name", ["base_collision", "torso_collision"])
def test_no_two_cells_overlap_or_touch(roboto, name):
    """Every pair is separated along at least one axis, by twice the shrink.
    Diagonal neighbours in the chessboard would share an edge or a corner
    without it."""
    assert SPLIT_CELL_SHRINK > 0.0
    split, _, boxes = roboto
    cells = _box_frame_cells(split, name, boxes[name])
    for (a, ca, ha), (b, cb, hb) in itertools.combinations(cells, 2):
        gap = np.abs(ca - cb) - (ha + hb)
        assert gap.max() >= 2 * SPLIT_CELL_SHRINK - 1e-9, (a, b, gap)


@pytest.mark.parametrize("name", ["base_collision", "torso_collision"])
def test_every_face_of_the_box_is_reached_by_a_cell(roboto, name):
    """The chessboard keeps the box's outer extent: each of the six face
    planes has a cell whose own face lies within the shrink of it."""
    split, _, boxes = roboto
    half_box = np.asarray(boxes[name].size)
    cells = _box_frame_cells(split, name, boxes[name])
    for axis, sign in itertools.product(range(3), (-1.0, 1.0)):
        reach = max(sign * centre[axis] + half[axis] for _, centre, half in cells)
        assert reach >= half_box[axis] - SPLIT_CELL_SHRINK - 1e-9, (axis, sign, reach)


def test_no_cell_edge_is_longer_than_10_cm(roboto):
    split, _, boxes = roboto
    for name, box in boxes.items():
        for cell, _, half in _box_frame_cells(split, name, box):
            assert 2 * half.max() <= MAX_CELL_EDGE_M, (cell, 2 * half)


def test_the_split_moves_no_mass(roboto):
    """Both boxes sit on bodies with an explicit <inertial>, so the split is
    a collision detail only. Compared bit for bit."""
    split, unsplit, _ = roboto
    for attr in ("body_mass", "body_inertia", "body_ipos", "body_iquat"):
        np.testing.assert_array_equal(getattr(split, attr), getattr(unsplit, attr), err_msg=attr)


def test_cells_keep_the_box_contact_attributes_body_and_orientation(roboto):
    split, unsplit, boxes = roboto
    for name, box in boxes.items():
        whole = unsplit.geom(name)
        for cell, _, _ in split_box_cells(name, box):
            geom = split.geom(cell)
            for attr in CONTACT_ATTRS:
                np.testing.assert_array_equal(
                    getattr(geom, attr), getattr(whole, attr), err_msg=f"{cell}.{attr}"
                )
            assert geom.bodyid[0] == whole.bodyid[0], cell
            np.testing.assert_array_equal(geom.quat, whole.quat, err_msg=cell)
            assert geom.type[0] == mujoco.mjtGeom.mjGEOM_BOX


# -- synthetic fixture --------------------------------------------------------

_PRESET_YAML = """
model: pd
groups:
  leg: {kp: 50.0, kd: 2.0, effort_limit: 40.0}
"""

_ROBOT_YAML = """
name: split_test
model_xml: robot.xml
actuated_joints: [leg_joint]
joint_groups:
  leg: [leg_joint]
passive_joints: {{}}
foot_sites: []
foot_geoms: []
model_patches:
  geoms:
    torso_box: {{body: {body}, type: box, size: [0.12, 0.09, 0.06], {pos}
                quat: {quat}{split}}}
"""

# Every contact attribute the box ends up with differs from mujoco's own
# default, which test_cells_inherit_what_the_whole_box_would enforces, so
# "inherits what the whole box would" cannot pass by both falling back to
# mujoco's defaults. The torso body's childclass overrides some of the main
# default's attributes, so a cell added under any class other than the
# body's own fails the comparison.
_XML = """
<mujoco model="split_test">
  <default>
    <geom contype="2" conaffinity="0" condim="4" friction="0.7 0.1 0.1" solref="0.004 1.5"
          solimp="0.8 0.9 0.002 0.4 3" solmix="0.5" margin="0.002" gap="0.001" priority="1"/>
    <default class="torso_class">
      <geom condim="6" friction="0.6 0.05 0.05" priority="2"/>
    </default>
  </default>
  <worldbody>
    <geom name="ground" type="plane" size="0 0 1" conaffinity="15"/>
    <body name="torso" pos="0 0 0.5" childclass="torso_class">
      <freejoint/>
      <inertial pos="0 0 0" mass="5.0" diaginertia="0.05 0.04 0.03"/>
      <geom name="torso_geom" type="sphere" size="0.05" contype="0" conaffinity="0"/>
      <body name="link1" pos="0 0 -0.2">
        <joint name="leg_joint" type="hinge" axis="0 1 0" range="-1.2 1.2"/>
        <geom name="link1_geom" type="capsule" fromto="0 0 0 0 0 -0.25" size="0.04" mass="2.0"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

# 90 degrees about z, then 90 degrees about x: no cell offset survives it unchanged.
_QUAT = (0.5, 0.5, 0.5, 0.5)


_POS = (0.02, -0.01, 0.03)


def _synthetic(tmp_path, *, body="torso", split=True, compiler: str = "", pos=_POS):
    """The synthetic robot's directory. `pos=None` leaves the box's pos out of
    robot.yaml, so it sits at the body origin."""
    robot_dir = tmp_path / ("split" if split else "whole")
    (robot_dir / "actuators").mkdir(parents=True)
    (robot_dir / "actuators" / "test.yaml").write_text(_PRESET_YAML)
    (robot_dir / "robot.xml").write_text(_XML.replace("<worldbody>", compiler + "\n  <worldbody>"))
    (robot_dir / "robot.yaml").write_text(
        _ROBOT_YAML.format(
            body=body,
            pos="" if pos is None else f"pos: {list(pos)},",
            quat=list(_QUAT),
            split=", split: [3, 2, 2]" if split else "",
        )
    )
    return robot_dir


@pytest.mark.parametrize("pos", [_POS, None], ids=["pos", "no-pos"])
def test_a_rotated_box_splits_in_its_own_frame(tmp_path, pos):
    """Cell centres are the box pos plus the box rotation applied to the
    in-box offset, and every cell carries the box's quat. A box with no pos
    is centred on the body origin."""
    robot_dir = _synthetic(tmp_path, pos=pos)
    model = compile_spec(build_spec(robot_dir, "test"))
    box = load_robot_spec(robot_dir).model_patches.geoms["torso_box"]
    assert box.pos == pos
    centre = np.zeros(3) if box.pos is None else np.asarray(box.pos)

    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, np.asarray(_QUAT))
    rot = rot.reshape(3, 3)
    hx, hy, hz = box.size
    cells = split_box_cells("torso_box", box)
    assert len(cells) == 6
    for cell, _, half in cells:
        i, j, k = (int(c) for c in cell.rsplit("_", 1)[1])
        offset = np.array(
            [-hx + (2 * i + 1) * hx / 3, -hy + (2 * j + 1) * hy / 2, -hz + (2 * k + 1) * hz / 2]
        )
        geom = model.geom(cell)
        np.testing.assert_allclose(geom.pos, centre + rot @ offset, atol=1e-12)
        np.testing.assert_allclose(geom.size, half, atol=1e-12)
        np.testing.assert_allclose(geom.quat, _QUAT, atol=1e-12)


def test_cells_inherit_what_the_whole_box_would(tmp_path):
    split_dir = _synthetic(tmp_path)
    split = compile_spec(build_spec(split_dir, "test"))
    whole = compile_spec(build_spec(_synthetic(tmp_path, split=False), "test"))
    reference = whole.geom("torso_box")
    assert reference.condim[0] == 6 and reference.priority[0] == 2  # the body's childclass, applied
    bare = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><geom name="g" type="box" size="1 1 1"/></worldbody></mujoco>'
    ).geom("g")
    for attr in CONTACT_ATTRS:
        assert not np.array_equal(getattr(reference, attr), getattr(bare, attr)), attr
    box = load_robot_spec(split_dir).model_patches.geoms["torso_box"]
    for cell, _, _ in split_box_cells("torso_box", box):
        for attr in CONTACT_ATTRS:
            np.testing.assert_array_equal(
                getattr(split.geom(cell), attr), getattr(reference, attr), err_msg=attr
            )
    for attr in ("body_mass", "body_inertia", "body_ipos", "body_iquat"):
        np.testing.assert_array_equal(getattr(split, attr), getattr(whole, attr), err_msg=attr)


def test_a_split_on_a_body_whose_mass_comes_from_its_geoms_is_refused(tmp_path):
    """link1 has no <inertial>: its mass is its geoms', so cells would move it."""
    with pytest.raises(ValueError, match="no explicit <inertial>"):
        build_spec(_synthetic(tmp_path, body="link1"), "test")


def test_a_split_under_inertiafromgeom_true_is_refused(tmp_path):
    """torso has an explicit <inertial>, but inertiafromgeom="true" takes its
    mass from its geoms anyway, so cells would move it. The same compiler
    setting without a split builds."""
    compiler = '<compiler inertiafromgeom="true"/>'
    with pytest.raises(ValueError, match="inertiafromgeom"):
        build_spec(_synthetic(tmp_path, compiler=compiler), "test")
    compile_spec(build_spec(_synthetic(tmp_path, split=False, compiler=compiler), "test"))
