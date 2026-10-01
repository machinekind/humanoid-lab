"""Mirror-map properties, model-free (humanoid_lab.envs.symmetry only).

The XML checks parse robots/roboto_origin/source/mjcf/rpo.xml as text: the
sign table is derived from it, so it is checked against it here, without
compiling a model. tests/integration/test_symmetry.py checks the same signs
against the compiled model's kinematics.
"""

import xml.etree.ElementTree as ET

import numpy as np
import pytest

from humanoid_lab import paths
from humanoid_lab.envs import symmetry
from humanoid_lab.envs.joystick import default_config
from humanoid_lab.robot.spec import load_robot_spec

ROBOT_DIR = paths.ROBOTS_DIR / "roboto_origin"
SPEC = load_robot_spec(ROBOT_DIR)
JOINTS = list(SPEC.actuated_joints)
FEET = list(SPEC.foot_sites)
NJ = len(JOINTS)
NF = len(FEET)

SIZES = {
    "gyro": 3,
    "gravity": 3,
    "joint_pos": NJ,
    "joint_vel": NJ,
    "last_action": NJ,
    "command": 3,
    "phase": 2 * NF,
    "linvel": 3,
    "height": 1,
    "contacts": NF,
    "actuator_force": NJ,
}


def _apply(perm, sign, x):
    return sign * x[perm]


def test_roboto_origin_has_23_joints_and_two_feet():
    assert NJ == 23
    assert FEET == ["left_foot", "right_foot"]


def test_joint_mirror_is_an_involution():
    perm, sign = symmetry.joint_mirror(JOINTS, "roboto_origin")
    assert perm.shape == sign.shape == (NJ,)
    assert sorted(perm.tolist()) == list(range(NJ))
    assert np.array_equal(perm[perm], np.arange(NJ))
    assert np.array_equal(sign * sign[perm], np.ones(NJ))
    x = np.random.default_rng(0).normal(size=NJ)
    assert np.allclose(_apply(perm, sign, _apply(perm, sign, x)), x)


def test_joint_mirror_swaps_twins_and_keeps_torso_in_place():
    perm, sign = symmetry.joint_mirror(JOINTS, "roboto_origin")
    for i, name in enumerate(JOINTS):
        if name.startswith("left_"):
            assert JOINTS[perm[i]] == "right_" + name[len("left_"):]
        elif name.startswith("right_"):
            assert JOINTS[perm[i]] == "left_" + name[len("right_"):]
        else:
            assert perm[i] == i
    torso = JOINTS.index("torso_joint")
    assert sign[torso] == -1.0


def test_joint_mirror_matches_the_robot_yaml_symmetry_pairs():
    perm, _ = symmetry.joint_mirror(JOINTS, "roboto_origin")
    index = {n: i for i, n in enumerate(JOINTS)}
    for left, right in SPEC.symmetry.items():
        assert perm[index[left]] == index[right]
        assert perm[index[right]] == index[left]


def test_the_home_keyframe_is_its_own_mirror_image():
    perm, sign = symmetry.joint_mirror(JOINTS, "roboto_origin")
    home = SPEC.keyframes["home"]
    q = np.array([home.joints.get(n, 0.0) for n in JOINTS])
    assert np.allclose(_apply(perm, sign, q), q)


def _xml_axes():
    """Joint name -> axis, read from rpo.xml, plus every body element."""
    root = ET.parse(SPEC.model_xml_path).getroot()
    axes = {
        j.get("name"): [float(v) for v in j.get("axis", "0 0 1").split()]
        for j in root.iter("joint")
        if j.get("name") and j.get("type", "hinge") == "hinge"
    }
    return axes, list(root.iter("body"))


def test_no_body_frame_in_the_source_xml_is_rotated():
    """The sign rule reads XML axes as base-frame axes, which holds only
    while no body frame on any chain is rotated."""
    _, bodies = _xml_axes()
    for body in bodies:
        for attr in ("quat", "euler", "axisangle", "xyaxes", "zaxis"):
            assert body.get(attr) is None, (body.get("name"), attr)


def test_sign_table_matches_the_source_xml_axes():
    axes, _ = _xml_axes()
    _, sign = symmetry.joint_mirror(JOINTS, "roboto_origin")
    for i, name in enumerate(JOINTS):
        twin = name
        if name.startswith("left_"):
            twin = "right_" + name[len("left_"):]
        elif name.startswith("right_"):
            twin = "left_" + name[len("right_"):]
        assert symmetry.axis_mirror_sign(axes[name], axes[twin]) == sign[i], name


def test_axis_mirror_sign_rule():
    # roll about x on both sides: the image is the same axis -> negate
    assert symmetry.axis_mirror_sign([1, 0, 0], [1, 0, 0]) == -1.0
    # pitch about y on both sides: the image is -y -> keep
    assert symmetry.axis_mirror_sign([0, 1, 0], [0, 1, 0]) == 1.0
    # a right side already flipped: roll about -x -> keep
    assert symmetry.axis_mirror_sign([1, 0, 0], [-1, 0, 0]) == 1.0
    with pytest.raises(ValueError):
        symmetry.axis_mirror_sign([1, 0, 0], [0, 0, 1])


def test_unknown_robot_raises_naming_it():
    with pytest.raises(KeyError, match="asimov_v1"):
        symmetry.joint_mirror(JOINTS, "asimov_v1")


def test_a_joint_without_a_twin_raises():
    with pytest.raises(ValueError, match="right_knee_joint"):
        symmetry.joint_mirror(["left_knee_joint", "torso_joint"], "roboto_origin")


@pytest.mark.parametrize("which", ["state", "privileged"])
def test_obs_mirror_spans_the_default_obs_lists(which):
    names = list(default_config().obs[which])
    perm, sign = symmetry.obs_mirror(names, JOINTS, FEET, "roboto_origin")
    n = sum(SIZES[m] for m in names)
    assert perm.shape == sign.shape == (n,)
    assert sorted(perm.tolist()) == list(range(n))
    assert set(np.abs(sign).tolist()) == {1.0}
    x = np.random.default_rng(1).normal(size=n)
    assert np.allclose(_apply(perm, sign, _apply(perm, sign, x)), x)


@pytest.mark.parametrize("name", sorted(SIZES))
def test_every_catalog_component_has_its_size(name):
    perm, sign = symmetry.component_mirror(name, JOINTS, FEET, "roboto_origin")
    assert len(perm) == len(sign) == SIZES[name]


def test_command_mirror_flips_vy_and_wz():
    perm, sign = symmetry.component_mirror("command", JOINTS, FEET)
    assert np.allclose(_apply(perm, sign, np.array([0.4, 0.2, -1.0])), [0.4, -0.2, 1.0])


def test_gyro_and_gravity_mirror_as_pseudo_and_true_vectors():
    v = np.array([1.0, 2.0, 3.0])
    perm, sign = symmetry.component_mirror("gyro", JOINTS, FEET)
    assert np.allclose(_apply(perm, sign, v), [-1.0, 2.0, -3.0])
    perm, sign = symmetry.component_mirror("gravity", JOINTS, FEET)
    assert np.allclose(_apply(perm, sign, v), [1.0, -2.0, 3.0])
    perm, sign = symmetry.component_mirror("linvel", JOINTS, FEET)
    assert np.allclose(_apply(perm, sign, v), [1.0, -2.0, 3.0])


def test_phase_mirror_swaps_the_feet_in_both_halves():
    perm, sign = symmetry.component_mirror("phase", JOINTS, FEET)
    # (cos L, cos R, sin L, sin R) -> (cos R, cos L, sin R, sin L)
    assert _apply(perm, sign, np.array([0.0, 1.0, 2.0, 3.0])).tolist() == [1.0, 0.0, 3.0, 2.0]


def test_contacts_mirror_swaps_the_feet():
    perm, sign = symmetry.component_mirror("contacts", JOINTS, FEET)
    assert _apply(perm, sign, np.array([1.0, 0.0])).tolist() == [0.0, 1.0]


def test_height_is_unchanged():
    perm, sign = symmetry.component_mirror("height", JOINTS, FEET)
    assert _apply(perm, sign, np.array([0.7])).tolist() == [pytest.approx(0.7)]


def test_unknown_component_raises_naming_it():
    with pytest.raises(KeyError, match="no_such_obs"):
        symmetry.component_mirror("no_such_obs", JOINTS, FEET)
    with pytest.raises(KeyError, match="no_such_obs"):
        symmetry.obs_mirror(["gyro", "no_such_obs"], JOINTS, FEET)
