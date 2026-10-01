"""Left/right mirror maps for the joystick task's observation and action spaces.

Mirroring is about the body xz-plane (y -> -y). Each left_* joint or foot
swaps with its right_* twin; a joint or foot with no side prefix maps to
itself. Vectors mirror as (x, -y, z), pseudo-vectors (angular rates) as
(-x, y, -z).

A rotation by theta about axis a, reflected through the xz-plane
(M = diag(1, -1, 1)), is a rotation by -theta about M a. So the twin's angle
is -theta when the twin's axis equals M a, and +theta when the twin's axis
is -M a (the source model already flipped it). That holds as long as every
body frame on the chain is unrotated, which is the case in
robots/roboto_origin/source/mjcf/rpo.xml (no body carries quat, euler,
axisangle, xyaxes or zaxis), so the XML axes are base-frame axes at the
zero pose.

roboto_origin, axes from rpo.xml:

    joint        axis L               axis R               M axis L             sign
    thigh_yaw    (-0.5, 0, -0.866)    (-0.5, 0, -0.866)    (-0.5, 0, -0.866)    -1
    thigh_roll   (0.866, 0, -0.5)     (0.866, 0, -0.5)     (0.866, 0, -0.5)     -1
    thigh_pitch  (0, 1, 0)            (0, 1, 0)            (0, -1, 0)           +1
    knee         (0, 1, 0)            (0, 1, 0)            (0, -1, 0)           +1
    ankle_pitch  (0, 1, 0)            (0, 1, 0)            (0, -1, 0)           +1
    ankle_roll   (1, 0, 0)            (1, 0, 0)            (1, 0, 0)            -1
    torso        (0, 0, 1)            (itself)             (0, 0, 1)            -1
    arm_pitch    (0, 1, 0)            (0, 1, 0)            (0, -1, 0)           +1
    arm_roll     (1, 0, 0)            (1, 0, 0)            (1, 0, 0)            -1
    arm_yaw      (0, 0, -1)           (0, 0, -1)           (0, 0, -1)           -1
    elbow_pitch  (0, 1, 0)            (0, 1, 0)            (0, -1, 0)           +1
    elbow_yaw    (1, 0, 0)            (1, 0, 0)            (1, 0, 0)            -1

The joint ranges agree: thigh_yaw [-1, 0.2] vs [-0.2, 1], thigh_roll
[-0.2, 2] vs [-2, 0.2] and arm_roll [-0.25, 3.14] vs [-3.14, 0.25] are
negated images, and the pitch, knee and elbow_pitch ranges are identical.
The twin body positions mirror too, with two exceptions that move no sign:
the right thigh_roll pivot sits 1.5 mm from the left one's image along the
roll axis itself, and right_arm_pitch_link sits 1 mm lower than the left
one.

Everything here is plain numpy on names and sizes (no model, no env), so
tests/unit covers it. The env validates the assembled maps against its real
observation sizes at construction.
"""

from __future__ import annotations

import numpy as np

_LEFT = "left_"
_RIGHT = "right_"

# Per robot: side-free joint name -> sign of the mirrored angle, target,
# velocity, torque or action (the derivation is the module docstring's
# table). A robot not listed here cannot train with symmetry on until its
# rows are derived the same way.
JOINT_SIGNS = {
    "roboto_origin": {
        "thigh_yaw_joint": -1.0,
        "thigh_roll_joint": -1.0,
        "thigh_pitch_joint": 1.0,
        "knee_joint": 1.0,
        "ankle_pitch_joint": 1.0,
        "ankle_roll_joint": -1.0,
        "torso_joint": -1.0,
        "arm_pitch_joint": 1.0,
        "arm_roll_joint": -1.0,
        "arm_yaw_joint": -1.0,
        "elbow_pitch_joint": 1.0,
        "elbow_yaw_joint": -1.0,
    },
}

DEFAULT_ROBOT = "roboto_origin"
DEFAULT_FEET = ("left_foot", "right_foot")


def axis_mirror_sign(axis_left, axis_right) -> float:
    """Sign of the twin joint's angle, from the two joints' base-frame axes.

    -1 when axis_right equals the mirror image of axis_left, +1 when it
    equals the negated image. A joint on the mirror plane passes its own
    axis twice.
    """
    a = np.asarray(axis_left, dtype=np.float64)
    b = np.asarray(axis_right, dtype=np.float64)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    image = a * np.array([1.0, -1.0, 1.0])
    if np.allclose(b, image, atol=1e-5):
        return -1.0
    if np.allclose(b, -image, atol=1e-5):
        return 1.0
    raise ValueError(
        f"axes {a.tolist()} and {b.tolist()} are not mirror twins: the twin's "
        "axis must be the mirror image of the other or its negation"
    )


def _side_free(name: str) -> str:
    for prefix in (_LEFT, _RIGHT):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def _twin(name: str) -> str:
    if name.startswith(_LEFT):
        return _RIGHT + name[len(_LEFT):]
    if name.startswith(_RIGHT):
        return _LEFT + name[len(_RIGHT):]
    return name


def _swap_perm(names) -> np.ndarray:
    """Index of each name's left/right twin in `names`."""
    names = list(names)
    index = {n: i for i, n in enumerate(names)}
    perm = []
    for n in names:
        twin = _twin(n)
        if twin not in index:
            raise ValueError(f"{n!r} has no mirror twin {twin!r} in {names}")
        perm.append(index[twin])
    return np.array(perm, dtype=np.int64)


def joint_mirror(joint_names, robot: str = DEFAULT_ROBOT):
    """(perm, sign) for a vector in actuated-joint order.

    `sign * x[perm]` is the mirrored vector: entry i reads the twin of
    joint i, signed per JOINT_SIGNS.
    """
    if robot not in JOINT_SIGNS:
        raise KeyError(
            f"no joint mirror signs for robot {robot!r}; derive them from its "
            f"MJCF axes and add them to JOINT_SIGNS (known: {sorted(JOINT_SIGNS)})"
        )
    table = JOINT_SIGNS[robot]
    missing = [n for n in joint_names if _side_free(n) not in table]
    if missing:
        raise KeyError(f"no mirror sign for joint(s) {missing} of robot {robot!r}")
    perm = _swap_perm(joint_names)
    sign = np.array([table[_side_free(n)] for n in joint_names], dtype=np.float32)
    return perm, sign


def foot_mirror(foot_names=DEFAULT_FEET):
    """(perm, sign) for a per-foot vector in foot_sites order."""
    return _swap_perm(foot_names), np.ones(len(foot_names), dtype=np.float32)


def component_mirror(name, joint_names, foot_names=DEFAULT_FEET, robot: str = DEFAULT_ROBOT):
    """(perm, sign) arrays mirroring one observation catalog component.

    Covers HumanoidEnv._obs_catalog plus the joystick task's command and
    phase. An unknown component raises KeyError naming it.
    """
    n_feet = len(foot_names)

    def ident(size, sign):
        return np.arange(size, dtype=np.int64), np.asarray(sign, dtype=np.float32)

    if name == "gyro":
        # angular rate: pseudo-vector
        return ident(3, [-1.0, 1.0, -1.0])
    if name in ("gravity", "linvel"):
        # body-frame direction or velocity: vector
        return ident(3, [1.0, -1.0, 1.0])
    if name in ("joint_pos", "joint_vel", "last_action", "actuator_force"):
        return joint_mirror(joint_names, robot)
    if name == "command":
        # (vx, vy, wz)
        return ident(3, [1.0, -1.0, -1.0])
    if name == "phase":
        # cos(leg phases) ++ sin(leg phases), foot_sites order. Swapping the
        # feet swaps their clock offsets; the master clock is unchanged.
        fperm, fsign = foot_mirror(foot_names)
        return np.concatenate([fperm, fperm + n_feet]), np.concatenate([fsign, fsign])
    if name == "height":
        return ident(1, [1.0])
    if name == "contacts":
        return foot_mirror(foot_names)
    raise KeyError(
        f"no mirror map for obs component {name!r}; add it to "
        "humanoid_lab.envs.symmetry before training with symmetry on"
    )


def obs_mirror(names, joint_names, foot_names=DEFAULT_FEET, robot: str = DEFAULT_ROBOT):
    """(perm, sign) for a concatenated observation vector.

    `names` is the ordered component list (obs.state or obs.privileged).
    `sign * obs[perm]` is the observation the mirrored world would produce.
    """
    perms, signs, offset = [], [], 0
    for name in names:
        perm, sign = component_mirror(name, joint_names, foot_names, robot)
        perms.append(np.asarray(perm, dtype=np.int64) + offset)
        signs.append(np.asarray(sign, dtype=np.float32))
        offset += len(perm)
    if not perms:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32)
    return np.concatenate(perms), np.concatenate(signs)
