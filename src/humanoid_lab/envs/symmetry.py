"""Left/right mirror maps for the observation and action spaces (port 3.1).

Mirroring is about the robot's own xz-plane (y -> -y). Left and right joints
swap; vectors mirror as (x,-y,z) and pseudo-vectors (angular rates) as
(-x,y,-z); the command mirrors as (vx,-vy,-wz); the gait clock's feet swap.

Ported from w01-tek's wojtek_rl/symmetry.py with two deliberate changes.

1. The joint SIGNS are derived numerically from the compiled model instead of
   being read off axis conventions. w01-tek's module hardcodes a per-leg
   JOINT_SIGN triple justified by its XML's axes; no naming or axis rule
   survives both of our robots. Measured 2026-07-31: asimov_v1 mirrors with
   sign -1 on all twelve leg joints (its left/right pitch hinges carry
   opposite local y-axes), while roboto_origin needs -1 on the thigh
   yaw/roll and ankle roll and +1 on the thigh pitch, knee and ankle pitch
   (identical axes on both sides, including a compound 60-degree thigh
   yaw/roll pair). A rule fitted to either robot is wrong on the other.

2. The assembled observation map is validated against the env's OWN
   component sizes, not a static size table. w01-tek's docstring claims that
   check; its code compares against a hardcoded COMPONENT_SIZES dict, which
   cannot notice a robot whose joint count or foot count differs from the
   table's.

Everything below is plain numpy over names and shapes, so the algebra half
is unit-testable with no model (tests/unit/test_symmetry.py) and the
derivation half runs once per env at construction, never inside a trace
(tests/integration/test_symmetry.py).
"""

from __future__ import annotations

import mujoco
import numpy as np

# Probe amplitude for the sign derivation, rad. Small enough that the
# response is essentially linear in the joint angle, large enough that the
# foot moves millimetres rather than float noise.
PROBE_DELTA = 0.05

# The winning sign must fit to this many metres, and must fit at least
# MIN_DISCRIMINATION times better than the losing sign.
#
# Both numbers are set by measurement, not taste. A static left/right
# geometry error of e metres shows up in this probe as about e*PROBE_DELTA
# of residual, so the 1e-4 m fit budget accepts a robot whose two sides
# differ by about 2 mm and rejects anything worse. asimov_v1's vendored CAD
# export is asymmetric by 1.0e-4 m (a right ankle-pitch link offset 0.1 mm in
# y), which lands at 5.0e-6 m of fit residual -- twenty times inside the
# budget. roboto_origin measures 5.9e-7 m static, 3.0e-8 m of fit. The
# discrimination floor of 100 is w01-tek's; the worst measured margin on
# either robot is 770 (asimov_v1's hip yaw, whose probe point sits close to
# its own axis).
MAX_FIT_RESIDUAL = 1e-4
MIN_DISCRIMINATION = 100.0


class SymmetryError(ValueError):
    """A mirror map could not be built, or does not fit the model/env.

    One named error type for every failure in this module: an actuated joint
    robot.yaml never paired, a pairing the model's geometry contradicts, an
    observation component with no mirror entry, a map whose length disagrees
    with the env's own observation layout. Every one of them is a
    construction-time failure -- nothing here runs inside a trace.
    """


# -- joint pairing -----------------------------------------------------------


def joint_permutation(actuated_joints, symmetry_map) -> np.ndarray:
    """Involution over `actuated_joints` from robot.yaml's `symmetry` map.

    The map is written left -> right; both directions are filled in here. A
    centerline joint (a waist or torso yaw on the mirror plane) is written as
    its own partner, `torso_joint: torso_joint`, and maps to itself.

    Every actuated joint must appear. An unmapped joint would otherwise be
    left in place with sign +1, which is not a missing mirror but a wrong
    one, and would be invisible until a policy trained on it.
    """
    index = {name: i for i, name in enumerate(actuated_joints)}
    partner: dict[str, str] = {}
    for left, right in symmetry_map.items():
        for name in (left, right):
            if name not in index:
                raise SymmetryError(
                    f"robot.yaml's symmetry map names joint '{name}', which is not one of the "
                    f"actuated joints {list(actuated_joints)}"
                )
        for a, b in ((left, right), (right, left)):
            if partner.setdefault(a, b) != b:
                raise SymmetryError(
                    f"robot.yaml's symmetry map pairs joint '{a}' with both '{partner[a]}' and "
                    f"'{b}'; a mirror map has to be a permutation"
                )

    missing = [n for n in actuated_joints if n not in partner]
    if missing:
        raise SymmetryError(
            f"actuated joint(s) {missing} are missing from robot.yaml's symmetry map. Every "
            "actuated joint needs a partner (a centerline joint is written as its own, e.g. "
            "'torso_joint: torso_joint'); symmetry.enable cannot be used until they all have one"
        )
    return np.array([index[partner[n]] for n in actuated_joints], dtype=np.int32)


# -- per-component observation maps ------------------------------------------


def component_maps(jperm, jsign, foot_perm) -> dict:
    """name -> (perm, sign) for every entry of the env's observation catalog.

    `jperm`/`jsign` mirror a vector in actuated-joint order; `foot_perm`
    mirrors a per-foot vector, in robot.yaml's foot_sites order.
    """
    jperm = np.asarray(jperm, dtype=np.int32)
    jsign = np.asarray(jsign, dtype=np.float32)
    foot_perm = np.asarray(foot_perm, dtype=np.int32)
    n_feet = len(foot_perm)
    joint = (jperm, jsign)
    return {
        # body-frame angular rate: a pseudo-vector
        "gyro": (np.arange(3), np.array([-1.0, 1.0, -1.0], dtype=np.float32)),
        # body-frame gravity direction: a true vector
        "gravity": (np.arange(3), np.array([1.0, -1.0, 1.0], dtype=np.float32)),
        "joint_pos": joint,
        "joint_vel": joint,
        "last_action": joint,
        # A generalized force is conjugate to its joint angle, so it carries
        # the joint's own sign.
        "actuator_force": joint,
        # (vx, vy, wz). Fixed at three channels: a task that adds a fourth
        # (w01-tek commands a stand height) fails the size check in
        # obs_mirror rather than mirroring the wrong slots.
        "command": (np.arange(3), np.array([1.0, -1.0, -1.0], dtype=np.float32)),
        # cos(feet) ++ sin(feet): swapping the feet swaps which foot owns
        # which clock offset. The master clock itself does not move, so the
        # mirrored gait is the same schedule with left and right exchanged.
        "phase": (
            np.concatenate([foot_perm, foot_perm + n_feet]),
            np.ones(2 * n_feet, dtype=np.float32),
        ),
        "linvel": (np.arange(3), np.array([1.0, -1.0, 1.0], dtype=np.float32)),
        "height": (np.arange(1), np.ones(1, dtype=np.float32)),
        "contacts": (foot_perm, np.ones(n_feet, dtype=np.float32)),
    }


def obs_mirror(names, sizes, maps) -> tuple[np.ndarray, np.ndarray]:
    """(perm, sign) for one concatenated observation vector.

    `names` is a resolved, ordered component list (the env's actor list after
    the obs.include filter, or its privileged list). `sizes` maps each name to
    the length that component actually has in this env's catalog, measured on
    the env's own model -- never a static table. Applying `sign * obs[perm]`
    yields the observation the mirrored world would have produced.
    """
    perms, signs, offset = [], [], 0
    for name in names:
        if name not in maps:
            raise SymmetryError(
                f"no mirror map for obs component '{name}'; add it to envs/symmetry.py's "
                "component_maps before training with symmetry.enable=true"
            )
        perm, sign = maps[name]
        size = sizes[name]
        if len(perm) != size or len(sign) != size:
            raise SymmetryError(
                f"the mirror map for obs component '{name}' is {len(perm)} wide, but this env's "
                f"catalog produces {size} values for it"
            )
        perms.append(np.asarray(perm, dtype=np.int32) + offset)
        signs.append(np.asarray(sign, dtype=np.float32))
        offset += size
    if not perms:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float32)
    return np.concatenate(perms), np.concatenate(signs)
