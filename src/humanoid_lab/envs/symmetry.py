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
(tests/integration/test_symmetry_env.py).
"""

from __future__ import annotations

from typing import NamedTuple

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
# y) and its worst fit residual is 5.0e-6 m, twenty times inside the budget;
# roboto_origin measures 5.9e-7 m static and 1.8e-6 m of fit. The
# discrimination floor of 100 is w01-tek's. The worst margin measured on
# either robot is 290, on roboto_origin's centerline torso yaw, whose probe
# point (the subtree centre of mass below it) sits close to its own axis;
# asimov_v1's tightest is 770, at the hip yaw, for the same reason.
MAX_FIT_RESIDUAL = 1e-4
MIN_DISCRIMINATION = 100.0


class MirrorTables(NamedTuple):
    """The three permutations an env needs, all in canonical order.

    joint_perm/joint_sign mirror a vector in robot_spec.actuated_joints
    order (an action, a joint angle, a joint velocity, a joint torque);
    foot_perm mirrors a per-foot vector in robot_spec.foot_sites order.
    """

    joint_perm: np.ndarray
    joint_sign: np.ndarray
    foot_perm: np.ndarray


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


# -- deriving the tables from a compiled model -------------------------------


def derive(model, spec, qpos, *, symmetry_map=None, delta: float = PROBE_DELTA) -> MirrorTables:
    """Measure the mirror tables on `model` at the pose `qpos`.

    `spec` is the robot's RobotSpec; `symmetry_map` overrides its own
    left -> right map (tests use it to check that a wrong pairing is
    rejected). Plain MuJoCo on the CPU model, run once at construction,
    never traced -- about 3 mj_forward calls per joint pair.

    The signs are NOT read off the joint axes. Given the pairing, each pair
    is probed: perturb the left joint by `delta` from `qpos`, take its foot
    site's displacement in the base frame, mirror it about the robot's
    xz-plane, and ask which sign of the right joint reproduces it. The
    winner has to fit (MAX_FIT_RESIDUAL) and to beat the loser
    (MIN_DISCRIMINATION), so the probe doubles as validation that the robot
    really is mirror-symmetric under this pairing -- which is deliberate:
    the augmentation is only meaningful on a model that is.

    Displacements, not positions: asimov_v1's vendored model is asymmetric
    by 1.0e-4 m (0.1 mm of y offset on the right ankle-pitch link), and a
    raw position comparison would put every one of its residuals on that
    floor and tell the two candidate signs apart by a factor of 24 instead
    of a factor of 10^4. Subtracting the unperturbed pose removes the static
    error to first order and leaves the probe measuring what it is about,
    which is how the joint moves.
    """
    joints = list(spec.actuated_joints)
    perm = joint_permutation(joints, spec.symmetry if symmetry_map is None else symmetry_map)
    qpos = np.asarray(qpos, dtype=np.float64)

    qadr = np.array([model.joint(n).qposadr[0] for n in joints])
    anchor = qpos[qadr]
    mirrored_anchor = anchor[perm]

    chains = _foot_chains(model, spec, joints)
    foot_perm = _foot_pairing(spec, chains, perm)
    sign = _derive_signs(model, spec, joints, perm, chains, foot_perm, qpos, delta)

    # The probe pose has to be its own mirror, or "the mirrored world" is not
    # this world seen in a mirror. It is also the joint_pos observation's own
    # anchor (envs/base.py subtracts _default_pose), so a pose that is not
    # mirror-symmetric would break that observation's map too.
    off = np.abs(sign * mirrored_anchor - anchor)
    if off.max() > 1e-9:
        bad = joints[int(off.argmax())]
        raise SymmetryError(
            f"the probe pose of robot '{spec.name}' is not its own mirror: joint '{bad}' reads "
            f"{anchor[int(off.argmax())]} where the mirror of its partner reads "
            f"{(sign * mirrored_anchor)[int(off.argmax())]}. The reset keyframe has to be a "
            "mirror-symmetric pose before the augmentation can run"
        )
    return MirrorTables(perm, sign, foot_perm)


def _foot_chains(model, spec, joints) -> list[frozenset]:
    """Per foot site, the set of actuated joints on its body's ancestry.

    Structural: walks the kinematic tree from each foot site's body to the
    world body. No name parsing anywhere -- 'left' and 'right' are naming
    conventions, and one of our two robots numbers its foot geoms in the
    opposite order on the two sides.
    """
    index = {int(model.joint(n).id): i for i, n in enumerate(joints)}
    chains = []
    for site in spec.foot_sites:
        body = int(model.site_bodyid[model.site(site).id])
        found = set()
        while body != 0:
            start = int(model.body_jntadr[body])
            for j in range(start, start + int(model.body_jntnum[body])):
                if j in index:
                    found.add(index[j])
            body = int(model.body_parentid[body])
        chains.append(frozenset(found))
    return chains


def _foot_pairing(spec, chains, perm) -> np.ndarray:
    """Which foot is which foot's mirror, from the chains alone.

    Two feet are partners when their joint chains map onto each other under
    the joint pairing.
    """
    pairing = []
    for i, chain in enumerate(chains):
        mirrored = frozenset(int(perm[j]) for j in chain)
        matches = [k for k, other in enumerate(chains) if other == mirrored]
        if len(matches) != 1:
            raise SymmetryError(
                f"foot '{spec.foot_sites[i]}' of robot '{spec.name}' has no unique mirror partner: "
                f"its joint chain maps onto {[spec.foot_sites[k] for k in matches] or 'no foot'}. "
                "robot.yaml's foot_sites and symmetry map disagree about which feet pair up"
            )
        pairing.append(matches[0])
    foot_perm = np.array(pairing, dtype=np.int32)
    if foot_perm[foot_perm].tolist() != list(range(len(chains))):
        raise SymmetryError(
            f"the foot pairing derived for robot '{spec.name}' is not an involution: {pairing}"
        )
    return foot_perm


def _derive_signs(model, spec, joints, perm, chains, foot_perm, qpos, delta):
    """Per joint, the sign its angle takes in the mirrored world."""
    data = mujoco.MjData(model)
    free = [i for i in range(model.njnt) if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE]
    if len(free) != 1:
        raise SymmetryError(f"expected exactly one free joint on '{spec.name}', found {len(free)}")
    base_body = int(model.jnt_bodyid[free[0]])
    site_ids = [int(model.site(n).id) for n in spec.foot_sites]
    qadr = [int(model.joint(n).qposadr[0]) for n in joints]

    def forward(joint_index=None, perturb=0.0):
        data.qpos[:] = qpos
        data.qvel[:] = 0.0
        if joint_index is not None:
            data.qpos[qadr[joint_index]] += perturb
        mujoco.mj_forward(model, data)

    def in_base(point):
        rot = data.xmat[base_body].reshape(3, 3)
        return rot.T @ (point - data.xpos[base_body])

    def probe_point(joint_index):
        """A physical point the joint moves, and its mirror partner's.

        A foot site when the joint is on exactly one foot chain: the foot is
        the end of the chain, so every leg joint swings it. Otherwise the
        subtree centre of mass below the joint -- NOT the child body's own
        origin, which is where MuJoCo puts the hinge and therefore does not
        move at all when the joint turns (measured: 0.0 m for every one of
        roboto_origin's six unpaired-to-a-foot arm and torso joints).
        """
        feet = [f for f, chain in enumerate(chains) if joint_index in chain]
        partner_feet = [f for f, chain in enumerate(chains) if int(perm[joint_index]) in chain]
        if len(feet) == 1 and len(partner_feet) == 1 and int(foot_perm[feet[0]]) == partner_feet[0]:
            return (
                lambda: in_base(data.site_xpos[site_ids[feet[0]]]),
                lambda: in_base(data.site_xpos[site_ids[partner_feet[0]]]),
            )
        body = int(model.jnt_bodyid[model.joint(joints[joint_index]).id])
        partner_body = int(model.jnt_bodyid[model.joint(joints[int(perm[joint_index])]).id])
        return (
            lambda: in_base(data.subtree_com[body]),
            lambda: in_base(data.subtree_com[partner_body]),
        )

    signs = np.ones(len(joints), dtype=np.float32)
    for i, partner in enumerate(perm):
        partner = int(partner)
        if partner < i:
            continue  # the pair was measured from its other end
        own, other = probe_point(i)
        forward()
        rest_own, rest_other = own().copy(), other().copy()
        forward(i, delta)
        target = (own() - rest_own) * np.array([1.0, -1.0, 1.0])  # mirror about the xz-plane

        fits = {}
        for candidate in (1.0, -1.0):
            forward(partner, candidate * delta)
            fits[candidate] = float(np.abs((other() - rest_other) - target).max())
        best = min(fits, key=fits.get)
        worst = max(fits, key=fits.get)
        decisive = fits[worst] >= MIN_DISCRIMINATION * max(fits[best], 1e-15)
        if fits[best] > MAX_FIT_RESIDUAL or not decisive:
            raise SymmetryError(
                f"robot '{spec.name}' does not mirror under the pairing '{joints[i]}' <-> "
                f"'{joints[partner]}': turning the first by {delta} rad moves its probe point "
                f"{np.abs(target).max():.3e} m, and the second reproduces the mirrored motion to "
                f"{fits[1.0]:.3e} m with sign +1 and {fits[-1.0]:.3e} m with sign -1 (a sign has "
                f"to fit within {MAX_FIT_RESIDUAL} m and to beat the other by "
                f"{MIN_DISCRIMINATION:g}x). Either the pairing is wrong or the model is not "
                "mirror-symmetric; symmetry stays unavailable for this robot until one of them "
                "is fixed"
            )
        signs[i] = best
        signs[partner] = best
    return signs


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


# -- the deployment-frame rule -----------------------------------------------


def deployment_frame_overrides(env_overrides: dict) -> dict:
    """`env_overrides` with the mirror augmentation forced off.

    The rule, in one place: a measurement describes the frame the robot will
    be deployed in, and the mirror is a training-only stochastic augmentation
    that flips half the envs into the other one. A battery run under it would
    report a policy's spin_left as its spin_right, and a sizing rollout would
    attribute the left leg's torques to the right motor.

    Everything else the run recorded survives, `mirror_prob` included: only
    `enable` is measurement-only, the same split eval/battery.py's
    no_progress override makes. Callers: eval/battery.py (so eval/video.py
    too, which rebuilds through it) and sizing/collect.py.
    """
    out = dict(env_overrides or {})
    out["symmetry"] = {**(out.get("symmetry") or {}), "enable": False}
    return out
