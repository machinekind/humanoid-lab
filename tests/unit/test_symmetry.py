"""Mirror-map algebra, model-free (envs/symmetry.py's numpy half).

Everything here builds its maps from synthetic (perm, sign) tables, so no
model, no env and no device is involved -- the joint SIGNS of a real robot
are derived numerically from its compiled model, which is
tests/integration/test_symmetry_env.py's job.

Ported from w01-tek's training/tests/unit/test_symmetry.py, with two
differences. Ours is a biped (2 feet, 6 joints a side) against w01-tek's
quadruped, and ours never hardcodes a joint sign: w01-tek's module owns a
JOINT_SIGN constant read off its XML's axes, and this port takes the signs as
an argument because no axis-naming rule survives both of our robots (see the
integration file).
"""

from __future__ import annotations

import numpy as np
import pytest

from humanoid_lab.envs import symmetry

# A synthetic biped: two feet, three joints a side, canonical order
# left-then-right. The signs are made up -- this file tests the algebra that
# carries them, not their derivation.
JOINTS = ["l_hip", "l_knee", "l_ankle", "r_hip", "r_knee", "r_ankle"]
PAIRS = {"l_hip": "r_hip", "l_knee": "r_knee", "l_ankle": "r_ankle"}
JPERM = np.array([3, 4, 5, 0, 1, 2])
JSIGN = np.array([-1.0, 1.0, -1.0, -1.0, 1.0, -1.0])
FOOT_PERM = np.array([1, 0])

# The joystick task's own obs lists, at this synthetic robot's sizes.
SIZES = {
    "gyro": 3,
    "gravity": 3,
    "joint_pos": 6,
    "joint_vel": 6,
    "last_action": 6,
    "command": 3,
    "phase": 4,
    "linvel": 3,
    "height": 1,
    "contacts": 2,
    "actuator_force": 6,
}
ACTOR = ["gyro", "gravity", "joint_pos", "joint_vel", "last_action", "command", "phase"]
PRIVILEGED = ACTOR + ["linvel", "height", "contacts", "actuator_force"]


def maps():
    return symmetry.component_maps(JPERM, JSIGN, FOOT_PERM)


def apply(perm, sign, x):
    return sign * np.asarray(x)[perm]


# -- the joint permutation ---------------------------------------------------


def test_the_permutation_pairs_every_joint_with_its_partner():
    perm = symmetry.joint_permutation(JOINTS, PAIRS)

    assert perm.tolist() == JPERM.tolist()


def test_the_permutation_is_an_involution():
    perm = symmetry.joint_permutation(JOINTS, PAIRS)

    assert perm[perm].tolist() == list(range(len(JOINTS)))


def test_a_centerline_joint_paired_with_itself_maps_to_itself():
    """A joint on the mirror plane (a waist/torso yaw) is its own partner.
    robot.yaml declares that explicitly rather than leaving it out."""
    joints = JOINTS + ["torso"]
    perm = symmetry.joint_permutation(joints, {**PAIRS, "torso": "torso"})

    assert perm.tolist() == [3, 4, 5, 0, 1, 2, 6]


def test_an_actuated_joint_the_symmetry_map_never_mentions_raises():
    """Silence is not a pairing: an unmapped joint would be left in place
    with sign +1, which is a wrong mirror, not a missing one."""
    with pytest.raises(symmetry.SymmetryError, match="l_ankle"):
        symmetry.joint_permutation(JOINTS, {"l_hip": "r_hip", "l_knee": "r_knee"})


def test_a_symmetry_map_naming_a_joint_that_is_not_actuated_raises():
    with pytest.raises(symmetry.SymmetryError, match="l_wrist"):
        symmetry.joint_permutation(JOINTS, {**PAIRS, "l_wrist": "r_wrist"})


def test_a_symmetry_map_that_pairs_one_joint_twice_raises():
    """Two entries claiming the same partner cannot both hold: the result
    would not be a permutation."""
    with pytest.raises(symmetry.SymmetryError, match="r_hip"):
        symmetry.joint_permutation(JOINTS, {**PAIRS, "l_ankle": "r_hip"})


# -- the per-component maps --------------------------------------------------


def test_gyro_and_gravity_mirror_as_a_pseudo_vector_and_a_vector():
    perm, sign = maps()["gyro"]
    assert apply(perm, sign, [1.0, 2.0, 3.0]).tolist() == [-1.0, 2.0, -3.0]

    perm, sign = maps()["gravity"]
    assert apply(perm, sign, [1.0, 2.0, 3.0]).tolist() == [1.0, -2.0, 3.0]


def test_the_command_mirror_flips_vy_and_wz_only():
    perm, sign = maps()["command"]

    assert apply(perm, sign, [0.4, 0.2, -0.5]).tolist() == [0.4, -0.2, 0.5]


def test_the_joint_channels_all_carry_the_same_joint_map():
    a = np.arange(6.0)
    expected = apply(JPERM, JSIGN, a).tolist()

    for name in ("joint_pos", "joint_vel", "last_action", "actuator_force"):
        perm, sign = maps()[name]
        assert apply(perm, sign, a).tolist() == expected


def test_the_phase_mirror_swaps_the_feet_in_the_cos_and_sin_halves():
    """phase is cos(2 feet) ++ sin(2 feet): mirroring swaps which foot owns
    which clock offset and touches no sign."""
    perm, sign = maps()["phase"]

    assert apply(perm, sign, [0.0, 1.0, 2.0, 3.0]).tolist() == [1.0, 0.0, 3.0, 2.0]


def test_the_contact_mirror_swaps_the_feet():
    perm, sign = maps()["contacts"]

    assert apply(perm, sign, [1.0, 0.0]).tolist() == [0.0, 1.0]


def test_linvel_mirrors_as_a_vector_and_height_is_untouched():
    perm, sign = maps()["linvel"]
    assert apply(perm, sign, [1.0, 2.0, 3.0]).tolist() == [1.0, -2.0, 3.0]

    perm, sign = maps()["height"]
    assert apply(perm, sign, [0.7]).tolist() == [0.7]


# -- the assembled observation map -------------------------------------------


@pytest.mark.parametrize("names", [ACTOR, PRIVILEGED], ids=["actor", "privileged"])
def test_the_assembled_obs_mirror_is_an_involution(names):
    perm, sign = symmetry.obs_mirror(names, SIZES, maps())
    n = sum(SIZES[m] for m in names)

    assert perm.shape == sign.shape == (n,)
    assert sorted(perm.tolist()) == list(range(n))  # a permutation
    assert set(np.abs(sign).tolist()) == {1.0}  # unit signs
    x = np.random.default_rng(0).normal(size=n)
    np.testing.assert_allclose(apply(perm, sign, apply(perm, sign, x)), x)


def test_every_block_of_the_assembled_actor_map_lands_at_its_own_offset():
    """Hand-written expectations for the whole 31-dim actor vector, so a
    component reorder or a size change cannot slip past the involution
    property above."""
    perm, sign = symmetry.obs_mirror(ACTOR, SIZES, maps())
    gyro = np.array([1.0, 2.0, 3.0])
    gravity = np.array([4.0, 5.0, 6.0])
    jpos = 10.0 + np.arange(6.0)
    jvel = 20.0 + np.arange(6.0)
    last = 30.0 + np.arange(6.0)
    command = np.array([40.0, 41.0, 42.0])
    phase = np.array([50.0, 51.0, 52.0, 53.0])
    m = apply(perm, sign, np.concatenate([gyro, gravity, jpos, jvel, last, command, phase]))

    assert m.shape == (31,)  # 3+3+6+6+6+3+4
    np.testing.assert_allclose(m[0:3], [-1.0, 2.0, -3.0])
    np.testing.assert_allclose(m[3:6], [4.0, -5.0, 6.0])
    for base, block in ((6, jpos), (12, jvel), (18, last)):
        np.testing.assert_allclose(m[base : base + 6], apply(JPERM, JSIGN, block))
    np.testing.assert_allclose(m[24:27], [40.0, -41.0, -42.0])
    np.testing.assert_allclose(m[27:31], [51.0, 50.0, 53.0, 52.0])


def test_an_obs_component_with_no_mirror_entry_raises():
    """w01-tek's KeyError pattern, adapted: a new catalog signal must get a
    mirror map before anything trains with the augmentation on."""
    with pytest.raises(symmetry.SymmetryError, match="imu_accel"):
        symmetry.obs_mirror(["gyro", "imu_accel"], {**SIZES, "imu_accel": 3}, maps())


def test_a_map_that_does_not_match_the_env_s_own_component_size_raises():
    """The validation w01-tek's docstring claims and its code does not do: the
    sizes come from the env's real catalog, so a robot whose command grew a
    fourth channel fails here instead of silently mirroring the wrong slots."""
    with pytest.raises(symmetry.SymmetryError, match="command"):
        symmetry.obs_mirror(["command"], {**SIZES, "command": 4}, maps())


def test_a_repeated_component_is_mirrored_in_both_of_its_slots():
    """`obs.state` is an ordered list, not a set; the offsets accumulate."""
    perm, sign = symmetry.obs_mirror(["gyro", "gyro"], SIZES, maps())

    np.testing.assert_allclose(
        apply(perm, sign, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        [-1.0, 2.0, -3.0, -4.0, 5.0, -6.0],
    )


# -- the noise argument ------------------------------------------------------


def test_the_actor_noise_scales_are_invariant_under_the_obs_mirror():
    """Why the env may mirror the ALREADY-NOISY observation (envs/base.py's
    _get_obs) instead of mirroring first and adding noise after.

    _build_obs draws one uniform per element and scales it by a per-COMPONENT
    constant, so the noise vector is i.i.d. within each component with a
    single scale. The mirror is a signed permutation that never moves a value
    across a component boundary, so the permuted scale vector is the scale
    vector itself, and a sign flip leaves a symmetric uniform's distribution
    alone. Mirroring after the noise is therefore the same DISTRIBUTION as
    noising after the mirror -- not the same sample, which is why the order
    matters for bit-exactness and not for correctness.
    """
    noise = {"gyro": 0.01, "joint_pos": 0.01, "joint_vel": 0.1}
    scales = np.concatenate([np.full(SIZES[n], noise.get(n, 0.0)) for n in ACTOR])
    perm, _sign = symmetry.obs_mirror(ACTOR, SIZES, maps())

    np.testing.assert_array_equal(scales[perm], scales)


# -- the deployment-frame rule -----------------------------------------------


def test_the_deployment_frame_forces_the_mirror_off():
    overrides = symmetry.deployment_frame_overrides({"symmetry": {"enable": True}})

    assert overrides["symmetry"]["enable"] is False


def test_the_deployment_frame_keeps_every_other_recorded_setting():
    """Only `enable` is measurement-only: the rebuilt env is still the run's
    own in every respect the measurement does not need to change."""
    overrides = symmetry.deployment_frame_overrides(
        {"symmetry": {"enable": True, "mirror_prob": 0.25}, "real_pose_ref": True}
    )

    assert overrides["symmetry"]["mirror_prob"] == 0.25
    assert overrides["real_pose_ref"] is True


def test_the_deployment_frame_forces_the_mirror_off_for_a_run_that_never_mentioned_it():
    """A run.json recorded before the switch existed still gets the explicit
    off: the measurement env does not depend on what happened to be saved."""
    assert symmetry.deployment_frame_overrides({})["symmetry"]["enable"] is False


def test_the_deployment_frame_does_not_mutate_the_run_s_own_dict():
    recorded = {"symmetry": {"enable": True}}
    symmetry.deployment_frame_overrides(recorded)

    assert recorded["symmetry"]["enable"] is True


def test_the_sizing_collector_rebuilds_its_env_in_the_deployment_frame(monkeypatch):
    """sizing/collect.py rolls a checkpoint out to size motors, so it is a
    measurement too -- one whose whole output is per-JOINT torque and speed,
    which a mirrored rollout would report on the wrong side of the robot.
    Unlike the battery it keeps the run's pushes and command sampling: it
    wants the distribution the trained gait really produces."""
    from humanoid_lab import registry
    from humanoid_lab.sizing import collect

    seen = {}

    def recorder(task, robot_dir, preset_name, env_overrides, actuator_overrides):
        seen.update(env_overrides)
        return "env"

    monkeypatch.setattr(registry, "make" + "_env", recorder)
    run = {
        "task": "sizing",
        "hydra_config": {
            "robot": {"dir": "robots/asimov_v1"},
            "actuators": {"name": "sizing_ideal"},
            "task": {"env": {"symmetry": {"enable": True}, "push": {"enable": True}}},
        },
    }
    collect.make_env_for_run(run)

    assert seen["symmetry"]["enable"] is False
    assert seen["push"]["enable"] is True


# -- what the augmentation can and cannot train ------------------------------


def signed_perm(perm, sign):
    n = len(perm)
    mat = np.zeros((n, n))
    mat[np.arange(n), perm] = sign
    return mat


def test_fixed_flag_world_mirror_is_invisible_to_the_policy():
    """Ported verbatim in intent from w01-tek's own test of the same name --
    the terrain_blind_v3 post-mortem, as an algebraic fact.

    The env's augmentation presents sigma(obs) and un-mirrors the action
    before a mirror-equivariant physics step, with the flag fixed per env
    (brax's auto-reset never re-runs reset). For ANY policy -- however
    asymmetric -- the policy-frame (obs, action, reward) stream of a mirrored
    env is then IDENTICAL to a plain env's: the flag is unobservable, so PPO
    gets zero gradient toward pi(sigma s) = sigma pi(s). World mirroring
    cancels the chirality of the WORLD; it cannot symmetrize the POLICY.

    That is how w01-tek's terrain_blind_v3_20260728 trained with
    symmetry.enable=true and provably correct maps (mirror-wrapping the
    checkpoint swapped its spin_left/spin_right scores exactly: -33/+124 deg
    became +127/-35) and still could not turn one way. Its asymmetric turning
    was fixed by the reward mechanics of port items 1.1 to 1.6, not by this.
    A fix that acts on the POLICY has to couple the two frames inside the
    LEARNER -- mirrored-transition duplication in the PPO batch, a symmetry
    loss, or a mirror-equivariant network -- and would make the two streams
    below differ for an asymmetric policy.
    """
    perm, sign = symmetry.obs_mirror(ACTOR, SIZES, maps())
    mo = signed_perm(perm, sign)  # obs mirror
    ma = signed_perm(JPERM, JSIGN)  # action mirror
    n, nu = len(perm), len(JPERM)
    rng = np.random.default_rng(1)

    # Linear stand-in plant, made mirror-equivariant by symmetrization -- the
    # property tests/integration/test_symmetry_env.py verifies for the real env.
    a0 = rng.normal(size=(n, n)) / (2 * n)
    b0 = rng.normal(size=(n, nu)) / (2 * n)
    plant_a = (a0 + mo @ a0 @ mo) / 2.0
    plant_b = (b0 + mo @ b0 @ ma) / 2.0
    w = rng.normal(size=(nu, n))

    def pi(obs):
        return np.tanh(w @ obs)

    x0 = rng.normal(size=n)
    # the premise: a random policy is genuinely asymmetric in its own frame
    assert np.abs(pi(mo @ x0) - ma @ pi(x0)).max() > 0.05

    def stream(mirrored):
        x = mo @ x0 if mirrored else x0.copy()
        out = []
        for _ in range(50):
            obs = mo @ x if mirrored else x
            act = pi(obs)  # what the learner records
            u = ma @ act if mirrored else act  # what the plant receives
            reward = x @ x + u @ u  # any mirror-invariant reward
            out.append((obs, act, reward))
            x = plant_a @ x + plant_b @ u
        return out

    for (o_p, a_p, r_p), (o_m, a_m, r_m) in zip(stream(False), stream(True)):
        np.testing.assert_allclose(o_m, o_p, atol=1e-12)
        np.testing.assert_allclose(a_m, a_p, atol=1e-12)
        np.testing.assert_allclose(r_m, r_p, atol=1e-12)
