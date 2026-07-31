"""Mirror augmentation against real models and a real env (port item 3.1).

Two layers, mirroring w01-tek's training/tests/integration/test_symmetry.py.

1. Derivation. The joint signs are measured on the compiled model, never read
   off axis names, so this file pins what the measurement returns for both
   robots and proves the measurement rejects a pairing the geometry
   contradicts.
2. Wiring. A `mirror_prob=1.0` env must present exactly the mirrored view of
   the same real-frame world a `mirror_prob=0.0` env produces, with identical
   rewards, and must un-mirror the policy's action before the physics.

The maps are built at construction and only when `symmetry.enable` is set, so
this file builds its own envs instead of flipping the flag on a shared one.
"""

from __future__ import annotations

import jax
import jax.numpy as jp
import mujoco
import numpy as np
import pytest

from humanoid_lab import paths
from humanoid_lab.envs import symmetry
from humanoid_lab.envs.joystick import Joystick, default_config
from humanoid_lab.robot.build import build_spec, compile_spec
from humanoid_lab.robot.spec import load_robot_spec

ASIMOV = paths.ROBOTS_DIR / "asimov_v1"
ROBOTO = paths.ROBOTS_DIR / "roboto_origin"
PRESET = "sizing_ideal"


def compiled(robot_dir, preset=PRESET):
    """(spec, model) for a robot, without building an env or touching MJX."""
    spec = load_robot_spec(robot_dir)
    return spec, compile_spec(build_spec(robot_dir, preset, None))


@pytest.fixture(scope="module")
def asimov():
    return compiled(ASIMOV)


@pytest.fixture(scope="module")
def roboto():
    return compiled(ROBOTO, "deploy_pd")


# -- the derived tables ------------------------------------------------------
#
# DERIVED, not chosen. The signs below came out of symmetry.derive()'s probe
# on the compiled model: perturb one joint of a pair, mirror the resulting
# foot-site displacement about the robot's xz-plane, and see which sign of
# the partner joint reproduces it (envs/symmetry.py documents the procedure
# and its tolerances). Re-derive by running that function; do not hand-edit.
#
# The test exists to catch silent drift in the derivation -- a changed probe
# pose, a re-exported model, a reordered actuated_joints list would all move
# these tables, and every one of them changes what the augmentation means.
#
# The two robots disagree about every joint sign that a naming rule would
# have to guess, which is why nothing here is read off axis conventions:
# asimov_v1 mirrors with -1 everywhere (its left/right pitch hinges carry
# opposite local y-axes), roboto_origin needs +1 on the pitch chain and -1 on
# the yaw/roll chain (identical axes on both sides).

ASIMOV_JOINT_PERM = [6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5]
ASIMOV_JOINT_SIGN = [-1.0] * 12

# left leg, right leg, torso (centerline, its own partner), left arm, right arm
ROBOTO_JOINT_PERM = [6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 12, 18, 19, 20, 21, 22, 13, 14, 15, 16, 17]
ROBOTO_LEG_SIGN = [-1.0, -1.0, 1.0, 1.0, 1.0, -1.0]  # thigh yaw/roll/pitch, knee, ankle pitch/roll
ROBOTO_ARM_SIGN = [1.0, -1.0, -1.0, 1.0, -1.0]  # arm pitch/roll/yaw, elbow pitch/yaw
ROBOTO_JOINT_SIGN = ROBOTO_LEG_SIGN * 2 + [-1.0] + ROBOTO_ARM_SIGN * 2


def derive(spec, model, keyframe="home", symmetry_map=None):
    return symmetry.derive(model, spec, model.key(keyframe).qpos, symmetry_map=symmetry_map)


def test_the_derived_tables_for_asimov_v1_match_their_snapshot(asimov):
    spec, model = asimov
    tables = derive(*asimov)

    assert list(spec.actuated_joints[:1]) == ["left_hip_pitch_joint"]  # order pinned elsewhere
    assert tables.joint_perm.tolist() == ASIMOV_JOINT_PERM
    assert tables.joint_sign.tolist() == ASIMOV_JOINT_SIGN
    assert tables.foot_perm.tolist() == [1, 0]
    assert model.nu == 12


def test_the_derived_tables_for_roboto_origin_match_their_snapshot(roboto):
    tables = derive(*roboto)

    assert tables.joint_perm.tolist() == ROBOTO_JOINT_PERM
    assert tables.joint_sign.tolist() == ROBOTO_JOINT_SIGN
    assert tables.foot_perm.tolist() == [1, 0]


def test_the_centerline_torso_joint_mirrors_onto_itself_with_a_flipped_sign(roboto):
    """roboto_origin's torso yaw sits on the mirror plane: its partner is
    itself, and turning the torso left is turning it right in the mirror."""
    spec, _model = roboto
    tables = derive(*roboto)
    i = spec.actuated_joints.index("torso_joint")

    assert int(tables.joint_perm[i]) == i
    assert float(tables.joint_sign[i]) == -1.0


def test_the_signs_do_not_depend_on_which_keyframe_the_probe_starts_from(asimov):
    """The mirror map is a property of the model, not of the reset pose.
    asimov_v1's two keyframes differ at the knees and ankles (knees_bent
    holds +-0.4 rad, with opposite signs left and right) and must not move
    a single entry."""
    home = derive(*asimov, keyframe="home")
    bent = derive(*asimov, keyframe="knees_bent")

    assert home.joint_sign.tolist() == bent.joint_sign.tolist()
    assert home.joint_perm.tolist() == bent.joint_perm.tolist()


def test_a_pairing_the_geometry_contradicts_is_rejected(asimov):
    """The probe doubles as model validation. Pair each hip pitch with the
    other side's KNEE -- still an involution, still leg-to-leg, and still
    wrong: no sign of a knee reproduces a mirrored hip's foot displacement."""
    spec, _model = asimov
    doctored = dict(zip(spec.actuated_joints[:6], spec.actuated_joints[6:]))
    doctored["left_hip_pitch_joint"] = "right_knee_joint"
    doctored["left_knee_joint"] = "right_hip_pitch_joint"

    with pytest.raises(symmetry.SymmetryError, match="left_hip_pitch_joint"):
        derive(*asimov, symmetry_map=doctored)


def test_a_robot_with_no_symmetry_map_cannot_build_one(asimov):
    spec, _model = asimov

    with pytest.raises(symmetry.SymmetryError, match="left_hip_pitch_joint"):
        derive(*asimov, symmetry_map={})


# -- the physical assumption -------------------------------------------------
#
# w01-tek's test_model_is_statically_mirror_symmetric, adapted. The
# augmentation is only semantically right if the two sides of the robot are
# each other's mirror image; the numbers below are what these two models
# actually measure, so a re-export that breaks symmetry is visible here.


@pytest.mark.parametrize(
    "fixture,site_tol,mass_tol",
    [("asimov", 2e-4, 1e-3), ("roboto", 1e-5, 2e-3)],
)
def test_the_two_sides_of_each_robot_are_mirror_images(fixture, site_tol, mass_tol, request):
    """Measured 2026-07-31 at the home keyframe: asimov_v1's foot sites
    mirror to 1.0e-4 m and its paired leg links to 4.6e-4 kg (a vendored CAD
    export with a 0.1 mm offset on the right ankle-pitch link);
    roboto_origin's to 5.9e-7 m and 1.5e-3 kg. Residual asymmetry at this
    scale is exactly what the augmentation is insurance against; a
    centimetre of it would mean the mirrored world is a different robot.
    """
    spec, model = request.getfixturevalue(fixture)
    data = mujoco.MjData(model)
    data.qpos[:] = model.key("home").qpos
    mujoco.mj_forward(model, data)

    left, right = (model.site(n).id for n in spec.foot_sites)
    mirrored = data.site_xpos[left] * np.array([1.0, -1.0, 1.0])
    assert np.abs(mirrored - data.site_xpos[right]).max() < site_tol

    for a, b in spec.symmetry.items():
        ba = int(model.jnt_bodyid[model.joint(a).id])
        bb = int(model.jnt_bodyid[model.joint(b).id])
        assert abs(float(model.body_mass[ba] - model.body_mass[bb])) < mass_tol, a


# -- the env wiring ----------------------------------------------------------


def make_env(mirror_prob=None):
    cfg = default_config()
    cfg.episode_length = 50  # fast tests, not a training config
    if mirror_prob is not None:
        cfg.symmetry.enable = True
        cfg.symmetry.mirror_prob = mirror_prob
    return Joystick(ASIMOV, PRESET, cfg)


@pytest.fixture(scope="module")
def env_stock():
    return make_env()


@pytest.fixture(scope="module")
def env_real():
    return make_env(0.0)


@pytest.fixture(scope="module")
def env_mirror():
    return make_env(1.0)


def mirror_obs(env, obs):
    return {
        "state": np.asarray(env._state_sign) * np.asarray(obs["state"])[np.asarray(env._state_perm)],
        "privileged_state": np.asarray(env._priv_sign)
        * np.asarray(obs["privileged_state"])[np.asarray(env._priv_perm)],
    }


def test_off_draws_no_mirror_flag_and_consumes_no_rng(env_stock):
    """The off-switch gate: the stock 3-way reset split is untouched, so the
    whole downstream RNG stream is the pre-port one."""
    key = jax.random.PRNGKey(0)
    state = env_stock.reset(key)

    assert "mirror" not in state.info
    np.testing.assert_array_equal(
        np.asarray(state.info["rng"]), np.asarray(jax.random.split(key, 3)[0])
    )


def test_on_draws_the_flag_once_at_reset_from_its_own_key(env_mirror):
    """Enabled, reset splits one extra key -- the no_progress pattern -- and
    the flag it draws is fixed for the whole episode."""
    key = jax.random.PRNGKey(0)
    state = env_mirror.reset(key)

    assert bool(state.info["mirror"]) is True
    np.testing.assert_array_equal(
        np.asarray(state.info["rng"]), np.asarray(jax.random.split(key, 4)[0])
    )
    stepped = env_mirror.step(state, jp.zeros(env_mirror.action_size))
    assert bool(stepped.info["mirror"]) is True


def test_the_flag_is_false_at_probability_zero(env_real):
    state = env_real.reset(jax.random.PRNGKey(0))

    assert bool(state.info["mirror"]) is False


def test_the_maps_are_as_wide_as_the_observations_the_env_actually_builds(env_mirror):
    """The validation w01-tek's docstring claims: the maps are sized from this
    env's own catalog, so an obs list change cannot leave them stale."""
    state = env_mirror.reset(jax.random.PRNGKey(0))

    assert env_mirror._state_perm.shape == state.obs["state"].shape
    assert env_mirror._priv_perm.shape == state.obs["privileged_state"].shape


def test_a_catalog_signal_with_no_mirror_map_fails_at_construction():
    """A new observation must get a mirror entry before it can be trained
    with the augmentation on."""

    class ExtraSignal(Joystick):
        def _obs_catalog(self, data, info):
            catalog = super()._obs_catalog(data, info)
            catalog["imu_accel"] = self._gyro(data)  # any 3-vector
            return catalog

    cfg = default_config()
    cfg.symmetry.enable = True
    cfg.obs.state = tuple(cfg.obs.state) + ("imu_accel",)

    with pytest.raises(symmetry.SymmetryError, match="imu_accel"):
        ExtraSignal(ASIMOV, PRESET, cfg)


def test_the_mirrored_env_presents_the_mirrored_view_of_the_same_world(env_real, env_mirror):
    """The wiring, end to end: same rng, so the same real-frame world; the
    observations are its mirror image; the policy's mirrored action is
    un-mirrored before the physics, so the two envs land on the same qpos
    with the same reward, and last_action is stored in the real frame."""
    key = jax.random.PRNGKey(3)
    s_real = env_real.reset(key)
    s_mirror = env_mirror.reset(key)

    np.testing.assert_array_equal(np.asarray(s_real.data.qpos), np.asarray(s_mirror.data.qpos))
    np.testing.assert_array_equal(
        np.asarray(s_real.info["command"]), np.asarray(s_mirror.info["command"])
    )
    for k, v in mirror_obs(env_real, s_real.obs).items():
        np.testing.assert_allclose(np.asarray(s_mirror.obs[k]), v, atol=1e-6)

    action = jax.random.uniform(
        jax.random.PRNGKey(4), (env_real.action_size,), minval=-0.5, maxval=0.5
    )
    mirrored_action = jp.asarray(env_mirror._act_sign) * action[jp.asarray(env_mirror._act_perm)]
    n_real = env_real.step(s_real, action)
    n_mirror = env_mirror.step(s_mirror, mirrored_action)

    np.testing.assert_allclose(
        np.asarray(n_real.data.qpos), np.asarray(n_mirror.data.qpos), atol=1e-6
    )
    np.testing.assert_allclose(float(n_real.reward), float(n_mirror.reward), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(
        np.asarray(n_mirror.info["last_action"]), np.asarray(action), atol=1e-6
    )
    for k, v in mirror_obs(env_real, n_real.obs).items():
        np.testing.assert_allclose(np.asarray(n_mirror.obs[k]), v, atol=1e-5)


def test_every_reward_term_is_identical_between_the_two_frames(env_real, env_mirror):
    """Physics, rewards and termination all run in the real frame, so a
    mirrored env cannot pay differently for the mirrored action -- term by
    term, not just in the total."""
    key = jax.random.PRNGKey(11)
    s_real, s_mirror = env_real.reset(key), env_mirror.reset(key)
    action = jax.random.uniform(
        jax.random.PRNGKey(12), (env_real.action_size,), minval=-0.5, maxval=0.5
    )
    mirrored_action = jp.asarray(env_mirror._act_sign) * action[jp.asarray(env_mirror._act_perm)]

    n_real = env_real.step(s_real, action)
    n_mirror = env_mirror.step(s_mirror, mirrored_action)

    for key_name, value in n_real.metrics.items():
        np.testing.assert_allclose(
            float(n_mirror.metrics[key_name]), float(value), rtol=1e-5, atol=1e-6, err_msg=key_name
        )
    assert float(n_real.done) == float(n_mirror.done)


def test_the_action_mirror_is_an_involution_on_the_real_env(env_mirror):
    perm = np.asarray(env_mirror._act_perm)
    sign = np.asarray(env_mirror._act_sign)
    action = np.arange(env_mirror.action_size, dtype=float)

    np.testing.assert_allclose(sign * (sign * action[perm])[perm], action)
