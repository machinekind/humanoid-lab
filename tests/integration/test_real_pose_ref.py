"""The settled-pose anchor.

`pose` and `stand_still` anchor on `_default_pose`, the reset keyframe's
COMMANDED joint values. Under gravity and finite gains the robot settles
BELOW that command, so both terms charge a residual the policy cannot remove
and the size of the charge depends on the actuator preset. With
`real_pose_ref` on, the env settles a quasi-rigid copy of the model once at
construction and anchors on the pose that copy came to rest in. That anchor
is a function of the geometry alone: the same for every gain set and every
actuator model.

The tests use **roboto_origin**, not asimov_v1. asimov_v1's two keyframes are
not standing equilibria -- held rigid at `home` it topples backward within a
second (its CoM sits about 2 cm behind the heel) and at `knees_bent` it holds
for about two seconds and then goes over. Both fail the degenerate guard, and
that is what the guard is for; they are the fixtures for the raising tests
below. roboto_origin's `home` settles and stays put (base height flat to
1e-5 m out to 10 simulated seconds).

The ctrl anchor is NOT part of this: `_default_pose` stays what
`ctrl_from_action` centers on and what the `joint_pos` observation subtracts.
Only the reward anchor and the reset pose move.
"""

from __future__ import annotations

import jax
import jax.numpy as jp
import numpy as np
import pytest

from humanoid_lab import paths
from humanoid_lab.envs.joystick import Joystick, default_config

ROBOT_DIR = paths.ROBOTS_DIR / "roboto_origin"
ASIMOV_DIR = paths.ROBOTS_DIR / "asimov_v1"
PRESETS = ("deploy_pd", "sizing_ideal")

# Every roboto_origin joint group, so an override can rewrite the whole gain
# set at once (see the invariance tests).
ROBOTO_GROUPS = (
    "thigh_yaw",
    "thigh_roll",
    "thigh_pitch",
    "knee",
    "torso",
    "ankle_pitch",
    "ankle_roll",
    "arm_pitch",
    "arm_roll",
    "arm_yaw",
    "elbow_pitch",
    "elbow_yaw",
)


def build(
    preset="deploy_pd",
    *,
    real_pose_ref=False,
    robot_dir=ROBOT_DIR,
    reset_keyframe="home",
    reset_noise=None,
    actuator_overrides=None,
):
    cfg = default_config()
    cfg.episode_length = 50  # fast tests, not a training config
    cfg.real_pose_ref = real_pose_ref
    cfg.reset_keyframe = reset_keyframe
    if reset_noise is not None:
        cfg.reset_noise = reset_noise
    return Joystick(robot_dir, preset, cfg, actuator_overrides=actuator_overrides)


@pytest.fixture(scope="module")
def legacy_env():
    """The default: the keyframe anchor, no settle."""
    return build()


@pytest.fixture(scope="module")
def anchored_envs():
    """One settled env per actuator preset."""
    return {preset: build(preset, real_pose_ref=True) for preset in PRESETS}


@pytest.fixture(scope="module")
def settled_env():
    """Settled, with the reset pose noise off so reset lands exactly on it."""
    return build(real_pose_ref=True, reset_noise=0.0)


def rewards_at(env, data, command):
    info = dict(env.reset(jax.random.PRNGKey(0)).info)
    info["command"] = command
    n_feet = env._n_feet
    rewards, _fall = env._compute_rewards(
        data,
        info,
        jp.zeros(env.action_size),
        jp.zeros(n_feet, dtype=bool),
        jp.zeros(n_feet, dtype=bool),
    )
    return rewards


# -- off by default ---------------------------------------------------------


def test_the_flag_is_off_by_default():
    assert default_config().real_pose_ref is False


def test_off_the_pose_anchor_is_the_default_pose_itself(legacy_env):
    """Identity, not equality: off, the anchor is not a settled copy that
    happens to agree, it is the same array the legacy code used."""
    assert legacy_env._pose_anchor is legacy_env._default_pose


def test_off_the_reset_qpos_is_the_home_qpos_itself(legacy_env):
    assert legacy_env._reset_qpos is legacy_env._home_qpos


def test_off_nothing_is_settled(legacy_env):
    """The settle costs nothing while the flag is off."""
    assert legacy_env._settle_ctrl is None


# -- the settled anchor -----------------------------------------------------


@pytest.mark.parametrize("preset", PRESETS)
def test_the_settled_anchor_sags_below_the_keyframe_pose(anchored_envs, preset):
    """Gravity sag exists and is worth anchoring on: the settled pose is a
    measurable distance from the commanded one."""
    env = anchored_envs[preset]
    sag = np.abs(np.asarray(env._pose_anchor) - np.asarray(env._default_pose))
    assert float(sag.sum()) > 0.01


@pytest.mark.parametrize("preset", PRESETS)
def test_the_settled_anchor_is_a_standing_pose(anchored_envs, preset):
    """The guard's own premise: what got anchored is a robot on its feet."""
    env = anchored_envs[preset]
    height = float(np.asarray(env._reset_qpos)[env._base_qadr + 2])
    assert height > env._config.fall.min_height


def test_the_settled_anchor_is_identical_under_two_gain_sets():
    """The whole point of settling a QUASI-RIGID copy: the anchor is a
    function of the geometry, so a preset with a third of the stiffness
    anchors on exactly the same pose. Bit-identical, not close."""
    stock = build(real_pose_ref=True)
    soft = build(
        real_pose_ref=True,
        actuator_overrides={"groups": {g: {"kp": 37.0, "kd": 0.7} for g in ROBOTO_GROUPS}},
    )
    np.testing.assert_array_equal(
        np.asarray(soft._pose_anchor), np.asarray(stock._pose_anchor)
    )


def test_the_settled_anchor_is_identical_under_two_actuator_models():
    """The same, across actuator MODELS. An ideal-torque preset injects
    biastype NONE actuators whose ctrl is a torque; overwriting gainprm and
    biasprm on those alone would leave `force = 400*ctrl` and settle a
    different robot, so the settle forces the actuator TYPES too."""
    servo = build(real_pose_ref=True)
    torque = build(real_pose_ref=True, actuator_overrides={"model": "ideal_torque"})
    np.testing.assert_array_equal(
        np.asarray(torque._pose_anchor), np.asarray(servo._pose_anchor)
    )


def test_the_two_actuator_models_still_agree_where_the_clip_bites():
    """The case the test above cannot see. roboto_origin's home pose sits
    inside its 0.9 soft limits, so the settle clip is a no-op there and both
    models settle an unclipped pose. At 0.8 four joints fall outside and the
    clip has teeth -- which is where a clip taken from `_ctrl_lo`/`_ctrl_hi`
    diverges, because those are joint angles for a `pd` preset and the
    actuator forcerange in Nm for an `ideal_torque` one, so the same np.clip
    moves the pd targets and cannot touch the torque ones."""
    tight = {"soft_limit_factor": 0.8}
    servo = build(real_pose_ref=True, actuator_overrides=tight)
    torque = build(real_pose_ref=True, actuator_overrides={**tight, "model": "ideal_torque"})

    # Non-vacuity: the clip really does bite at this factor.
    assert np.any(np.asarray(servo._settle_ctrl) != np.asarray(servo._default_pose))
    np.testing.assert_array_equal(
        np.asarray(torque._settle_ctrl), np.asarray(servo._settle_ctrl)
    )
    np.testing.assert_array_equal(
        np.asarray(torque._pose_anchor), np.asarray(servo._pose_anchor)
    )


@pytest.mark.parametrize(
    "model",
    [pytest.param({}, id="pd"), pytest.param({"model": "ideal_torque"}, id="ideal_torque")],
)
def test_a_tighter_soft_limit_factor_moves_the_settled_anchor(model):
    """`soft_limit_factor` is the one config axis the anchor is NOT invariant
    to, and it moves the anchor for BOTH actuator models. It never reaches
    the compiled model (both models delete it -- see actuators/models.py), so
    the settle physics is identical and the clip is the only thing that can
    move the result: the runtime envelope is preset policy, and a pose
    settled outside it is one the policy can never command.

    Both roboto_origin presets are `pd`, so the model has to be overridden
    here rather than picked from PRESETS."""
    stock = build(real_pose_ref=True, actuator_overrides=model)
    tight = build(
        real_pose_ref=True, actuator_overrides={**model, "soft_limit_factor": 0.8}
    )

    moved = np.abs(np.asarray(tight._pose_anchor) - np.asarray(stock._pose_anchor)).max()
    assert moved > 1e-3


def test_two_presets_settle_to_the_same_pose(anchored_envs):
    """Close, but not bit-identical, and deliberately not asserted as such:
    roboto_origin's two presets set different joint ARMATURE (0.01 against
    0.0306/0.064/0.032), which the settle copy keeps. Armature is a property
    of the mechanism, not a gain, so it changes how fast the settle converges
    and leaves a ~5e-5 rad residual at the two-second cut. Gains and the
    actuator model, which are what the anchor promises invariance to, are
    bit-exact above."""
    a, b = (np.asarray(anchored_envs[p]._pose_anchor) for p in PRESETS)
    np.testing.assert_allclose(a, b, atol=1e-3)


# -- the settle procedure ---------------------------------------------------


@pytest.mark.parametrize("preset", PRESETS)
def test_the_settle_ctrl_is_the_soft_joint_limit_clip(anchored_envs, preset):
    """In the ANGLE domain, unconditionally. `_ctrl_lo`/`_ctrl_hi` are these
    same numbers for a `pd` preset and the actuator forcerange in Nm for an
    `ideal_torque` one, so clipping a radian target against them is a unit
    error that happens to be invisible on a pd preset."""
    env = anchored_envs[preset]
    expected = np.clip(np.asarray(env._default_pose), env._soft_lo, env._soft_hi)
    np.testing.assert_array_equal(np.asarray(env._settle_ctrl), expected)


def test_the_settle_ctrl_is_not_the_raw_ctrlrange_clip(anchored_envs):
    """A `pd` preset's actuators are deliberately
    ctrllimited=False, so their raw ctrlrange is [0, 0]: clipping the settle
    targets to it would command every joint to zero and settle a different
    robot. The runtime bounds -- the soft limits step() clips motor targets
    to -- are the honest envelope."""
    env = anchored_envs["deploy_pd"]
    raw = np.asarray(env._mj_model.actuator_ctrlrange)
    raw_clip = np.clip(np.asarray(env._default_pose), raw[:, 0], raw[:, 1])
    assert not np.allclose(raw_clip, np.asarray(env._settle_ctrl))


def test_the_settle_clips_its_targets_to_the_soft_joint_limits():
    """The clip with teeth. roboto_origin's home pose sits inside its own
    soft limits, so on a stock env the clip is a no-op and the assertion
    above cannot tell a clipped target from an unclipped one. Tightening the
    upper bound on a built env and calling the settle directly makes the clip
    bite well past the 0.029 rad a `soft_limit_factor` of 0.8 buys. The env is
    thrown away afterwards; `_settle_pose` reads the bounds and returns what
    it used, so nothing else has to move."""
    env = build()  # off: no settle at construction, nothing to undo
    env._soft_hi = np.minimum(env._soft_hi, 0.05)
    ctrl, _qpos = env._settle_pose()

    expected = np.clip(np.asarray(env._default_pose), env._soft_lo, env._soft_hi)
    np.testing.assert_array_equal(np.asarray(ctrl), expected)
    assert np.any(np.asarray(ctrl) != np.asarray(env._default_pose))


# -- the ctrl anchor does not move ------------------------------------------


@pytest.mark.parametrize("preset", PRESETS)
def test_the_default_pose_still_centers_the_action(anchored_envs, preset):
    """HARD CONTRACT. The settled pose is a REWARD anchor. What a zero action
    commands, and what the joint_pos observation subtracts, stay on
    `_default_pose`. Moving them would silently re-center the policy's
    action space on a sagged pose."""
    env = anchored_envs[preset]
    legacy = build(preset)

    np.testing.assert_array_equal(
        np.asarray(env._default_pose), np.asarray(legacy._default_pose)
    )
    np.testing.assert_array_equal(
        np.asarray(env._neutral_ctrl), np.asarray(legacy._neutral_ctrl)
    )

    state = env.reset(jax.random.PRNGKey(0))
    catalog = env._obs_catalog(state.data, state.info)
    np.testing.assert_array_equal(
        np.asarray(catalog["joint_pos"]),
        np.asarray(state.data.qpos[env._qadr] - env._default_pose),
    )


# -- reset and the rewards --------------------------------------------------


def test_reset_starts_at_the_settled_pose(settled_env):
    """With the pose noise off, reset lands on the settled state exactly:
    the robot starts at rest instead of dropping into its own sag.

    Exact at the joints and at the base position. The base QUATERNION is
    compared loosely because reset runs mjx.forward, which re-normalizes it:
    the settled quat is unit in the settle's float64 and a few 1e-8 off it
    once cast to the env's float32."""
    state = settled_env.reset(jax.random.PRNGKey(3))
    qpos = np.asarray(state.data.qpos)
    settled = np.asarray(settled_env._reset_qpos)
    adr = settled_env._base_qadr

    qadr = np.asarray(settled_env._qadr)
    np.testing.assert_array_equal(qpos[qadr], settled[qadr])
    np.testing.assert_array_equal(qpos[adr : adr + 3], settled[adr : adr + 3])
    np.testing.assert_allclose(qpos[adr + 3 : adr + 7], settled[adr + 3 : adr + 7], atol=1e-6)


def test_reset_noise_still_rides_on_the_settled_pose(anchored_envs):
    """The stock reset noise is unchanged; it just centers somewhere else."""
    env = anchored_envs["deploy_pd"]
    noise = env._config.reset_noise
    assert noise > 0.0

    qpos = np.asarray(env.reset(jax.random.PRNGKey(3)).data.qpos)
    offset = qpos[np.asarray(env._qadr)] - np.asarray(env._pose_anchor)
    assert np.all(np.abs(offset) <= noise)
    assert np.any(offset != 0.0)


def test_the_pose_reward_is_zero_at_the_settled_pose(settled_env):
    """The floor the mechanism exists to remove. At the pose the robot
    actually holds, the pose term charges nothing at all."""
    state = settled_env.reset(jax.random.PRNGKey(3))
    assert float(rewards_at(settled_env, state.data, jp.array([0.5, 0.0, 0.0]))["pose"]) == 0.0


def test_the_stand_still_penalty_is_zero_at_the_settled_pose(settled_env):
    state = settled_env.reset(jax.random.PRNGKey(3))
    rewards = rewards_at(settled_env, state.data, jp.zeros(3))
    assert float(rewards["stand_still"]) == 0.0


def test_the_keyframe_anchor_charges_for_that_same_pose(legacy_env, settled_env):
    """The same state, scored by the legacy anchor, is not free -- which is
    the sag the policy would otherwise be paying for forever."""
    state = settled_env.reset(jax.random.PRNGKey(3))
    legacy = rewards_at(legacy_env, state.data, jp.zeros(3))
    assert float(legacy["pose"]) > 0.0
    assert float(legacy["stand_still"]) > 0.01


def test_the_anchor_touches_no_other_reward_term(legacy_env, settled_env):
    """Two envs identical but for the anchor must agree on every term that
    does not read it."""
    state = settled_env.reset(jax.random.PRNGKey(3))
    command = jp.array([0.5, 0.0, 0.0])
    settled = rewards_at(settled_env, state.data, command)
    legacy = rewards_at(legacy_env, state.data, command)

    assert list(settled) == list(legacy)
    for key in legacy:
        if key in ("pose", "stand_still"):
            continue
        assert float(settled[key]) == float(legacy[key]), key


# -- the degenerate settle --------------------------------------------------


def test_a_settle_that_collapses_raises():
    """asimov_v1's `home` keyframe stands the robot on straight legs with its
    CoM about 2 cm behind the heel. Held rigid it topples backward and comes
    to rest on the floor at 0.111 m. A fallen robot must never become a
    silent reward anchor."""
    with pytest.raises(ValueError, match="real_pose_ref"):
        build(real_pose_ref=True, robot_dir=ASIMOV_DIR)


def test_the_message_names_the_robot_and_the_height():
    with pytest.raises(ValueError) as excinfo:
        build(real_pose_ref=True, robot_dir=ASIMOV_DIR)
    message = str(excinfo.value)
    assert "asimov" in message
    assert "0.11" in message


def test_a_settle_that_is_still_toppling_raises():
    """asimov_v1's `knees_bent` keyframe is marginal rather than collapsed:
    two seconds in it is still above the fall floor (0.614 m) but moving at
    0.15 rad/s and on its way over -- by four seconds it is on the floor. A
    height check alone would accept that snapshot, so the guard also requires
    the settle to have come to rest."""
    with pytest.raises(ValueError, match="real_pose_ref"):
        build(real_pose_ref=True, robot_dir=ASIMOV_DIR, reset_keyframe="knees_bent")
