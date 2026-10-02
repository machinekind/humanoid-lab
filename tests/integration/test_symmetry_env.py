"""Symmetry augmentation on roboto_origin: the joint signs against the
compiled model's kinematics, and the env wiring.

Kinematics: a joint configuration and its mirror image (through
symmetry.joint_mirror) must put every body in mirror-image places. The feet
are checked by position and every body by orientation, so the arm and
ankle_roll/elbow_yaw signs, which move no foot, are checked too. Positions
skip the bodies whose source-XML origins are not exact twins (see
POSITION_EXEMPT).

Wiring: a mirror_prob=1 env must present exactly the mirrored view of the
world a mirror_prob=0 env produces from the same key, mirror the action
back before the physics, and store the real-frame action.
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

ROBOT_DIR = paths.ROBOTS_DIR / "roboto_origin"
PRESET = "deploy_pd"
M = np.diag([1.0, -1.0, 1.0])
# Bodies whose frame ORIGINS are not mirror twins in the source XML (their
# orientations are): the right thigh_roll frame sits 1.5 mm from the left
# one's image along the roll axis itself, which moves nothing below it, and
# right_arm_pitch_link sits 1 mm lower, which moves the whole right arm.
POSITION_EXEMPT = ("thigh_roll", "arm", "elbow")


def _make_env(enable=True, mirror_prob=0.5):
    cfg = default_config()
    cfg.episode_length = 50
    cfg.push.enable = False
    cfg.symmetry.enable = enable
    cfg.symmetry.mirror_prob = mirror_prob
    return Joystick(ROBOT_DIR, PRESET, cfg)


@pytest.fixture(scope="module")
def env_real():
    return _make_env(mirror_prob=0.0)


@pytest.fixture(scope="module")
def env_mirror():
    return _make_env(mirror_prob=1.0)


def _twin(name):
    if name.startswith("left_"):
        return "right_" + name[len("left_"):]
    if name.startswith("right_"):
        return "left_" + name[len("right_"):]
    return name


def _kinematics(env, joint_q):
    """CPU mj_kinematics at a base on the mirror plane (origin, upright)."""
    m = env.mj_model
    d = mujoco.MjData(m)
    d.qpos[:] = np.asarray(env._home_qpos)
    d.qpos[env._base_qadr : env._base_qadr + 7] = [0.0, 0.0, 0.75, 1.0, 0.0, 0.0, 0.0]
    d.qpos[np.asarray(env._qadr)] = joint_q
    mujoco.mj_kinematics(m, d)
    return d


def _random_and_mirrored_q(env, seed):
    rs = env.robot_spec
    perm, sign = symmetry.joint_mirror(rs.actuated_joints, rs.name)
    lo = np.asarray(env._soft_lo)
    hi = np.asarray(env._soft_hi)
    q = np.random.default_rng(seed).uniform(lo, hi)
    return q, sign * q[perm]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_mirrored_joints_put_the_feet_in_mirror_image_places(env_real, seed):
    env = env_real
    m = env.mj_model
    q, q_m = _random_and_mirrored_q(env, seed)
    d = _kinematics(env, q)
    d_m = _kinematics(env, q_m)
    for foot in env.robot_spec.foot_sites:
        p = d.site_xpos[m.site(_twin(foot)).id]
        p_m = d_m.site_xpos[m.site(foot).id]
        np.testing.assert_allclose(p_m, M @ p, atol=1e-4, err_msg=foot)


def test_a_self_mirrored_pose_puts_the_feet_mirror_to_each_other(env_real):
    """x equal, y negated, z equal."""
    env = env_real
    m = env.mj_model
    q, q_m = _random_and_mirrored_q(env, 3)
    d = _kinematics(env, 0.5 * (q + q_m))
    left = d.site_xpos[m.site("left_foot").id]
    right = d.site_xpos[m.site("right_foot").id]
    assert abs(left[1]) > 0.02
    np.testing.assert_allclose(left, M @ right, atol=1e-4)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_mirrored_joints_put_every_body_in_a_mirror_image_orientation(env_real, seed):
    env = env_real
    m = env.mj_model
    q, q_m = _random_and_mirrored_q(env, seed)
    d = _kinematics(env, q)
    d_m = _kinematics(env, q_m)
    for b in range(1, m.nbody):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        twin = m.body(_twin(name)).id
        R = d.xmat[twin].reshape(3, 3)
        R_m = d_m.xmat[b].reshape(3, 3)
        np.testing.assert_allclose(R_m, M @ R @ M, atol=1e-4, err_msg=name)
        if not any(tag in name for tag in POSITION_EXEMPT):
            np.testing.assert_allclose(d_m.xpos[b], M @ d.xpos[twin], atol=1e-4, err_msg=name)


def test_the_sign_table_matches_the_compiled_joint_axes(env_real):
    env = env_real
    m = env.mj_model
    rs = env.robot_spec
    _, sign = symmetry.joint_mirror(rs.actuated_joints, rs.name)
    d = _kinematics(env, np.zeros(env.action_size))
    for i, name in enumerate(rs.actuated_joints):
        axis = d.xaxis[m.joint(name).id]
        axis_twin = d.xaxis[m.joint(_twin(name)).id]
        assert symmetry.axis_mirror_sign(axis, axis_twin) == sign[i], name


def _mirror_obs(env, obs):
    return {
        "state": np.asarray(env._state_sign) * np.asarray(obs["state"])[np.asarray(env._state_perm)],
        "privileged_state": np.asarray(env._priv_sign)
        * np.asarray(obs["privileged_state"])[np.asarray(env._priv_perm)],
    }


def test_a_mirrored_env_presents_the_mirrored_view_of_the_same_world(env_real, env_mirror):
    rng = jax.random.PRNGKey(3)
    s_real = jax.jit(env_real.reset)(rng)
    s_mir = jax.jit(env_mirror.reset)(rng)
    assert not bool(s_real.info["mirror"])
    assert bool(s_mir.info["mirror"])

    np.testing.assert_array_equal(np.asarray(s_real.data.qpos), np.asarray(s_mir.data.qpos))
    np.testing.assert_array_equal(
        np.asarray(s_real.info["command"]), np.asarray(s_mir.info["command"])
    )
    for k, v in _mirror_obs(env_real, s_real.obs).items():
        np.testing.assert_allclose(np.asarray(s_mir.obs[k]), v, atol=1e-6, err_msg=k)

    # The mirrored env receives the mirrored action and maps it back:
    # identical real-frame physics and reward, real-frame last_action.
    act = jax.random.uniform(
        jax.random.PRNGKey(4), (env_real.action_size,), minval=-0.5, maxval=0.5
    )
    act_m = env_mirror._act_sign * act[env_mirror._act_perm]
    n_real = jax.jit(env_real.step)(s_real, act)
    n_mir = jax.jit(env_mirror.step)(s_mir, act_m)
    np.testing.assert_allclose(np.asarray(n_real.data.qpos), np.asarray(n_mir.data.qpos), atol=1e-6)
    np.testing.assert_allclose(float(n_real.reward), float(n_mir.reward), atol=1e-5)
    np.testing.assert_allclose(np.asarray(n_mir.info["last_action"]), np.asarray(act), atol=1e-6)
    for k, v in _mirror_obs(env_real, n_real.obs).items():
        np.testing.assert_allclose(np.asarray(n_mir.obs[k]), v, atol=1e-5, err_msg=k)

    # The policy reads back the action it emitted.
    names = env_mirror.actor_obs_names
    sizes = {
        "gyro": 3, "gravity": 3, "command": 3, "phase": 2 * env_mirror._n_feet,
        "joint_pos": env_mirror.action_size, "joint_vel": env_mirror.action_size,
        "last_action": env_mirror.action_size,
    }
    start = sum(sizes[n] for n in names[: names.index("last_action")])
    seen = np.asarray(n_mir.obs["state"])[start : start + env_mirror.action_size]
    np.testing.assert_allclose(seen, np.asarray(act_m), atol=1e-6)


def test_symmetry_off_adds_no_info_key():
    env = _make_env(enable=False)
    state = jax.jit(env.reset)(jax.random.PRNGKey(5))
    assert "mirror" not in state.info
    assert not hasattr(env, "_act_perm")
