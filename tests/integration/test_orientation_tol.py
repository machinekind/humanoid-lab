"""The tolerance cone on the orientation penalty (port item 1.8).

`reward.orientation` is `sum(gravity_xy²)`, which is `sin²` of the base's
tilt from vertical. With `reward.orientation_tol_deg` set it becomes
`max(sin²(tilt) - sin²(tol), 0)`: free inside the cone, and rising
continuously from its edge. A flat-referenced tilt penalty taxes the body
pitch that locomotion needs, such as leaning into acceleration. w01-tek runs
20 degrees, and rejected 10 because tilt is measured against gravity rather
than the local surface.

Unlike the tracking-kernel tests, these build one env per tolerance instead
of flipping a flag on a live config: `sin²(radians(tol))` is precomputed in
`Joystick.__init__`, so a live env never re-reads the config key. Two envs,
identical but for the tolerance, is the whole fixture set.

The tilts are manufactured by overwriting the base quaternion with a
rotation of a known angle from vertical and re-running `mjx.forward`, so
`sin²(tilt)` is known by construction and also recoverable from the state's
own gravity vector. The tests assert against the recovered value, so nothing
depends on the keyframe's pose.
"""

from __future__ import annotations

import jax
import jax.numpy as jp
import numpy as np
import pytest
from mujoco import mjx

from humanoid_lab import paths
from humanoid_lab.envs.joystick import Joystick, default_config
from humanoid_lab.rewards import terms

ROBOT_DIR = paths.ROBOTS_DIR / "asimov_v1"
PRESET = "sizing_ideal"

TOL_DEG = 20.0  # w01-tek's own tolerance
COMMAND = jp.array([0.5, 0.0, 0.0])


def build_env(tol_deg: float) -> Joystick:
    cfg = default_config()
    cfg.episode_length = 50  # fast tests, not a training config
    cfg.reward.orientation_tol_deg = tol_deg
    return Joystick(ROBOT_DIR, PRESET, cfg)


@pytest.fixture(scope="module")
def legacy_env():
    """Tolerance 0: the pre-1.8 penalty."""
    return build_env(0.0)


@pytest.fixture(scope="module")
def cone_env():
    return build_env(TOL_DEG)


@pytest.fixture(scope="module")
def reset_state(legacy_env):
    return legacy_env.reset(jax.random.PRNGKey(0))


def tilted(env, state, deg, axis=(0.0, 1.0, 0.0)):
    """`state` with the base rotated `deg` from vertical about `axis`."""
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    half = np.radians(deg) / 2.0
    quat = jp.array([np.cos(half), *(np.sin(half) * axis)])

    adr = env._base_qadr
    qpos = state.data.qpos.at[adr + 3 : adr + 7].set(quat)
    return state.replace(data=mjx.forward(env._mjx_model, state.data.replace(qpos=qpos)))


def sin2_tilt(env, state):
    """`sin²(tilt)` read back out of the state: the legacy penalty's value."""
    return float(terms.orientation(env._gravity_body(state.data)[:2]))


def orientation_reward(env, state):
    info = dict(state.info)
    info["command"] = COMMAND
    n_feet = env._n_feet
    rewards, _fall = env._compute_rewards(
        state.data,
        info,
        jp.zeros(env.action_size),
        jp.zeros(n_feet, dtype=bool),
        jp.zeros(n_feet, dtype=bool),
    )
    return float(rewards["orientation"])


# -- off by default --------------------------------------------------------


def test_the_tolerance_is_zero_by_default():
    assert default_config().reward.orientation_tol_deg == 0.0


def test_a_zero_tolerance_compiles_to_the_legacy_expression(legacy_env):
    """0 is falsy, so the `if self._orientation_tol` branch is not traced at
    all and the penalty is the untouched `sum(gravity_xy²)`."""
    assert legacy_env._orientation_tol == 0.0


@pytest.mark.parametrize("deg", [0.0, 5.0, 20.0, 35.0])
def test_a_zero_tolerance_scores_the_legacy_penalty_exactly(legacy_env, reset_state, deg):
    state = tilted(legacy_env, reset_state, deg)
    assert orientation_reward(legacy_env, state) == sin2_tilt(legacy_env, state)


def test_the_manufactured_tilt_is_the_tilt_it_claims_to_be(legacy_env, reset_state):
    """Guards the fixture: the penalty is sin² of the angle from vertical,
    so a 30 degree tilt must read sin²(30°) = 0.25 whatever the pose."""
    state = tilted(legacy_env, reset_state, 30.0)
    assert sin2_tilt(legacy_env, state) == pytest.approx(0.25, rel=1e-4)


# -- the cone --------------------------------------------------------------


@pytest.mark.parametrize("deg", [0.0, 5.0, 12.0, 19.0])
def test_a_tilt_inside_the_cone_is_free(legacy_env, cone_env, reset_state, deg):
    """Exactly zero, not merely small: inside the cone the term contributes
    nothing and has no gradient, which is what makes leaning into an
    acceleration untaxed."""
    state = tilted(legacy_env, reset_state, deg)
    assert orientation_reward(cone_env, state) == 0.0
    if deg:
        assert orientation_reward(legacy_env, state) > 0.0


@pytest.mark.parametrize("deg", [25.0, 30.0, 45.0])
def test_a_tilt_outside_the_cone_pays_the_legacy_penalty_less_the_cone(
    legacy_env, cone_env, reset_state, deg
):
    state = tilted(legacy_env, reset_state, deg)
    cone = np.square(np.sin(np.radians(TOL_DEG)))
    expected = sin2_tilt(legacy_env, state) - cone

    assert expected > 0.0
    assert orientation_reward(cone_env, state) == pytest.approx(expected, rel=1e-5)


def test_the_penalty_is_continuous_at_the_cone_edge(legacy_env, cone_env, reset_state):
    """A subtraction, not a switch: the penalty leaves the cone at zero and
    grows from there, so there is no step for the optimizer to fall off."""
    edge = orientation_reward(cone_env, tilted(legacy_env, reset_state, TOL_DEG))
    just_outside = [
        orientation_reward(cone_env, tilted(legacy_env, reset_state, TOL_DEG + d))
        for d in (0.1, 0.5, 2.0)
    ]

    assert edge == pytest.approx(0.0, abs=1e-7)
    assert just_outside == sorted(just_outside)
    assert just_outside[0] < 1e-3
    assert just_outside[-1] > 0.0


def test_the_cone_still_prices_a_nosedive(legacy_env, cone_env, reset_state):
    """The cone forgives a lean, not a fall: at 60 degrees it hands back
    sin²(20°) = 0.117 of a 0.75 penalty and leaves the rest standing."""
    state = tilted(legacy_env, reset_state, 60.0)
    legacy = orientation_reward(legacy_env, state)
    cone = orientation_reward(cone_env, state)

    assert legacy == pytest.approx(0.75, rel=1e-4)
    assert cone > 0.8 * legacy


def test_the_cone_is_round(legacy_env, cone_env, reset_state):
    """The tolerance is a cone around upright, not a per-axis band: the same
    angle costs the same whether it is pitch, roll, or a mix."""
    scores = [
        orientation_reward(cone_env, tilted(legacy_env, reset_state, 30.0, axis=axis))
        for axis in ((0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (1.0, -2.0, 0.0))
    ]
    for score in scores[1:]:
        assert score == pytest.approx(scores[0], rel=1e-5)
    assert scores[0] > 0.0


def test_the_tolerance_touches_no_other_reward_term(legacy_env, cone_env, reset_state):
    """The cone is subtracted from one term. Two envs identical but for the
    tolerance must agree on everything else, at the same state."""
    state = tilted(legacy_env, reset_state, 30.0)
    info = dict(state.info)
    info["command"] = COMMAND
    n_feet = legacy_env._n_feet
    args = (
        state.data,
        info,
        jp.zeros(legacy_env.action_size),
        jp.zeros(n_feet, dtype=bool),
        jp.zeros(n_feet, dtype=bool),
    )
    legacy, _ = legacy_env._compute_rewards(*args)
    cone, _ = cone_env._compute_rewards(*args)

    assert list(cone) == list(legacy)
    for key in legacy:
        if key == "orientation":
            continue
        assert float(cone[key]) == float(legacy[key]), key
    assert float(cone["orientation"]) < float(legacy["orientation"])
