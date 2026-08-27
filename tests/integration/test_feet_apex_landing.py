"""feet_apex and feet_landing wired into the env.

The two terms' math is unit-tested model-free in tests/unit/test_rewards.py.
What is tested here is the wiring: where the swing-apex tracker sits in
`step()` relative to the contact bookkeeping and the reward call, which feet
it pays and when, and the masks the env applies on top of the term math.

Most of the file reads one scripted swing (the `swing` fixture): the whole
robot is lifted 3.5 cm off the floor with 0.5 m/s of upward base velocity and
stepped with a zero action until both feet are planted again. That is a real
ballistic arc through MJX -- clearance rises for three control steps, peaks,
falls for three more, and the two feet touch down one step apart -- so the
tracker has to hold a peak it stopped seeing, pay the foot that lands, and
leave the other foot's swing running. Nothing about the numbers is
hardcoded: every expectation is recomputed from the recorded frames, and the
shape of the arc is asserted separately in
`test_the_scripted_swing_has_the_shape_the_other_tests_read`.

`apex_target`, `glide_height` and the scales are read inside
`_compute_rewards`, so the flag-flipping style of
tests/integration/test_tracking_kernels.py works here too: one model for the
whole file.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass

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

# Enough command to clear the `moving` mask (_cmd_speed 0.8 > 0.05) and slow
# enough that the tracking kernels stay mid-range, so a gate test can tell a
# gated term from an ungated one.
MOVING_CMD = jp.array([0.8, 0.0, 0.0])
# Fast enough that a robot at rest tracks essentially none of it.
UNTRACKED_CMD = jp.array([1.5, 0.0, 0.0])

# The scripted swing: lift, upward base velocity, and how many control steps
# to record. 11 steps reaches four steps of two-footed stance after the
# second touchdown.
LIFT = 0.035
RISE = 0.5
STEPS = 11


@pytest.fixture(scope="module")
def env():
    cfg = default_config()
    cfg.episode_length = 50  # fast tests, not a training config
    return Joystick(ROBOT_DIR, PRESET, cfg)


@contextlib.contextmanager
def reward_flags(env, **flags):
    """Set reward config keys for the body of the with-block and restore them
    after. Every key here is read inside `_compute_rewards`, so a live env
    picks the new value up on the next call."""
    r = env._config.reward
    saved = {k: r[k] for k in flags}
    r.update(flags)
    try:
        yield
    finally:
        r.update(saved)


@dataclass(frozen=True)
class Frame:
    """One recorded control step of the scripted swing."""

    state: object
    contact: np.ndarray  # per-foot, from the data this step ended on
    clearance: np.ndarray
    apex: np.ndarray  # info["swing_apex"] as the step left it
    feet_apex: float  # metrics["reward/feet_apex"], unscaled term value
    feet_landing: float


@pytest.fixture(scope="module")
def swing(env):
    """The scripted swing, as a list of `Frame`s one control step apart.

    The start state is the reset state with the base raised by `LIFT` and
    given `RISE` of upward velocity, relabelled as a robot mid-swing:
    airborne (`last_contact` false) with a fresh air-time counter. The apex
    tracker is left exactly as `reset()` seeded it.
    """
    state = env.reset(jax.random.PRNGKey(0))
    qpos = state.data.qpos.at[env._base_qadr + 2].add(LIFT)
    qvel = state.data.qvel.at[env._base_vadr + 2].set(RISE)
    data = mjx.forward(env._mjx_model, state.data.replace(qpos=qpos, qvel=qvel))

    info = dict(state.info)
    info["command"] = MOVING_CMD
    info["last_contact"] = jp.zeros(env._n_feet, dtype=bool)
    info["feet_air_time"] = jp.zeros(env._n_feet)
    state = state.replace(data=data, info=info)

    frames = []
    for _ in range(STEPS):
        state = env.step(state, jp.zeros(env.action_size))
        frames.append(
            Frame(
                state=state,
                contact=np.asarray(env._foot_contact(state.data)),
                clearance=np.asarray(env._foot_clearance(state.data)),
                apex=np.asarray(state.info["swing_apex"]),
                feet_apex=float(state.metrics["reward/feet_apex"]),
                feet_landing=float(state.metrics["reward/feet_landing"]),
            )
        )
    return frames


def airborne(frames):
    """Indices of the frames that ended with every foot off the floor."""
    return [i for i, f in enumerate(frames) if not f.contact.any()]


def landed(frames, i):
    """Per-foot mask of the feet whose swing ended on frame `i` -- in
    contact now, off the floor on the frame before. That is exactly
    `step()`'s `first_contact`: a foot already in contact had its air time
    zeroed last step, so it cannot fire again."""
    was = frames[i - 1].contact if i else np.zeros_like(frames[i].contact)
    return frames[i].contact & ~was


def prev_apex(frames, i):
    return frames[i - 1].apex if i else np.zeros_like(frames[i].apex)


# -- defaults: the mechanism is off and costs nothing ----------------------


def test_the_new_terms_are_off_by_default(env):
    r = env._config.reward
    assert r.scales.feet_apex == 0.0
    assert r.scales.feet_landing == 0.0


def test_the_new_tuning_keys_carry_their_starting_values(env):
    """0.05 m of apex and a 0.03 m glide band are untuned starting values
    for a 0.21 m four-bar quadruped leg. Re-derive both for asimov's leg."""
    r = env._config.reward
    assert r.apex_target == 0.05
    assert r.glide_height == 0.03


def test_reset_seeds_the_tracker_at_zero_whatever_the_scales_say(env):
    """The tracker is unconditional state, like feet_air_time: it exists at
    weight 0 too, so the info pytree does not change shape with the config."""
    state = env.reset(jax.random.PRNGKey(0))
    assert "swing_apex" in state.info
    assert state.info["swing_apex"].shape == (env._n_feet,)
    assert np.asarray(state.info["swing_apex"]) == pytest.approx(0.0)


def test_the_new_terms_are_appended_at_the_end_of_the_reward_dict(env):
    """The scaled sum adds the terms in dict order, so a new key inserted
    mid-dict would shift every later float addition and break the pre-port
    golden. New keys go at the end."""
    state = env.reset(jax.random.PRNGKey(0))
    n_feet = env._n_feet
    rewards, _fall = env._compute_rewards(
        state.data,
        dict(state.info),
        jp.zeros(env.action_size),
        jp.zeros(n_feet, dtype=bool),
        jp.zeros(n_feet, dtype=bool),
    )
    assert list(rewards)[-11:] == [
        "feet_apex",
        "feet_landing",
        "pose_l1",
        "joint_pos_limits",
        "joint_vel",
        "joint_acc",
        "upward",
        "feet_distance",
        "knee_distance",
        "feet_contact_without_cmd",
        "feet_air_time_biped",
    ]


# -- the scripted swing ----------------------------------------------------


def test_the_scripted_swing_has_the_shape_the_other_tests_read(swing):
    """Guards the fixture. The arc must be: several airborne frames whose
    peak clearance is strictly inside the airborne stretch (so holding the
    peak is a real requirement), then one foot landing, then the other,
    then two-footed stance."""
    air = airborne(swing)
    assert air == list(range(len(air))), "the swing must start airborne and land once"
    assert len(air) >= 4

    peak = max(air, key=lambda i: swing[i].clearance.max())
    assert peak < air[-1], "the peak must be behind the tracker by the last airborne frame"

    landings = [i for i in range(len(swing)) if landed(swing, i).any()]
    assert len(landings) == 2, "the two feet must touch down on separate frames"
    assert [landed(swing, i).sum() for i in landings] == [1, 1]
    assert all(f.contact.all() for f in swing[landings[-1] :])


def test_the_tracker_is_the_running_maximum_of_clearance_while_airborne(env, swing):
    running = np.zeros(env._n_feet)
    for i in airborne(swing):
        running = np.maximum(running, swing[i].clearance)
        assert swing[i].apex == pytest.approx(running), f"frame {i}"


def test_the_tracker_holds_the_peak_after_the_foot_starts_descending(swing):
    """The point of tracking an apex at all: on the way down the tracker
    reports the peak, not the clearance it can currently see."""
    last = airborne(swing)[-1]
    assert (swing[last].apex > swing[last].clearance + 0.01).all()


def test_the_tracker_pays_the_completed_swings_apex_at_touchdown(env, swing):
    """The payout is the apex as the step BEFORE touchdown left it: at first
    contact `contact_filt` is already true, so this step's update is skipped
    and the term reads the swing that just finished."""
    paid = False
    for i in range(len(swing)):
        expected = float(
            np.sum(np.clip(prev_apex(swing, i) / env._config.reward.apex_target, 0.0, 1.0) * landed(swing, i))
        )
        assert swing[i].feet_apex == pytest.approx(expected, rel=1e-5), f"frame {i}"
        paid = paid or expected > 0.0
    assert paid, "the swing never paid -- the fixture is not exercising the term"


def test_only_the_foot_that_landed_is_paid(env, swing):
    """The two feet land a step apart, and the first landing must pay that
    foot's apex alone -- not the other foot's still-running swing."""
    i = next(i for i in range(len(swing)) if landed(swing, i).any())
    one, other = np.flatnonzero(landed(swing, i))[0], np.flatnonzero(~landed(swing, i))[0]
    target = env._config.reward.apex_target

    assert swing[i].feet_apex == pytest.approx(
        float(np.clip(prev_apex(swing, i)[one] / target, 0.0, 1.0)), rel=1e-5
    )
    # Both feet swung about as high, so paying both would roughly double it.
    assert prev_apex(swing, i)[other] > 0.5 * prev_apex(swing, i)[one]


def test_the_tracker_zeroes_on_the_foot_that_landed_and_leaves_the_other(swing):
    i = next(i for i in range(len(swing)) if landed(swing, i).any())
    just_landed = landed(swing, i)
    assert swing[i].apex[just_landed] == pytest.approx(0.0)
    assert (swing[i].apex[~just_landed] > 0.0).all()


def test_a_foot_in_continued_contact_keeps_a_zero_tracker_and_pays_nothing(swing):
    """After the second touchdown both feet stay planted: nothing accrues
    and the swing is not paid twice."""
    stance = [i for i in range(len(swing)) if swing[i].contact.all() and not landed(swing, i).any()]
    assert len(stance) >= 3, "the fixture must reach two-footed stance"
    for i in stance:
        assert swing[i].apex == pytest.approx(0.0), f"frame {i}"
        assert swing[i].feet_apex == pytest.approx(0.0), f"frame {i}"


# -- feet_landing, on the same swing ---------------------------------------


def test_feet_landing_is_zero_while_the_feet_are_above_the_glide_band(env, swing):
    glide = env._config.reward.glide_height
    above = [i for i in airborne(swing) if (swing[i].clearance > glide).all()]
    assert len(above) >= 3, "the fixture must spend time above the glide band"
    for i in above:
        assert swing[i].feet_landing == pytest.approx(0.0), f"frame {i}"


def test_feet_landing_prices_the_descent_before_contact(env, swing):
    """The reason the penalty is not measured at contact: by the step the
    contact is seen the solver has already absorbed the impact. Here the
    descending feet are billed while they are still in the air."""
    glide = env._config.reward.glide_height
    approach = [
        i for i in airborne(swing) if (swing[i].clearance < glide).any() and (swing[i].clearance > 0).all()
    ]
    assert approach, "the fixture must enter the glide band before touching down"
    for i in approach:
        data = swing[i].state.data
        expected = float(
            terms.feet_landing(env._foot_linvel(data)[:, 2], env._foot_clearance(data), glide)
        )
        assert swing[i].feet_landing == pytest.approx(expected, rel=1e-5), f"frame {i}"
    assert swing[approach[-1]].feet_landing > 0.0


# -- the masks the env puts on top of the term math ------------------------


def descending_state(env, swing):
    """The last airborne frame of the swing: feet inside the glide band and
    moving down, so both new terms have something to score."""
    i = airborne(swing)[-1]
    return swing[i].state


def rewards_for(env, state, command, first_contact=True):
    info = dict(state.info)
    info["command"] = jp.asarray(command, dtype=float)
    n_feet = env._n_feet
    rewards, _fall = env._compute_rewards(
        state.data,
        info,
        jp.zeros(env.action_size),
        jp.full(n_feet, first_contact, dtype=bool),
        jp.zeros(n_feet, dtype=bool),
    )
    return {k: float(v) for k, v in rewards.items()}


def test_both_terms_are_masked_off_by_a_zero_command(env, swing):
    """`moving` gates both: a robot told to stand is not shaped into a gait,
    and it is not billed for putting its feet down either."""
    state = descending_state(env, swing)
    standing = rewards_for(env, state, (0.0, 0.0, 0.0))
    walking = rewards_for(env, state, MOVING_CMD)

    assert standing["feet_apex"] == pytest.approx(0.0)
    assert standing["feet_landing"] == pytest.approx(0.0)
    assert walking["feet_apex"] > 0.0
    assert walking["feet_landing"] > 0.0


def test_the_shaping_gate_takes_feet_apex_and_leaves_feet_landing(env, swing):
    """feet_apex joins the 1.4 gated set: lifting a leg while the command
    goes unserved pays nothing. feet_landing is a PENALTY and stays ungated
    -- gating it on the tracking kernel would relax it exactly when tracking
    is failing, which is when feet are being slammed into the floor."""
    state = descending_state(env, swing)
    ungated = rewards_for(env, state, UNTRACKED_CMD)
    with reward_flags(env, shaping_tracking_gate=True):
        gated = rewards_for(env, state, UNTRACKED_CMD)

    assert ungated["feet_apex"] > 0.5
    assert gated["feet_apex"] < 1e-3
    assert gated["feet_landing"] == ungated["feet_landing"]
    assert gated["feet_landing"] > 0.0
