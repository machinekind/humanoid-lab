"""Each promoted task.env knob on, in the wrapped training layout.

The env is built through the registry and wrapped by make_wrap_env_fn, as
the trainer does. Outputs stay finite and the knob's state behaves.
"""

import jax
import jax.numpy as jp
import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from humanoid_lab import paths
from humanoid_lab.envs import wrappers
from humanoid_lab.envs.wrappers import make_wrap_env_fn
from humanoid_lab.registry import make_env

ROBOT = paths.ROBOTS_DIR / "roboto_origin"
LIN, YAW = 0.5, 1.0
TOL = 1e-6


def _assert_finite(tree, where):
    for path, leaf in jax.tree_util.tree_leaves_with_path(tree):
        a = np.asarray(leaf)
        if np.issubdtype(a.dtype, np.floating):
            assert np.isfinite(a).all(), f"{where}{jax.tree_util.keystr(path)} is not finite"


def test_the_command_ramp_slews_from_zero_and_stays_finite():
    env = make_env(
        "joystick",
        ROBOT,
        "deploy_pd",
        {
            "sim": {"backend": "jax"},
            "episode_length": 20,
            "command": {"ramp": {"lin": LIN, "yaw": YAW}, "resample_steps": 10},
        },
    )
    assert float(env.dt) == 0.02
    w = make_wrap_env_fn(env._config)(env, episode_length=20, action_repeat=1)
    state = jax.jit(w.reset)(jax.random.split(jax.random.PRNGKey(0), 4))
    assert "cmd_target" in state.info
    np.testing.assert_array_equal(np.asarray(state.info["command"]), np.zeros((4, 3), np.float32))
    assert np.abs(np.asarray(state.info["cmd_target"])).max() > 0.0

    step = jax.jit(w.step)
    key = jax.random.PRNGKey(1)
    limit = np.array([LIN, LIN, YAW]) * 0.02 + TOL
    prev = np.asarray(state.info["command"])
    moved = False
    for _ in range(30):
        key, k = jax.random.split(key)
        action = jax.random.uniform(k, (4, env.action_size), minval=-1.0, maxval=1.0)
        state = step(state, action)
        cmd = np.asarray(state.info["command"])
        assert np.all(np.abs(cmd - prev) <= limit), np.abs(cmd - prev).max(axis=0)
        moved = moved or bool(np.any(cmd != prev))
        prev = cmd
        _assert_finite(state.reward, "reward")
        _assert_finite(state.obs, "obs")
        _assert_finite(state.metrics, "metrics")
        _assert_finite({k: v for k, v in state.info.items() if k != "rng"}, "info")
    assert moved


def test_a_default_env_has_no_ramp_state():
    env = make_env("joystick", ROBOT, "deploy_pd", {"sim": {"backend": "jax"}})
    state = jax.jit(env.reset)(jax.random.PRNGKey(0))
    assert "cmd_target" not in state.info
    assert "land_frac" not in state.info
    assert jp.shape(state.info["command"]) == (3,)
    for t in LANDING_TERMS:
        assert f"reward/{t}" not in state.metrics, t


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"command": {"ramp": {"lin": 0.5, "yaw": 0.0}}}, "command.ramp.lin"),
        ({"reward": {"scales": {"foot_reach": 1.0}}}, "reward.foot_reach.target_a"),
        ({"gait": {"frozen_phase_obs": "offsets"}}, "gait.frozen_phase_obs"),
    ],
    ids=["ramp_one_rate", "foot_reach_no_targets", "unknown_frozen_phase_obs"],
)
def test_joystick_construction_refuses_a_bad_promoted_knob(overrides, match):
    """The refusals run in Joystick.__init__. Construction only, no jit."""
    with pytest.raises(ValueError, match=match):
        make_env("joystick", ROBOT, "deploy_pd", {"sim": {"backend": "jax"}, **overrides})


def test_the_root_body_is_the_floating_base():
    env = make_env("joystick", ROBOT, "deploy_pd", {"sim": {"backend": "jax"}})
    assert env._root_body == env._mj_model.body(env.robot_spec.termination_bodies[0]).id


LANDING_TERMS = ("step_length", "foot_reach", "knee_extend", "forward_lean")
# The parameters of the parity suite's _env(), with pair_min on for the
# three landing terms. roboto's home knee is 0.3 rad, and flexion is
# positive.
LANDING_REWARD = {
    "scales": {"step_length": 1.0, "foot_reach": 1.0, "knee_extend": 0.5, "forward_lean": 0.5},
    "step_length": {"pair_min": True},
    "foot_reach": {"target_a": 0.05, "target_b": 0.15, "pair_min": True},
    "knee_extend": {"flex_lo": 0.1, "flex_hi": 0.6, "td_mult": 0.5, "pair_min": True},
    "forward_lean": {"max_deg": 5.0, "vx_max": 1.0, "sigma": 0.05},
}


def test_the_landing_terms_pay_finite_rewards_and_land_frac_restarts_on_respawn():
    env = make_env(
        "joystick",
        ROBOT,
        "deploy_pd",
        {
            "sim": {"backend": "jax"},
            "episode_length": 20,
            "reward": LANDING_REWARD,
        },
    )
    n_feet = env._n_feet
    w = make_wrap_env_fn(env._config)(env, episode_length=20, action_repeat=1)
    state = jax.jit(w.reset)(jax.random.split(jax.random.PRNGKey(0), 4))
    assert jp.shape(state.info["land_frac"]) == (4, n_feet, 3)
    for t in LANDING_TERMS:
        assert f"reward/{t}" in state.metrics, t

    step = jax.jit(w.step)
    key = jax.random.PRNGKey(1)
    respawned = 0
    for _ in range(25):
        key, k = jax.random.split(key)
        action = jax.random.uniform(k, (4, env.action_size), minval=-1.0, maxval=1.0)
        state = step(state, action)
        land = np.asarray(state.info["land_frac"])
        assert land.shape == (4, n_feet, 3)
        done = np.asarray(state.done) > 0.0
        if done.any():
            respawned += int(done.sum())
            np.testing.assert_array_equal(land[done], np.zeros_like(land[done]))
        for t in LANDING_TERMS:
            assert np.isfinite(np.asarray(state.metrics[f"reward/{t}"])).all(), t
        _assert_finite(state.reward, "reward")
        _assert_finite(state.obs, "obs")
        _assert_finite(state.metrics, "metrics")
        _assert_finite({k: v for k, v in state.info.items() if k != "rng"}, "info")
    # The truncation at step 20 respawns every world at least once.
    assert respawned >= 4


def _roll_wrapped(env, steps, n=4, episode_length=20):
    """Reset and step the wrapped env with random actions. Every reward,
    obs, metric and info leaf must stay finite. Returns the wrapped env and
    the last state."""
    w = make_wrap_env_fn(env._config)(env, episode_length=episode_length, action_repeat=1)
    state = jax.jit(w.reset)(jax.random.split(jax.random.PRNGKey(0), n))
    step = jax.jit(w.step)
    key = jax.random.PRNGKey(1)
    for i in range(steps):
        key, k = jax.random.split(key)
        action = jax.random.uniform(k, (n, env.action_size), minval=-1.0, maxval=1.0)
        state = step(state, action)
        _assert_finite(state.reward, f"step {i} reward")
        _assert_finite(state.obs, f"step {i} obs")
        _assert_finite(state.metrics, f"step {i} metrics")
        _assert_finite({k: v for k, v in state.info.items() if k != "rng"}, f"step {i} info")
    return w, state


def _layers(w):
    """The wrapper types from the outside in."""
    out = []
    while hasattr(w, "env"):
        out.append(type(w))
        w = w.env
    return out


def test_every_promoted_knob_on_stays_finite():
    env = make_env(
        "joystick",
        ROBOT,
        "deploy_pd",
        {
            "sim": {"backend": "jax"},
            "episode_length": 20,
            "command": {"ramp": {"lin": LIN, "yaw": YAW}, "straight_prob": 0.3},
            "gait": {"lift": {"v_full": 0.5, "min_frac": 0.25}},
            "reward": LANDING_REWARD,
        },
    )
    _, state = _roll_wrapped(env, 30)
    assert "cmd_target" in state.info
    assert "land_frac" in state.info
    for t in LANDING_TERMS:
        assert f"reward/{t}" in state.metrics, t


def test_the_v40_preset_builds_and_steps():
    with initialize_config_dir(version_base=None, config_dir=str(paths.CONFIGS_DIR)):
        cfg = compose(config_name="config", overrides=["experiment=roboto_gait_v40"])
    env_overrides = OmegaConf.to_container(cfg.task.env, resolve=True)
    env_overrides.update({"sim": {"backend": "jax"}, "episode_length": 20})
    env = make_env(
        cfg.task.name,
        paths.REPO_ROOT / cfg.robot.dir,
        cfg.actuators.name,
        env_overrides,
        actuator_overrides=OmegaConf.to_container(cfg.actuators.overrides, resolve=True),
    )
    assert env._config.obs.state[-1] == "heading_err"
    w, state = _roll_wrapped(env, 20)
    layers = _layers(w)
    assert wrappers.HeadingReseedWrapper in layers
    assert wrappers.LandReseedWrapper in layers
    assert "yaw_ref" in state.info
    assert "land_frac" in state.info
    for t in LANDING_TERMS:
        assert f"reward/{t}" in state.metrics, t
