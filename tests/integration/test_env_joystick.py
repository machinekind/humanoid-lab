"""The joystick task constructs on robots/asimov_v1 and runs a jitted
reset/step without NaN. CPU-fast on purpose (tiny episode length, no
training loop) -- this is the per-task compile/NaN smoke test, not a
training run (that's `./run.sh smoke`).
"""

from __future__ import annotations

import jax
import jax.numpy as jp
import pytest
import yaml

from humanoid_lab import paths
from humanoid_lab.envs import base
from humanoid_lab.envs.joystick import Joystick, default_config

ROBOT_DIR = paths.ROBOTS_DIR / "asimov_v1"
PRESET = "sizing_ideal"

# Sim-only signals: must never appear on the actor's (deployable) obs list.
PRIVILEGED_ONLY = {"linvel", "height", "contacts", "actuator_force"}


@pytest.fixture(scope="module")
def env():
    cfg = default_config()
    cfg.episode_length = 50  # fast tests, not a training config
    return Joystick(ROBOT_DIR, PRESET, cfg)


def test_actor_obs_excludes_privileged_only_signals(env):
    names = env.actor_obs_names
    assert PRIVILEGED_ONLY.isdisjoint(names)
    assert list(names) == [
        "gyro",
        "gravity",
        "joint_pos",
        "joint_vel",
        "last_action",
        "command",
        "phase",
    ]


def test_reset_gives_finite_obs_with_documented_sizes(env):
    state = env.reset(jax.random.PRNGKey(0))
    state_obs = state.obs["state"]
    priv_obs = state.obs["privileged_state"]

    assert jp.all(jp.isfinite(state_obs))
    assert jp.all(jp.isfinite(priv_obs))

    nu = env.action_size
    n_feet = env._n_feet
    # gyro(3) + gravity(3) + joint_pos(nu) + joint_vel(nu) + last_action(nu)
    # + command(3) + phase(2*n_feet)
    expected_actor = 3 + 3 + nu + nu + nu + 3 + 2 * n_feet
    # privileged adds linvel(3) + height(1) + contacts(n_feet) +
    # actuator_force(nu)
    expected_privileged = expected_actor + 3 + 1 + n_feet + nu
    assert state_obs.shape == (expected_actor,)
    assert priv_obs.shape == (expected_privileged,)


def test_step_is_finite_and_metrics_keyset_matches_reset(env):
    state = env.reset(jax.random.PRNGKey(0))
    step_fn = jax.jit(env.step)
    action = jp.zeros(env.action_size)
    next_state = step_fn(state, action)

    # CRITICAL for brax's scan-carry: the metrics pytree structure must not
    # change shape between reset() and step().
    assert set(next_state.metrics.keys()) == set(state.metrics.keys())
    assert bool(jp.isfinite(next_state.reward))
    assert jp.all(jp.isfinite(next_state.obs["state"]))
    assert jp.all(jp.isfinite(next_state.obs["privileged_state"]))
    assert jp.all(jp.isfinite(next_state.data.qpos))
    assert jp.all(jp.isfinite(next_state.data.qvel))


def test_termination_triggers_when_base_drops_below_min_height(env):
    state = env.reset(jax.random.PRNGKey(0))
    low_qpos = state.data.qpos.at[env._base_qadr + 2].set(0.1)
    state = state.replace(data=state.data.replace(qpos=low_qpos))

    action = jp.zeros(env.action_size)
    next_state = env.step(state, action)

    assert bool(next_state.done)
    assert bool(jp.isfinite(next_state.reward))


def test_joystick_yaml_obs_and_fall_match_env_defaults():
    """Pin configs/task/joystick.yaml's env: mirror to the code defaults,
    so the two cannot silently drift apart.
    """
    task_cfg = yaml.safe_load((paths.CONFIGS_DIR / "task" / "joystick.yaml").read_text())
    default = default_config()

    assert task_cfg["env"]["reset_keyframe"] == default.reset_keyframe
    assert task_cfg["env"]["reset_noise"] == pytest.approx(default.reset_noise)
    assert list(task_cfg["env"]["obs"]["state"]) == list(default.obs.state)
    assert list(task_cfg["env"]["obs"]["privileged"]) == list(default.obs.privileged)
    assert task_cfg["env"]["fall"] == default.fall.to_dict()
    assert task_cfg["env"]["obs_noise"] == default.obs_noise.to_dict()
    assert task_cfg["env"]["command"]["vx"] == list(default.command.vx)
    assert task_cfg["env"]["command"]["vy"] == list(default.command.vy)
    assert task_cfg["env"]["command"]["wz"] == list(default.command.wz)


def test_armed_distance_term_refuses_a_robot_without_the_pair():
    """toy_robot has no ankle_roll group. With the term off it constructs
    (conformance covers that); armed, it must refuse."""
    cfg = default_config()
    cfg.episode_length = 50
    cfg.reset_keyframe = "standing"  # toy_robot has no "home"
    cfg.reward.scales.feet_distance = 0.1
    toy = paths.REPO_ROOT / "tests" / "data" / "toy_robot"
    with pytest.raises(ValueError, match="ankle_roll"):
        Joystick(toy, "pd_test", cfg)


def test_the_flat_hooks_return_the_legacy_values(env):
    """The hooks a terrain task overrides are identities here: each returns
    exactly what the inlined code computed before it became a hook."""
    state = env.reset(jax.random.PRNGKey(0))
    data = state.data
    b = env._base_qadr

    assert jp.array_equal(env._base_height(data), data.qpos[b + 2])

    for z, gz in ((0.3, -1.0), (0.7, -1.0), (0.7, 0.0)):
        posed = data.replace(qpos=data.qpos.at[b + 2].set(z))
        gravity = jp.array([0.0, 0.0, gz])
        legacy = (posed.qpos[b + 2] < env._config.fall.min_height) | (
            gravity[2] > env._config.fall.max_tilt_gz
        )
        assert bool(env._fall(posed, gravity)) == bool(legacy)

    key = jax.random.PRNGKey(3)
    assert jp.array_equal(env._next_command(key, state.info), env._sample_command(key))
    assert jp.array_equal(
        env._draw_command(key, env._config.command), env._sample_command(key)
    )

    npg = env._config.no_progress
    assert env._no_progress_params(state.info) == (npg.grace_sec, npg.p_max)

    qpos = env._reset_qpos
    placed, extra = env._place_base(key, qpos)
    assert placed is qpos and extra == {}

    assert env._settle_model() is env._mj_model


def test_the_ccd_budget_reaches_make_data(monkeypatch):
    """sim.naccdmax_per_env reaches the data builder as its last argument.
    The default config passes None, so the flat path's data is unchanged."""
    seen = []
    real = base.make_data_fn

    def record(*args):
        seen.append(args)
        return real(*args)

    monkeypatch.setattr(base, "make_data_fn", record)
    for naccd in (None, 24):
        cfg = default_config()
        cfg.episode_length = 50
        cfg.sim.naccdmax_per_env = naccd
        Joystick(ROBOT_DIR, PRESET, cfg)
    assert [args[-1] for args in seen] == [None, 24]
