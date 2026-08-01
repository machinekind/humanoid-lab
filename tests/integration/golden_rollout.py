"""The rollout recorded in tests/data/golden/, shared by the generator and the test.

The gate is that with every optional mechanism off, an env rollout is
bit-exact against the recorded baseline. This module defines that rollout
once so `generate_golden.py` and `test_golden_baseline.py` cannot drift
apart: if the
recording procedure changes, both change together and the npz files must be
regenerated deliberately.

Not a `test_*.py` file, so pytest never collects it. Both importers get
tests/integration on sys.path for free -- pytest prepends the test file's
directory, and Python prepends a script's own directory.

260 steps is deliberate. It crosses the push event at step 199
(push.interval_steps=200) and the command resample at step 250
(command.resample_steps=250), so both of step()'s RNG consumers are inside
the recording. A shorter rollout would not notice a change to either.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import jax
import numpy as np

from humanoid_lab import paths
from humanoid_lab.envs.joystick import Joystick, default_config

GOLDEN_DIR = Path(__file__).resolve().parent.parent / "data" / "golden"

STEPS = 260
RESET_SEED = 2026
ACTION_SEED = 7


@dataclass(frozen=True)
class GoldenCase:
    """One (robot, actuator preset) pair with its recording."""

    name: str
    robot_dir: Path
    preset: str
    # Stock default_config() values that the robot cannot satisfy. The toy
    # fixture's only keyframe is "standing"; default_config() asks for
    # "home", which robots/asimov_v1 has and tests/data/toy_robot does not.
    # Nothing else is overridden: the whole point is to record stock behavior.
    config_overrides: dict = field(default_factory=dict)

    @property
    def path(self) -> Path:
        return GOLDEN_DIR / f"{self.name}.npz"


CASES = (
    GoldenCase(
        name="toy_robot__ideal_test",
        robot_dir=paths.REPO_ROOT / "tests" / "data" / "toy_robot",
        preset="ideal_test",
        config_overrides={"reset_keyframe": "standing"},
    ),
    GoldenCase(
        name="toy_robot__pd_test",
        robot_dir=paths.REPO_ROOT / "tests" / "data" / "toy_robot",
        preset="pd_test",
        config_overrides={"reset_keyframe": "standing"},
    ),
    GoldenCase(
        name="asimov_v1__sizing_ideal",
        robot_dir=paths.ROBOTS_DIR / "asimov_v1",
        preset="sizing_ideal",
    ),
    GoldenCase(
        name="asimov_v1__deploy_pd",
        robot_dir=paths.ROBOTS_DIR / "asimov_v1",
        preset="deploy_pd",
    ),
)

CASE_IDS = tuple(c.name for c in CASES)


def rollout(case: GoldenCase) -> dict[str, np.ndarray]:
    """Run `case` and return the arrays that go into (or get compared to) its npz.

    Every value is converted to numpy eagerly so the caller holds no device
    buffers and np.testing can compare bit patterns directly.
    """
    config = default_config()
    for key, value in case.config_overrides.items():
        setattr(config, key, value)

    env = Joystick(case.robot_dir, case.preset, config)

    actions = jax.random.uniform(
        jax.random.PRNGKey(ACTION_SEED),
        (STEPS, env.action_size),
        minval=-1.0,
        maxval=1.0,
    )

    step = jax.jit(env.step)
    state = env.reset(jax.random.PRNGKey(RESET_SEED))

    metric_names = sorted(k for k in state.metrics if k.startswith("reward/"))
    rewards, dones, obs_state, obs_priv, metrics = [], [], [], [], []

    for t in range(STEPS):
        state = step(state, actions[t])
        rewards.append(np.asarray(state.reward))
        dones.append(np.asarray(state.done))
        obs_state.append(np.asarray(state.obs["state"]))
        obs_priv.append(np.asarray(state.obs["privileged_state"]))
        metrics.append(np.asarray([np.asarray(state.metrics[k]) for k in metric_names]))

    return {
        "reward": np.stack(rewards),
        "done": np.stack(dones),
        "obs_state": np.stack(obs_state),
        "obs_privileged_state": np.stack(obs_priv),
        "metric_names": np.array(metric_names),
        "metric_values": np.stack(metrics),
        "final_rng": np.asarray(jax.random.key_data(state.info["rng"])),
        "final_phase": np.asarray(state.info["phase"]),
        "final_command": np.asarray(state.info["command"]),
    }
