"""configs/experiment/novel_pure_cmd.yaml: yolo_v4 plus three pure command
draws. Pins the composed values, that every armed draw redraws inside
roboto_origin's command box, and the command mix the sampler produces from
them. Model-free: `_sample_command` and the `_draw_command` it delegates to
read only `self._config.command`, so they run against a stub holding the
composed config, no env built.
"""

from __future__ import annotations

import types

import jax
import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from humanoid_lab import paths
from humanoid_lab.envs.joystick import Joystick, check_pure_draw_ranges, default_config
from humanoid_lab.registry import _apply_overrides

EXPERIMENT = "novel_pure_cmd"


def _compose(extra=()):
    with initialize_config_dir(version_base=None, config_dir=str(paths.CONFIGS_DIR)):
        return compose(config_name="config", overrides=[f"experiment={EXPERIMENT}", *extra])


def _env_cfg(cfg):
    env_cfg = default_config()
    _apply_overrides(env_cfg, OmegaConf.to_container(cfg.task.env, resolve=True))
    return env_cfg


def test_novel_pure_cmd_arms_clean_draws_inside_the_roboto_box():
    cfg = _compose()
    command = cfg.task.env.command
    assert cfg.robot.name == "roboto_origin"
    assert cfg.actuators.name == "deploy_pd"
    assert command.pure_fast_prob == pytest.approx(0.15)
    assert command.pure_slow_prob == pytest.approx(0.10)
    assert command.pure_wz_prob == pytest.approx(0.10)
    assert "pure_back_prob" not in command
    assert "pure_vy_prob" not in command
    assert cfg.ppo.num_evals == 24
    assert cfg.ppo.num_timesteps == pytest.approx(1.0e9)

    env_cfg = _env_cfg(cfg)
    c = env_cfg.command
    assert tuple(c.vx) == (-0.6, 1.0)
    assert c.pure_back_prob == 0.0
    assert c.pure_vy_prob == 0.0
    assert c.vx[0] <= c.slow_vx[0] and c.slow_vx[1] <= c.vx[1]
    assert c.vx[0] <= c.fast_vx[0] and c.fast_vx[1] <= c.vx[1]
    check_pure_draw_ranges(c)


def test_novel_pure_cmd_refuses_the_back_draw_on_roboto():
    # back_vx (-0.8, -0.2) reaches below roboto_origin's vx floor of -0.6.
    cfg = _compose(["++task.env.command.pure_back_prob=0.1"])
    with pytest.raises(ValueError, match="back_vx"):
        check_pure_draw_ranges(_env_cfg(cfg).command)


def test_novel_pure_cmd_command_mix_matches_the_draw_order():
    """Draw order wz, vy, slow, fast, back, each overwriting the one before,
    then zero_prob overrides everything: fast 0.15*0.85, slow
    0.10*0.85*0.85, wz 0.10*0.90*0.85*0.85, stand 0.15, box the rest."""
    env_cfg = _env_cfg(_compose())
    c = env_cfg.command
    stub = types.SimpleNamespace(_config=env_cfg)
    stub._draw_command = lambda rng, cmd_cfg: Joystick._draw_command(stub, rng, cmd_cfg)
    n = 40_000
    keys = jax.random.split(jax.random.PRNGKey(0), n)
    cmds = np.asarray(jax.jit(jax.vmap(lambda k: Joystick._sample_command(stub, k)))(keys))
    vx, vy, wz = cmds.T

    stand = (vx == 0) & (vy == 0) & (wz == 0)
    straight = (vy == 0) & (wz == 0) & ~stand
    fast = straight & (vx >= c.fast_vx[0]) & (vx <= c.fast_vx[1])
    slow = straight & (vx >= c.slow_vx[0]) & (vx <= c.slow_vx[1])
    spin = (vx == 0) & (vy == 0) & (wz != 0)
    box = ~(stand | straight | spin)
    # Every straight draw is one of the two ranges; they do not overlap.
    assert np.all(fast | slow | ~straight)
    assert not np.any(fast & slow)
    # Every command sits inside the box.
    assert np.all((vx >= c.vx[0]) & (vx <= c.vx[1]))
    assert np.all((vy >= c.vy[0]) & (vy <= c.vy[1]))
    assert np.all((wz >= c.wz[0]) & (wz <= c.wz[1]))

    z = c.zero_prob
    expected = {
        "fast": 0.15 * (1 - z),
        "slow": 0.10 * (1 - 0.15) * (1 - z),
        "spin": 0.10 * (1 - 0.10) * (1 - 0.15) * (1 - z),
        "stand": z,
        "box": (1 - 0.10) * (1 - 0.10) * (1 - 0.15) * (1 - z),
    }
    observed = {
        "fast": fast.mean(),
        "slow": slow.mean(),
        "spin": spin.mean(),
        "stand": stand.mean(),
        "box": box.mean(),
    }
    # ~5 binomial standard deviations at n=40k for the largest share.
    for name, p in expected.items():
        assert observed[name] == pytest.approx(p, abs=0.012), (name, observed, expected)
