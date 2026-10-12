"""configs/torch_warp.yaml and the README's override table.

Model-free: Hydra composes only. The README is read as text, so this file
needs no torch.
"""

import re

import pytest
from hydra import compose, initialize_config_dir
from hydra.errors import ConfigCompositionException
from omegaconf import OmegaConf

from humanoid_lab import paths, registry
from humanoid_lab.envs import joystick

README = paths.REPO_ROOT / "src" / "humanoid_lab" / "torch_warp" / "README.md"

WTRAIN_FLAGS = (
    "--src",
    "--name",
    "--steps",
    "--num-envs",
    "--seed",
    "--ckpt-minutes",
    "--lr",
    "--sym-coef",
    "--critic-warmup-iters",
    "--max-minutes",
    "--src-ckpt",
    "--resume",
    "--reset-keyframe",
    "--act-scale",
    "--warp-reward",
    "--pose-l1",
    "--scale",
    "--warp-cmd-ramp",
    "--cmd-straight",
    "--warp-lift",
    "--env",
)


def _compose(config_name, overrides=()):
    with initialize_config_dir(version_base=None, config_dir=str(paths.CONFIGS_DIR)):
        return compose(config_name=config_name, overrides=list(overrides))


def _flag_rows():
    """{flag: override cell} from the README's wtrain flags table."""
    text = README.read_text(encoding="utf-8")
    section = text.split("## wtrain flags", 1)[1]
    rows = {}
    for line in section.splitlines():
        m = re.match(r"^\|\s*`(--[a-z0-9-]+)`\s*\|(.*)\|\s*$", line)
        if m:
            rows[m.group(1)] = m.group(2).strip()
    return rows


def _readme_overrides():
    """Every backticked override string in a cell that is not 'dropped:'."""
    found = []
    for flag, cell in _flag_rows().items():
        if cell.startswith("dropped:"):
            continue
        for i, s in enumerate(re.findall(r"`([^`]+)`", cell)):
            found.append(pytest.param(s, id=f"{flag}-{i}"))
    return found


def test_torch_warp_composes_alone_and_with_an_experiment():
    cfg = _compose("torch_warp")
    assert cfg.torch_warp.device == "cuda:0"
    assert cfg.torch_warp.resume is False
    assert cfg.bench.iters == 3
    cfg = _compose("torch_warp", ["experiment=roboto_gait_v40"])
    assert cfg.ppo.sym_coef > 0
    assert "torch_warp" in cfg and "bench" in cfg


def test_config_rejects_the_torch_only_blocks():
    for override in ("torch_warp.device=cpu", "bench.iters=1"):
        with pytest.raises(ConfigCompositionException):
            _compose("config", [override])


def test_sym_coef_needs_no_plus_on_either_config():
    for name in ("config", "torch_warp"):
        assert _compose(name, ["ppo.sym_coef=0.5"]).ppo.sym_coef == 0.5


@pytest.mark.parametrize("override", _readme_overrides())
def test_every_readme_override_composes_and_lands_on_a_real_key(override):
    assert "<" not in override and "..." not in override
    cfg = _compose("torch_warp", override.split())
    registry._apply_overrides(joystick.default_config(), OmegaConf.to_container(cfg.task.env, resolve=True))


def test_the_readme_lists_every_wtrain_flag():
    rows = _flag_rows()
    assert len(WTRAIN_FLAGS) == 21
    assert sorted(rows) == sorted(WTRAIN_FLAGS)
    for flag, cell in rows.items():
        assert cell.startswith("dropped:") or re.search(r"`[^`]+=[^`]*`", cell), flag


def test_the_num_timesteps_example_composes():
    cfg = _compose("torch_warp", ["++ppo.num_timesteps=3e8", "run_name=roboto_walk_v1"])
    assert cfg.ppo.num_timesteps == 3e8
    assert cfg.run_name == "roboto_walk_v1"
