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
from humanoid_lab.ppo_config import compose_ppo_params

README = paths.REPO_ROOT / "src" / "humanoid_lab" / "torch_warp" / "README.md"


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


def test_torch_warp_is_config_plus_its_own_blocks_alone_and_with_an_experiment():
    for overrides in ([], ["experiment=roboto_gait_v40"]):
        torch_cfg = OmegaConf.to_container(_compose("torch_warp", overrides), resolve=True)
        brax_cfg = OmegaConf.to_container(_compose("config", overrides), resolve=True)
        assert set(torch_cfg) - set(brax_cfg) == {"torch_warp", "bench"}
        assert {k: torch_cfg[k] for k in brax_cfg} == brax_cfg
        if overrides:
            # The experiment lands: its sym_coef replaces the default 0.
            assert torch_cfg["ppo"]["sym_coef"] > 0


def test_config_rejects_the_torch_only_blocks():
    for override in ("torch_warp.device=cpu", "bench.iters=1"):
        with pytest.raises(ConfigCompositionException):
            _compose("config", [override])


def test_sym_coef_needs_no_plus_on_either_config():
    for name in ("config", "torch_warp"):
        cfg = _compose(name, ["ppo.sym_coef=0.5"])
        assert cfg.ppo.sym_coef == 0.5
        assert compose_ppo_params(cfg).sym_coef == 0.5


@pytest.mark.parametrize("override", _readme_overrides())
def test_every_readme_override_composes_and_lands_on_a_real_key(override):
    assert "<" not in override and "..." not in override
    cfg = _compose("torch_warp", override.split())
    registry._apply_overrides(joystick.default_config(), OmegaConf.to_container(cfg.task.env, resolve=True))
