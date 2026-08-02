"""Pins the yaml config mirrors to the Python defaults that back them, so
the two sources cannot silently drift apart.

Env configs are composed yaml-over-code (registry.make_env builds the code
default_config, then merges the task yaml's env block onto it), so wherever
the yaml restates a code default for visibility there are two copies of the
value. These tests are what keeps the copies equal: retuning either source
alone fails here instead of shipping a yaml that silently disagrees with
the code path that bypasses Hydra (tests, check CLIs, direct make_env).
"""

import pytest
import yaml

from humanoid_lab import paths
from humanoid_lab.dr import randomize
from humanoid_lab.envs.joystick import default_config

DR_YAML = paths.CONFIGS_DIR / "dr" / "default.yaml"
JOYSTICK_YAML = paths.CONFIGS_DIR / "task" / "joystick.yaml"

# joystick.yaml env blocks that restate the code defaults verbatim.
# `command` is deliberately absent: the yaml carries only the vx/vy/wz
# envelope and leaves the sampler knobs (resample_steps, pure-draw probs,
# speed bands) to the code -- the envelope subset is pinned separately
# below.
MIRRORED_ENV_BLOCKS = ("reset_keyframe", "reset_noise", "obs", "obs_noise", "fall")


def _listify(value):
    """ConfigDict.to_dict() holds tuples where yaml holds lists."""
    if isinstance(value, dict):
        return {k: _listify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_listify(v) for v in value]
    return value


def _env_yaml():
    return yaml.safe_load(JOYSTICK_YAML.read_text())["env"]


def test_dr_yaml_matches_code_defaults():
    cfg = yaml.safe_load(DR_YAML.read_text())
    assert _listify(cfg) == _listify(randomize._DEFAULT_DR)


@pytest.mark.parametrize("block", MIRRORED_ENV_BLOCKS)
def test_joystick_yaml_mirrors_match_env_defaults(block):
    code = default_config()[block]
    code = code.to_dict() if hasattr(code, "to_dict") else code
    assert _listify(_env_yaml()[block]) == _listify(code)


def test_joystick_yaml_command_envelope_matches_env_defaults():
    """Every command key the yaml does carry must equal the code default,
    so an envelope retune edits both sources or fails here."""
    code = default_config().command.to_dict()
    for key, value in _env_yaml()["command"].items():
        assert _listify(value) == _listify(code[key]), key
