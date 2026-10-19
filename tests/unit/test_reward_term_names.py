"""reward_term_names: the reward terms the Torch env logs, by name.

Model-free. The integration test test_resolved_constants.py checks the names
against a real compute_rewards call.
"""

import pytest

from humanoid_lab.envs.joystick import default_config, reward_term_names


def test_an_unknown_scales_key_is_refused():
    scales = default_config().reward.scales
    scales.unlock()
    scales["no_such_term"] = 1.0
    with pytest.raises(KeyError, match="no_such_term"):
        reward_term_names(scales)
