"""reward_term_names: the reward terms the Torch env logs, by name.

Model-free. The integration test test_resolved_constants.py checks the names
against a real compute_rewards call.
"""

import pytest

from humanoid_lab.envs import joystick
from humanoid_lab.envs.joystick import _STYLE_TERMS, default_config, reward_term_names


def test_term_names_follow_the_metric_names():
    """The same names as the reward/* metrics, in compute_rewards order. The
    metrics follow the scales, which a ConfigDict iterates sorted."""
    scales = default_config().reward.scales
    names = reward_term_names(scales)
    stripped = [k.removeprefix("reward/") for k in joystick._reward_metric_names(scales)]
    assert sorted(names) == sorted(stripped)
    assert len(names) == len(set(names))
    assert not [n for n in names if n.startswith("reward/")]
    order = joystick._REWARD_ORDER
    assert [order.index(n) for n in names] == sorted(order.index(n) for n in names)
    assert names[:2] == ["tracking_lin_vel", "tracking_ang_vel"]


def test_an_unknown_scales_key_is_refused():
    scales = default_config().reward.scales
    scales.unlock()
    scales["no_such_term"] = 1.0
    with pytest.raises(KeyError, match="no_such_term"):
        reward_term_names(scales)


@pytest.mark.parametrize("term", _STYLE_TERMS)
def test_style_terms_at_zero_scale_are_absent(term):
    scales = default_config().reward.scales
    scales[term] = 0.0
    assert term not in reward_term_names(scales)
    scales[term] = 0.5
    assert term in reward_term_names(scales)
