"""check_pure_draw_ranges: an armed pure draw must redraw inside the
command box. Enforced at env construction; export inherits the guarantee
because loading a checkpoint rebuilds its env through the same
constructor."""

from __future__ import annotations

import pytest

from humanoid_lab.envs.joystick import check_pure_draw_ranges, default_config


def test_an_armed_pure_draw_outside_the_command_box_raises():
    command = default_config().to_dict()["command"]
    command["pure_fast_prob"] = 0.2
    command["fast_vx"] = (0.9, 1.4)  # above command.vx
    with pytest.raises(ValueError, match="fast_vx"):
        check_pure_draw_ranges(command)


def test_a_disarmed_pure_draw_outside_the_command_box_is_ignored():
    command = default_config().to_dict()["command"]
    command["pure_fast_prob"] = 0.0
    command["fast_vx"] = (0.9, 1.4)
    check_pure_draw_ranges(command)


def test_an_armed_pure_draw_inside_the_command_box_passes():
    command = default_config().to_dict()["command"]
    command["pure_back_prob"] = 0.3
    check_pure_draw_ranges(command)
