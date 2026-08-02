"""The in-frame overlays draw from explicit inputs, with no model behind
them: eval/overlays.py takes arrays and numbers and returns a frame, so
every panel is checkable here on synthetic inputs. The SceneView that
feeds it (eval/render.py) needs a scene, so it belongs to the slow suite.
"""

import argparse

import numpy as np
import pytest

from humanoid_lab.eval import overlays
from humanoid_lab.eval.render import frame_size

SIZE = (640, 480)
# Actuator (column) order deliberately differs from group order, matching
# how a robot.yaml interleaves left/right joints.
NAMES = ["hip_l", "knee_l", "hip_r", "knee_r"]
GROUPS = {"hip": ["hip_l", "hip_r"], "knee": ["knee_l", "knee_r"]}
CAPS = np.array([60.0, 90.0, 60.0, 90.0])


def blank(size=SIZE):
    return np.zeros((size[1], size[0], 3), dtype=np.uint8)


@pytest.mark.parametrize(
    "text,expected", [("640x480", (640, 480)), ("1280X720", (1280, 720))]
)
def test_frame_size_parses_wxh(text, expected):
    assert frame_size(text) == expected


@pytest.mark.parametrize("text", ["640", "640x", "640x0", "-640x480", "wide"])
def test_frame_size_rejects_junk(text):
    with pytest.raises(argparse.ArgumentTypeError):
        frame_size(text)


def test_torque_strip_stays_in_the_bottom_band():
    frame = blank()
    out = overlays.draw_torques(frame, np.zeros(4), CAPS, NAMES, GROUPS)
    assert out.shape == frame.shape
    assert not frame.any()  # the input frame is not modified
    half = frame.shape[0] // 2
    assert not out[:half].any()
    assert out[half:].any()


def test_torque_bars_grow_with_the_torque():
    idle = overlays.draw_torques(blank(), np.zeros(4), CAPS, NAMES, GROUPS)

    def bar_pixels(fraction):
        out = overlays.draw_torques(blank(), fraction * CAPS, CAPS, NAMES, GROUPS)
        return int(np.any(out != idle, axis=-1).sum())

    assert bar_pixels(0.9) > bar_pixels(0.2) > 0


def test_bars_are_normalized_by_each_joints_own_cap():
    """Equal fractions of unequal caps draw the identical strip."""
    hetero = overlays.draw_torques(blank(), 0.5 * CAPS, CAPS, NAMES, GROUPS)
    uniform = overlays.draw_torques(
        blank(), np.full(4, 30.0), np.full(4, 60.0), NAMES, GROUPS
    )
    assert np.array_equal(hetero, uniform)


def test_limit_lines_are_drawn():
    out = overlays.draw_torques(blank(), np.zeros(4), CAPS, NAMES, GROUPS)
    assert (out == overlays.LIMIT_RGB).all(axis=-1).any()


def test_an_ungrouped_joint_raises():
    with pytest.raises(KeyError, match="ankle_l"):
        overlays.draw_torques(
            blank(), np.zeros(5), np.ones(5), NAMES + ["ankle_l"], GROUPS
        )


def test_a_group_member_missing_from_joint_names_raises():
    groups = {**GROUPS, "ankle": ["ankle_l", "ankle_r"]}
    with pytest.raises(KeyError, match="ankle_l"):
        overlays.draw_torques(blank(), np.zeros(4), CAPS, NAMES, groups)


def test_mismatched_torque_length_raises():
    with pytest.raises(ValueError):
        overlays.draw_torques(blank(), np.zeros(3), CAPS, NAMES, GROUPS)


def test_panels_scale_with_the_frame():
    assert overlays.scale_of((480, 360)) < 1.0 < overlays.scale_of((1920, 1440))
    assert overlays.scale_of((overlays.REF_W, overlays.REF_H)) == 1.0
    assert overlays.font_px((320, 240)) < overlays.font_px((1920, 1440))
