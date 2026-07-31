"""eval/plots.py's panel builders, exercised on synthetic per-step arrays --
no rollout, no checkpoint, no mujoco model. Mirrors test_sizing_report.py's
pattern (plain numpy in, checks on the returned frame_at(t) closure out).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from humanoid_lab.eval import plots
from humanoid_lab.eval.plots import cursor_strip, joint_grid, joint_zoom, torque_strip

# Asimov v1's 12 actuated leg joints and 6 joint_groups (robots/asimov_v1/robot.yaml).
JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
]
JOINT_GROUPS = {
    "hip_pitch": ["left_hip_pitch_joint", "right_hip_pitch_joint"],
    "hip_roll": ["left_hip_roll_joint", "right_hip_roll_joint"],
    "hip_yaw": ["left_hip_yaw_joint", "right_hip_yaw_joint"],
    "knee": ["left_knee_joint", "right_knee_joint"],
    "ankle_pitch": ["left_ankle_pitch_joint", "right_ankle_pitch_joint"],
    "ankle_roll": ["left_ankle_roll_joint", "right_ankle_roll_joint"],
}
T = 50


def _times():
    return np.arange(T) * 0.02


def _synthetic_traces(n_joints=12):
    # Deterministic, no RNG seed needed: a phase-shifted sine per column.
    times = _times()
    phase = np.linspace(0, np.pi, n_joints)
    base = np.sin(times[:, None] * 3.0 + phase[None, :])
    return base, base + 0.05  # (position/torque-like, target/second-trace)


def _torque_caps(n_joints=12):
    return np.linspace(20.0, 150.0, n_joints)


def test_torque_strip_frame_shape_at_start_mid_end():
    times = _times()
    torques, _ = _synthetic_traces()
    caps = _torque_caps()

    frame_at = torque_strip(times, torques, caps, JOINT_NAMES, JOINT_GROUPS, width=320, height=240)

    for t in (times[0], times[len(times) // 2], times[-1]):
        frame = frame_at(t)
        assert frame.shape == (240, 320, 3)
        assert frame.dtype == np.uint8


def test_torque_strip_normalized_values_at_cap_dont_raise():
    # torques == caps broadcast -> every plotted ratio is 1.0. No pixel-level
    # assertion (the render-once design makes that fragile); just confirm
    # this doesn't raise and produces the requested frame size.
    times = _times()
    caps = _torque_caps()
    torques = np.tile(caps, (T, 1))

    frame_at = torque_strip(times, torques, caps, JOINT_NAMES, JOINT_GROUPS, width=200, height=150)

    assert frame_at(times[0]).shape == (150, 200, 3)


def test_torque_strip_raises_keyerror_for_ungrouped_joint():
    times = _times()
    names = JOINT_NAMES + ["left_wrist_yaw_joint"]
    torques, _ = _synthetic_traces(n_joints=len(names))
    caps = _torque_caps(n_joints=len(names))

    with pytest.raises(KeyError, match="left_wrist_yaw_joint"):
        torque_strip(times, torques, caps, names, JOINT_GROUPS, width=200, height=150)


def test_joint_grid_frame_shape_six_groups():
    times = _times()
    positions, targets = _synthetic_traces()

    frame_at = joint_grid(times, targets, positions, JOINT_NAMES, JOINT_GROUPS, width=480, height=480)

    for t in (times[0], times[len(times) // 2], times[-1]):
        frame = frame_at(t)
        assert frame.shape == (480, 480, 3)
        assert frame.dtype == np.uint8


def test_joint_grid_single_group_squeeze_edge_case():
    names = ["left_knee_joint", "right_knee_joint"]
    groups = {"knee": names}
    times = _times()
    positions, targets = _synthetic_traces(n_joints=2)

    frame_at = joint_grid(times, targets, positions, names, groups, width=320, height=160)

    frame = frame_at(times[0])
    assert frame.shape == (160, 320, 3)
    assert frame.dtype == np.uint8


def _row_ylims(monkeypatch, times, targets, positions, names, groups):
    """joint_grid's per-axes y-limits, in row-major order.

    `cursor_strip` renders and then closes the figure, so the axes cannot be
    inspected after joint_grid returns; this spies on the handoff instead of
    changing either function's contract to make the test possible.
    """
    captured = {}
    real = plots.cursor_strip

    def spy(fig, axes, t, **kwargs):
        captured["ylims"] = [ax.get_ylim() for ax in axes]
        return real(fig, axes, t, **kwargs)

    monkeypatch.setattr(plots, "cursor_strip", spy)
    joint_grid(times, targets, positions, names, groups, width=320, height=320)
    return captured["ylims"]


def test_joint_grid_rows_share_a_y_range_so_asymmetry_is_visible(monkeypatch):
    """The whole point of the grid: a left knee swinging twice as far as the
    right has to READ as twice as far. Per-axes autoscaling would rescale
    each column to fill its own box and the two traces would look the same.
    """
    names = ["left_knee_joint", "right_knee_joint", "left_hip_pitch_joint", "right_hip_pitch_joint"]
    groups = {"knee": names[:2], "hip_pitch": names[2:]}
    times = _times()
    signal = np.sin(times * 3.0)
    # Row 0 is asymmetric (left swings 4x the right); row 1 is symmetric and
    # an order of magnitude smaller.
    positions = np.stack([4.0 * signal, signal, 0.1 * signal, 0.1 * signal], axis=-1)

    ylims = _row_ylims(monkeypatch, times, positions, positions, names, groups)

    assert ylims[0] == ylims[1], "the two columns of a group must share their y-range"
    assert ylims[0][1] >= 4.0, "the shared range has to span the taller of the two traces"
    # Per ROW, not globally: row 1's 0.1 rad wiggle would be a flat line
    # inside row 0's range.
    assert ylims[2] != ylims[0]
    assert ylims[2] == ylims[3]


def test_joint_grid_raises_keyerror_for_ungrouped_joint():
    times = _times()
    names = JOINT_NAMES + ["left_wrist_yaw_joint"]
    positions, targets = _synthetic_traces(n_joints=len(names))

    with pytest.raises(KeyError, match="left_wrist_yaw_joint"):
        joint_grid(times, targets, positions, names, JOINT_GROUPS, width=320, height=160)


def test_joint_zoom_frame_shape():
    times = _times()
    positions, targets = _synthetic_traces()

    frame_at = joint_zoom(times, targets[:, 3], positions[:, 3], "left_knee_joint", width=400, height=240)

    for t in (times[0], times[len(times) // 2], times[-1]):
        frame = frame_at(t)
        assert frame.shape == (240, 400, 3)
        assert frame.dtype == np.uint8


def _build_torque_strip_frame_at(times):
    torques, _ = _synthetic_traces()
    return torque_strip(times, torques, _torque_caps(), JOINT_NAMES, JOINT_GROUPS, 200, 150)


def _build_joint_grid_frame_at(times):
    positions, targets = _synthetic_traces()
    return joint_grid(times, targets, positions, JOINT_NAMES, JOINT_GROUPS, 200, 200)


def _build_joint_zoom_frame_at(times):
    positions, targets = _synthetic_traces()
    return joint_zoom(times, targets[:, 0], positions[:, 0], "left_hip_pitch_joint", 200, 150)


@pytest.mark.parametrize(
    "build_frame_at",
    [_build_torque_strip_frame_at, _build_joint_grid_frame_at, _build_joint_zoom_frame_at],
)
def test_cursor_moves_and_frame_at_is_pure(build_frame_at):
    times = _times()
    frame_at = build_frame_at(times)

    first = frame_at(times[0])
    last = frame_at(times[-1])
    assert not np.array_equal(first, last)  # cursor moved

    # Calling frame_at twice at the same t must not mutate the base image.
    first_again = frame_at(times[0])
    assert np.array_equal(first, first_again)

    mid = frame_at(times[len(times) // 2])  # noqa: F841 -- exercised for side effects
    first_yet_again = frame_at(times[0])
    assert np.array_equal(first, first_yet_again)


def test_cursor_strip_handles_single_step_episode():
    times = np.array([0.0])
    target = np.array([0.1])
    position = np.array([0.12])

    frame_at = joint_zoom(times, target, position, "left_knee_joint", width=200, height=150)

    frame = frame_at(0.0)
    assert frame.shape == (150, 200, 3)
    assert frame.dtype == np.uint8


def _torque_strip_awkward_size(times):
    torques, _ = _synthetic_traces()
    return torque_strip(times, torques, _torque_caps(), JOINT_NAMES, JOINT_GROUPS, width=481, height=239)


def _joint_grid_awkward_size(times):
    positions, targets = _synthetic_traces()
    return joint_grid(times, targets, positions, JOINT_NAMES, JOINT_GROUPS, width=481, height=239)


def _joint_zoom_awkward_size(times):
    positions, targets = _synthetic_traces()
    return joint_zoom(times, targets[:, 0], positions[:, 0], "left_hip_pitch_joint", width=481, height=239)


@pytest.mark.parametrize(
    "build_frame_at",
    [_torque_strip_awkward_size, _joint_grid_awkward_size, _joint_zoom_awkward_size],
)
def test_frame_shape_is_exact_for_awkward_size(build_frame_at):
    # width=481, height=239 truncate under naive figsize=(w/_DPI, h/_DPI) ->
    # Agg canvas math (int(figwidth * dpi)); the builder must still hand
    # back exactly the requested (height, width, 3).
    times = _times()
    frame_at = build_frame_at(times)

    frame = frame_at(times[0])
    assert frame.shape == (239, 481, 3)
    assert frame.dtype == np.uint8


def test_cursor_strip_direct_on_trivial_figure():
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(2.0, 1.5), dpi=100)
    ax.plot([0.0, 1.0], [0.0, 1.0])

    frame_at = cursor_strip(fig, [ax], [0.0, 1.0])

    start = frame_at(0.0)
    end = frame_at(1.0)
    assert start.shape == end.shape == (150, 200, 3)
    assert not np.array_equal(start, end)
