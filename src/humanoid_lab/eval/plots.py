"""Matplotlib panel builders for rollout videos: torque and joint-angle
traces stitched under (or beside) the rendered frames.

Every builder here (torque_strip, joint_grid, joint_zoom) draws its axes
exactly ONCE for the whole episode, then returns a `frame_at(t)` closure
that copies the pre-rendered RGB image and stamps a moving cursor column
into it. Video assembly then costs one matplotlib pass per panel plus O(T)
cheap array copies, instead of O(T) matplotlib redraws -- replotting per
frame is what makes video rendering slow, not the rendering itself.

Joint grouping comes from RobotSpec.joint_groups (left/right pairs per
named group, robot.yaml) rather than from parsing leg-name prefixes: group
membership is already an explicit, validated part of the robot spec.
"""

from __future__ import annotations

from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_DPI = 100
_CURSOR_COLOR = (220, 40, 40)


def _group_of(joint_name: str, joint_groups: dict) -> str:
    """joint_groups key containing `joint_name`. Mirrors RobotSpec.group_of
    (src/humanoid_lab/robot/spec.py) but takes the dict directly: builders
    here are handed joint_groups as plain data, not a RobotSpec, so callers
    that already loaded a RobotSpec don't need to keep it around just for
    this lookup.
    """
    for group, joints in joint_groups.items():
        if joint_name in joints:
            return group
    raise KeyError(f"joint '{joint_name}' is not assigned to any group in joint_groups")


def _check_all_grouped(joint_names, joint_groups: dict) -> None:
    for name in joint_names:
        _group_of(name, joint_groups)  # raises KeyError naming the joint


def _check_all_named(joint_groups: dict, joint_names) -> None:
    known = set(joint_names)
    for group, joints in joint_groups.items():
        for name in joints:
            if name not in known:
                raise KeyError(f"joint '{name}' in group '{group}' is not in joint_names")


def _exact_size(img: np.ndarray, height: int, width: int) -> np.ndarray:
    """Slice or edge-pad `img` (H, W, 3) to exactly (height, width, 3).

    Agg sizes the canvas as int(figwidth * dpi); figwidth = width / _DPI is
    a float division, so it truncates to one pixel short of the requested
    size for some non-round widths/heights (e.g. requested 481 -> 480 wide).
    Callers vstack these panels with a fixed-size render frame, so a
    mismatched width fails far from here; a truncated height can also flip
    an even requested height to odd, breaking H.264's even-dimension
    requirement. Edge-padding is a 1px, visually inert fix for the
    truncated case; slicing covers the (unobserved but possible) oversize
    case the same way.
    """
    h, w = img.shape[:2]
    if h > height:
        img = img[:height]
    elif h < height:
        img = np.pad(img, ((0, height - h), (0, 0), (0, 0)), mode="edge")
    w = img.shape[1]
    if w > width:
        img = img[:, :width]
    elif w < width:
        img = np.pad(img, ((0, 0), (0, width - w), (0, 0)), mode="edge")
    return img


def cursor_strip(
    fig, axes, times, width: int | None = None, height: int | None = None
) -> Callable[[float], np.ndarray]:
    """Render `fig` once and return frame_at(t): a copy of the rendered RGB
    image with a 2px red vertical cursor column stamped inside each of
    `axes`, positioned at the fraction of `times` that `t` falls at.

    `width`/`height`, when given, are the caller's requested panel size in
    pixels; the rendered buffer is forced to exactly (height, width, 3) --
    see `_exact_size` for why. Callers that don't have fixed target
    dimensions (e.g. a caller-supplied `fig` of arbitrary size) may omit
    them and get the canvas's natural size back.

    `ax.get_window_extent()` is bottom-up (origin at the figure's
    bottom-left, matplotlib's display-coordinate convention); the rendered
    image array is top-down, so each axes' y-span is flipped (row = fig
    height - y) before it's used to index into the image.
    """
    fig.canvas.draw()
    base = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    if height is not None or width is not None:
        base = _exact_size(base, height if height is not None else base.shape[0],
                            width if width is not None else base.shape[1])
    h = base.shape[0]
    spans = [
        (e.x0, e.x1, int(h - e.y1), int(h - e.y0))
        for e in (ax.get_window_extent() for ax in axes)
    ]
    plt.close(fig)

    t0, t1 = times[0], times[-1]
    span = max(t1 - t0, 1e-9)  # guards T=1 (episode fell at step 0)

    def frame_at(t: float) -> np.ndarray:
        img = base.copy()
        frac = min(max((t - t0) / span, 0.0), 1.0)
        for x0, x1, r0, r1 in spans:
            col = int(x0 + frac * (x1 - x0))
            img[r0:r1, max(col - 1, 0):col + 1] = _CURSOR_COLOR
        return img

    return frame_at


def torque_strip(
    times, torques, torque_caps, joint_names, joint_groups: dict, width: int, height: int = 240
) -> Callable[[float], np.ndarray]:
    """Full-episode strip of every joint's torque, normalized by that
    joint's own actuator cap (torques[:, j] / torque_caps[j]).

    This robot's actuators have heterogeneous per-joint force ranges (a hip
    motor and an ankle motor don't share a torque scale), so a single N*m
    cap line across every joint -- which only makes sense when one motor
    spec covers every joint -- does not mean anything here. Normalizing puts
    every joint on the same +-1 actuator-limit scale regardless of its
    underlying N*m range, and the dashed lines at +-1 mark that shared
    limit.

    One color per joint group (plt.get_cmap("tab10"), indexed by the
    group's position in `joint_groups`); both joints of a left/right pair
    share a color. `torques` is [T, nu], `torque_caps` is [nu] (positive),
    `joint_names` has length nu in the same column order.
    """
    _check_all_grouped(joint_names, joint_groups)
    times = np.asarray(times, dtype=float)
    torques = np.asarray(torques, dtype=float)
    torque_caps = np.asarray(torque_caps, dtype=float)
    groups = list(joint_groups)
    colors = plt.get_cmap("tab10").colors

    fig, ax = plt.subplots(figsize=(width / _DPI, height / _DPI), dpi=_DPI)
    for j, name in enumerate(joint_names):
        group = _group_of(name, joint_groups)
        color = colors[groups.index(group) % len(colors)]
        is_first_of_group = joint_groups[group][0] == name
        ax.plot(
            times, torques[:, j] / torque_caps[j], color=color, linewidth=0.6,
            label=group if is_first_of_group else None,
        )
    ax.axhline(1.0, color="red", linestyle="--", linewidth=0.7)
    ax.axhline(-1.0, color="red", linestyle="--", linewidth=0.7)
    ax.set_ylim(-1.3, 1.3)
    ax.set_xlim(times[0], times[-1])
    ax.set_xlabel("t [s]", fontsize=8)
    ax.set_ylabel("torque / limit", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(loc="upper right", fontsize=6, ncol=4, framealpha=0.5)
    fig.tight_layout(pad=0.4)
    return cursor_strip(fig, [ax], times, width=width, height=height)


def joint_grid(
    times, targets, positions, joint_names, joint_groups: dict, width: int, height: int = 480
) -> Callable[[float], np.ndarray]:
    """Grid of achieved-vs-target position traces: one row per joint group
    (dict order), one column per position within the group's joint list
    (left, right for a pair). `sharex=True, sharey="row"` so left/right
    asymmetry within a group is visible at a glance. Achieved position is
    solid in the group's color, target dashed black.

    Column titles "left"/"right" are added only when every group has
    exactly 2 joints; a mixed group size (e.g. a future singleton waist
    group) has no single "left"/"right" reading that fits every column, so
    the titles are skipped rather than mislabeled.

    `targets`/`positions` are [T, nu], columns aligned with `joint_names`.
    """
    _check_all_grouped(joint_names, joint_groups)
    _check_all_named(joint_groups, joint_names)  # both directions, before fig creation
    times = np.asarray(times, dtype=float)
    targets = np.asarray(targets, dtype=float)
    positions = np.asarray(positions, dtype=float)
    name_to_idx = {n: i for i, n in enumerate(joint_names)}
    groups = list(joint_groups)
    colors = plt.get_cmap("tab10").colors

    ncols = max(len(joints) for joints in joint_groups.values())
    label_columns = ncols == 2 and all(len(j) == 2 for j in joint_groups.values())

    fig, axes = plt.subplots(
        len(groups), ncols, figsize=(width / _DPI, height / _DPI), dpi=_DPI,
        sharex=True, sharey="row",
    )
    # plt.subplots squeezes away size-1 dimensions (a single group returns a
    # flat row of axes, single-joint groups return a flat column, 1x1
    # returns a bare Axes) -- reshape straight back to (n_groups, ncols)
    # rather than branching on which squeeze case applies; the element
    # count always matches.
    axes = np.asarray(axes).reshape(len(groups), ncols)

    plotted_axes = []
    for row, group in enumerate(groups):
        color = colors[row % len(colors)]
        group_joints = joint_groups[group]
        for col in range(ncols):
            ax = axes[row][col]
            if col >= len(group_joints):
                ax.axis("off")
                continue
            j = name_to_idx[group_joints[col]]
            ax.plot(times, positions[:, j], color=color, linewidth=0.6, label="state")
            ax.plot(times, targets[:, j], color="black", linewidth=0.5, linestyle="--", label="target")
            ax.tick_params(labelsize=6)
            plotted_axes.append(ax)
        axes[row][0].set_ylabel(group, fontsize=7)

    if label_columns:
        axes[0][0].set_title("left", fontsize=7)
        axes[0][1].set_title("right", fontsize=7)

    axes[0][0].set_xlim(times[0], times[-1])
    axes[0][0].legend(loc="upper right", fontsize=5, framealpha=0.5)
    fig.tight_layout(pad=0.3)
    return cursor_strip(fig, plotted_axes, times, width=width, height=height)


def joint_zoom(
    times, target, position, joint_name: str, width: int, height: int = 240
) -> Callable[[float], np.ndarray]:
    """One full-width, readable panel for a single joint: raw angle in
    radians, achieved state solid, policy target dashed black -- the
    zoomed-in alternative to joint_grid for inspecting one joint closely.
    """
    times = np.asarray(times, dtype=float)
    target = np.asarray(target, dtype=float)
    position = np.asarray(position, dtype=float)

    fig, ax = plt.subplots(figsize=(width / _DPI, height / _DPI), dpi=_DPI)
    ax.plot(times, position, color="tab:blue", linewidth=1.2, label="state")
    ax.plot(times, target, color="black", linewidth=0.9, linestyle="--", label="target")
    ax.set_xlim(times[0], times[-1])
    ax.set_xlabel("t [s]", fontsize=9)
    ax.set_ylabel("angle [rad]", fontsize=9)
    ax.set_title(joint_name, fontsize=10)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.3, linewidth=0.4)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.5)
    fig.tight_layout(pad=0.4)
    return cursor_strip(fig, [ax], times, width=width, height=height)
