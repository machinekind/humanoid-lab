"""In-frame overlays drawn onto rendered rollout frames.

Pure functions over explicit inputs (a frame array, torques and caps, the
joint grouping), so everything here is checkable on synthetic inputs with
no model behind it. These draw ONTO the frame; the full-episode trace
panels stitched UNDER the frame are eval/plots.py's.
"""

import functools
from pathlib import Path

import numpy as np

REF_W, REF_H = 960, 720  # the size the panel constants below were drawn at
MARGIN = 12
STRIP_H = 152
FONT_PX = 14

PANEL_RGBA = (18, 18, 22, 170)
TEXT_RGB = (236, 236, 240)
EDGE_RGB = (210, 210, 216)
LIMIT_RGB = (235, 70, 60)


@functools.lru_cache(maxsize=4)
def _font(px=FONT_PX):
    """Monospace face shipped with matplotlib, so no new font dependency."""
    import matplotlib
    from PIL import ImageFont

    ttf = Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSansMono.ttf"
    try:
        return ImageFont.truetype(str(ttf), px)
    except OSError:
        return ImageFont.load_default()


def _group_colors(joint_groups):
    import matplotlib

    colors = matplotlib.colormaps["tab10"].colors
    return {
        group: tuple(round(255 * c) for c in colors[i % len(colors)])
        for i, group in enumerate(joint_groups)
    }


def scale_of(size):
    """Panel scale for a frame of `size`, 1.0 at the reference size."""
    w, h = size
    return min(w / REF_W, h / REF_H)


def font_px(size):
    """Text size for a frame of `size`; a small frame gets a smaller face,
    so the labels stay inside the panels they belong to."""
    return max(9, round(FONT_PX * scale_of(size)))


def draw_torques(frame, torque, caps, joint_names, joint_groups):
    """A new frame with a signed bar per actuator drawn into its bottom band.

    Bars are normalized by each joint's own cap (torque[j] / caps[j]) --
    the same +-1 actuator-limit scale as eval/plots.py's torque_strip, and
    for the same reason: the per-joint force ranges are heterogeneous, so
    a shared N*m axis means nothing. The red lines mark the +-1 limit.
    Groups follow `joint_groups` (dict order, one color per group), joints
    within a group in that group's list order; `torque`/`caps` columns are
    in `joint_names` order. The input frame is not modified.
    """
    from PIL import Image, ImageDraw

    torque = np.asarray(torque, dtype=float)
    caps = np.asarray(caps, dtype=float)
    if len(torque) != len(joint_names) or len(caps) != len(joint_names):
        raise ValueError(
            f"torque ({len(torque)}) and caps ({len(caps)}) must match "
            f"joint_names ({len(joint_names)})"
        )
    index = {name: j for j, name in enumerate(joint_names)}
    grouped = [name for joints in joint_groups.values() for name in joints]
    unknown = [name for name in grouped if name not in index]
    if unknown:
        raise KeyError(f"joints {unknown} in joint_groups are not in joint_names")
    ungrouped = [name for name in joint_names if name not in set(grouped)]
    if ungrouped:
        raise KeyError(f"joints {ungrouped} are not assigned to any group in joint_groups")

    im = Image.fromarray(np.asarray(frame))
    draw = ImageDraw.Draw(im, "RGBA")
    px = font_px(im.size)
    font = _font(px)
    colors = _group_colors(joint_groups)
    w, h = im.size
    strip_h = max(round(STRIP_H * scale_of(im.size)), 2 * px + 40)
    x0, x1 = MARGIN, w - MARGIN
    y1 = h - MARGIN
    y0 = y1 - strip_h
    draw.rectangle([x0, y0, x1, y1], fill=PANEL_RGBA)
    pad, group_gap, bar_gap = 14, 30, 6
    top, bot = y0 + 3 * px + 18, y1 - 8
    mid = (top + bot) / 2
    half = (bot - top) / 2
    scale = 0.85 * half  # a bar at its limit reaches 85% of the half-band
    groups = list(joint_groups)
    n_bars = len(grouped)
    group_w_total = (x1 - x0) - 2 * pad - group_gap * (len(groups) - 1)
    bar_w = (group_w_total - bar_gap * (n_bars - len(groups))) / n_bars
    bx = x0 + pad
    for g, group in enumerate(groups):
        gx0 = bx
        for name in joint_groups[group]:
            j = index[name]
            tip = mid - float(np.clip(torque[j] / caps[j] * scale, -half, half))
            draw.rectangle(
                [bx, min(mid, tip), bx + bar_w, max(mid, tip)], fill=colors[group]
            )
            bx += bar_w + bar_gap
        cx = (gx0 + bx - bar_gap) / 2
        # Staggered over two rows: a humanoid has enough groups with long
        # names that a single row of labels collides.
        ty = y0 + px + 8 + (px + 2) * (g % 2)
        draw.text((cx, ty), group, fill=colors[group], font=font, anchor="ma")
        bx += group_gap - bar_gap
    draw.line([x0 + pad, mid, x1 - pad, mid], fill=EDGE_RGB, width=1)
    for sign in (-1, 1):
        y = mid - sign * scale
        draw.line([x0 + pad, y, x1 - pad, y], fill=LIMIT_RGB, width=2)
    draw.text(
        (x0 + pad, y0 + 4), "actuator torque / limit, red = ±1",
        fill=TEXT_RGB, font=font,
    )
    return np.asarray(im)
