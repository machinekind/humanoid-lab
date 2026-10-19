"""Gait KPIs from a battery rollout record.

Pure numpy: record arrays in, a dict of numbers out, so this is unit-tested
directly in tests/unit/test_gait_metrics.py with no env, model or checkpoint.

The module holds two families of metrics.

- The swing KPIs (gait_metrics) read per-foot clearance and vertical
  velocity. They score rec[settle:].
- The gait-shape metrics (gait_shape) read the base pose, the command,
  the foot contacts, the foot positions and the knee angles. They score
  rec[settle:fell_at]. They run the shared
  signals module on the record's numpy arrays, with the record's step axis
  as the batch axis.

These are RAW METRICS. Nothing here folds into a score, a gate, or the
fell/completed logic. They exist because velocity tracking error cannot
tell a walking policy from two failure modes that track velocity perfectly
well: a skimming gait that never clears the ground, and a stand-and-lift
farm that collects swing-shaping reward without going anywhere. Both are
visible in the apex and the touchdown numbers.

Works for any foot count -- the record's second axis is the feet.
"""

from __future__ import annotations

import numpy as np

from humanoid_lab.envs import signals

# Clearance above which a foot counts as airborne, metres. A physical
# height: _foot_clearance reads the sole's height above the floor (a
# planted foot ~0), so this is a real 5 mm band. A biped foot is an
# extended plate that rolls heel-to-toe while the per-foot site is a single
# point, so partial contact near the band edge classifies by the site's
# height; treat swing counts near the threshold with care.
AIRBORNE_M = 0.005

# Shortest run of airborne steps that counts as a swing. A single step over
# the band is contact-detection noise, not a step taken.
MIN_SWING_STEPS = 2

G = 9.81  # m/s^2, the free-fall reference softness normalizes against

_FIELDS = (
    "swing_apex_med_m",
    "swing_apex_p90_m",
    "swings",
    "touchdown_v_med",
    "touchdown_softness_med",
)

# Commanded distance below which speed_frac reads None, metres. Below it the
# fraction divides by a near-zero number.
SPEED_FRAC_MIN_M = 0.1


def foot_sides(names) -> tuple[str, ...]:
    """The side token of each foot site in `names`, in the same order.

    A foot's side is "left" or "right" when exactly one of the two tokens
    occurs in its site name. The match is case-sensitive, the same rule
    Joystick._foot_ordered_group_qidx uses. If any name has no single token,
    or two names share a side, every foot gets "foot0", "foot1", and so on.
    The per-foot battery keys are suffixed with these tokens.
    """
    names = list(names)
    fallback = tuple(f"foot{i}" for i in range(len(names)))
    sides = []
    for name in names:
        tokens = [t for t in ("left", "right") if t in name]
        if len(tokens) != 1:
            return fallback
        sides.append(tokens[0])
    if len(set(sides)) != len(sides):
        return fallback
    return tuple(sides)


def _swings(clear_f: np.ndarray, vz_f: np.ndarray):
    """(apex, touchdown speed) for every scorable swing of one foot.

    Both arguments are float columns of the trimmed record, one foot's.

    A swing is a contiguous run of `clear_f > AIRBORNE_M`. Two runs are
    dropped because they were not fully observed rather than because they
    were bad: one already airborne at the first sample (its apex may have
    happened before the window opened) and one still airborne at the last
    (it has no touchdown). Both would report a floor as if it were a peak.
    """
    airborne = clear_f > AIRBORNE_M
    n = len(airborne)
    out = []
    t = 0
    while t < n:
        if not airborne[t]:
            t += 1
            continue
        start = t
        while t < n and airborne[t]:
            t += 1
        if start == 0 or t >= n or t - start < MIN_SWING_STEPS:
            continue
        apex = float(clear_f[start:t].max())
        # Touchdown speed is the DOWNWARD vertical speed on the last
        # airborne step, so a foot still rising there arrived at 0, not at a
        # negative speed. This is the one velocity in these KPIs that is
        # world-frame rather than body-frame, on purpose: how hard a foot
        # hits the ground is a question about the foot and the ground, and
        # the ground does not rotate with the robot.
        td = max(0.0, -float(vz_f[t - 1]))
        out.append((apex, td))
    return out


def gait_metrics(rec: dict, *, settle_steps: int) -> dict:
    """Swing-apex and touchdown-softness KPIs over `rec`'s post-settle window.

    `rec` carries `foot_clear` and `foot_vz`, both (steps, feet). The first
    `settle_steps` steps are dropped as the reset transient (the caller owns
    the number; eval/battery.py passes settle_steps(dt)).

    Per swing: the apex is the peak clearance, the touchdown speed the
    downward vertical speed on the last airborne step, and
    `softness = touchdown_v / sqrt(2*G*apex)` normalizes that speed by what
    free fall from this swing's own apex would have delivered. 1.0 means the
    foot fell like a brick from its peak; lower means it was flown in.

    Every field is always present. With no scorable swing, `swings` is 0 and
    every median is None -- an unmeasured apex reported as 0.0 would average
    into a keeper comparison as though a foot had been measured on the floor.
    `swings` is the sample count behind all four medians.
    """
    clear = np.asarray(rec["foot_clear"], dtype=float)[settle_steps:]
    vz = np.asarray(rec["foot_vz"], dtype=float)[settle_steps:]

    measured = []
    for f in range(clear.shape[1] if clear.ndim == 2 else 0):
        measured += _swings(clear[:, f], vz[:, f])

    if not measured:
        return {**{k: None for k in _FIELDS}, "swings": 0}

    apexes = np.array([a for a, _ in measured])
    tds = np.array([t for _, t in measured])
    softness = tds / np.sqrt(2.0 * G * apexes)
    return {
        "swing_apex_med_m": round(float(np.median(apexes)), 4),
        "swing_apex_p90_m": round(float(np.percentile(apexes, 90)), 4),
        "swings": len(measured),
        "touchdown_v_med": round(float(np.median(tds)), 3),
        "touchdown_softness_med": round(float(np.median(softness)), 3),
    }


# -- gait shape --------------------------------------------------------------


def metric_window(n_rows: int, settle_steps: int, fell_at: int | None) -> slice:
    """The rows the gait-shape metrics score: rec[settle_steps:fell_at].

    The window starts after the reset transient. It leaves out the row of
    the step that tripped done. The battery record keeps that fallen row.
    The end is n_rows when fell_at is None, and it never goes below
    settle_steps, so a fall inside the settle window gives an empty slice.
    """
    end = n_rows if fell_at is None else min(fell_at, n_rows)
    return slice(settle_steps, max(end, settle_steps))


def _r(value, nd: int):
    """value rounded to nd places as a Python float. None passes through.

    battery.json goes through json.dumps, which cannot write numpy floats.
    """
    return None if value is None else round(float(value), nd)


def _bounds(rec: dict, settle_steps: int, fell_at: int | None) -> tuple[int, int]:
    w = metric_window(len(rec["cmd"]), settle_steps, fell_at)
    return w.start, w.stop


def travel_metrics(rec: dict, *, dt: float, settle_steps: int, fell_at: int | None) -> dict:
    """Distance, heading and lateral drift against the command, over rec[settle:fell_at].

    The start heading is the IMU yaw of the first scored row. Travel is
    measured from the first scored row to the last one, in that heading's
    frame. Lateral is positive to the left.

    Row i is the state after step i under cmd[i]. The motion between the
    first scored row s and the last one e-1 therefore comes from the commands
    of rows s+1 to e-1. A walk that moves exactly cmd_vx * dt per row reads a
    speed_frac of 1.0.

    - speed_frac is the distance along the start heading over the commanded
      distance. It is None below SPEED_FRAC_MIN_M of commanded travel.
    - heading_drift_deg is the yaw change minus the commanded yaw change.
      The yaw series is unwrapped over the window, so a missed spin keeps
      its full size and sign.
    - lateral_drift_m is the displacement across the start heading.

    speed_frac, lateral drift and pelvis sway read cleanest on straight
    commands. On a turn the path curves away from the start heading. On
    strafe the lateral drift is the commanded sideways travel, and
    speed_frac is None because no forward distance is commanded.

    Every key is None with fewer than 2 scored rows.
    """
    s, e = _bounds(rec, settle_steps, fell_at)
    out = {"speed_frac": None, "heading_drift_deg": None, "lateral_drift_m": None}
    if e - s < 2:
        return out
    q = np.asarray(rec["base_quat"], dtype=np.float64)
    p = np.asarray(rec["base_pos"], dtype=np.float64)
    cmd = np.asarray(rec["cmd"], dtype=np.float64)

    yaw0 = signals.base_yaw(q[s])
    dx, dy = p[e - 1, :2] - p[s, :2]
    along = np.cos(yaw0) * dx + np.sin(yaw0) * dy
    lateral = -np.sin(yaw0) * dx + np.cos(yaw0) * dy

    commanded = dt * cmd[s + 1 : e, 0].sum()
    if abs(commanded) >= SPEED_FRAC_MIN_M:
        out["speed_frac"] = _r(along / commanded, 3)
    turn = dt * cmd[s + 1 : e, 2].sum()
    # Unwrapped over the window, so a 2 deg turn across 180 deg reads 2 deg
    # and a missed multi-turn spin keeps its full size and sign. The yaw moves
    # far less than pi per row, so the unwrap is exact.
    yaw = np.unwrap(signals.base_yaw(q[s:e]))
    drift = yaw[-1] - yaw[0] - turn
    out["heading_drift_deg"] = _r(np.degrees(drift), 2)
    out["lateral_drift_m"] = _r(lateral, 4)
    return out


def support_fractions(rec: dict, *, settle_steps: int, fell_at: int | None) -> dict:
    """Share of rows in rec[settle:fell_at] with exactly 0, 1 or 2 feet down.

    A row's count is rec["contact"] summed over the foot axis. On a biped
    the three shares sum to 1. Every key is None for an empty window.
    """
    s, e = _bounds(rec, settle_steps, fell_at)
    keys = ("support_frac_0", "support_frac_1", "support_frac_2")
    if e <= s:
        return dict.fromkeys(keys)
    down = np.asarray(rec["contact"])[s:e].astype(np.int64).sum(axis=-1)
    return {k: _r(np.mean(down == n), 3) for n, k in enumerate(keys)}


def stability_metrics(rec: dict, *, settle_steps: int, fell_at: int | None) -> dict:
    """Roll spread and pelvis sway over rec[settle:fell_at].

    roll_std_deg is the std of the roll angle, read from the body-frame
    gravity vector. It is None for an empty window.

    pelvis_sway_cm is the std of the base's lateral position across the start
    heading, after its least-squares line over the row index is removed. The
    line takes out a steady drift. A turn's curve is not a line, so on a turn
    the curve dominates this number. It is None with fewer than 3 rows.
    """
    s, e = _bounds(rec, settle_steps, fell_at)
    out = {"roll_std_deg": None, "pelvis_sway_cm": None}
    if e <= s:
        return out
    q = np.asarray(rec["base_quat"], dtype=np.float64)
    p = np.asarray(rec["base_pos"], dtype=np.float64)

    g = signals.gravity_body(q[s:e])
    roll = np.degrees(np.arcsin(np.clip(g[:, 1], -1.0, 1.0)))
    out["roll_std_deg"] = _r(roll.std(), 2)

    if e - s >= 3:
        yaw0 = signals.base_yaw(q[s])
        dx = p[s:e, 0] - p[s, 0]
        dy = p[s:e, 1] - p[s, 1]
        lat = -np.sin(yaw0) * dx + np.cos(yaw0) * dy
        t = np.arange(e - s, dtype=np.float64)
        resid = lat - np.polyval(np.polyfit(t, lat, 1), t)
        out["pelvis_sway_cm"] = _r(resid.std() * 100.0, 2)
    return out


def _mean_or_none(values, nd: int):
    return _r(np.mean(values), nd) if len(values) else None


def landing_metrics(
    rec: dict, *, sides: tuple[str, ...], settle_steps: int, fell_at: int | None
) -> dict:
    """Landing position, step length and knee angle at touchdown, over rec[settle:fell_at].

    This module has two touchdown detectors. gait_metrics ends a swing at the
    end of a clearance run over AIRBORNE_M. landing_metrics uses the contact
    rising edge, where a foot geom comes within 5 mm of the floor. Their
    counts can differ. The keys here are named landing_* so the two are never
    mixed up.

    A touchdown of foot f is a scored row t where contact[t, f] is True and
    contact[t - 1, f] is False. At the first scored row the edge compares
    with the last settle row. Each foot's first touchdown in the window is
    dropped, because it lands from the reset pose. The rest are its scored
    landings.

    Foot positions come from signals.foot_pos_base. They are relative to the
    base, in the base frame. x is forward. y is signed and left is +y, so a
    sign change shows a foot crossing the midline.

    - landing_x_m_<side> and landing_y_m_<side> are the mean foot x and y
      over the foot's scored landings.
    - step_length_m is the mean, over both feet's scored landings, of the
      landing foot's x minus the other foot's x. It needs exactly two feet.
    - knee_td_rad_<side> is the mean knee angle magnitude over the foot's
      scored landings. It takes the magnitude because the knee sign is
      robot-specific. It is None when the record has no knee_q.

    `sides` names each foot for the key suffixes, in the record's foot
    order. Every key is always present, and None when unmeasured.
    """
    contact = np.asarray(rec["contact"], dtype=bool)
    n_feet = contact.shape[1]
    if len(sides) != n_feet:
        raise ValueError(f"sides has {len(sides)} entries for {n_feet} feet")
    s, e = _bounds(rec, settle_steps, fell_at)

    q = np.asarray(rec["base_quat"], dtype=np.float64)
    base = np.asarray(rec["base_pos"], dtype=np.float64)
    feet = np.asarray(rec["foot_pos"], dtype=np.float64)
    # base_pos stands in for xpos with root_body 0, and foot_pos for
    # site_xpos with ids 0..F-1.
    pb = signals.foot_pos_base(feet, base[:, None, :], np.arange(n_feet), 0, q)
    knee = np.abs(np.asarray(rec["knee_q"], dtype=np.float64)) if "knee_q" in rec else None

    landings = []
    for f in range(n_feet):
        rows = [t for t in range(max(s, 1), e) if contact[t, f] and not contact[t - 1, f]]
        landings.append(rows[1:])

    out = {}
    for f, side in enumerate(sides):
        rows = landings[f]
        out[f"landing_x_m_{side}"] = _mean_or_none(pb[rows, f, 0], 4)
        out[f"landing_y_m_{side}"] = _mean_or_none(pb[rows, f, 1], 4)
    steps = []
    if n_feet == 2:
        for f in range(2):
            steps += [pb[t, f, 0] - pb[t, 1 - f, 0] for t in landings[f]]
    out["step_length_m"] = _mean_or_none(steps, 4)
    for f, side in enumerate(sides):
        rows = landings[f]
        out[f"knee_td_rad_{side}"] = None if knee is None else _mean_or_none(knee[rows, f], 3)
    return out


def gait_shape(
    rec: dict,
    *,
    dt: float,
    settle_steps: int,
    fell_at: int | None,
    sides: tuple[str, ...] | None = None,
) -> dict:
    """travel_metrics, support_fractions, stability_metrics and landing_metrics in one dict.

    `sides` names each foot for the per-foot keys. When it is None the feet
    are named foot0, foot1, and so on.

    Every key is always present, and None when unmeasured. This is
    gait_metrics' convention, so battery.json's shape never depends on what
    the policy did.
    """
    if sides is None:
        sides = tuple(f"foot{i}" for i in range(np.asarray(rec["contact"]).shape[1]))
    return {
        **travel_metrics(rec, dt=dt, settle_steps=settle_steps, fell_at=fell_at),
        **support_fractions(rec, settle_steps=settle_steps, fell_at=fell_at),
        **stability_metrics(rec, settle_steps=settle_steps, fell_at=fell_at),
        **landing_metrics(rec, sides=sides, settle_steps=settle_steps, fell_at=fell_at),
    }
