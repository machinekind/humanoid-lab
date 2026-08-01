"""Swing-apex and touchdown-softness KPIs from a battery rollout record.

Pure numpy: per-foot clearance and vertical velocity arrays in, a dict of
numbers out, so this is unit-tested directly in
tests/unit/test_gait_metrics.py with no env, model or checkpoint.

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

# Clearance above which a foot counts as airborne, metres. An inherited
# 5 mm ground band, unchanged.
#
# One caveat: _foot_clearance is measured
# against the RESET KEYFRAME's site height, not the floor, and the keyframes
# float the robot a few mm (4.92 mm on asimov_v1, 3.11 mm on roboto_origin).
# A planted foot therefore reads about -5 mm, so this 5 mm band sits ~10 mm
# above the floor on asimov and every apex here reads low by that offset.
# See docs/lessons/foot-clearance.md, which owns the numbers and the
# deferred fix; nothing in this module is worth re-deriving before it lands.
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
    the number; eval/battery.py passes its SETTLE_STEPS).

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
