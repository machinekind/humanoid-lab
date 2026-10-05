"""Geometry rows: nominal speed, dry floor, no kick. Only the shape varies.

Waypoints are in the course frame (origin at the post-settle base, +x along
its heading). Every curved shape starts behind geometry.lead_in(). The point
counts below are part of each row's definition (see geometry.py).
"""

from __future__ import annotations

import math
from types import MappingProxyType

from humanoid_lab.eval.courses.geometry import (
    LEAD_IN_M,
    arc,
    circle,
    join,
    lead_in,
    line,
    rounded_square,
    sine_slalom,
)
from humanoid_lab.eval.courses.spec import SLALOM_WAVELENGTHS, CourseParams, PathCourse

FAMILY = "geometry"

# The straight and the 2 m circle are shared with the speed, floor and
# disturbance rows, whose geometry must equal their baseline's. Those rows
# record the same shape parameters as their baseline.
STRAIGHT_M = 10.0
CIRCLE_R_M = 2.0
CIRCLE_N = 128
STRAIGHT_GEOMETRY = MappingProxyType({"length_m": STRAIGHT_M})
CIRCLE_R2_GEOMETRY = MappingProxyType({"radius_m": CIRCLE_R_M, "lead_in_m": LEAD_IN_M})

# The square's corners use r_tight, so they ask for the same share of the cap
# as every other tight turn. Each edge keeps at least SQUARE_MIN_EDGE_M of
# straight between two corners, Asimov's edge at r_tight 1.25 m. A robot whose
# r_tight is wider gets a larger side, recorded in geometry.side_m. Shrinking
# the radius instead would raise the corners' demand above R_TIGHT_DEMAND of
# the cap.
SQUARE_SIDE_M = 3.0
SQUARE_MIN_EDGE_M = 0.5


def straight():
    return line(STRAIGHT_M)


def circle_r2():
    return circle(CIRCLE_R_M, n=CIRCLE_N)


def square_side(r_tight):
    """SQUARE_SIDE_M, or the smallest side that leaves SQUARE_MIN_EDGE_M of
    edge when the corners need more. The tolerance absorbs the last-ulp
    rounding of a derived r_tight: Asimov's is 1.25 m plus one ulp."""
    need = 2 * r_tight + SQUARE_MIN_EDGE_M
    return SQUARE_SIDE_M if need <= SQUARE_SIDE_M + 1e-9 else need


def u_turn(r):
    """3 m out, a half circle of radius `r` to the left, 3 m back."""
    return lead_in(join(
        line(3.0),
        arc(r, math.pi, start=(3.0, 0.0), n=48),
        line(3.0, start=(3.0, 2 * r), heading=math.pi),
    ))


def courses(p: CourseParams) -> list[PathCourse]:
    nom = (p.v_nom,)
    lam = p.slalom_wavelength
    side = square_side(p.r_tight)
    return [
        PathCourse("straight_10m", FAMILY, "heading hold", straight(), nom,
                   geometry=STRAIGHT_GEOMETRY),
        PathCourse("arc_r3_90deg", FAMILY, "gentle curvature", lead_in(arc(3.0, math.pi / 2)), nom,
                   geometry={"radius_m": 3.0, "sweep_rad": math.pi / 2, "lead_in_m": LEAD_IN_M}),
        PathCourse("circle_r2", FAMILY, "sustained mild yaw", circle_r2(), nom,
                   geometry=CIRCLE_R2_GEOMETRY),
        PathCourse("circle_tight", FAMILY, "hard yaw", circle(p.r_tight, n=CIRCLE_N), nom,
                   geometry={"radius_m": p.r_tight, "lead_in_m": LEAD_IN_M}),
        PathCourse(
            "figure_eight_r15", FAMILY, "curvature reversal, self-crossing",
            lead_in(join(arc(1.5, 2 * math.pi, n=96), arc(1.5, -2 * math.pi, n=96))), nom,
            geometry={"radius_m": 1.5, "lead_in_m": LEAD_IN_M},
        ),
        PathCourse("square_3m", FAMILY, "90 deg corners", lead_in(rounded_square(side, p.r_tight)),
                   nom,
                   geometry={"side_m": side, "corner_radius_m": p.r_tight, "lead_in_m": LEAD_IN_M}),
        PathCourse(
            "slalom_05m", FAMILY, "alternating curvature",
            lead_in(sine_slalom(SLALOM_WAVELENGTHS * lam, p.slalom_amplitude, lam, n=192)), nom,
            geometry={"amplitude_m": p.slalom_amplitude, "wavelength_m": lam,
                      "wavelengths": SLALOM_WAVELENGTHS, "lead_in_m": LEAD_IN_M},
        ),
        PathCourse("u_turn", FAMILY, "reversal", u_turn(p.r_tight), nom,
                   geometry={"leg_m": 3.0, "radius_m": p.r_tight, "lead_in_m": LEAD_IN_M}),
    ]
