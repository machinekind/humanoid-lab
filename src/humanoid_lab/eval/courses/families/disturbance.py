"""Disturbance rows: one planar kick of the robot's training push.vel, to the
left of the heading, at spec.PUSH_AT_M of progress.

The kick is the disturbance the robot trains against. A linear inverted
pendulum needs a recovery step for it: the capture point moves
dv sqrt(z_com / g), 0.129 m on Roboto (COM 0.651 m at the home keyframe),
1.8 stance half-widths. The fixed side keeps the two rows different only in
speed.
"""

from __future__ import annotations

from humanoid_lab.eval.courses.families.geometry_paths import (
    STRAIGHT_GEOMETRY,
    straight,
)
from humanoid_lab.eval.courses.spec import PUSH_AT_M, CourseParams, PathCourse

FAMILY = "disturbance"


def courses(p: CourseParams) -> list[PathCourse]:
    kick = f"{p.push_vel:g} m/s kick at {PUSH_AT_M:g} m"
    return [
        PathCourse("straight_push", FAMILY, kick, straight(), (p.v_nom,),
                   baseline="straight_10m", push_at_m=PUSH_AT_M, push_vel=p.push_vel,
                   geometry=STRAIGHT_GEOMETRY),
        PathCourse("straight_push_fast", FAMILY, "the kick at speed", straight(), (p.v_fast,),
                   baseline="straight_fast", push_at_m=PUSH_AT_M, push_vel=p.push_vel,
                   geometry=STRAIGHT_GEOMETRY),
    ]
