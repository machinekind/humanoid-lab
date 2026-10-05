"""Spin rows: one held [0, 0, wz] command, one full turn.

The left/right pair differs only in sign, so a policy that can turn one way
and not the other cannot hide it. The slow and fast rows differ from
spin_left only in rate. Rates are spec.SPIN_FRACS of the yaw cap, floored at
spec.SPIN_MIN_RAD_S.
"""

from __future__ import annotations

from humanoid_lab.eval.courses.spec import CourseParams, SpinCourse

FAMILY = "spin"


def courses(p: CourseParams) -> list[SpinCourse]:
    return [
        SpinCourse("spin_left", FAMILY, "yaw left", wz=p.spin_nom),
        SpinCourse("spin_right", FAMILY, "chirality", wz=-p.spin_nom, baseline="spin_left"),
        SpinCourse("spin_slow", FAMILY, "slow pivot", wz=p.spin_slow, baseline="spin_left"),
        SpinCourse("spin_fast", FAMILY, "fast pivot", wz=p.spin_fast, baseline="spin_left"),
    ]
