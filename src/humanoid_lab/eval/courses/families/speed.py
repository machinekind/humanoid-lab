"""Speed rows: the geometry of a nominal row, only the commanded speed changes."""

from __future__ import annotations

import numpy as np

from humanoid_lab.eval.courses.families.geometry_paths import (
    CIRCLE_R2_GEOMETRY,
    STRAIGHT_GEOMETRY,
    STRAIGHT_M,
    circle_r2,
    straight,
)
from humanoid_lab.eval.courses.spec import CourseParams, PathCourse

FAMILY = "speed"


def courses(p: CourseParams) -> list[PathCourse]:
    # speed_steps_straight cuts the shared straight into one equal block per
    # speed step, four blocks of 2.5 m. It is the only row whose command
    # changes along the path.
    n = len(p.speed_steps)
    x = np.linspace(0.0, STRAIGHT_M, n + 1)
    return [
        PathCourse("straight_slow", FAMILY, "slow gait", straight(), (p.v_slow,),
                   baseline="straight_10m", geometry=STRAIGHT_GEOMETRY),
        PathCourse("straight_fast", FAMILY, "fast gait", straight(), (p.v_fast,),
                   baseline="straight_10m", geometry=STRAIGHT_GEOMETRY),
        PathCourse("circle_r2_fast", FAMILY, "yaw at speed", circle_r2(), (p.v_fast,),
                   baseline="circle_r2", geometry=CIRCLE_R2_GEOMETRY),
        PathCourse("speed_steps_straight", FAMILY, "speed changes mid-path",
                   np.stack([x, np.zeros_like(x)], axis=-1), tuple(p.speed_steps),
                   baseline="straight_10m",
                   geometry={**STRAIGHT_GEOMETRY, "block_m": STRAIGHT_M / n}),
    ]
