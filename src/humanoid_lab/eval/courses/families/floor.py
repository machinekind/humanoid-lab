"""Floor rows: a nominal row on spec.SLIPPERY_MU, set on the floor and every
foot (model_ids.friction_geom_ids says why the feet too)."""

from __future__ import annotations

from humanoid_lab.eval.courses.families.geometry_paths import (
    CIRCLE_R2_GEOMETRY,
    STRAIGHT_GEOMETRY,
    circle_r2,
    straight,
)
from humanoid_lab.eval.courses.spec import SLIPPERY_MU, CourseParams, PathCourse

FAMILY = "floor"


def courses(p: CourseParams) -> list[PathCourse]:
    nom = (p.v_nom,)
    return [
        PathCourse("straight_slippery", FAMILY, f"mu {SLIPPERY_MU:g}", straight(), nom,
                   baseline="straight_10m", friction=SLIPPERY_MU,
                   geometry=STRAIGHT_GEOMETRY),
        PathCourse("circle_r2_slippery", FAMILY, f"mu {SLIPPERY_MU:g} while turning", circle_r2(),
                   nom, baseline="circle_r2", friction=SLIPPERY_MU,
                   geometry=CIRCLE_R2_GEOMETRY),
    ]
