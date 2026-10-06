"""Model element ids the course lanes need, from plain model arrays.

numpy only, so the rule is unit-tested on hand-made arrays.
"""

from __future__ import annotations

import numpy as np


def friction_geom_ids(geom_bodyid, contype, conaffinity, foot_ids) -> np.ndarray:
    """Geoms whose sliding friction a floor row sets: the ground and the feet.

    The ground is every colliding geom of the world body (body 0), so the
    rule needs no geom name (Roboto's floor is `ground`, Asimov's `floor`)
    and covers a heightfield and its boxes the same way. The feet are in it
    because MuJoCo combines the friction of two equal-priority geoms by
    element-wise max: lowering the floor alone leaves the feet's higher value
    in every contact. Sorted, without duplicates.
    """
    bodyid = np.asarray(geom_bodyid)
    collides = (np.asarray(contype) != 0) | (np.asarray(conaffinity) != 0)
    world = np.flatnonzero((bodyid == 0) & collides)
    feet = np.atleast_1d(np.asarray(foot_ids, dtype=np.int64))
    return np.unique(np.concatenate([world, feet])).astype(np.int32)
