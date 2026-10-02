"""The model-free parts of terrain scene assembly: aprons, box quaternions,
the names the scene adds, and which geoms pair with the ground.
"""

from __future__ import annotations

import dataclasses
import math

import mujoco
import numpy as np

from humanoid_lab.terrain import scene

# A rectangle off the origin, so a strip placed about (0, 0) instead of the
# rectangle's centre shows.
X_MIN, X_MAX, Y_MIN, Y_MAX = -3.0, 5.0, -2.0, 1.0


def _covers(box, x, y):
    (cx, cy, _), (hx, hy, _) = box.pos, box.half
    return (np.abs(x - cx) <= hx + 1e-12) & (np.abs(y - cy) <= hy + 1e-12)


def test_apron_strips_cover_the_corners_and_sit_at_zero():
    strips = scene.apron_boxes(X_MIN, X_MAX, Y_MIN, Y_MAX)
    assert len(strips) == len(scene.APRON_SIDES) == 4
    for b in strips:
        assert b.yaw == 0.0
        assert b.pos[2] + b.half[2] == 0.0
        assert b.half[2] == scene.APRON_HALF_Z

    north, south, east, west = strips
    assert north.pos[1] - north.half[1] == Y_MAX
    assert south.pos[1] + south.half[1] == Y_MIN
    assert east.pos[0] - east.half[0] == X_MAX
    assert west.pos[0] + west.half[0] == X_MIN
    e = scene.APRON_EXTENT
    assert north.pos[1] + north.half[1] == Y_MAX + e
    assert south.pos[1] - south.half[1] == Y_MIN - e
    assert east.pos[0] + east.half[0] == X_MAX + e
    assert west.pos[0] - west.half[0] == X_MIN - e
    for b in (north, south):  # the full width plus both corners
        assert (b.pos[0] - b.half[0], b.pos[0] + b.half[0]) == (X_MIN - e, X_MAX + e)
    for b in (east, west):  # stop at the rectangle and leave the corners to N/S
        assert (b.pos[1] - b.half[1], b.pos[1] + b.half[1]) == (Y_MIN, Y_MAX)

    rng = np.random.default_rng(0)
    x = rng.uniform(X_MIN - e, X_MAX + e, 20000)
    y = rng.uniform(Y_MIN - e, Y_MAX + e, 20000)
    # The four corner squares by name, and random points over the frame.
    corners = np.array(
        [(X_MIN - e / 2, Y_MIN - e / 2), (X_MIN - e / 2, Y_MAX + e / 2),
         (X_MAX + e / 2, Y_MIN - e / 2), (X_MAX + e / 2, Y_MAX + e / 2)]
    )
    x = np.concatenate([x, corners[:, 0]])
    y = np.concatenate([y, corners[:, 1]])
    inside = (x > X_MIN) & (x < X_MAX) & (y > Y_MIN) & (y < Y_MAX)
    hits = np.sum([_covers(b, x, y) for b in strips], axis=0)
    assert np.all(hits[~inside] >= 1), "a point of the frame lies on no strip"
    assert np.all(hits[inside] == 0), "a strip reaches into the rectangle"


def test_box_quat_is_yaw_only_and_identity_at_zero():
    assert scene.box_quat(0.0) == [1.0, 0.0, 0.0, 0.0]
    for yaw in (0.3, -0.7, 1.0, math.pi / 2, 2.9):
        w, x, y, z = scene.box_quat(yaw)
        assert x == 0.0 and y == 0.0
        assert math.isclose(w * w + z * z, 1.0, rel_tol=0, abs_tol=1e-15)
        # The rotation angle about +z, and the sense: +yaw turns +x towards +y.
        assert math.isclose(2 * math.atan2(z, w), yaw, abs_tol=1e-12)


def test_added_names_share_the_prefix():
    added = [scene.HFIELD_GEOM, *scene.APRON_GEOMS, *(scene.box_geom_name(k) for k in range(5000))]
    assert all(name.startswith(scene.PREFIX) for name in added)
    assert len(set(added)) == len(added)
    assert scene.APRON_GEOMS == tuple(f"{scene.PREFIX}apron_{s}" for s in scene.APRON_SIDES)


def test_contact_kwargs_carry_exactly_the_contact_fields():
    gc = scene.GroundContact(
        contype=1, conaffinity=15, condim=1, friction=(0.9, 0.2, 0.2),
        solref=(0.001, 2.0), solimp=(0.9, 0.95, 0.001, 0.5, 2.0), solmix=1.0,
        priority=0, margin=0.0, gap=0.0, material="matplane",
        rgba=(0.5, 0.5, 0.5, 1.0), group=0,
    )
    kwargs = gc.contact_kwargs()
    assert tuple(kwargs) == scene.CONTACT_FIELDS
    assert kwargs["friction"] == [0.9, 0.2, 0.2]
    assert kwargs["solref"] == [0.001, 2.0]


def test_ground_pairing_reads_both_directions_and_skips_world_geoms():
    """Roboto Origin's colliders are contype 1, conaffinity 0 against a 1/15
    floor. Asimov's are 1/1 against a 1/1 floor. On both robots, the
    collider's contype against the floor's conaffinity gives the full set.
    A rule that dropped the other term would still pass on them."""
    plane, box = mujoco.mjtGeom.mjGEOM_PLANE, mujoco.mjtGeom.mjGEOM_BOX
    spec = mujoco.MjSpec()
    world = spec.worldbody
    world.add_geom(name="floor", type=plane, size=[1, 1, 0.1], contype=1, conaffinity=15)
    # A world geom that pairs with the floor is still not a robot collider.
    world.add_geom(name="world_box", type=box, size=[0.1, 0.1, 0.1], contype=1, conaffinity=1)
    body = world.add_body(name="robot")
    roles = {"both": (1, 1), "by_type": (1, 0), "by_affinity": (0, 1),
             "neither": (0, 0), "other_bit": (2, 0)}
    for name, (contype, conaffinity) in roles.items():
        body.add_geom(name=name, type=box, size=[0.1, 0.1, 0.1],
                      contype=contype, conaffinity=conaffinity)

    gc = scene.ground_contact(scene.floor_plane(spec))
    assert {g.name for g in scene.ground_pairing_geoms(spec, gc)} == {
        "both", "by_type", "by_affinity", "other_bit"
    }
    # 'other_bit' pairs only through the floor's conaffinity bit 2.
    narrow = dataclasses.replace(gc, conaffinity=1)
    assert {g.name for g in scene.ground_pairing_geoms(spec, narrow)} == {
        "both", "by_type", "by_affinity"
    }
