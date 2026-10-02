"""Terrain scene assembly on a robot's MjSpec.

`attach_terrain` replaces the spec's floor plane with an arena: one
heightfield, the arena's static boxes, and four apron boxes that continue
the flat border outward. Everything it adds sits on the world body and is
named with `PREFIX`.

Contact physics. Every ground geom copies the replaced plane's contact
attributes (`CONTACT_FIELDS`). MuJoCo mixes a pair's parameters from its two
geoms. Each robot collider therefore meets the terrain with the parameters
it met the floor with. Roboto Origin's condim 1 floor against its condim 4
colliders stays a condim 4 contact. A geom added under a robot's default
class would take that class's values instead. Roboto Origin's default is
contype 1, conaffinity 0, and such a geom never pairs with its colliders.
The attributes are always copied explicitly for that reason.

MuJoCo skips contacts between geoms of one body. The arena therefore adds
no terrain-terrain pairs. Margin stays 0, because jax `put_model` refuses a
heightfield pair with a nonzero margin. Gap stays 0 too. On the C engine
and MJWarp, a gap records contacts up to that distance above the ground.
Those contacts apply no force but take slots in the warp contact pool. jax
ignores gap. The robot's bodies, joints, inertias and actuators are
untouched. Its geom ids shift. Consumers resolve
robot geoms by name. Code reads the heightfield as `m.hfield(HFIELD_ASSET)`
and `m.geom(HFIELD_GEOM)`, never by index.

Contact cap on jax. Only the jax MJX backend reads the optional
`max_contact_points` numeric. It keeps the deepest n contacts of each
condim group. A pile-up with more than n penetrating contacts loses the
shallowest ones. The C engine and MJWarp ignore it, so warp and C physics
are unchanged. The `max_geom_pairs` numeric is refused. MJX groups geom
pairs by geom types and condim. The numeric keeps the k pairs of each group
with the smallest centre distance minus both bounding radii. A box's
bounding radius is half its diagonal, so an apron's is over 10 m. Apron
pairs, and pairs with longer boxes nearby, can outrank the pair of a foot
and the tread it stands on. Roboto Origin standing 2 mm into a stair tread
of the CPU arena gets no foot-stair contact on jax at k = 48. It gets all
of it at k = 60.

Pinning `stat` keeps the free camera, znear and zfar as on the flat model.
Unpinned, the arena and its aprons set `stat.extent`. It is 159 with the
default arena, against 1.26 for Roboto Origin on its plane.
`stat.meaninertia` depends on the robot only.

This module imports mujoco. `humanoid_lab.terrain` does not import it, so
the generator package stays numpy only.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import mujoco
import numpy as np

from humanoid_lab.terrain.arena import Arena, Box

HFIELD_ASSET = "terrain"
HFIELD_GEOM = "terrain_hfield"
# Every geom attach_terrain adds starts with this.
PREFIX = "terrain_"
APRON_SIDES = ("north", "south", "east", "west")
APRON_GEOMS = tuple(f"{PREFIX}apron_{side}" for side in APRON_SIDES)
# Aprons reach this far (m) beyond the heightfield edge, tops at z = 0.
APRON_EXTENT = 25.0
APRON_HALF_Z = 0.5

CONTACT_FIELDS = (
    "contype",
    "conaffinity",
    "condim",
    "friction",
    "solref",
    "solimp",
    "solmix",
    "priority",
    "margin",
    "gap",
)
JAX_CONTACT_NUMERIC = "max_contact_points"
JAX_PAIR_NUMERIC = "max_geom_pairs"

# Arena boxes alternate between two greys so neighbouring steps read apart.
BOX_RGBA = ((0.55, 0.55, 0.55, 1.0), (0.45, 0.45, 0.45, 1.0))

# The aprons continue the heightfield at z = 0, so its outer node ring must
# be 0 to within this (m).
BORDER_TOL = 1e-6


@dataclass(frozen=True)
class GroundContact:
    """The floor plane's contact attributes, plus how it renders."""

    contype: int
    conaffinity: int
    condim: int
    friction: tuple[float, float, float]
    solref: tuple[float, float]
    solimp: tuple[float, ...]
    solmix: float
    priority: int
    margin: float
    gap: float
    material: str
    rgba: tuple[float, ...]
    group: int

    def contact_kwargs(self) -> dict:
        """`CONTACT_FIELDS` as `add_geom` keyword arguments."""
        out = {}
        for name in CONTACT_FIELDS:
            value = getattr(self, name)
            out[name] = list(value) if isinstance(value, tuple) else value
        return out


def box_geom_name(k: int) -> str:
    """Name of the geom for `arena.boxes[k]`."""
    return f"{PREFIX}box_{k}"


def floor_plane(spec: mujoco.MjSpec) -> mujoco.MjsGeom:
    """The world body's one plane geom. Raises unless there is exactly one."""
    planes = [g for g in spec.worldbody.geoms if g.type == mujoco.mjtGeom.mjGEOM_PLANE]
    if len(planes) != 1:
        raise ValueError(
            f"a terrain scene replaces the floor plane, and the world body holds "
            f"{len(planes)} plane geoms, not one"
        )
    return planes[0]


def ground_contact(plane: mujoco.MjsGeom) -> GroundContact:
    """The contact attributes of `plane`, read off the spec geom.

    A parsed spec geom already carries its default class's values. Raises on
    a nonzero margin or gap."""
    gc = GroundContact(
        contype=int(plane.contype),
        conaffinity=int(plane.conaffinity),
        condim=int(plane.condim),
        friction=tuple(float(v) for v in plane.friction),
        solref=tuple(float(v) for v in plane.solref),
        solimp=tuple(float(v) for v in plane.solimp),
        solmix=float(plane.solmix),
        priority=int(plane.priority),
        margin=float(plane.margin),
        gap=float(plane.gap),
        material=str(plane.material),
        rgba=tuple(float(v) for v in plane.rgba),
        group=int(plane.group),
    )
    if gc.margin != 0.0 or gc.gap != 0.0:
        raise ValueError(
            f"floor plane '{plane.name}' has margin {gc.margin} and gap {gc.gap}. "
            "The terrain copies both. jax put_model refuses a heightfield pair "
            "with a nonzero margin. A nonzero gap adds forceless contacts on C "
            "and warp."
        )
    return gc


def ground_pairing_geoms(spec: mujoco.MjSpec, gc: GroundContact) -> list[mujoco.MjsGeom]:
    """Robot geoms that collide with ground carrying `gc`'s contype and conaffinity."""
    world = spec.worldbody
    return [
        g
        for g in spec.geoms
        if g.parent != world
        and ((g.contype & gc.conaffinity) | (gc.contype & g.conaffinity)) != 0
    ]


def apron_boxes(x_min: float, x_max: float, y_min: float, y_max: float) -> list[Box]:
    """Four strips around the rectangle, in `APRON_SIDES` order.

    Their tops sit at z = 0. The north and south strips span the full width
    plus both corners."""
    cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
    e = APRON_EXTENT / 2
    hx = (x_max - x_min) / 2 + APRON_EXTENT
    hy = (y_max - y_min) / 2
    z = -APRON_HALF_Z
    return [
        Box((cx, y_max + e, z), (hx, e, APRON_HALF_Z)),
        Box((cx, y_min - e, z), (hx, e, APRON_HALF_Z)),
        Box((x_max + e, cy, z), (e, hy, APRON_HALF_Z)),
        Box((x_min - e, cy, z), (e, hy, APRON_HALF_Z)),
    ]


def box_quat(yaw: float) -> list[float]:
    """Unit quaternion for a rotation of `yaw` about +z. Exactly the
    identity at 0."""
    if yaw == 0.0:
        return [1.0, 0.0, 0.0, 0.0]
    return [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]


def _refuse_conflicts(spec: mujoco.MjSpec, arena: Arena) -> None:
    taken = sorted(g.name for g in spec.geoms if g.name.startswith(PREFIX))
    if taken:
        raise ValueError(f"the spec already has geoms named with '{PREFIX}': {taken}")
    if any(h.name == HFIELD_ASSET for h in spec.hfields):
        raise ValueError(f"the spec already has a heightfield named '{HFIELD_ASSET}'")
    for name in (JAX_CONTACT_NUMERIC, JAX_PAIR_NUMERIC):
        if any(n.name == name for n in spec.numerics):
            raise ValueError(f"the spec already has a numeric named '{name}'")
    using = sorted(g.name or f"#{i}" for i, g in enumerate(spec.geoms) if g.hfieldname)
    if using:
        raise ValueError(
            f"geoms {using} use a heightfield. A terrain scene holds exactly one, "
            f"'{HFIELD_ASSET}'."
        )
    ring = np.concatenate(
        [arena.lookup[0], arena.lookup[-1], arena.lookup[:, 0], arena.lookup[:, -1]]
    )
    worst = float(np.abs(ring).max())
    if worst > BORDER_TOL:
        raise ValueError(
            f"the arena's outer node ring reaches {worst:.3g} m. The aprons "
            "continue it at z = 0, so it must be flat at 0."
        )


def attach_terrain(
    spec: mujoco.MjSpec,
    arena: Arena,
    stat: tuple[float, Sequence[float]] | None = None,
    max_contact_points: int | None = None,
) -> GroundContact:
    """Replace `spec`'s floor plane with `arena`. Returns the plane's contact.

    `stat` is (extent, center) to pin on the spec, normally the flat
    model's. `max_contact_points` adds the jax-only contact cap. Raises
    before changing the spec when it already holds a name this adds, a
    heightfield in use, a `max_contact_points` or `max_geom_pairs` numeric,
    or when the arena's border is not flat at 0."""
    plane = floor_plane(spec)
    gc = ground_contact(plane)
    _refuse_conflicts(spec, arena)
    if max_contact_points is not None and int(max_contact_points) < 1:
        raise ValueError(f"max_contact_points must be positive, got {max_contact_points}")

    spec.delete(plane)
    # No geom uses a heightfield any more (checked above), so every
    # heightfield asset left is dead weight. Roboto Origin's XML ships one.
    for hfield in list(spec.hfields):
        spec.delete(hfield)

    contact = gc.contact_kwargs()
    world = spec.worldbody
    s = arena.spec
    hf = s.hfield
    spec.add_hfield(
        name=HFIELD_ASSET,
        size=[hf.radius_x, hf.radius_y, hf.elevation_z, hf.base_z],
        nrow=hf.nrow,
        ncol=hf.ncol,
        userdata=arena.hfield_data.ravel().tolist(),
    )
    world.add_geom(
        name=HFIELD_GEOM,
        type=mujoco.mjtGeom.mjGEOM_HFIELD,
        hfieldname=HFIELD_ASSET,
        pos=[(s.x_min + s.x_max) / 2, (s.y_min + s.y_max) / 2, hf.pos_z],
        material=gc.material,
        rgba=list(gc.rgba),
        group=gc.group,
        **contact,
    )
    for k, b in enumerate(arena.boxes):
        world.add_geom(
            name=box_geom_name(k),
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=list(b.pos),
            size=list(b.half),
            quat=box_quat(b.yaw),
            material="",
            rgba=list(BOX_RGBA[k % 2]),
            group=gc.group,
            **contact,
        )
    for name, b in zip(APRON_GEOMS, apron_boxes(s.x_min, s.x_max, s.y_min, s.y_max)):
        world.add_geom(
            name=name,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=list(b.pos),
            size=list(b.half),
            material=gc.material,
            rgba=list(gc.rgba),
            group=gc.group,
            **contact,
        )
    if stat is not None:
        extent, center = stat
        spec.stat.extent = float(extent)
        spec.stat.center = [float(v) for v in center]
    if max_contact_points is not None:
        spec.add_numeric(name=JAX_CONTACT_NUMERIC, data=[float(int(max_contact_points))])
    return gc


def is_terrain_model(m: mujoco.MjModel) -> bool:
    """Whether `m` holds the terrain heightfield geom."""
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, HFIELD_GEOM) >= 0


def ground_geom_ids(m: mujoco.MjModel) -> np.ndarray:
    """Sorted ids of the world body's colliding geoms.

    On a terrain model these are the heightfield, the arena boxes and the
    aprons. On a flat robot model it is the floor plane."""
    world = np.asarray(m.geom_bodyid) == 0
    collides = (np.asarray(m.geom_contype) != 0) | (np.asarray(m.geom_conaffinity) != 0)
    return np.flatnonzero(world & collides).astype(np.int32)
