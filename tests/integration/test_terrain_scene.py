"""Terrain scene assembly: `attach_terrain` on a robot's built spec.

Plain MuJoCo unless a test says jax. Each scene is `build_spec`, then
`attach_terrain` with the CPU arena (terrain/config.py), unless a test names
the default arena.

Every ray below starts 1e-7 m off any heightfield node. A ray cast exactly
onto a node can return the wrong surface: at 4355 random nodes of the
default arena, 68 came back more than 1e-4 m off, the worst by 1.0 m, and
the 1e-7 m offset brought the worst to 1.7e-6 m.
"""

from __future__ import annotations

import copy
import dataclasses
import functools
from dataclasses import dataclass

import jax
import mujoco
import numpy as np
import pytest
from mujoco import mjx

from humanoid_lab import paths, terrain
from humanoid_lab.robot.build import build_spec, compile_spec
from humanoid_lab.robot.spec import load_robot_spec
from humanoid_lab.terrain import scene
from humanoid_lab.terrain.config import CPU_ARENA, arena_for, params_from_config

ROBOTS = ("roboto_origin", "asimov_v1")
PRESET = "deploy_pd"
# Robot geoms that pair with the floor: Roboto Origin's 15 chessboard cells
# and 14 capsules, and Asimov's 33 capsules.
GROUND_COLLIDERS = {"roboto_origin": 29, "asimov_v1": 33}
# The ray offset from every heightfield node (see the module docstring).
OFF_NODE = 1e-7

_MODEL_FIELDS = {
    "contype": "geom_contype",
    "conaffinity": "geom_conaffinity",
    "condim": "geom_condim",
    "friction": "geom_friction",
    "solref": "geom_solref",
    "solimp": "geom_solimp",
    "solmix": "geom_solmix",
    "priority": "geom_priority",
    "margin": "geom_margin",
    "gap": "geom_gap",
}


def _arena(name: str) -> terrain.Arena:
    if name == "cpu":
        return arena_for(params_from_config(CPU_ARENA))
    return arena_for(terrain.ArenaParams())


@dataclass(frozen=True, eq=False)
class Scene:
    robot: str
    arena: terrain.Arena
    flat: mujoco.MjModel
    plane: str
    spec: mujoco.MjSpec
    gc: scene.GroundContact
    model: mujoco.MjModel


@functools.cache
def _scene(robot: str, arena: str = "cpu", cap: int | None = None) -> Scene:
    robot_dir = paths.ROBOTS_DIR / robot
    flat_spec = build_spec(robot_dir, PRESET)
    plane = scene.floor_plane(flat_spec).name
    flat = compile_spec(flat_spec)
    spec = build_spec(robot_dir, PRESET)
    a = _arena(arena)
    gc = scene.attach_terrain(
        spec, a, stat=(flat.stat.extent, flat.stat.center), max_contact_points=cap
    )
    return Scene(robot, a, flat, plane, spec, gc, compile_spec(spec))


@pytest.fixture(scope="module", params=ROBOTS)
def cpu_scene(request) -> Scene:
    return _scene(request.param)


def _home(m: mujoco.MjModel, xy, dz: float) -> np.ndarray:
    """The home keyframe moved to base xy and raised by dz."""
    q = m.key("home").qpos.copy()
    b = int(m.jnt_qposadr[0])
    q[b : b + 2] = xy
    q[b + 2] += dz
    return q


def _ground_data(m: mujoco.MjModel) -> mujoco.MjData:
    """Data with the robot lifted 50 m, clear of every downward ray."""
    d = mujoco.MjData(m)
    d.qpos[:] = _home(m, (0.0, 0.0), 50.0)
    mujoco.mj_forward(m, d)
    return d


def _rays(m, d, x, y, z0=10.0):
    """Height and geom id of the first surface straight down at each (x, y)."""
    z = np.full(len(x), np.nan)
    hit = np.full(len(x), -1)
    gid = np.zeros(1, np.int32)
    down = np.array([0.0, 0.0, -1.0])
    for i, (xi, yi) in enumerate(zip(x, y)):
        dist = mujoco.mj_ray(m, d, np.array([xi, yi, z0]), down, None, 1, -1, gid)
        if dist >= 0:
            z[i], hit[i] = z0 - dist, int(gid[0])
    return z, hit


def _lookup(arena, x, y):
    return terrain.bilinear(arena.lookup, *terrain.sample_frame(arena.spec), x, y)


def _box_table(arena):
    return np.array([b.pos + b.half + (b.yaw,) for b in arena.boxes])


def _box_distances(boxes, x, y, chunk=1000):
    """Yield (slice, held, dist) over chunks of the points. held[i, k] says
    box k holds point i in xy; dist[i, k] is the distance from point i to
    box k's footprint boundary, inside or out."""
    c, s = np.cos(boxes[:, 6]), np.sin(boxes[:, 6])
    for i0 in range(0, len(x), chunk):
        sl = slice(i0, i0 + chunk)
        u = x[sl, None] - boxes[None, :, 0]
        v = y[sl, None] - boxes[None, :, 1]
        ex = np.abs(u * c + v * s) - boxes[:, 3]
        ey = np.abs(v * c - u * s) - boxes[:, 4]
        held = (ex <= 0) & (ey <= 0)
        out = np.hypot(np.maximum(ex, 0), np.maximum(ey, 0))
        yield sl, held, np.where(held, np.minimum(-ex, -ey), out)


def _box_geometry(boxes, x, y):
    """Per point: inside any box, distance to the nearest box boundary, and
    the highest top among the boxes holding it (-inf outside every box)."""
    inside = np.zeros(len(x), bool)
    edge = np.full(len(x), np.inf)
    top = np.full(len(x), -np.inf)
    tops = boxes[:, 2] + boxes[:, 5]
    for sl, held, dist in _box_distances(boxes, x, y):
        inside[sl] = held.any(1)
        edge[sl] = dist.min(1)
        top[sl] = np.where(held, tops, -np.inf).max(1)
    return inside, edge, top


# -- the scene ---------------------------------------------------------------


def test_the_floor_plane_is_replaced(cpu_scene):
    m, hf = cpu_scene.model, cpu_scene.arena.spec.hfield
    assert np.any(cpu_scene.flat.geom_type == mujoco.mjtGeom.mjGEOM_PLANE)
    assert not np.any(m.geom_type == mujoco.mjtGeom.mjGEOM_PLANE)
    assert m.nhfield == 1
    assert m.hfield(scene.HFIELD_ASSET).id == 0
    if cpu_scene.robot == "roboto_origin":
        assert cpu_scene.flat.hfield("hf0").nrow[0] == 200
    assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_HFIELD, "hf0") == -1
    assert (m.hfield_nrow[0], m.hfield_ncol[0]) == (hf.nrow, hf.ncol)
    np.testing.assert_allclose(
        m.hfield_size[0], [hf.radius_x, hf.radius_y, hf.elevation_z, hf.base_z]
    )
    np.testing.assert_array_equal(m.hfield_data, cpu_scene.arena.hfield_data.ravel())
    g = m.geom(scene.HFIELD_GEOM)
    assert m.geom_type[g.id] == mujoco.mjtGeom.mjGEOM_HFIELD
    assert m.geom_dataid[g.id] == 0
    np.testing.assert_allclose(m.geom_pos[g.id], [0.0, 0.0, hf.pos_z])


def test_ground_geoms_copy_the_floor_contact(cpu_scene):
    assert set(_MODEL_FIELDS) == set(scene.CONTACT_FIELDS)
    flat, m = cpu_scene.flat, cpu_scene.model
    plane = flat.geom(cpu_scene.plane).id
    ground = scene.ground_geom_ids(m)
    assert len(ground) == 1 + len(cpu_scene.arena.boxes) + len(scene.APRON_GEOMS)
    for field, attr in _MODEL_FIELDS.items():
        want = getattr(flat, attr)[plane]
        got = getattr(m, attr)[ground]
        np.testing.assert_array_equal(got, np.broadcast_to(want, got.shape), err_msg=field)
    assert np.all(m.geom_group[ground] == flat.geom_group[plane])
    material = flat.geom_matid[plane]
    assert material >= 0
    floor_material = flat.material(material).name
    for name in (scene.HFIELD_GEOM, *scene.APRON_GEOMS):
        assert m.material(m.geom_matid[m.geom(name).id]).name == floor_material


def test_ground_geoms_copy_contact_values_the_default_class_lacks():
    """Both robots' floors match their default class on most contact
    fields. A field the copy missed compiles to the class value, and the
    test above would still pass. Here the floor differs from the class on
    every field the copy carries except margin and gap, which stay 0."""
    spec = build_spec(paths.ROBOTS_DIR / "roboto_origin", PRESET)
    plane, default = scene.floor_plane(spec), spec.default.geom
    want = {
        "contype": 2,
        "conaffinity": 3,
        "condim": 6,
        "friction": (0.7, 0.03, 0.002),
        "solref": (0.005, 1.5),
        "solimp": (0.8, 0.9, 0.002, 0.4, 3.0),
        "solmix": 0.6,
        "priority": 2,
    }
    assert set(want) == set(scene.CONTACT_FIELDS) - {"margin", "gap"}
    for field, value in want.items():
        # A geom that missed the copy would compile to the class value.
        assert not np.allclose(getattr(default, field), value), field
        setattr(plane, field, list(value) if isinstance(value, tuple) else value)
    scene.attach_terrain(spec, _arena("cpu"))
    m = compile_spec(spec)
    ground = scene.ground_geom_ids(m)
    for field, value in want.items():
        got = getattr(m, _MODEL_FIELDS[field])[ground]
        np.testing.assert_allclose(
            got, np.broadcast_to(value, got.shape), rtol=0, atol=0, err_msg=field
        )


def test_every_robot_collider_pairs_with_the_ground(cpu_scene):
    m = cpu_scene.model
    ct, ca = m.geom_contype, m.geom_conaffinity
    robot = np.flatnonzero((m.geom_bodyid != 0) & ((ct | ca) != 0))
    ground = scene.ground_geom_ids(m)
    pairs = (ct[robot][:, None] & ca[ground][None, :]) | (ct[ground][None, :] & ca[robot][:, None])
    assert np.all(pairs != 0)
    assert len(robot) == GROUND_COLLIDERS[cpu_scene.robot]

    names = sorted(m.geom(i).name for i in robot)
    assert sorted(g.name for g in scene.ground_pairing_geoms(cpu_scene.spec, cpu_scene.gc)) == names
    # The same colliders met the floor plane on the flat model.
    flat = cpu_scene.flat
    fct, fca = flat.geom_contype, flat.geom_conaffinity
    plane = flat.geom(cpu_scene.plane).id
    on_floor = [
        i for i in range(flat.ngeom)
        if flat.geom_bodyid[i] != 0 and ((fct[i] & fca[plane]) | (fct[plane] & fca[i]))
    ]
    assert sorted(flat.geom(i).name for i in on_floor) == names


def test_ground_geom_ids_flat_is_the_plane_and_terrain_is_every_world_collider(cpu_scene):
    flat, m = cpu_scene.flat, cpu_scene.model
    assert scene.ground_geom_ids(flat).tolist() == [flat.geom(cpu_scene.plane).id]
    assert not scene.is_terrain_model(flat)
    assert scene.is_terrain_model(m)
    names = {m.geom(int(i)).name for i in scene.ground_geom_ids(m)}
    boxes = {scene.box_geom_name(k) for k in range(len(cpu_scene.arena.boxes))}
    assert names == {scene.HFIELD_GEOM, *scene.APRON_GEOMS, *boxes}
    world = {m.geom(i).name for i in range(m.ngeom) if m.geom_bodyid[i] == 0}
    assert world == names


def test_the_robot_is_unchanged_by_the_scene(cpu_scene):
    flat, m = cpu_scene.flat, cpu_scene.model
    for n in ("nbody", "njnt", "nq", "nv", "nu", "nsensor", "nkey"):
        assert getattr(m, n) == getattr(flat, n), n
    for attr in (
        "body_mass", "body_inertia", "body_ipos", "body_iquat", "body_pos", "body_quat",
        "jnt_type", "jnt_range", "jnt_qposadr", "dof_damping", "dof_armature",
        "dof_frictionloss", "actuator_gainprm", "actuator_biasprm", "actuator_forcerange",
        "actuator_ctrlrange", "actuator_trnid", "key_qpos", "sensor_adr",
    ):
        np.testing.assert_array_equal(getattr(m, attr), getattr(flat, attr), err_msg=attr)
    # Robot geoms keep their order and every attribute; only their ids shift.
    rf, rm = flat.geom_bodyid != 0, m.geom_bodyid != 0
    for attr in ("geom_bodyid", "geom_type", "geom_size", "geom_pos", "geom_quat",
                 *_MODEL_FIELDS.values()):
        np.testing.assert_array_equal(getattr(m, attr)[rm], getattr(flat, attr)[rf], err_msg=attr)
    assert m.opt.timestep == flat.opt.timestep
    assert m.opt.ccd_iterations == flat.opt.ccd_iterations


def test_render_statistics_match_the_flat_model(cpu_scene):
    flat, m = cpu_scene.flat, cpu_scene.model
    assert m.stat.extent == flat.stat.extent
    np.testing.assert_array_equal(m.stat.center, flat.stat.center)
    assert m.stat.meaninertia == flat.stat.meaninertia
    # Unpinned, the arena and its aprons set the extent.
    spec = build_spec(paths.ROBOTS_DIR / cpu_scene.robot, PRESET)
    scene.attach_terrain(spec, cpu_scene.arena)
    assert compile_spec(spec).stat.extent > 50 * flat.stat.extent


def _names(spec: mujoco.MjSpec) -> tuple:
    """The spec's geom, heightfield and numeric names."""
    return (
        sorted(g.name for g in spec.geoms),
        sorted(h.name for h in spec.hfields),
        sorted(n.name for n in spec.numerics),
    )


# One raised node per side of the outer ring. Each sits off the corners. A
# corner node lies on a row and a column, so one side's check would hide a
# missing other side.
RING_NODES = {
    "ring_first_row": (0, 5),
    "ring_last_row": (-1, 5),
    "ring_first_col": (5, 0),
    "ring_last_col": (5, -1),
}


def _refusal_case(case: str, spec: mujoco.MjSpec, a: terrain.Arena) -> tuple:
    """Apply one refusal case to `spec`. Returns attach_terrain's arena and
    keyword arguments."""
    kwargs = {}
    if case == "prefix":
        spec.worldbody.add_geom(
            name=f"{scene.PREFIX}x", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.1, 0.1, 0.1]
        )
    elif case == "hfield_name":
        spec.add_hfield(name=scene.HFIELD_ASSET, nrow=2, ncol=2, size=[1, 1, 1, 1])
    elif case == "contact_numeric":
        spec.add_numeric(name=scene.JAX_CONTACT_NUMERIC, data=[8.0])
    elif case == "pair_numeric":
        spec.add_numeric(name=scene.JAX_PAIR_NUMERIC, data=[16.0])
    elif case == "hfield_user":
        spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_HFIELD, hfieldname="hf0")
    elif case == "cap_zero":
        kwargs["max_contact_points"] = 0
    elif case == "margin":
        scene.floor_plane(spec).margin = 0.01
    elif case == "gap":
        scene.floor_plane(spec).gap = 0.01
    elif case in RING_NODES:
        lookup = a.lookup.copy()
        lookup[RING_NODES[case]] = 0.01
        a = dataclasses.replace(a, lookup=lookup)
    else:
        raise AssertionError(case)
    return a, kwargs


@pytest.mark.parametrize(
    ("case", "match"),
    [
        ("prefix", "named with 'terrain_'"),
        # Redundant while unused heightfields are deleted: this pins config
        # hygiene, not an assembly failure.
        ("hfield_name", "heightfield named 'terrain'"),
        ("contact_numeric", scene.JAX_CONTACT_NUMERIC),
        ("pair_numeric", scene.JAX_PAIR_NUMERIC),
        ("hfield_user", "use a heightfield"),
        ("cap_zero", "must be positive, got 0"),
        ("margin", "margin 0.01"),
        ("gap", "gap 0.01"),
        *((case, "outer node ring") for case in RING_NODES),
    ],
)
def test_attach_refuses_a_spec_it_cannot_assemble(case, match):
    spec = build_spec(paths.ROBOTS_DIR / "roboto_origin", PRESET)
    a, kwargs = _refusal_case(case, spec, _arena("cpu"))
    before = _names(spec)
    with pytest.raises(ValueError, match=match):
        scene.attach_terrain(spec, a, **kwargs)
    assert _names(spec) == before


@pytest.mark.parametrize("n", [0, 2])
def test_attach_refuses_a_spec_without_one_floor_plane(n):
    a = _arena("cpu")
    spec = build_spec(paths.ROBOTS_DIR / "roboto_origin", PRESET)
    if n == 0:
        scene.attach_terrain(spec, a)
    else:
        spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[1, 1, 0.1])
    before = _names(spec)
    with pytest.raises(ValueError, match=f"{n} plane geoms"):
        scene.attach_terrain(spec, a)
    assert _names(spec) == before


# -- heights: the lookup against the physics surface ---------------------------


@pytest.fixture(scope="module")
def default_scene():
    sc = _scene("roboto_origin", "default")
    return sc, _ground_data(sc.model)


def test_lookup_matches_mj_ray(default_scene):
    """The lookup (bilinear) against mj_ray on the default arena.

    - Pads read their pad height to 1e-6 m.
    - Bare heightfield more than 1.5 cells from any box. MuJoCo splits each
      cell into two triangles and bilinear blends four nodes. The two differ
      by at most a quarter of the cell's twist |h00 - h01 - h10 + h11|. The
      per-point bound is checked, not a fixed tolerance: the default arena
      peaks at 4.3 mm on the d = 1 slope creases.
    - Box interiors more than 1.5 cells from every box edge read the top to
      1e-6 m.
    - Within one cell outside an axis-aligned stair box, where the surface is
      below the box top, the lookup reads at or above it (to 1e-6 m). Within
      one cell inside an edge that falls mid-cell it reads low. That side is
      not asserted here. Exact box edges are excluded: the lookup counts an
      edge node as box top and the ray misses the box there.

    Both surfaces read the same float32 node heights. Away from the creases
    and box edges they agree to under 5e-8 m on the default arena.
    """
    sc, d = default_scene
    m, a = sc.model, sc.arena
    s = a.spec
    cell = s.params.cell_size
    hfield = m.geom(scene.HFIELD_GEOM).id
    boxes = _box_table(a)
    rng = np.random.default_rng(0)

    # Pads.
    r_pad = s.params.pad_radius - 1.5 * cell
    xs, ys, want = [], [], []
    for t in s.tiles:
        r = r_pad * np.sqrt(rng.uniform(size=20))
        th = rng.uniform(0, 2 * np.pi, 20)
        xs += [t.origin[0], *(t.origin[0] + r * np.cos(th))]
        ys += [t.origin[1], *(t.origin[1] + r * np.sin(th))]
        want += [t.pad_height] * 21
    x, y, want = np.array(xs) + OFF_NODE, np.array(ys) + OFF_NODE, np.array(want)
    z, _ = _rays(m, d, x, y)
    np.testing.assert_allclose(z, want, rtol=0, atol=1e-6)
    np.testing.assert_allclose(_lookup(a, x, y), z, rtol=0, atol=1e-6)

    # Bare heightfield and box interiors, over the whole arena.
    n = 20000
    x = rng.uniform(s.x_min, s.x_max, n) + OFF_NODE
    y = rng.uniform(s.y_min, s.y_max, n) + OFF_NODE
    inside, edge, top = _box_geometry(boxes, x, y)
    z, hit = _rays(m, d, x, y)
    h = _lookup(a, x, y)
    clear = edge > 1.5 * cell

    bare = ~inside & clear
    assert bare.sum() > 10000
    assert np.all(hit[bare] == hfield)
    x0, dx, y0, dy = terrain.sample_frame(s)
    c0 = np.floor((x[bare] - x0) / dx).astype(int)
    r0 = np.floor((y[bare] - y0) / dy).astype(int)
    grid = a.lookup.astype(float)
    twist = grid[r0, c0] - grid[r0, c0 + 1] - grid[r0 + 1, c0] + grid[r0 + 1, c0 + 1]
    err = np.abs(h[bare] - z[bare])
    assert np.all(err <= np.abs(twist) / 4 + 1e-6)
    assert np.median(err) < 1e-5

    deep = inside & clear
    assert deep.sum() > 500
    assert np.all(hit[deep] != hfield)
    np.testing.assert_allclose(z[deep], top[deep], rtol=0, atol=1e-6)
    np.testing.assert_allclose(h[deep], z[deep], rtol=0, atol=1e-6)

    # Stair low sides.
    low = []
    for t in (t for t in s.tiles if t.terrain_type in terrain.STAIR_TYPES):
        mine = boxes[(np.abs(boxes[:, 0] - t.origin[0]) < 2) & (np.abs(boxes[:, 1] - t.origin[1]) < 2)]
        assert np.all(mine[:, 6] == 0.0)
        reach = t.feature_radius + 2 * cell
        x = t.origin[0] + rng.uniform(-reach, reach, 1000) + OFF_NODE
        y = t.origin[1] + rng.uniform(-reach, reach, 1000) + OFF_NODE
        z, _ = _rays(m, d, x, y)
        h = _lookup(a, x, y)
        out = np.maximum(
            np.abs(x[:, None] - mine[None, :, 0]) - mine[:, 3],
            np.abs(y[:, None] - mine[None, :, 1]) - mine[:, 4],
        )
        below = z[:, None] < (mine[:, 2] + mine[:, 5])[None, :] - 1e-6
        side = ((out > 1e-6) & (out < cell) & below).any(1)
        low.append(h[side] - z[side])
    low = np.concatenate(low)
    assert len(low) > 1000
    assert low.min() >= -1e-6


def test_rotated_box_top_matches_ray(default_scene):
    """Points inside every yawed box on the default arena hit that box at its
    top. They reach 1 mm from its edges and corners, so a mirrored or
    misplaced yaw puts many of them outside a non-square box. The points
    more than 1.5 cells inside read the top in the lookup too."""
    sc, d = default_scene
    m, a = sc.model, sc.arena
    deep = 1.5 * a.spec.params.cell_size
    edge = 1e-3
    boxes = _box_table(a)
    rng = np.random.default_rng(1)
    ks, xs, ys = [], [], []
    for k, (cx, cy, _, hx, hy, _, yaw) in enumerate(boxes):
        if yaw == 0.0:
            continue
        u = rng.uniform(-hx + edge, hx - edge, 8)
        v = rng.uniform(-hy + edge, hy - edge, 8)
        xs += list(cx + u * np.cos(yaw) - v * np.sin(yaw))
        ys += list(cy + u * np.sin(yaw) + v * np.cos(yaw))
        ks += [k] * 8
    ks, x, y = np.array(ks), np.array(xs) + OFF_NODE, np.array(ys) + OFF_NODE
    # Keep points that no other box holds or reaches within 1 mm. Mark the
    # ones deep inside their own box and 1.5 cells from every other box.
    alone = np.ones(len(ks), bool)
    inner = np.ones(len(ks), bool)
    for sl, held, dist in _box_distances(boxes, x, y):
        own = np.zeros_like(held)
        own[np.arange(held.shape[0]), ks[sl]] = True
        alone[sl] = np.all(own | (~held & (dist > edge)), axis=1)
        inner[sl] = np.all(np.where(own, dist > deep, ~held & (dist > deep)), axis=1)
    ks, x, y, inner = ks[alone], x[alone], y[alone], inner[alone]
    assert len(np.unique(ks)) > 1000
    z, hit = _rays(m, d, x, y)
    want_geom = np.array([m.geom(scene.box_geom_name(int(k))).id for k in ks])
    np.testing.assert_array_equal(hit, want_geom)
    top = boxes[ks, 2] + boxes[ks, 5]
    np.testing.assert_allclose(z, top, rtol=0, atol=1e-6)
    assert inner.sum() > 500
    np.testing.assert_allclose(_lookup(a, x[inner], y[inner]), top[inner], rtol=0, atol=1e-6)


def test_aprons_continue_the_border():
    sc = _scene("roboto_origin")
    m, s = sc.model, sc.arena.spec
    d = _ground_data(m)
    aprons = {m.geom(name).id for name in scene.APRON_GEOMS}
    cx, cy = (s.x_min + s.x_max) / 2, (s.y_min + s.y_max) / 2
    points = []
    for gap in (0.01, 1.0, 20.0):
        for sx in (-1, 0, 1):
            for sy in (-1, 0, 1):
                if sx == sy == 0:
                    continue
                x = cx if sx == 0 else (s.x_max + gap if sx > 0 else s.x_min - gap)
                y = cy if sy == 0 else (s.y_max + gap if sy > 0 else s.y_min - gap)
                points.append((x, y))
    x, y = np.array(points).T
    z, hit = _rays(m, d, x + OFF_NODE, y + OFF_NODE)
    assert set(hit.tolist()) == aprons
    np.testing.assert_array_equal(z, 0.0)
    # The heightfield's own edge, just inside, is at 0 too.
    z, hit = _rays(m, d, np.array([s.x_max - 0.01, s.x_min + 0.01]) + OFF_NODE, np.array([cy, cy]) + OFF_NODE)
    assert np.all(hit == m.geom(scene.HFIELD_GEOM).id)
    np.testing.assert_allclose(z, 0.0, rtol=0, atol=1e-6)


# -- physics -----------------------------------------------------------------


def _stand(m: mujoco.MjModel, q: np.ndarray, steps: int) -> mujoco.MjData:
    """`steps` mj_step from qpos `q`, PD targets held at the home keyframe."""
    rs = load_robot_spec(paths.ROBOTS_DIR / "roboto_origin")
    d = mujoco.MjData(m)
    d.qpos[:] = q
    d.ctrl[:] = m.key("home").qpos[[m.joint(n).qposadr[0] for n in rs.actuated_joints]]
    for _ in range(steps):
        mujoco.mj_step(m, d)
    return d


def _tile(arena, terrain_type, row=1):
    return next(t for t in arena.spec.tiles if t.terrain_type == terrain_type and t.row == row)


def test_physics_smoke_on_a_rough_pad():
    sc = _scene("roboto_origin")
    m = sc.model
    t = _tile(sc.arena, "rough_uniform")
    d = _stand(m, _home(m, t.origin[:2], t.pad_height), 500)
    assert np.all(np.isfinite(d.qpos)) and np.all(np.isfinite(d.qvel))
    height = d.qpos[int(m.jnt_qposadr[0]) + 2] - t.pad_height
    assert 0.6 < height < 0.8
    dist = np.array([d.contact[k].dist for k in range(d.ncon)])
    assert d.ncon > 0 and dist.min() > -1e-3
    ground = set(scene.ground_geom_ids(m).tolist())
    on = {int(g) for k in range(d.ncon) for g in d.contact[k].geom if g in ground}
    assert on == {m.geom(scene.HFIELD_GEOM).id}


def test_the_contact_cap_leaves_c_physics_unchanged():
    plain, capped = _scene("roboto_origin").model, _scene("roboto_origin", cap=116).model
    assert plain.nnumeric == 0
    assert capped.numeric(scene.JAX_CONTACT_NUMERIC).data[0] == 116
    t = _tile(_arena("cpu"), "pyramid_stairs")
    q = _home(plain, t.origin[:2], t.pad_height)
    a, b = _stand(plain, q, 200), _stand(capped, q, 200)
    assert a.ncon > 0
    np.testing.assert_array_equal(a.qpos, b.qpos)
    np.testing.assert_array_equal(a.qvel, b.qvel)


def test_jax_put_model_accepts_the_cpu_arena(cpu_scene):
    mx = mjx.put_model(cpu_scene.model, impl="jax")
    assert mx.ngeom == cpu_scene.model.ngeom
    mjx.make_data(mx)


def test_jax_refuses_a_heightfield_margin_and_accepts_a_gap():
    """attach_terrain refuses both. Only the margin refusal is jax's own."""
    m = copy.copy(_scene("roboto_origin").model)
    g = m.geom(scene.HFIELD_GEOM).id
    m.geom_gap[g] = 0.01
    mjx.put_model(m, impl="jax")
    m.geom_gap[g] = 0.0
    m.geom_margin[g] = 0.01
    with pytest.raises(NotImplementedError, match="margin"):
        mjx.put_model(m, impl="jax")


def test_the_contact_cap_bounds_the_jax_contact_count():
    """jax sizes its contact arrays from every geom pair it may test. The cap
    of 4 per ground-pairing collider (116 for Roboto Origin) bounds them.
    Roboto Origin's colliders pair with the ground only and all contacts
    are condim 4, so there is one condim group and one cap."""
    sc = _scene("roboto_origin")
    cap = 4 * len(scene.ground_pairing_geoms(sc.spec, sc.gc))
    assert cap == 116
    uncapped = mjx.make_data(mjx.put_model(sc.model, impl="jax"))._impl.contact.dist.shape[0]
    capped_model = _scene("roboto_origin", cap=cap).model
    capped = mjx.make_data(mjx.put_model(capped_model, impl="jax"))._impl.contact.dist.shape[0]
    assert uncapped > 10 * cap
    assert capped == cap


def test_a_robot_on_a_stair_tread_keeps_its_foot_contacts_on_jax():
    """Roboto Origin standing 2 mm into the first tread of the pyramid stairs.

    Every foot capsule meets the tread box on jax, with the contact cap on.
    The same scene with `max_geom_pairs` 48 has no foot-stair contact at
    all. The cull ranks pairs by centre distance minus both bounding radii.
    Apron pairs and the longer steps below take every slot. At 60 every
    foot-stair contact is back. That is why attach_terrain refuses the
    numeric."""
    rs = load_robot_spec(paths.ROBOTS_DIR / "roboto_origin")
    a = _arena("cpu")
    t = _tile(a, "pyramid_stairs")
    riser = t.pad_height / t.n_steps
    tread_top = (t.n_steps - 1) * riser
    xy = (t.origin[0], t.origin[1] + a.spec.stair_platform_half + t.stair_tread / 2)

    def penetrating_foot_box(pairs):
        spec = build_spec(paths.ROBOTS_DIR / "roboto_origin", PRESET)
        scene.attach_terrain(spec, a, max_contact_points=116)
        if pairs is not None:
            spec.add_numeric(name=scene.JAX_PAIR_NUMERIC, data=[float(pairs)])
        m = compile_spec(spec)
        k = mujoco.MjData(m)
        k.qpos[:] = m.key("home").qpos
        mujoco.mj_kinematics(m, k)
        feet = [m.geom(n).id for n in rs.foot_geoms]
        sole = min(k.geom_xpos[g, 2] - m.geom_size[g, 0] for g in feet)
        q = _home(m, xy, tread_top - 0.002 - sole)
        mx = mjx.put_model(m, impl="jax")
        out = jax.jit(mjx.forward)(mx, mjx.make_data(mx).replace(qpos=q))
        geom = np.asarray(out._impl.contact.geom)
        dist = np.asarray(out._impl.contact.dist)
        boxes = {m.geom(scene.box_geom_name(i)).id for i in range(len(a.boxes))}
        hits = [
            ({g1, g2} & set(feet)).pop()
            for (g1, g2), dk in zip(geom.tolist(), dist)
            if dk < 0 and {g1, g2} & boxes and {g1, g2} & set(feet)
        ]
        return hits, set(feet)

    hits, feet = penetrating_foot_box(None)
    assert len(hits) >= 6
    assert set(hits) == feet
    assert penetrating_foot_box(48)[0] == []
    assert sorted(penetrating_foot_box(60)[0]) == sorted(hits)
