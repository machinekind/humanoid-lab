"""The critic's height scan, `height_scan_clean`, in the joystick and
terrain envs. The default-list test also covers the sizing env.

roboto_origin under deploy_pd, and under sizing_ideal for the sizing
task. The terrain env runs on the CPU arena, jax on CPU. Scan tests pose
the robot with mjx.forward and swap the terrain env's ground through
`_tables._replace`, so they need no rollout.
"""

from __future__ import annotations

import functools
import math

import jax
import jax.numpy as jp
import numpy as np
import pytest
from mujoco import mjx

from humanoid_lab import paths
from humanoid_lab.envs import height_scan as hs
from humanoid_lab.envs import terrain_geometry as tg
from humanoid_lab.envs.joystick import default_config as joystick_default_config
from humanoid_lab.registry import make_env
from humanoid_lab.terrain import Box
from humanoid_lab.terrain.config import CPU_ARENA

ROBOT_DIR = paths.ROBOTS_DIR / "roboto_origin"
PRESETS = {"joystick": "deploy_pd", "terrain": "deploy_pd", "sizing": "sizing_ideal"}
# A point on the CPU arena's flat row. The scan placed there at any yaw
# lies on the row.
FLAT_XY = (-3.1, -2.3)
JOYSTICK_CRITIC = tuple(joystick_default_config().obs.privileged)
WITH_SCAN = {"obs": {"privileged": [*JOYSTICK_CRITIC, hs.NAME]}}
# Every task's catalog under its default lists, in order.
DEFAULT_CATALOG = [
    "gyro",
    "gravity",
    "joint_pos",
    "joint_vel",
    "last_action",
    "linvel",
    "height",
    "actuator_force",
    "contacts",
    "command",
    "phase",
]


def build(task, overrides=None):
    """The task's env with `overrides` merged, the terrain env on the CPU
    arena."""
    overrides = dict(overrides or {})
    if task == "terrain":
        overrides["terrain"] = {"arena": CPU_ARENA, **overrides.get("terrain", {})}
    return make_env(task, ROBOT_DIR, PRESETS[task], overrides)


@pytest.fixture(scope="module")
def flat_scan():
    return build("joystick", WITH_SCAN)


@pytest.fixture(scope="module")
def terrain_scan():
    return build("terrain", WITH_SCAN)


@functools.cache
def _forward_fn(e):
    return jax.jit(lambda q: mjx.forward(e.mjx_model, e._make_data().replace(qpos=q, ctrl=e._neutral_ctrl)))


def posed(e, xy=FLAT_XY, dz=0.0, yaw=0.0, q=None):
    """Forward data with the base at `xy`, raised by dz and turned by yaw.
    The joints come from `q`, the reset qpos by default."""
    b = e._base_qadr
    q = e._reset_qpos if q is None else q
    q = q.at[b : b + 2].set(jp.asarray(xy, jp.float32)).at[b + 2].add(dz)
    q = q.at[b + 3 : b + 7].set(tg.quat_mul(tg.yaw_quat(jp.float32(yaw)), e._reset_qpos[b + 3 : b + 7]))
    return _forward_fn(e)(q)


def with_boxes(e, boxes):
    """A copy of `e` whose ground is 0 except under `boxes` (terrain.Box),
    with no flat band. The lookup holds the boxes rasterized onto its
    nodes, as the generator writes them."""
    t = e._tables
    x0, dx, y0, dy = t.frame
    nr, nc = t.lookup.shape
    xs, ys = np.meshgrid(x0 + dx * np.arange(nc), y0 + dy * np.arange(nr))
    lookup = np.zeros((nr, nc))
    for b in boxes:
        c, s = math.cos(b.yaw), math.sin(b.yaw)
        ox, oy = xs - b.pos[0], ys - b.pos[1]
        inside = (np.abs(ox * c + oy * s) <= b.half[0] + 1e-9) & (np.abs(oy * c - ox * s) <= b.half[1] + 1e-9)
        lookup[inside] = np.maximum(lookup[inside], b.pos[2] + b.half[2])
    cell, rows = tg.box_tables(boxes, t.frame, (nr, nc))
    clone = object.__new__(type(e))
    clone.__dict__.update(e.__dict__)
    clone._tables = t._replace(
        lookup=jp.asarray(lookup, jp.float32),
        heights=jp.zeros_like(t.heights),
        cell_boxes=jp.asarray(cell),
        boxes=jp.asarray(rows),
        flat_band=tg.EMPTY_BAND,
    )
    return clone


def sole_z(e, data):
    """Each foot geom's sole z, in numpy: centre z less radius."""
    ids = np.asarray(e._foot_geom_ids)
    return np.asarray(data.geom_xpos)[ids, 2] - np.asarray(e._foot_geom_radius)


def flat_floor_scan(e, data):
    """The flat floor's scan, in numpy: minus the lowest sole's z,
    clipped."""
    return np.full(hs.SIZE, np.clip(np.float32(0.0) - sole_z(e, data).min(), -hs.CLIP, hs.CLIP), np.float32)


@pytest.mark.parametrize("task", ["joystick", "terrain"])
def test_privileged_width_is_joystick_plus_the_scan(task, flat_scan, terrain_scan):
    """The scan adds its 98 values after joystick's critic list and
    changes nothing else. The terrain env spawns on its terrain row."""
    level = {"terrain": {"spawn": {"level": 1}}} if task == "terrain" else {}
    e = flat_scan if task == "joystick" else build(task, {**WITH_SCAN, **level})
    plain = build(task, level)
    key = jax.random.PRNGKey(3)
    s = jax.jit(e.reset)(key)
    p = jax.jit(plain.reset)(key)
    priv = np.asarray(s.obs["privileged_state"])
    joystick_width = np.asarray(p.obs["privileged_state"]).shape[0]
    assert joystick_width == sum(plain.obs_component_sizes()[n] for n in JOYSTICK_CRITIC)
    assert priv.shape == (joystick_width + hs.SIZE,)
    np.testing.assert_array_equal(priv[:joystick_width], p.obs["privileged_state"])
    np.testing.assert_array_equal(s.obs["state"], p.obs["state"])

    slices = e.obs_slices("privileged")
    assert list(slices)[-1] == hs.NAME
    assert slices[hs.NAME] == slice(joystick_width, joystick_width + hs.SIZE)
    np.testing.assert_allclose(priv[slices[hs.NAME]], e._height_scan_clean(s.data), atol=1e-6)

    catalog = e._obs_catalog(s.data, s.info)
    assert list(catalog) == [*DEFAULT_CATALOG[:-2], hs.NAME, *DEFAULT_CATALOG[-2:]]
    assert {k: int(v.shape[0]) for k, v in catalog.items()} == e.obs_component_sizes()


@pytest.mark.parametrize("task", ["joystick", "terrain"])
def test_the_actor_scan_is_refused(task):
    state = [*joystick_default_config().obs.state, hs.NAME]
    with pytest.raises(ValueError, match="obs.state names 'height_scan_clean'.*obs.privileged only"):
        build(task, {"obs": {"state": state}})


def test_the_flat_env_serves_the_flat_floor_scan(flat_scan):
    """Every value is minus the lowest sole's z, clipped. At the reset pose
    the soles float 3.1 mm."""
    scans = []
    for xy, dz, yaw in [((0.0, 0.0), 0.0, 0.0), ((5.0, -2.0), 0.2, 1.1), ((-1.0, 3.0), 0.8, -2.5)]:
        data = posed(flat_scan, xy, dz, yaw)
        scan = np.asarray(flat_scan._height_scan_clean(data))
        assert scan.shape == (hs.SIZE,) and scan.dtype == np.float32
        np.testing.assert_array_equal(scan, flat_floor_scan(flat_scan, data))
        scans.append(float(scan[0]))
    assert scans[0] == pytest.approx(-0.0031, abs=2e-4)
    assert scans[1] == pytest.approx(-0.2031, abs=2e-4)
    assert scans[2] == -hs.CLIP

    # A bent left knee lifts the left sole, and the scan stays on the right.
    knee = int(flat_scan._mj_model.joint("left_knee_joint").qposadr[0])
    data = posed(flat_scan, (0.0, 0.0), q=flat_scan._reset_qpos.at[knee].add(0.4))
    soles = sole_z(flat_scan, data)
    assert soles.max() - soles.min() > 0.02
    scan = np.asarray(flat_scan._height_scan_clean(data))
    np.testing.assert_array_equal(scan, flat_floor_scan(flat_scan, data))
    assert float(scan[0]) == pytest.approx(-0.0031, abs=2e-4)


def test_the_terrain_scan_on_the_flat_row_equals_the_flat_env_scan(terrain_scan, flat_scan):
    for dz, yaw in [(0.0, 0.0), (0.004, 0.4), (0.3, -2.0)]:
        points = hs.world_xy(hs.body_grid(), np.asarray(FLAT_XY), yaw)
        assert bool(np.all(tg.in_band(terrain_scan._tables, points)))
        t = np.asarray(terrain_scan._height_scan_clean(posed(terrain_scan, dz=dz, yaw=yaw)))
        f = np.asarray(flat_scan._height_scan_clean(posed(flat_scan, dz=dz, yaw=yaw)))
        np.testing.assert_array_equal(t.view(np.uint32), f.view(np.uint32))


def test_the_scan_sees_a_riser_ahead_at_the_right_index(terrain_scan):
    """A riser 0.41 m ahead shows from row ix = 9 (0.5 m) on, across every
    column. Row 8 (0.4 m) sits mid-cell, a quarter lookup cell (0.01 m)
    short of the face. The CPU arena's lookup cell is 0.04 m. The scan
    reads the exact ground there, where the lookup reads half the riser.

    Turned 90 degrees, a riser 0.15 m to the robot's left shows from
    column iy = 5 (0.2 m) on, along every row. Taller than CLIP, it reads
    CLIP."""
    e = terrain_scan
    x0, dx, _, _ = e._tables.frame
    node = x0 + dx * round((2.4 - x0) / dx)
    base = np.array([node + dx / 2 - 0.4, 1.0])
    face = node + 0.75 * dx
    top = 0.12
    ahead = with_boxes(e, [Box((face + 0.5, 1.0, top / 2), (0.5, 0.5, top / 2))])
    data = posed(e, base)
    ref = float(sole_z(e, data).min())
    scan = np.asarray(ahead._height_scan_clean(data)).reshape(hs.NX, hs.NY)
    rows = np.where(np.arange(hs.NX) >= 9, top, 0.0) - ref
    np.testing.assert_allclose(scan, np.repeat(rows[:, None], hs.NY, axis=1), atol=1e-6)
    row8 = hs.world_xy(hs.body_grid(), base, 0.0)[8 * hs.NY : 9 * hs.NY]
    np.testing.assert_allclose(tg.height(ahead._tables, jp.asarray(row8, jp.float32)), top / 2, atol=1e-5)

    tall = 0.8
    # At yaw pi / 2 the robot's left is world -x, and its heading world +y.
    left = with_boxes(e, [Box((base[0] - 0.15 - 0.5, base[1] + 0.25, tall / 2), (0.5, 1.0, tall / 2))])
    data = posed(e, base, yaw=math.pi / 2)
    ref = float(sole_z(e, data).min())
    scan = np.asarray(left._height_scan_clean(data)).reshape(hs.NX, hs.NY)
    cols = np.clip(np.where(np.arange(hs.NY) >= 5, tall, 0.0) - ref, -hs.CLIP, hs.CLIP)
    np.testing.assert_allclose(scan, np.repeat(cols[None, :], hs.NX, axis=0), atol=1e-6)
    assert float(scan.max()) == hs.CLIP


@pytest.mark.parametrize("task", ["joystick", "sizing", "terrain"])
def test_default_lists_leave_the_catalog_keys_unchanged(task):
    """Under the default lists no env builds the scan, and the catalog is
    the one every task served before it. The sizes still list the scan."""
    e = build(task)
    assert hs.NAME not in e._config.obs.privileged
    assert not e._scan_listed
    catalog = e._obs_catalog(e._make_data(), e._catalog_probe_info())
    assert list(catalog) == DEFAULT_CATALOG
    sizes = e.obs_component_sizes()
    assert sizes[hs.NAME] == hs.SIZE
    assert {k: int(v.shape[0]) for k, v in catalog.items()} == {k: n for k, n in sizes.items() if k != hs.NAME}
