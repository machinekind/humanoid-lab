"""TerrainJoystick (task=terrain): construction, terrain-relative
measurements, spawns, reset and step.

roboto_origin under deploy_pd on the CPU arena, on jax, unless a test says
otherwise. Measurement tests pose geoms directly in the data, so they need
no rollout. Lookups swap in through `_tables._replace`, so a test can put
any surface under the robot.
"""

from __future__ import annotations

import functools
import math

import jax
import jax.numpy as jp
import mujoco
import numpy as np
import pytest
from mujoco import mjx
from mujoco_playground._src import mjx_env

from humanoid_lab import paths
from humanoid_lab.envs import curriculum, height_scan
from humanoid_lab.envs import terrain_geometry as tg
from humanoid_lab.envs.terrain_joystick import (
    JAX_BOX_LIMIT,
    TERRAIN_METRICS,
    TerrainJoystick,
)
from humanoid_lab.envs.terrain_joystick import default_config as terrain_default_config
from humanoid_lab.eval.terrain_suite import ROBOTO_SUITE
from humanoid_lab.registry import _apply_overrides, make_env
from humanoid_lab.terrain import TYPES, Box, scene
from humanoid_lab.terrain.config import CPU_ARENA

ROBOT = "roboto_origin"
ROBOT_DIR = paths.ROBOTS_DIR / ROBOT
PRESET = "deploy_pd"
CPU = {"arena": CPU_ARENA}
# A point on the CPU arena's flat row, away from its edges.
FLAT_XY = (-3.1, -2.3)
# The info keys a terrain reset adds: (shape, dtype). terrain_rng is a raw
# uint32 key pair.
TERRAIN_INFO = {
    "terrain_type": ((), np.dtype("int32")),
    "terrain_level": ((), np.dtype("int32")),
    "spawn_kind": ((), np.dtype("int32")),
    "since_spawn": ((), np.dtype("int32")),
    "curriculum_strikes": ((), np.dtype("int32")),
    "spawn_xy": ((2,), np.dtype("float32")),
    "last_xy": ((2,), np.dtype("float32")),
    "tile_origin": ((2,), np.dtype("float32")),
    "cheby_min": ((), np.dtype("float32")),
    "cheby_max": ((), np.dtype("float32")),
    "commanded_dist": ((), np.dtype("float32")),
    "served_dist": ((), np.dtype("float32")),
    "curriculum_free": ((), np.dtype("bool")),
    "terrain_rng": ((2,), np.dtype("uint32")),
}
# The keys a step leaves as they are. Only a reset writes them.
SPAWN_INFO = (
    "terrain_rng",
    "terrain_type",
    "terrain_level",
    "spawn_xy",
    "spawn_kind",
    "tile_origin",
    "curriculum_strikes",
    "curriculum_free",
)


def build(robot=ROBOT, env=None, terrain=None, cls=None):
    """The terrain env on the CPU arena with `terrain` merged into its block."""
    overrides = {**(env or {}), "terrain": {**CPU, **(terrain or {})}}
    if cls is None:
        return make_env("terrain", paths.ROBOTS_DIR / robot, PRESET, overrides)
    cfg = terrain_default_config()
    _apply_overrides(cfg, overrides)
    return cls(paths.ROBOTS_DIR / robot, PRESET, cfg)


@pytest.fixture(scope="module")
def env():
    return build()


@pytest.fixture(scope="module")
def flat():
    return make_env("joystick", ROBOT_DIR, PRESET)


@functools.cache
def _forward_fn(e):
    return jax.jit(lambda q: mjx.forward(e.mjx_model, e._make_data().replace(qpos=q, ctrl=e._neutral_ctrl)))


def posed(e, xy=FLAT_XY, dz=0.0, yaw=0.0):
    """Forward data with the base at `xy`, raised by dz and turned by yaw."""
    b = e._base_qadr
    q = e._reset_qpos.at[b : b + 2].set(jp.asarray(xy)).at[b + 2].add(dz)
    q = q.at[b + 3 : b + 7].set(tg.quat_mul(tg.yaw_quat(jp.float32(yaw)), e._reset_qpos[b + 3 : b + 7]))
    return _forward_fn(e)(q)


def with_tables(e, **fields):
    """A shallow copy of `e` that reads replaced tables."""
    clone = object.__new__(type(e))
    clone.__dict__.update(e.__dict__)
    clone._tables = e._tables._replace(**fields)
    return clone


def with_ground(e, grid, **fields):
    """A copy of `e` whose lookup and heightfield grid are both `grid`, with
    no box."""
    t = e._tables
    no_box = jp.full_like(t.cell_boxes, t.boxes.shape[0] - 1)
    return with_tables(e, lookup=grid, heights=grid, cell_boxes=no_box, **fields)


def with_boxes(e, boxes):
    """A copy of `e` whose ground is 0 under `boxes` (terrain.Box), off the
    flat band. The lookup holds the boxes rasterized onto its nodes, as the
    generator writes them."""
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
    return with_tables(
        e,
        lookup=jp.asarray(lookup, jp.float32),
        heights=jp.zeros_like(t.heights),
        cell_boxes=jp.asarray(cell),
        boxes=jp.asarray(rows),
        flat_band=tg.EMPTY_BAND,
    )


def plane(z_of_xy):
    """A lookup grid over the CPU arena's frame with height z_of_xy(x, y)."""
    def build_grid(t):
        x0, dx, y0, dy = t.frame
        nr, nc = t.lookup.shape
        xs, ys = np.meshgrid(x0 + dx * np.arange(nc), y0 + dy * np.arange(nr))
        return jp.asarray(z_of_xy(xs, ys), jp.float32)

    return build_grid


def set_geoms(data, ids, pos=None, mat=None):
    xpos, xmat = data.geom_xpos, data.geom_xmat
    if pos is not None:
        xpos = xpos.at[ids].set(jp.asarray(pos, jp.float32))
    if mat is not None:
        xmat = xmat.at[ids].set(jp.asarray(mat, jp.float32))
    return data.replace(geom_xpos=xpos, geom_xmat=xmat)


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def lay_foot(e, data, foot, centre, pitch, yaw=0.0, lift=0.0, ground=None):
    """Pose `foot`'s capsules level across, pitched by `pitch` along their
    axis, the middle capsule's centre at `centre` (xy). Each capsule rests
    on `ground(x, y)` (a plane through its centre point) plus `lift`, one
    value for all or one per capsule. Every other foot geom goes 1 m up."""
    ids = np.asarray(e._foot_geom_ids)
    owner = np.asarray(e._foot_geom_foot_idx)
    r = float(e._foot_geom_radius[0])
    R = rot_z(yaw) @ rot_y(-pitch)  # +x axis tilts up by pitch
    mine = ids[owner == foot]
    lifts = np.broadcast_to(np.asarray(lift, float), (len(mine),))
    side = R @ np.array([0.0, 1.0, 0.0])
    pos = np.array(data.geom_xpos)[ids]
    pos[:, 2] += 1.0
    mats = np.array(data.geom_xmat)[ids]
    for k, gid in enumerate(mine):
        off = (k - (len(mine) - 1) / 2) * 0.017
        c = np.array([centre[0], centre[1], 0.0]) + off * side
        c[2] = ground(c[0], c[1]) + r / math.cos(pitch) + lifts[k]
        pos[ids == gid] = c
        mats[ids == gid] = R @ rot_y(math.pi / 2)  # capsule axis (local z) along +x
    return set_geoms(data, ids, pos, mats)


# -- construction --------------------------------------------------------------


def test_actor_obs_match_the_joystick_env(env, flat):
    assert env.actor_obs_names == flat.actor_obs_names
    assert env.obs_slices("state") == flat.obs_slices("state")
    assert env.obs_slices("privileged") == flat.obs_slices("privileged")


@pytest.mark.parametrize("which", ["joystick", "terrain"])
def test_obs_component_sizes_match_the_catalog(which, env, flat):
    """Every catalog entry has its size. The sizes also list the height
    scan, which the catalog builds only when a list names it."""
    e = env if which == "terrain" else flat
    data = posed(e)
    catalog = e._obs_catalog(data, e._catalog_probe_info())
    sizes = e.obs_component_sizes()
    assert set(sizes) - set(catalog) == {height_scan.NAME}
    assert {k: int(v.shape[0]) for k, v in catalog.items()} == {k: sizes[k] for k in catalog}
    for which_list in ("state", "privileged"):
        slices = e.obs_slices(which_list)
        assert list(slices) == list(e._config.obs[which_list])
        assert max(s.stop for s in slices.values()) == sum(
            e.obs_component_sizes()[n] for n in e._config.obs[which_list]
        )


def test_sample_points_cover_every_ground_collider(env):
    """190 points: 8 corners on each of the 15 chessboard cells and 5 axis
    points on each of the 14 capsules, 30 of them on the foot capsules."""
    m = env.mj_model
    owner = env._spawn_owner
    assert env._spawn_u.shape == (190, 2)
    assert env._sole_idx.size == 30
    colliders = [
        g
        for g in range(m.ngeom)
        if m.geom_bodyid[g] != 0 and (m.geom_contype[g] or m.geom_conaffinity[g])
    ]
    assert sorted(set(owner.tolist())) == colliders
    counts = {g: int((owner == g).sum()) for g in colliders}
    for g, n in counts.items():
        want = 8 if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX else 5
        assert n == want, mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g)
    assert float(env._spawn_lift.min()) == 0.0
    assert env._feet_reach == pytest.approx(0.1498, abs=1e-3)


def test_footprint_reach_is_within_the_suite_constant(env):
    """The scan suite's footprint bounds the reach the env measures, within
    0.01 m. It sets r_out and keeps every start on the pad."""
    assert env._feet_reach <= ROBOTO_SUITE.footprint_reach
    assert ROBOTO_SUITE.footprint_reach - env._feet_reach < 0.01


def _edited(edit):
    class Edited(TerrainJoystick):
        def _customize_spec(self, spec):
            edit(spec)
            super()._customize_spec(spec)

    return Edited


def _no_termination_colliders(spec):
    for g in spec.geoms:
        if g.parent.name in ("base_link", "torso_link"):
            g.contype = 0
            g.conaffinity = 0


def _add_geom(**kw):
    def edit(spec):
        spec.body("left_knee_link").add_geom(name="extra", size=[0.03, 0.03, 0.03], **kw)

    return edit


@pytest.mark.parametrize(
    "case",
    [
        "mode",
        "pad_fit",
        "no_termination_geoms",
        "bad_collider_type",
        "unpaired_collider",
        "naccdmax_above_naconmax",
        "init_level_frac_above_one",
        "warp_without_budgets",
        "bias_out_of_box",
        "feature_candidates_zero",
        "crowded_arena",
    ],
)
def test_construction_refusals(case):
    if case == "mode":
        with pytest.raises(ValueError, match="mode must be 'pad' or 'feature'"):
            build(terrain={"spawn": {"mode": "hover"}})
    elif case == "crowded_arena":
        with pytest.raises(ValueError, match=r"meets 10 boxes.*Lower terrain\.arena\.discrete_count") as info:
            build(terrain={"arena": {**CPU_ARENA, "discrete_count": 200, "seed": 0}})
        # It refuses in the spec hook, before any model compiles.
        assert any(entry.name == "_customize_spec" for entry in info.traceback)
    elif case == "feature_candidates_zero":
        with pytest.raises(ValueError, match="feature_candidates must be at least 1 in feature mode, got 0"):
            build(terrain={"spawn": {"mode": "feature", "feature_candidates": 0}})
        # Pad mode never reads the candidate count.
        build(terrain={"spawn": {"mode": "pad", "feature_candidates": 0}})
    elif case == "pad_fit":
        with pytest.raises(ValueError, match="off the pad.*0.4 m pad radius"):
            build(terrain={"spawn": {"pad_jitter": 0.2}})
    elif case == "init_level_frac_above_one":
        with pytest.raises(ValueError, match="init_level_frac 1.5 draws first levels past"):
            build(terrain={"spawn": {"init_level_frac": 1.5}})
    elif case == "warp_without_budgets":
        # robot.yaml's flat-floor sim_budget does not stand in.
        with pytest.raises(
            ValueError, match=r"naconmax_per_env and task\.env\.sim\.njmax are unset.*flat-floor measurement"
        ):
            build(env={"sim": {"backend": "warp"}})
    elif case == "bias_out_of_box":
        # roboto_origin's vx box. back_vx lies outside it. The base command
        # config alone arms no back draw, so it builds.
        box = {"command": {"vx": [-0.6, 1.0]}}
        with pytest.raises(ValueError, match=r"pure_back_prob is armed but command\.back_vx"):
            build(env=box, terrain={"command_bias": {"enable": True, "pure_back_prob": 0.1}})
        build(env=box, terrain={"command_bias": {"enable": False, "pure_back_prob": 0.1}})
    elif case == "no_termination_geoms":
        with pytest.raises(ValueError, match="carry no collision geom"):
            build(cls=_edited(_no_termination_colliders))
        build(terrain={"base_contact": {"terminate": False}}, cls=_edited(_no_termination_colliders))
    elif case == "bad_collider_type":
        cyl = _add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, contype=1, conaffinity=0)
        with pytest.raises(ValueError, match="extra.*not a box, capsule or sphere"):
            build(cls=_edited(cyl))
    elif case == "unpaired_collider":
        loner = _add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, contype=16, conaffinity=0)
        with pytest.raises(ValueError, match="extra.*do not pair"):
            build(cls=_edited(loner))
    else:
        with pytest.raises(ValueError, match="naccdmax_per_env 200 exceeds.*100"):
            build(env={"sim": {"naconmax_per_env": 100, "naccdmax_per_env": 200}})


def test_small_contact_budget_warns(capsys):
    build(env={"sim": {"naconmax_per_env": 112}})
    assert "naconmax_per_env 112 is under 4 contacts for each of the 29" in capsys.readouterr().out
    build(env={"sim": {"naconmax_per_env": 116}})
    assert "naconmax_per_env" not in capsys.readouterr().out


def test_a_thin_fall_margin_warns(env, capsys):
    """The reset height must clear fall.min_height by the arena's largest
    lookup step plus 0.05 m, or construction warns."""
    edge = env._z0 - env._tables.max_step - 0.05
    build(env={"fall": {"min_height": edge + 0.01}})
    assert "largest lookup step" in capsys.readouterr().out
    build(env={"fall": {"min_height": edge - 0.01}})
    assert "largest lookup step" not in capsys.readouterr().out


def test_real_pose_ref_settles_on_the_flat_model():
    terrain_env = build(env={"real_pose_ref": True})
    flat_env = make_env("joystick", ROBOT_DIR, PRESET, {"real_pose_ref": True})
    assert terrain_env._settle_model() is terrain_env._flat_model
    assert not scene.is_terrain_model(terrain_env._flat_model)
    np.testing.assert_array_equal(np.asarray(terrain_env._pose_anchor), np.asarray(flat_env._pose_anchor))
    np.testing.assert_array_equal(np.asarray(terrain_env._reset_qpos), np.asarray(flat_env._reset_qpos))


@pytest.mark.parametrize("cap, want", [(None, 116), (64, 64)])
def test_the_jax_contact_cap_is_four_per_ground_collider_unless_set(cap, want, env):
    e = env if cap is None else build(terrain={"jax_contacts": {"max_contact_points": cap}})
    assert e._jax_contact_cap == want
    assert int(e.mj_model.numeric(scene.JAX_CONTACT_NUMERIC).data[0]) == want
    assert e.arena_record()["jax_max_contact_points"] == want


# -- measurements ------------------------------------------------------------------


def test_terrain_relative_measurements(env):
    """On a ramp z = 0.3 + 0.1 x, every reading subtracts the surface under
    the point it measures."""
    ramp = with_ground(env, plane(lambda x, y: 0.3 + 0.1 * x)(env._tables), flat_band=tg.EMPTY_BAND)
    data = posed(env, xy=(2.0, 1.0))
    b = env._base_qadr
    z = float(data.qpos[b + 2])
    assert float(ramp._base_height(data)) == pytest.approx(z - 0.5, abs=1e-5)
    assert float(ramp._obs_catalog(data, env._catalog_probe_info())["height"][0]) == pytest.approx(
        z - 0.5, abs=1e-5
    )
    # A uniform raise of the ground and the robot leaves every reading.
    t = env._tables
    raised = with_tables(
        env,
        lookup=t.lookup + 0.25,
        heights=t.heights + 0.25,
        boxes=t.boxes.at[:, 6].add(0.25),
        flat_band=tg.EMPTY_BAND,
    )
    low = with_tables(env, flat_band=tg.EMPTY_BAND)
    up = posed(env, xy=(2.0, 1.0), dz=0.25)
    np.testing.assert_allclose(raised._base_height(up), low._base_height(data), atol=1e-5)
    np.testing.assert_allclose(raised._foot_clearance(up), low._foot_clearance(data), atol=1e-5)
    np.testing.assert_array_equal(raised._foot_contact(up), low._foot_contact(data))


def test_contact_and_clearance_on_a_20deg_ramp(env):
    """A foot lying along a 20 degree slope reads gap r / cos(20 deg) - r,
    under a millimetre. It reads contact. Lifted 1 cm it reads no contact.

    A foot is in contact when any of its capsules is, and its clearance is
    its lowest capsule's. One capsule planted and the others 2 cm up reads
    as planted, whichever capsule it is. All three up by different amounts
    reads the lowest."""
    t = math.radians(20.0)
    slope = math.tan(t)
    ramp = with_ground(env, plane(lambda x, y: slope * x)(env._tables), flat_band=tg.EMPTY_BAND)
    data = posed(env, xy=(2.0, 1.0))

    def ground(x, y):
        return slope * x

    planted = lay_foot(env, data, 0, (2.0, 1.1), t, ground=ground)
    assert bool(ramp._foot_contact(planted)[0])
    assert abs(float(ramp._foot_clearance(planted)[0])) < 0.003
    lifted = lay_foot(env, data, 0, (2.0, 1.1), t, lift=0.01, ground=ground)
    assert not bool(ramp._foot_contact(lifted)[0])
    assert float(ramp._foot_clearance(lifted)[0]) == pytest.approx(0.01, abs=0.003)

    r = float(env._foot_geom_radius[0])
    gap = r / math.cos(t) - r
    n = int((np.asarray(env._foot_geom_foot_idx) == 0).sum())
    assert n == 3
    for k in range(n):
        lift = np.full(n, 0.02)
        lift[k] = 0.0
        one_down = lay_foot(env, data, 0, (2.0, 1.1), t, lift=lift, ground=ground)
        assert bool(ramp._foot_contact(one_down)[0]), k
        assert float(ramp._foot_clearance(one_down)[0]) == pytest.approx(gap, abs=1e-4), k
    staggered = lay_foot(env, data, 0, (2.0, 1.1), t, lift=(0.01, 0.015, 0.02), ground=ground)
    assert not bool(ramp._foot_contact(staggered)[0])
    assert float(ramp._foot_clearance(staggered)[0]) == pytest.approx(0.01 + gap, abs=1e-4)


def test_a_foot_across_a_riser_reads_the_upper_tread(env):
    """A level foot whose toe rests on a 10 cm tread and whose heel hangs
    over the lower one reads contact and clearance 0."""
    # The riser falls mid-cell at x = 2.01.
    step = with_boxes(env, [Box((2.51, 1.0, 0.05), (0.5, 0.5, 0.05))])
    data = posed(env, xy=(2.0, 1.0))
    # The capsules lie along +x with their +x end-cap centres at x = 2.06,
    # on the tread. The rest of each foot hangs over the lower tread.
    foot = lay_foot(env, data, 0, (2.06 - 0.067, 1.1), 0.0, ground=lambda x, y: 0.1)
    assert bool(step._foot_contact(foot)[0])
    assert abs(float(step._foot_clearance(foot)[0])) < 1e-4


def test_a_foot_on_a_box_edge_reads_the_box_top(env):
    """The feet read box tops exactly up to the edge, where the lookup
    reads them low.

    A level foot rests its heel end caps 1 cm inside the edge of a box
    yawed 0.5 rad, and the rest of it hangs off the box. It reads contact
    and clearance 0. The lookup reads every heel more than 5 mm low. Read
    on the lookup at its axis points, the foot would read no contact.

    A level foot rests on a box edge that runs along it, 0.5 r out from
    its outer capsule's axis. No axis point is over the box. The samples
    0.8 r out are, so it reads contact, with the clearance 0.27 r those
    samples see."""
    r, half, top = float(env._foot_geom_radius[0]), float(env._foot_half[0]), 0.08
    data = posed(env, xy=(2.0, 1.0))
    yaw = 0.5
    fwd = np.array([math.cos(yaw), math.sin(yaw)])
    box = Box((2.3, 1.0, top / 2), (0.1, 0.1, top / 2), yaw)
    yawed = with_boxes(env, [box])
    centre = np.array(box.pos[:2]) + (0.09 + half) * fwd
    foot = lay_foot(env, data, 0, centre, 0.0, yaw=yaw, ground=lambda x, y: top)
    assert bool(yawed._foot_contact(foot)[0])
    assert abs(float(yawed._foot_clearance(foot)[0])) < 1e-4
    ids = np.asarray(env._foot_geom_ids)[np.asarray(env._foot_geom_foot_idx) == 0]
    heels = np.asarray(foot.geom_xpos)[ids, :2] - half * fwd
    assert float(tg.height(yawed._tables, jp.asarray(heels, jp.float32)).max()) < top - 0.005

    # The box covers y >= 1.03, mid-cell.
    side = with_boxes(env, [Box((2.3, 1.23, top / 2), (0.3, 0.2, top / 2))])
    rest = top + math.sqrt(r * r - (0.5 * r) ** 2)
    foot = lay_foot(env, data, 0, (2.3, 1.03 - 0.5 * r - 0.017), 0.0, ground=lambda x, y: rest - r)
    assert bool(side._foot_contact(foot)[0])
    assert float(side._foot_clearance(foot)[0]) == pytest.approx(rest - 0.6 * r - top, abs=1e-5)


def test_a_toe_beside_a_riser_face_reads_its_own_ground(env):
    """A riser whose face lies on the node line x = 2.04. A level foot
    3 cm up, its toe end-cap centre 2 cm short of the face, reads no
    contact and its 3 cm clearance. The lookup reads half the 10 cm riser
    under that end cap. Read on the lookup at its axis points, the foot
    would read contact and a clearance of -2 cm."""
    half = float(env._foot_half[0])
    riser = with_boxes(env, [Box((2.54, 1.0, 0.05), (0.5, 0.5, 0.05))])
    data = posed(env, xy=(2.0, 1.0))
    foot = lay_foot(env, data, 0, (2.02 - half, 1.0), 0.0, lift=0.03, ground=lambda x, y: 0.0)
    assert not bool(riser._foot_contact(foot)[0])
    assert float(riser._foot_clearance(foot)[0]) == pytest.approx(0.03, abs=1e-5)
    assert float(tg.height(riser._tables, jp.array([2.02, 1.0]))) == pytest.approx(0.05, abs=1e-5)


def test_flat_row_measurements_equal_the_flat_env(flat):
    """With base contact off, every measurement on the flat row is the flat
    floor's own expression, bit for bit, and so is the reward dict.

    A foot pitched 10 degrees, its capsule centres one radius plus 8 mm
    up, separates the two foot rules. The flat rule reads no contact. The
    terrain rule reads the low end cap in contact, with a clearance of
    about -3.6 mm. On the flat row both read the flat rule.

    A level foot with one capsule down and the others 2 cm up reads
    contact on both, whichever capsule is down. On the flat row clearance
    is the foot site's expression and does not depend on the capsules, so
    that case checks contact only."""
    env = build(terrain={"base_contact": {"terminate": False}})
    off = with_tables(env, flat_band=tg.EMPTY_BAND)
    phi = math.radians(10.0)
    r = float(env._foot_geom_radius[0])
    lift = 0.008 + r - r / math.cos(phi)
    for foot in (0, 1):
        td = lay_foot(env, posed(env, yaw=0.4), foot, FLAT_XY, phi, ground=lambda x, y: 0.0, lift=lift)
        fd = lay_foot(flat, posed(flat, yaw=0.4), foot, FLAT_XY, phi, ground=lambda x, y: 0.0, lift=lift)
        for name in ("_foot_contact", "_foot_clearance"):
            np.testing.assert_array_equal(getattr(env, name)(td), getattr(flat, name)(fd), err_msg=name)
        assert not bool(env._foot_contact(td)[foot])
        assert float(env._foot_clearance(td)[foot]) > 0.0
        # The pose is not vacuous: off the flat row the same data reads the
        # other way.
        assert bool(off._foot_contact(td)[foot])
        low_cap = 0.008 - float(env._foot_half[0]) * math.sin(phi)
        assert float(off._foot_clearance(td)[foot]) == pytest.approx(low_cap, abs=1e-3)
        n = int((np.asarray(env._foot_geom_foot_idx) == foot).sum())
        level = functools.partial(lay_foot, foot=foot, centre=FLAT_XY, pitch=0.0, ground=lambda x, y: 0.0)
        for k in range(n):
            one_down = np.full(n, 0.02)
            one_down[k] = 0.0
            td = level(env, posed(env, yaw=0.4), lift=one_down)
            fd = level(flat, posed(flat, yaw=0.4), lift=one_down)
            np.testing.assert_array_equal(env._foot_contact(td), flat._foot_contact(fd), err_msg=str(k))
            assert bool(flat._foot_contact(fd)[foot]), k
    for dz in (0.0, 0.004, 0.05):
        td = posed(env, dz=dz, yaw=0.4)
        fd = posed(flat, dz=dz, yaw=0.4)
        for name in ("_base_height", "_foot_clearance", "_foot_contact"):
            np.testing.assert_array_equal(getattr(env, name)(td), getattr(flat, name)(fd), err_msg=name)
        info = {
            "command": jp.array([0.4, 0.1, -0.2]),
            "last_action": jp.full(env.action_size, 0.1),
            "last_last_action": jp.zeros(env.action_size),
            "last_torque": jp.zeros(env.action_size),
            "feet_air_time": jp.array([0.1, 0.0]),
            "swing_apex": jp.array([0.02, 0.0]),
            "phase": jp.array(0.7),
            "since_spawn": jp.array(3),
            "terrain_level": jp.array(0),
        }
        action = jp.linspace(-0.2, 0.2, env.action_size)
        tc, fc = env._foot_contact(td), flat._foot_contact(fd)
        tr, tfall = env._compute_rewards(td, info, action, tc, tc)
        fr, ffall = flat._compute_rewards(fd, info, action, fc, fc)
        assert list(tr) == list(fr)
        for k in tr:
            np.testing.assert_array_equal(tr[k], fr[k], err_msg=k)
        np.testing.assert_array_equal(tfall, ffall)


def _cells_low(env, data, corner_z, xy=(2.0, 1.0)):
    """Termination cells 1 m up, except the first, rolled 30 degrees and
    pitched 20 with its lowest corner at `corner_z` above xy. The cell's
    half extents come from the model."""
    ids = env._term_box
    R = rot_z(0.3) @ rot_y(math.radians(20.0)) @ np.array(
        [[1, 0, 0], [0, math.cos(0.5), -math.sin(0.5)], [0, math.sin(0.5), math.cos(0.5)]]
    )
    half = env.mj_model.geom_size[ids[0]]
    low = -(R @ (np.sign(R[2, :]) * half))
    pos = np.array(data.geom_xpos)[ids]
    pos[:, 2] += 1.0
    pos[0] = np.array([xy[0], xy[1], corner_z]) - low
    mats = np.array(data.geom_xmat)[ids]
    mats[0] = R
    return set_geoms(data, ids, pos, mats)


def _capsule_low(env, data, bottom_z, xy=(2.0, 1.0)):
    """Termination capsules 1 m up, except the first, pitched 20 degrees
    with the bottom of its low end cap at `bottom_z` above xy. The radius
    and half length come from the model."""
    ids = env._term_capsule
    r, half = env.mj_model.geom_size[ids[0], :2]
    R = rot_y(math.radians(20.0))
    axis = R[:, 2]  # points up, so the low end cap is at -half along it
    pos = np.array(data.geom_xpos)[ids]
    pos[:, 2] += 1.0
    pos[0] = np.array([xy[0], xy[1], bottom_z + r]) + half * axis
    mats = np.array(data.geom_xmat)[ids]
    mats[0] = R
    return set_geoms(data, ids, pos, mats)


@pytest.fixture(scope="module")
def asimov():
    return build(robot="asimov_v1")


def test_termination_tables_come_from_the_model(env, asimov):
    """roboto_origin ends an episode on its 15 base and torso chessboard
    cells, asimov_v1 on its pelvis and waist capsules. Each table holds
    the model's own geom sizes."""
    m = env.mj_model
    cells = [
        g
        for g in range(m.ngeom)
        if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith(
            ("base_collision_", "torso_collision_")
        )
    ]
    assert len(cells) == 15
    assert sorted(env._term_box.tolist()) == cells
    assert env._term_capsule.size == 0 and env._term_sphere.size == 0
    half = m.geom_size[env._term_box].astype(np.float32)
    np.testing.assert_array_equal(np.asarray(env._term_box_half), half)

    m = asimov.mj_model
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, int(g)) for g in asimov._term_capsule]
    assert names == ["pelvis_collision", "waist_yaw_link_collision"]
    assert asimov._term_box.size == 0 and asimov._term_sphere.size == 0
    size = m.geom_size[asimov._term_capsule].astype(np.float32)
    np.testing.assert_array_equal(np.asarray(asimov._term_capsule_half), size[:, 1])
    np.testing.assert_array_equal(np.asarray(asimov._term_capsule_r), size[:, 0])


@pytest.mark.parametrize("robot", [ROBOT, "asimov_v1"])
def test_base_contact_reads_the_lowest_point(robot, env, asimov):
    """A box cell's lowest corner, or a capsule's lowest end-cap point,
    within base_contact.tol of the ground reads contact."""
    e, low = (env, _cells_low) if robot == ROBOT else (asimov, _capsule_low)
    level = jp.full_like(e._tables.lookup, 0.2)
    raised = with_tables(e, lookup=level, lookup_spawn=level)
    data = posed(e, xy=(2.0, 1.0))
    tol = e._config.terrain.base_contact.tol
    assert not bool(raised._base_contact(data))
    assert bool(raised._base_contact(low(e, data, 0.2 + tol - 0.002)))
    assert not bool(raised._base_contact(low(e, data, 0.2 + tol + 0.002)))


def test_base_contact_reads_a_tread_lip_on_the_spawn_grid(env):
    """A box whose edge falls mid-cell: 4 cm in, the lookup reads the top
    at a node, and 5 mm in it reads the blend. A cell corner resting 5 mm
    inside the edge reads contact, because base contact reads the dilated
    grid."""
    lookup = plane(lambda x, y: np.where(x >= 2.02, 0.15, 0.0))(env._tables)
    lip = with_tables(env, lookup=lookup, lookup_spawn=jp.asarray(tg.dilate_max(np.asarray(lookup), 1)))
    data = _cells_low(env, posed(env, xy=(2.0, 1.0)), 0.152, xy=(2.025, 1.0))
    assert float(tg.height(lip._tables, jp.array([2.025, 1.0]))) < 0.15 - 0.05
    assert bool(lip._base_contact(data))


def test_base_contact_and_terrain_height_end_the_episode(env):
    """The fall a step uses, from `_compute_rewards`: base contact ends the
    episode only when enabled, and the height check reads the base above
    the ground under it."""
    off = build(terrain={"base_contact": {"terminate": False}})
    upright = jp.array([0.0, 0.0, -1.0])
    action = jp.zeros(env.action_size)
    info = {
        "command": jp.zeros(3),
        "last_action": action,
        "last_last_action": action,
        "last_torque": action,
        "feet_air_time": jp.zeros(2),
        "swing_apex": jp.zeros(2),
        "phase": jp.array(0.0),
        "since_spawn": jp.array(10),
        "terrain_level": jp.array(0),
    }
    for e, want in ((env, True), (off, False)):
        data = _cells_low(e, posed(e), 0.005, xy=FLAT_XY)
        assert bool(e._base_contact(data))
        assert bool(e._fall(data, upright)) is want
        assert not bool(e._fall(posed(e), upright))
        c = e._foot_contact(data)
        r, fall = e._compute_rewards(data, info, action, c, c)
        assert bool(fall) is want
        assert float(r["termination"]) == float(want)

    # Ground 0.5 m up: the free joint's z clears fall.min_height, and the
    # height above the ground does not.
    high = with_tables(off, lookup=off._tables.lookup + 0.5, flat_band=tg.EMPTY_BAND)
    d = posed(off)
    b = off._base_qadr
    assert float(d.qpos[b + 2]) > off._config.fall.min_height > float(high._base_height(d))
    c = high._foot_contact(d)
    r, fall = high._compute_rewards(d, info, action, c, c)
    assert bool(fall)
    assert float(r["termination"]) == 1.0


def test_spawn_grace_waives_only_the_termination_reward():
    env = build(terrain={"spawn": {"grace_sec": 0.1}})
    assert env._grace_steps == 5
    data = posed(env, dz=-0.5)
    action = jp.zeros(env.action_size)
    contact = env._foot_contact(data)
    info = {
        "command": jp.zeros(3),
        "last_action": action,
        "last_last_action": action,
        "last_torque": action,
        "feet_air_time": jp.zeros(2),
        "swing_apex": jp.zeros(2),
        "phase": jp.array(0.0),
        "terrain_level": jp.array(1),
    }
    early, fall_early = env._compute_rewards(data, {**info, "since_spawn": jp.array(4)}, action, contact, contact)
    late, fall_late = env._compute_rewards(data, {**info, "since_spawn": jp.array(5)}, action, contact, contact)
    assert bool(fall_early) and bool(fall_late)
    assert float(early["termination"]) == 0.0 and float(late["termination"]) == 1.0
    assert list(early) == list(late)
    for k in early:
        if k != "termination":
            np.testing.assert_array_equal(early[k], late[k], err_msg=k)


def test_no_progress_patience_applies_off_the_flat_row(env):
    base = env._config.no_progress
    assert env._no_progress_params({"terrain_level": jp.array(1)}) == (base.grace_sec, base.p_max)
    patient = build(terrain={"no_progress": {"grace_sec": 3.5, "p_max_scale": 0.5}})
    for level, want in ((0, (base.grace_sec, base.p_max)), (1, (3.5, base.p_max * 0.5))):
        grace, p_max = patient._no_progress_params({"terrain_level": jp.array(level)})
        assert float(grace) == pytest.approx(want[0]) and float(p_max) == pytest.approx(want[1])
    # Without a flat row, level 0 is terrain.
    no_flat = with_tables(patient, flat_row=False, flat_band=tg.EMPTY_BAND)
    assert not bool(no_flat._on_flat(jp.array(0)))
    grace, p_max = no_flat._no_progress_params({"terrain_level": jp.array(0)})
    assert float(grace) == pytest.approx(3.5) and float(p_max) == pytest.approx(base.p_max * 0.5)


def test_command_bias_applies_off_the_flat_row_and_stays_in_the_box():
    """The bias replaces the command off the flat row at reset and at every
    resample, and a step reads the terrain no-progress patience. The
    no-progress cut here fires surely on the flat row (p_max 1). Off it,
    both halves of the patience hold it back: p_max_scale 0 never cuts, and
    a 100 s grace keeps a 20 s old command unarmed."""
    env = build(
        env={
            "command": {"resample_steps": 1},
            "no_progress": {"enable": True, "p_max": 1.0, "risk_below": 1.0},
        },
        terrain={
            "spawn": {"init_level_frac": 1.0},
            "command_bias": {"enable": True, "zero_prob": 0.0, "pure_wz_prob": 0.0, "pure_vy_prob": 0.0,
                             "pure_slow_prob": 1.0},
            "no_progress": {"p_max_scale": 0.0},
        },
    )
    c = env._config.command

    def in_slow_box(cmds):
        cmds = np.asarray(cmds)
        return (
            np.all(cmds[..., 1:] == 0.0)
            and np.all((cmds[..., 0] >= c.slow_vx[0]) & (cmds[..., 0] <= c.slow_vx[1]))
            and np.all((cmds[..., 0] >= c.vx[0]) & (cmds[..., 0] <= c.vx[1]))
        )

    keys = jax.random.split(jax.random.PRNGKey(0), 256)
    for level in (0, 1):
        info = {"terrain_level": jp.array(level)}
        cmds = np.asarray(jax.vmap(lambda k, info=info: env._next_command(k, info))(keys))
        if level == 0:
            np.testing.assert_array_equal(cmds, np.asarray(jax.vmap(env._sample_command)(keys)))
        else:
            assert in_slow_box(cmds)
    # Without a flat row, level 0 takes the bias too.
    no_flat = with_tables(env, flat_row=False, flat_band=tg.EMPTY_BAND)
    info = {"terrain_level": jp.array(0)}
    assert in_slow_box(jax.vmap(lambda k: no_flat._next_command(k, info))(keys))

    # Reset: Joystick's draw on the flat row, the bias elsewhere, and the
    # progress meter and the observation follow the command that stands.
    keys = jax.random.split(jax.random.PRNGKey(3), 64)
    states = jax.jit(jax.vmap(env.reset))(keys)
    lvl = np.asarray(states.info["terrain_level"])
    assert set(lvl.tolist()) == {0, 1}
    cmd = np.asarray(states.info["command"])
    joystick = np.asarray(jax.vmap(lambda k: env._sample_command(jax.random.split(k, 3)[1]))(keys))
    np.testing.assert_array_equal(cmd[lvl == 0], joystick[lvl == 0])
    assert in_slow_box(cmd[lvl == 1])
    np.testing.assert_array_equal(
        np.asarray(states.info["progress_ema"]), np.asarray(jax.jit(jax.vmap(env._cmd_speed))(cmd))
    )
    # Batched and fused, the gravity components differ by an ulp. A stale
    # command would differ by more than 0.1.
    np.testing.assert_allclose(
        np.asarray(states.obs["state"]),
        np.asarray(jax.vmap(env._build_obs)(states.data, states.info)["state"]),
        rtol=0,
        atol=1e-6,
    )

    # Step: a command 0.8 m/s ahead with negative progress, drawn 1000
    # steps (20 s) ago, long past Joystick's grace. The hazard clips to
    # p_max.
    s0 = jax.tree.map(lambda x: x[0], states)

    def stalled(level):
        return s0.replace(
            info={
                **s0.info,
                "terrain_level": jp.asarray(level, s0.info["terrain_level"].dtype),
                "command": jp.array([0.8, 0.0, 0.0]),
                "progress_ema": jp.asarray(-5.0, s0.info["progress_ema"].dtype),
                "steps_since_cmd": jp.asarray(1000, s0.info["steps_since_cmd"].dtype),
            }
        )

    step = jax.jit(env.step)
    for level in (0, 1):
        s1 = step(stalled(level), jp.zeros(env.action_size))
        cut = float(s1.metrics["no_progress_cut"])
        if level == 0:
            assert cut == 1.0
        else:
            assert cut == 0.0
            assert in_slow_box(s1.info["command"])
    # The same stalled state under a 100 s terrain grace and the full p_max.
    patient = build(
        env={"no_progress": {"enable": True, "p_max": 1.0, "risk_below": 1.0}},
        terrain={"no_progress": {"grace_sec": 100.0}},
    )
    step = jax.jit(patient.step)
    cuts = {}
    for level in (0, 1):
        s1 = step(stalled(level), jp.zeros(patient.action_size))
        cuts[level] = float(s1.metrics["no_progress_cut"])
    assert cuts == {0: 1.0, 1: 0.0}


# -- spawns -------------------------------------------------------------------------


@pytest.fixture(scope="module")
def place(env):
    return jax.jit(jax.vmap(lambda k: env._place_base(k, env._reset_qpos)))


def test_initial_level_lies_in_the_lower_fraction(env, place):
    keys = jax.random.split(jax.random.PRNGKey(1), 256)
    _, info = place(keys)
    # Two levels at frac 0.5: round(1.0) = 1, so every first spawn is level 0.
    assert np.all(np.asarray(info["terrain_level"]) == 0)
    wide = build(terrain={"spawn": {"init_level_frac": 1.0}})
    _, info = jax.vmap(lambda k: wide._place_base(k, wide._reset_qpos))(keys)
    assert set(np.asarray(info["terrain_level"]).tolist()) == {0, 1}
    assert set(np.asarray(info["terrain_type"]).tolist()) == set(range(len(TYPES)))


def test_spawn_level_pins_the_first_spawn():
    pinned = build(terrain={"spawn": {"level": 1}})
    keys = jax.random.split(jax.random.PRNGKey(2), 64)
    _, info = jax.vmap(lambda k: pinned._place_base(k, pinned._reset_qpos))(keys)
    assert np.all(np.asarray(info["terrain_level"]) == 1)
    with pytest.raises(ValueError, match="spawn.level 2 is past"):
        build(terrain={"spawn": {"level": 2}})


def test_pad_spawn_z_is_pad_height_plus_z0(env, place):
    """The pads are flat, so a pad spawn's base height is the pad's height
    plus the reset height: exactly on the flat row, and to float32 rounding
    of the bilinear blend elsewhere."""
    wide = build(terrain={"spawn": {"init_level_frac": 1.0}})
    keys = jax.random.split(jax.random.PRNGKey(4), 512)
    qpos, info = jax.jit(jax.vmap(lambda k: wide._place_base(k, wide._reset_qpos)))(keys)
    b = env._base_qadr
    level, ttype = np.asarray(info["terrain_level"]), np.asarray(info["terrain_type"])
    pad_h = np.asarray(env._tables.pad_h)[level, ttype]
    z = np.asarray(qpos)[:, b + 2]
    assert np.all(np.asarray(info["spawn_kind"]) == tg.SPAWN_PAD)
    assert (pad_h != 0).any()
    np.testing.assert_array_equal(z[level == 0], np.float32(env._z0))
    np.testing.assert_allclose(z, np.float32(env._z0) + pad_h, rtol=0, atol=2.4e-7)


def test_spawn_qpos_reads_only_xy_yaw_and_kind(env):
    """The base pose comes from the drawn xy, yaw and kind, never from the
    qpos given. The yaw turns the reset quaternion from the left. Kind 1
    reads the spawn grid, kinds 0 and 2 the lookup. A tilted reset
    quaternion and a spawn grid 0.1 m above the lookup tell each rule
    apart."""
    b = env._base_qadr
    e = with_tables(env, lookup_spawn=env._tables.lookup_spawn + 0.1)
    reset_quat = np.array([0.8, -0.2, 0.4, 0.3])
    e._reset_quat = jp.asarray(reset_quat / np.linalg.norm(reset_quat), jp.float32)
    tilt = np.array([0.9, 0.3, 0.2, 0.1])
    tilt = jp.asarray(tilt / np.linalg.norm(tilt), jp.float32)
    q = env._reset_qpos.at[b + 2].set(2.0).at[b + 3 : b + 7].set(tilt)
    xy, yaw = jp.asarray(FLAT_XY, jp.float32), jp.float32(0.7)
    kinds = (tg.SPAWN_PAD, tg.SPAWN_FEATURE, tg.SPAWN_FALLBACK)
    out = {k: np.asarray(e.spawn_qpos(q, xy, yaw, jp.int32(k))) for k in kinds}
    ref = np.asarray(e.spawn_qpos(e._reset_qpos, xy, yaw, jp.int32(tg.SPAWN_PAD)))
    want = np.asarray(tg.quat_mul(tg.yaw_quat(yaw), e._reset_quat))
    rest = np.ones(q.shape[0], bool)
    rest[b : b + 7] = False
    for o in out.values():
        np.testing.assert_array_equal(o[b : b + 2], np.asarray(xy))
        np.testing.assert_allclose(o[b + 3 : b + 7], want, atol=1e-6)
        np.testing.assert_array_equal(o[rest], np.asarray(q)[rest])
    z = {k: float(o[b + 2]) for k, o in out.items()}
    assert z[tg.SPAWN_PAD] == float(ref[b + 2])
    assert z[tg.SPAWN_FALLBACK] == z[tg.SPAWN_PAD]
    assert z[tg.SPAWN_FEATURE] == pytest.approx(z[tg.SPAWN_PAD] + 0.1, abs=1e-5)


@pytest.fixture(scope="module")
def feature_env():
    return build(env={"reset_noise": 0.0}, terrain={"spawn": {"mode": "feature", "init_level_frac": 1.0}})


def test_feature_spawn_is_level_footed_or_falls_back(feature_env):
    e = feature_env
    t = e._tables
    keys = jax.random.split(jax.random.PRNGKey(5), 512)
    qpos, info = jax.jit(jax.vmap(lambda k: e._place_base(k, e._reset_qpos)))(keys)
    kind = np.asarray(info["spawn_kind"])
    level = np.asarray(info["terrain_level"])
    xy = info["spawn_xy"]
    # The reset quaternion is the identity, so the base quaternion is the yaw's.
    b = e._base_qadr
    yaw = 2 * jp.arctan2(qpos[:, b + 6], qpos[:, b + 3])
    assert np.all(kind[level == 0] == tg.SPAWN_PAD)
    assert {tg.SPAWN_FEATURE, tg.SPAWN_FALLBACK} <= set(kind[level == 1].tolist())
    spread = np.asarray(jax.vmap(lambda p, a: tg.support_spread(t, e._spawn_u_sole, p, a))(xy, yaw))
    assert np.all(spread[kind == tg.SPAWN_FEATURE] <= e._config.terrain.spawn.feature_max_spread)
    offset = np.abs(np.asarray(xy) - np.asarray(info["tile_origin"]))
    assert np.all(offset[kind != tg.SPAWN_FEATURE] <= e._config.terrain.spawn.pad_jitter)
    assert np.all(offset <= e._feature_half)


def _collider_bottoms(e, pos, mat):
    """(xy (P, 2), bottom z (P,)) of every collider's sample points, from
    geom positions (ngeom, 3) and rotations (ngeom, 3, 3)."""
    m = e.mj_model
    xy, bottom = [], []
    for g in sorted(set(e._spawn_owner.tolist())):
        c, R, size = pos[g], mat[g], m.geom_size[g]
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
            pts = [c + R @ ((2 * np.array(s) - 1) * size) for s in np.ndindex(2, 2, 2)]
            r = 0.0
        else:
            pts = [c + t * size[1] * R[:, 2] for t in tg.SPAWN_CAPSULE_POINTS]
            r = size[0]
        for p in pts:
            xy.append(p[:2])
            bottom.append(p[2] - r)
    return np.array(xy), np.array(bottom)


def test_spawns_keep_every_collider_clear_of_the_ground(feature_env):
    """Every sample point's bottom stays above the grid its spawn read, and
    the binding one keeps the reset pose's float: the gap the lowest point
    has over the flat floor at the reset keyframe (3.1 mm for roboto_origin).
    Poses come from plain MuJoCo kinematics."""
    e = feature_env
    m = e.mj_model
    d = mujoco.MjData(m)

    def bottoms(qpos):
        d.qpos[:] = qpos
        mujoco.mj_kinematics(m, d)
        return _collider_bottoms(e, d.geom_xpos, d.geom_xmat.reshape(-1, 3, 3))

    # The float is an absolute z, so the keyframe's xy does not matter.
    _, flat_bottom = bottoms(np.asarray(e._reset_qpos))
    keyframe_float = float(flat_bottom.min())
    assert 0.0 < keyframe_float < 0.01

    keys = jax.random.split(jax.random.PRNGKey(6), 48)
    qpos, info = jax.jit(jax.vmap(lambda k: e._place_base(k, e._reset_qpos)))(keys)
    qpos = np.asarray(qpos)
    kinds = np.asarray(info["spawn_kind"])
    assert set(kinds.tolist()) >= {tg.SPAWN_PAD, tg.SPAWN_FEATURE}
    for i in range(len(keys)):
        xy, bottom = bottoms(qpos[i])
        grid = "spawn" if kinds[i] == tg.SPAWN_FEATURE else "lookup"
        ground = np.asarray(tg.height(e._tables, jp.asarray(xy, jp.float32), grid))
        assert float((bottom - ground).min()) == pytest.approx(keyframe_float, abs=1e-5)


@pytest.fixture(scope="module")
def arena_feature_env():
    """Feature spawns on the default arena."""
    return make_env("terrain", ROBOT_DIR, PRESET, {"terrain": {"spawn": {"mode": "feature"}}})


def _tiles(e, types, min_difficulty=0.3):
    """(level, type index) of the arena's tiles of `types` above
    `min_difficulty`."""
    return [
        (t.row, TYPES.index(t.terrain_type))
        for t in e._arena.spec.tiles
        if t.terrain_type in types and t.difficulty > min_difficulty
    ]


def _accepted_feature_spawns(e, tiles, per_tile):
    """(qpos (n, nq), type index (n,)) of the feature spawns `per_tile` keys
    give on each tile, pad fallbacks dropped."""

    @jax.jit
    def spawn(keys, level, ttype):
        def one(key):
            xy, yaw, kind = e.draw_spawn(key, ttype, level)
            return e.spawn_qpos(e._reset_qpos, xy, yaw, kind), kind

        return jax.vmap(one)(keys)

    out, types = [], []
    for i, (level, ttype) in enumerate(tiles):
        qpos, kind = spawn(jax.random.split(jax.random.PRNGKey(i), per_tile), level, ttype)
        out.append(np.asarray(qpos)[np.asarray(kind) == tg.SPAWN_FEATURE])
        types.append(np.full(len(out[-1]), ttype))
    return np.concatenate(out), np.concatenate(types)


def _foot_gaps_by_ray(e, m, d, qpos):
    """Per foot, the smallest drop from the bottom of a foot capsule's
    spawn sample points to the ground, by mj_ray straight down in plain
    MuJoCo. Each ray starts 5 mm above the capsule's bottom, so a point
    a hair inside the ground still finds its surface. Ground geoms are in
    group 0, and the filter keeps the rays off the robot's own geoms."""
    d.qpos[:] = qpos
    mujoco.mj_kinematics(m, d)
    ground_only = np.array([1, 0, 0, 0, 0, 0], np.uint8)
    hit = np.zeros(1, np.int32)
    ids = np.asarray(e._foot_geom_ids)
    feet = np.asarray(e._foot_geom_foot_idx)
    gaps = np.full(e._n_feet, np.inf)
    for g, foot in zip(ids, feet):
        r, half = m.geom_size[g, :2]
        axis = d.geom_xmat[g].reshape(3, 3)[:, 2]
        for t in tg.SPAWN_CAPSULE_POINTS:
            p = d.geom_xpos[g] + t * half * axis
            start = np.array([p[0], p[1], p[2] - r + 0.005])
            dist = mujoco.mj_ray(m, d, start, np.array([0.0, 0.0, -1.0]), ground_only, 1, -1, hit)
            gaps[foot] = min(gaps[foot], dist - 0.005)
    return gaps


def test_feature_spawns_rest_both_feet_on_the_ground(arena_feature_env):
    """Plain MuJoCo on the default arena, 64 draws per stair and rubble
    tile above difficulty 0.3. No accepted feature spawn on stairs hangs a
    foot 3 cm over the ground. The worst over about 240,000 stair spawns is
    2.3 cm. On rubble the bound is 6 cm. Nothing caps the rubble gap. The
    worst over about 118,000 rubble spawns is 4.7 cm. On these keys both
    types stay within 2.3 cm. The spread reads both the spawn grid and the
    lookup. Reading the spawn grid alone, the same keys hang a foot 15.3 cm
    over a stair and 7.4 cm over rubble."""
    e = arena_feature_env
    m = e.mj_model
    d = mujoco.MjData(m)
    tiles = _tiles(e, ("pyramid_stairs", "inverted_pyramid_stairs", "random_grid"))
    qpos, types = _accepted_feature_spawns(e, tiles, 64)
    assert len(qpos) > 1000
    gaps = np.array([_foot_gaps_by_ray(e, m, d, q) for q in qpos])
    assert np.isfinite(gaps).all() and gaps.min() > -1e-3
    worst = gaps.max(axis=1)
    rubble = types == TYPES.index("random_grid")
    assert rubble.sum() > 300 and (~rubble).sum() > 600
    assert worst[~rubble].max() < 0.03
    assert worst[rubble].max() < 0.06


def test_feature_spawns_do_not_penetrate_the_arena(arena_feature_env):
    """Plain MuJoCo on the default arena: 1000 feature spawns on obstacle
    and rubble tiles above difficulty 0.3 leave every robot-ground contact
    within 0.5 mm. None has a robot-ground contact at all. Spawn heights
    from the plain lookup put 9 of the same spawns more than 0.5 mm deep,
    the deepest 10.7 mm."""
    e = arena_feature_env
    tiles = _tiles(e, ("discrete_obstacles", "random_grid"))

    @jax.jit
    def spawn(key, level, ttype):
        xy, yaw, kind = e.draw_spawn(key, ttype, level)
        return e.spawn_qpos(e._reset_qpos, xy, yaw, kind), kind

    m = e.mj_model
    d = mujoco.MjData(m)
    ground = set(scene.ground_geom_ids(m).tolist())
    accepted, worst = 0, np.inf
    for i in range(2000):
        level, ttype = tiles[i % len(tiles)]
        qpos, kind = spawn(jax.random.PRNGKey(i), level, ttype)
        if int(kind) != tg.SPAWN_FEATURE:
            continue
        d.qpos[:] = np.asarray(qpos)
        mujoco.mj_forward(m, d)
        for (g1, g2), dist in zip(d.contact.geom, d.contact.dist):
            if (g1 in ground) != (g2 in ground):
                worst = min(worst, float(dist))
        accepted += 1
        if accepted == 1000:
            break
    assert accepted == 1000
    assert worst > -5e-4


def test_the_exact_ground_matches_mj_ray_on_the_default_arena(arena_feature_env):
    """Plain MuJoCo on the default arena, the robot lifted 50 m: the exact
    ground the env built reads what a ray straight down hits. That is
    MuJoCo's own heightfield triangles and the box tops. The points are
    uniform on every stair and box tile above difficulty 0.3, inside the
    inverted-stairs pit corner cells, and within 2 cm of a box edge.
    Float32 coordinates round a point by up to about 1.6e-4 of a cell, so
    the bound grows with the drop across the point's cell. Points within
    1e-5 m of a box edge are skipped. The same rounding can move them
    across it."""
    e = arena_feature_env
    t, m = e._tables, e.mj_model
    d = mujoco.MjData(m)
    d.qpos[:] = np.asarray(e._reset_qpos)
    d.qpos[e._base_qadr + 2] += 50.0
    mujoco.mj_kinematics(m, d)
    x0, dx, y0, dy = t.frame
    h = np.asarray(t.heights, float)
    corner = np.stack([h[:-1, :-1], h[:-1, 1:], h[1:, :-1], h[1:, 1:]])
    drop = corner.max(axis=0) - corner.min(axis=0)
    twist = np.abs(corner[0] - corner[1] - corner[2] + corner[3])
    rng = np.random.default_rng(0)

    tiles = _tiles(e, ("pyramid_stairs", "inverted_pyramid_stairs", "random_grid", "discrete_obstacles"))
    origins = np.asarray(t.origin_xy)[tuple(np.asarray(tiles).T)]
    half = t.tile_size / 2
    uniform = (origins[:, None, :] + rng.uniform(-half, half, (len(tiles), 1500, 2))).reshape(-1, 2)
    # The pit's corner cells twist by the whole pit depth.
    rows, cols = np.nonzero(twist > 0.05)
    assert len(rows) >= 8
    k = np.repeat(np.arange(len(rows)), 100)
    pit = np.stack([x0 + (cols[k] + rng.random(len(k))) * dx, y0 + (rows[k] + rng.random(len(k))) * dy], axis=1)
    # Boxes as (x, y, half x, half y, cos, sin), then a row that meets no point.
    boxes = np.array(
        [(b.pos[0], b.pos[1], b.half[0], b.half[1], math.cos(b.yaw), math.sin(b.yaw)) for b in e._arena.boxes]
        + [(0.0, 0.0, -1.0, -1.0, 1.0, 0.0)]
    )
    b = boxes[rng.integers(len(boxes) - 1, size=8000)]
    across = rng.integers(2, size=len(b)) == 1
    out = rng.choice([-1.0, 1.0], len(b))
    along = rng.uniform(-1.0, 1.0, len(b))
    off = rng.uniform(-0.02, 0.02, len(b))
    u = np.where(across, along * b[:, 2], out * (b[:, 2] + off))
    v = np.where(across, out * (b[:, 3] + off), along * b[:, 3])
    edges = np.stack([b[:, 0] + u * b[:, 4] - v * b[:, 5], b[:, 1] + u * b[:, 5] + v * b[:, 4]], axis=1)
    pts = np.concatenate([uniform, pit, edges]).astype(np.float32)

    got = np.asarray(jax.jit(lambda p: tg.ground(t, p))(jp.asarray(pts)))
    down, hit = np.array([0.0, 0.0, -1.0]), np.zeros(1, np.int32)
    ground_only = np.array([1, 0, 0, 0, 0, 0], np.uint8)
    dist = np.array([mujoco.mj_ray(m, d, np.array([p[0], p[1], 10.0]), down, ground_only, 1, -1, hit) for p in pts])
    assert (dist > 0).all()
    err = np.abs(got - (10.0 - dist))

    p = pts.astype(float)
    nr, nc = drop.shape
    col = np.clip(np.floor((p[:, 0] - x0) / dx), 0, nc - 1).astype(int)
    row = np.clip(np.floor((p[:, 1] - y0) / dy), 0, nr - 1).astype(int)
    listed = boxes[np.asarray(t.cell_boxes)[row, col]]  # (n, K, 6)
    ox, oy = p[:, None, 0] - listed[..., 0], p[:, None, 1] - listed[..., 1]
    du = np.abs(ox * listed[..., 4] + oy * listed[..., 5]) - listed[..., 2]
    dv = np.abs(oy * listed[..., 4] - ox * listed[..., 5]) - listed[..., 3]
    keep = ~(np.abs(np.maximum(du, dv)) <= 1e-5).any(axis=1)
    assert keep.mean() > 0.99
    bound = 2e-5 + 2e-4 * drop[row, col]
    worst = np.argmax(err[keep] / bound[keep])
    assert (err[keep] < bound[keep]).all(), (pts[keep][worst], err[keep][worst], bound[keep][worst])


# -- reset and step -------------------------------------------------------------------


def test_reset_draws_the_joystick_command_and_pose_noise(env, flat):
    """Joystick's command and pose noise, and the base `_place_base` puts
    down for the same key. Separately compiled, the quaternion can differ
    by an ulp."""
    reset_t, reset_f = jax.jit(env.reset), jax.jit(flat.reset)
    place = jax.jit(lambda k: env._place_base(k, env._reset_qpos))
    b = env._base_qadr
    for seed in (0, 1, 2):
        key = jax.random.PRNGKey(seed)
        ts = reset_t(key)
        fs = reset_f(key)
        np.testing.assert_array_equal(np.asarray(ts.info["command"]), np.asarray(fs.info["command"]))
        q = np.asarray(env._qadr)
        np.testing.assert_array_equal(np.asarray(ts.data.qpos)[q], np.asarray(fs.data.qpos)[q])
        np.testing.assert_array_equal(np.asarray(ts.info["rng"]), np.asarray(fs.info["rng"]))
        qpos, info = place(key)
        np.testing.assert_allclose(
            np.asarray(ts.data.qpos)[b : b + 7], np.asarray(qpos)[b : b + 7], rtol=0, atol=1e-6
        )
        for k in ("spawn_kind", "spawn_xy"):
            np.testing.assert_array_equal(np.asarray(ts.info[k]), np.asarray(info[k]), err_msg=k)


@pytest.fixture(scope="module")
def stepping():
    """An env whose command resamples on every step."""
    e = build(env={"command": {"resample_steps": 1}})
    return e, jax.jit(e.reset), jax.jit(e.step)


def test_info_and_metric_keys_match_between_reset_and_step(stepping):
    e, reset, step = stepping
    s0 = reset(jax.random.PRNGKey(0))
    s1 = step(s0, jp.zeros(e.action_size))
    for a, b in ((s0.info, s1.info), (s0.metrics, s1.metrics)):
        assert set(a) == set(b)
        for k in a:
            assert (jp.shape(a[k]), jp.result_type(a[k])) == (jp.shape(b[k]), jp.result_type(b[k])), k
    assert set(TERRAIN_INFO) <= set(s0.info)
    for k, want in TERRAIN_INFO.items():
        assert (jp.shape(s0.info[k]), jp.result_type(s0.info[k])) == want, k
    assert int(s0.info["curriculum_strikes"]) == 0
    assert float(s0.info["served_dist"]) == 0.0 and bool(s0.info["curriculum_free"])
    assert float(s0.info["cheby_min"]) == float(s0.info["cheby_max"])
    for k in SPAWN_INFO:
        np.testing.assert_array_equal(np.asarray(s1.info[k]), np.asarray(s0.info[k]), err_msg=k)
    assert set(TERRAIN_METRICS) <= set(s0.metrics)
    for k in TERRAIN_METRICS:
        assert float(s0.metrics[k]) == 0.0
    assert "nefc_peak" not in s0.info  # warp only


def test_terrain_metrics_after_a_step(stepping):
    e, reset, step = stepping
    a = jp.zeros(e.action_size)
    s0 = reset(jax.random.PRNGKey(0))
    assert int(s0.info["terrain_level"]) == 0
    assert int(s0.info["spawn_kind"]) == tg.SPAWN_PAD
    b = e._base_qadr

    def stepped(dz=0.0, **info):
        info = {k: jp.asarray(v, s0.info[k].dtype) for k, v in info.items()}
        data = s0.data.replace(qpos=s0.data.qpos.at[b + 2].add(dz))
        return step(s0.replace(data=data, info={**s0.info, **info}), a)

    def metrics(s):
        return {k.removeprefix("terrain/"): float(s.metrics[k]) for k in TERRAIN_METRICS}

    s1 = stepped()
    assert float(s1.done) == 0.0
    assert metrics(s1) == {
        "level_per_step": 0.0,
        "level_free_per_step": 0.0,
        "free_per_step": 1.0,
        "on_flat_per_step": 1.0,
        "spawn_fallback_per_step": 0.0,
        "base_contact_at_done": 0.0,
        "early_end": 0.0,
    }
    for kind, want in ((tg.SPAWN_PAD, 0.0), (tg.SPAWN_FEATURE, 0.0), (tg.SPAWN_FALLBACK, 1.0)):
        assert metrics(stepped(spawn_kind=kind))["spawn_fallback_per_step"] == want, kind
    m = metrics(stepped(terrain_level=1))
    assert m["level_per_step"] == 1.0 and m["on_flat_per_step"] == 0.0
    assert (m["level_free_per_step"], m["free_per_step"]) == (1.0, 1.0)
    # A pinned env keeps its level in level_per_step only.
    m = metrics(stepped(terrain_level=1, curriculum_free=False))
    assert (m["level_per_step"], m["level_free_per_step"], m["free_per_step"]) == (1.0, 0.0, 0.0)

    # Base 0.5 m down: the height check ends the episode, and no
    # termination cell is near the floor. An end within EARLY_SEC of the
    # spawn, that bound included, is early.
    s = stepped(dz=-0.5, since_spawn=e._early_steps - 1)
    assert float(s.done) == 1.0
    m = metrics(s)
    assert m["early_end"] == 1.0 and m["base_contact_at_done"] == 0.0
    assert metrics(stepped(dz=-0.5, since_spawn=e._early_steps))["early_end"] == 0.0

    # Base 0.70 m down: the termination cells reach the floor.
    s = stepped(dz=-0.70)
    assert bool(e._base_contact(s.data))
    assert float(s.done) == 1.0
    assert metrics(s)["base_contact_at_done"] == 1.0


def test_step_tracks_the_band_with_the_command_that_drove_it(stepping):
    e, reset, step = stepping
    s0 = reset(jax.random.PRNGKey(1))
    # The planar speed is 0.5. Its vx alone, its norm with the yaw rate,
    # and the resampled command all give a different distance.
    s0 = s0.replace(info={**s0.info, "command": jp.array([0.3, 0.4, 0.8])})
    s1 = step(s0, jp.zeros(e.action_size))
    b = e._base_qadr
    xy = np.asarray(s1.data.qpos[b : b + 2])
    assert not np.array_equal(np.asarray(s1.info["command"]), [0.3, 0.4, 0.8])
    assert float(s1.info["commanded_dist"]) == pytest.approx(0.5 * e.dt)
    served = curriculum.served_step(e._local_linvel(s1.data)[:2], jp.array([0.3, 0.4, 0.8]), e.dt)
    assert float(s1.info["served_dist"]) == pytest.approx(float(served), abs=1e-9)
    np.testing.assert_array_equal(np.asarray(s1.info["last_xy"]), xy)
    r = float(np.abs(xy - np.asarray(s0.info["tile_origin"])).max())
    r0 = float(s0.info["cheby_min"])
    assert float(s1.info["cheby_min"]) == pytest.approx(min(r0, r))
    assert float(s1.info["cheby_max"]) == pytest.approx(max(r0, r))
    assert int(s1.info["since_spawn"]) == 1
    np.testing.assert_array_equal(np.asarray(s1.info["spawn_xy"]), np.asarray(s0.info["spawn_xy"]))


def test_a_tracked_arc_serves_its_commanded_distance(env, monkeypatch):
    """The physics step swapped for a base that holds the body-frame planar
    velocity and yaw rate it reads from data.qvel. Three runs start on the
    flat row under the command (0.6, 0, 6.0) m/s, rad/s. Each base moves
    at 0.6 m/s and turns 4.8 rad in 40 steps on a 0.1 m circle. It ends
    about 0.135 m from its spawn. served_dist reads the env's own linvel
    sensor. Tracking the command serves the 0.48 m commanded. Crabbing
    across it serves nothing. Backing against it serves -0.48 m."""
    b, v = env._base_qadr, env._base_vadr
    dt = env.dt
    vx, vy, wz = command = (0.6, 0.0, 6.0)
    n = 40

    def turn(yaw, xy):
        c, s = jp.cos(yaw), jp.sin(yaw)
        return jp.stack([c * xy[0] - s * xy[1], s * xy[0] + c * xy[1]])

    def kinematic(model, data, action, n_substeps=1):
        quat = data.qpos[b + 3 : b + 7]
        yaw = height_scan.yaw_from_quat(quat, jp)
        rate = data.qvel[v + 5]
        body = turn(-yaw, data.qvel[v : v + 2])
        # One step's displacement in the frame of the yaw it starts from.
        s, c = jp.sin(rate * dt) / rate, (1.0 - jp.cos(rate * dt)) / rate
        arc = jp.stack([s * body[0] - c * body[1], c * body[0] + s * body[1]])
        qpos = data.qpos.at[b : b + 2].add(turn(yaw, arc))
        qpos = qpos.at[b + 3 : b + 7].set(tg.quat_mul(tg.yaw_quat(rate * dt), quat))
        qvel = data.qvel.at[v : v + 2].set(turn(yaw + rate * dt, body))
        return mjx.forward(model, data.replace(qpos=qpos, qvel=qvel, ctrl=action))

    monkeypatch.setattr(mjx_env, "step", kinematic)
    step = jax.jit(lambda state, action: env.step(state, action))
    s0 = jax.jit(env.reset)(jax.random.PRNGKey(0))
    assert int(s0.info["terrain_level"]) == 0
    # reset returns a weak-typed phase and step a strong one. The cast lets
    # every step reuse the first compile.
    info = {**s0.info, "phase": s0.info["phase"].astype(jp.float32), "command": jp.array(command)}
    yaw0 = height_scan.yaw_from_quat(s0.data.qpos[b + 3 : b + 7], jp)
    action = jp.zeros(env.action_size)
    for body, sign in (((vx, vy), 1.0), ((vy, vx), 0.0), ((-vx, vy), -1.0)):
        base = jp.concatenate([turn(yaw0, jp.array(body)), jp.array([0.0, 0.0, 0.0, wz])])
        state = s0.replace(data=s0.data.replace(qvel=s0.data.qvel.at[v : v + 6].set(base)), info=info)
        for _ in range(n):
            state = step(state, action)
            assert float(state.done) == 0.0
        np.testing.assert_allclose(np.asarray(state.info["command"]), command)
        commanded = float(state.info["commanded_dist"])
        assert commanded == pytest.approx(vx * dt * n, rel=1e-5)
        served = float(state.info["served_dist"])
        if sign:
            assert served == pytest.approx(sign * commanded, rel=1e-4), body
        else:
            assert abs(served) < 1e-3 * commanded, body
        walked = np.linalg.norm(np.asarray(state.info["last_xy"]) - np.asarray(state.info["spawn_xy"]))
        assert walked == pytest.approx(2 * vx / wz * math.sin(wz * dt * n / 2), abs=0.01), body
        assert walked < 0.3 * commanded, body


@pytest.mark.parametrize("robot", ["roboto_origin", "asimov_v1"])
def test_ten_steps_are_finite(robot, stepping, asimov):
    """Asimov here checks the generic code paths only. Its terrain is
    untuned."""
    if robot == ROBOT:
        e, reset, step = stepping
    else:
        e = asimov
        reset, step = jax.jit(e.reset), jax.jit(e.step)
    s = reset(jax.random.PRNGKey(7))
    action = jp.zeros(e.action_size)
    for _ in range(10):
        s = step(s, action)
        for leaf in (s.data.qpos, s.data.qvel, s.obs["state"], s.obs["privileged_state"], s.reward):
            assert np.all(np.isfinite(np.asarray(leaf)))
    assert int(s.info["since_spawn"]) == 10


def test_jax_refuses_a_large_box_arena():
    """The default arena builds, so the C engine can use it. Tracing a jax
    reset or step on it raises."""
    e = make_env("terrain", ROBOT_DIR, PRESET)
    assert e._n_ground_boxes > JAX_BOX_LIMIT
    with pytest.raises(ValueError, match=f"{e._n_ground_boxes} ground boxes.*terrain_cpu"):
        jax.jit(e.reset)(jax.random.PRNGKey(0))
    with pytest.raises(ValueError, match="ground boxes"):
        e.step(None, None)


def test_arena_record_names_the_arena(env):
    rec = env.arena_record()
    assert rec["n_boxes"] == len(env._arena.boxes) == 33
    assert rec["n_ground_geoms"] == 1 + 33 + 4
    assert rec["jax_max_contact_points"] == 116
    assert rec["spawn_mode"] == "pad"
    assert rec["ccd_scratch"]["bytes_per_slot"] == 5480
    assert rec["params"]["difficulties"] == [0.5]
