"""Joystick velocity tracking on a procedural terrain arena (task=terrain).

The task is Joystick with its floor plane replaced by an arena
(terrain/scene.py) and one extra config block, `terrain`. Every reward
term, observation and command is Joystick's. What changes is where the
ground is:

- Base height, the `height` observation and the fall check read the base
  above the ground under it.
- Foot contact and clearance read each foot capsule's underside against
  the exact ground at sole samples across its footprint.
- On the flat row's rectangle both foot measurements are the flat floor's
  own expressions. The ground there reads exactly 0. Base height, foot
  contact and foot clearance there equal Joystick's bit for bit on the same
  data. The physics is not Joystick's. The flat row is part of the arena's
  ground, not Joystick's plane.
- Base contact can end an episode: a collision geom of a termination body
  whose lowest point comes within `base_contact.tol` of the ground.

Base height reads `terrain_geometry.height`, a bilinear blend of the
arena's lookup grid. Inside a box, within one cell of an edge that falls
between node lines, the blend reads low. Outside a box, beside a face on a
node line, it reads up to the full step high. terrain_geometry's module
docstring states both bands.

Feet read `terrain_geometry.ground`: the heightfield, or the top of a box
that contains the point. `terrain_geometry.SOLE_SAMPLES` lists the 17
samples per capsule. Measured on the default arena with a level foot
resting on the exact ground at a uniform xy and yaw, 4000 placements per
tile on the stair and box tiles:
- Foot centre on a box top: no contact in 0.16% of random_grid placements,
  0.06% on discrete_obstacles and 1.3% on stair treads. Three axis points
  per capsule read on the lookup miss 6.8%, 1.1% and 5.9%.
- A box edge under the foot: no contact in 4.4% of random_grid, 3.3% of
  discrete_obstacles and 2.8% of stair placements. Read on the lookup, the
  same feet miss 35%, 27% and 12%. Of the remaining misses, 80% on stairs
  and 47% on random_grid rest on a capsule's outer 0.2 r, where the
  surface normal is more than 53 degrees off vertical. The others rest on
  a box edge between two samples. A missed foot reads its clearance over
  the lower ground, up to a riser.
- No sample reads the capsule lower than it is, so clearance never reads
  below the true gap. Beside a riser whose face lies on a node line, the
  lookup reads the ground up to a full riser high, 15 cm at d = 1. Beside a
  face mid-cell it reads up to half a riser high. A level foot resting on a
  tread, read on the lookup at its three axis points per capsule, read a
  clearance as low as -10.7 cm. A toe near a riser face read on the lookup
  would read contact and a negative clearance.

Base contact reads the dilated spawn grid at each termination collider's
lowest point. A termination cell resting on the default arena's stair and
box tiles, at a uniform xy, yaw and tilt up to 45 degrees, reads late (at
least `tol`, 1 cm) in 7.9% of stair placements and 4.9% of rubble and
obstacle placements. The lookup reads 19% and 18% late. The exact ground
the feet read would read 19% and 16% late. A lowest point resting on a box
top reads on time on stairs. Within one cell of a rubble or obstacle
edge, 0.02% of top points read more than 1 cm low. The late reads are
cells tilted over a box edge with the lowest point past it. Of the late
reads, 64% on stairs and 81% on rubble and obstacles have the lowest
point within two cells of the edge. The dilated grid's cost is an early
read. It reads up to the box's step high within two cells beside a box,
and more than 1 cm high on 15% of the arena's surface. Lifted 2 cm, a
cell reads contact in 9.4% of stair and 9.0% of rubble and obstacle
placements. The exact ground never reads early.

Spawns. Each episode starts on one tile, a (level, terrain type) pair, at
its pad or, in feature mode, at a level-footed point among its features.
The base height is the lowest that keeps every robot collider out of the
ground: `terrain_geometry.spawn_z` over sample points on each collider at
the reset pose. The binding point keeps the float it has over the flat
floor at the reset keyframe. Pad spawns read the lookup. The pads are
flat, so a pad spawn sits at the pad height plus the reset height. Feature
spawns read the dilated spawn grid. A feature candidate passes when the
sole points' spread is small on both the spawn grid and the lookup. On the
default arena above difficulty 0.3, 1000 feature spawns on rubble and
obstacle tiles put no robot collider into the ground. Over about 240,000
stair spawns the worst foot hung 2.3 cm over its ground. Over about
118,000 rubble spawns the worst foot hung 4.7 cm. In that case the whole
robot sits that high beside a box, with both feet off the ground. Every
sole point lies within two cells of the box, where the spawn grid reads
high.

jax limits. The jax backend runs narrowphase on every robot-box pair, so
it refuses an arena of more than JAX_BOX_LIMIT ground boxes at trace time.
jax's heightfield collider misses the low-side prisms of a rolled or yawed
box, so a tilted box collider gets partial heightfield contact there. The
`max_contact_points` cap keeps the deepest contacts on jax only. Physics on
the C engine and MJWarp is unchanged by it.
"""

from __future__ import annotations

import math
from dataclasses import asdict
from typing import NamedTuple

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env

from humanoid_lab import sim_budget
from humanoid_lab.envs import terrain_geometry as tg
from humanoid_lab.envs.backend import resolve_backend
from humanoid_lab.envs.joystick import Joystick, check_pure_draw_ranges
from humanoid_lab.envs.joystick import default_config as joystick_default_config
from humanoid_lab.robot.build import compile_spec
from humanoid_lab.terrain import TYPES, ArenaParams, fingerprint, params_to_dict, scene
from humanoid_lab.terrain.arena import GENERATOR_VERSION
from humanoid_lab.terrain.config import (
    arena_for,
    ccd_scratch,
    config_from_params,
    params_from_config,
    require_terrain_budgets,
)

# fold_in domains of the terrain draws. `fold_in(key, i)` equals
# `split(key, n)[i]` for i < n, so these sit far above any split width the
# env uses, and above Joystick's pure-draw domain 0x100 + i.
TERRAIN_DRAW = 0x200
BIAS_DRAW = 0x201
# An episode that ends within this many seconds of its spawn counts as an
# early end (metric terrain/early_end).
EARLY_SEC = 1.0
# Feature spawn candidates keep every collider this far (m) inside the tile
# beyond the colliders' own reach.
SPAWN_EDGE_SLACK = 0.05
# More ground boxes than this refuse to trace on jax.
JAX_BOX_LIMIT = 128
# Contacts warp writes per heightfield pair at most. The contact budget
# warning reads it against the ground-pairing collider count.
HFIELD_CONTACTS_PER_PAIR = 4
# The command bias replaces these command probabilities.
BIAS_KEYS = (
    "zero_prob",
    "pure_wz_prob",
    "pure_vy_prob",
    "pure_slow_prob",
    "pure_fast_prob",
    "pure_back_prob",
)

_COLLIDER_TYPES = {
    int(mujoco.mjtGeom.mjGEOM_BOX): "box",
    int(mujoco.mjtGeom.mjGEOM_CAPSULE): "capsule",
    int(mujoco.mjtGeom.mjGEOM_SPHERE): "sphere",
}

TERRAIN_METRICS = (
    "terrain/level_per_step",
    "terrain/on_flat_per_step",
    "terrain/spawn_fallback_per_step",
    "terrain/base_contact_at_done",
    "terrain/early_end",
)


def default_config() -> config_dict.ConfigDict:
    """Joystick's config plus exactly one block, `terrain`. None of its
    numbers has been tuned by training."""
    cfg = joystick_default_config()
    cfg.terrain = config_dict.create(
        # Every ArenaParams field, with a cap for every terrain type (1.0 is
        # uncapped). See terrain/config.py.
        arena=config_dict.ConfigDict(config_from_params(ArenaParams())),
        spawn=config_dict.create(
            mode="pad",  # pad | feature
            yaw=True,  # U(-pi, pi) at every spawn
            # m, uniform per axis, so a square. The construction refuses a
            # jitter whose diagonal puts a foot past the pad.
            pad_jitter=0.15,
            # Candidates per feature spawn before the pad fallback. Over the
            # default arena's tiles above difficulty 0.3, 16 keeps each terrain
            # type's acceptance at 0.83 or more. The hardest inverted-slope
            # tiles accept about 0.78. 8 drops inverted slopes to 0.60.
            feature_candidates=16,
            # m, the largest spread (max minus min) under the sole points a
            # feature spawn accepts, read on both the dilated spawn grid and
            # the lookup. Rejects straddled risers, a foot beside a box, and
            # slopes above 5.1 to 8.5 degrees, by the yaw.
            feature_max_spread=0.02,
            # >= 0 pins each env's FIRST spawn to this level.
            level=-1,
            # The first level is U[0, max(1, round(levels * frac))).
            init_level_frac=0.5,
            # s after a spawn in which a fall pays no termination penalty.
            # The episode still ends. 0 is off.
            grace_sec=0.0,
        ),
        curriculum=config_dict.create(
            demote_fraction=0.5, demote_strikes=1, pinned_frac=0.0, pinned_flat_frac=0.0
        ),
        # End the episode when a termination body's collider comes within
        # tol (m) of the ground.
        base_contact=config_dict.create(terminate=True, tol=0.01),
        # Off the flat row only: no-progress grace (s, 0 keeps Joystick's)
        # and a scale on its p_max.
        no_progress=config_dict.create(grace_sec=0.0, p_max_scale=1.0),
        # Off the flat row only: these command probabilities replace
        # Joystick's. pure_back_prob stays 0, because roboto_origin's
        # back_vx lies outside its vx box.
        command_bias=config_dict.create(
            enable=False,
            zero_prob=0.10,
            pure_wz_prob=0.10,
            pure_vy_prob=0.05,
            pure_slow_prob=0.30,
            pure_fast_prob=0.0,
            pure_back_prob=0.0,
        ),
        # jax only: the deepest-contact cap. None derives it as 4 per
        # ground-pairing collider plus the robot-robot slots.
        jax_contacts=config_dict.create(max_contact_points=None),
    )
    return cfg


def bias_command_config(command, bias) -> config_dict.ConfigDict:
    """A copy of the command config with the bias block's probabilities."""
    c = config_dict.ConfigDict(command.to_dict() if hasattr(command, "to_dict") else command)
    for key in BIAS_KEYS:
        c[key] = float(bias[key])
    return c


class TerrainKeys(NamedTuple):
    type: jax.Array
    level: jax.Array
    spawn: jax.Array
    bias: jax.Array
    next: jax.Array


def _pairs(ct_a, ca_a, ct_b, ca_b) -> bool:
    return ((int(ct_a) & int(ca_b)) | (int(ct_b) & int(ca_a))) != 0


class TerrainJoystick(Joystick):
    def __init__(self, robot_dir, preset_name, config=None, config_overrides=None, actuator_overrides=None):
        # See _init_mirror_maps.
        self._mirror_maps_wait = True
        super().__init__(
            robot_dir, preset_name, config or default_config(), config_overrides, actuator_overrides
        )
        tc = self._config.terrain
        self._build_robot_tables()

        jitter_reach = tc.spawn.pad_jitter * math.sqrt(2.0) + self._feet_reach
        if jitter_reach > self._tables.pad_radius:
            raise ValueError(
                f"terrain.spawn.pad_jitter {tc.spawn.pad_jitter} puts a foot off the pad: "
                f"its diagonal {tc.spawn.pad_jitter * math.sqrt(2.0):.3f} m plus the "
                f"{self._feet_reach:.3f} m reach of the feet is {jitter_reach:.3f} m, past "
                f"the {self._tables.pad_radius} m pad radius"
            )
        if tc.spawn.level >= self._tables.n_rows:
            raise ValueError(
                f"terrain.spawn.level {tc.spawn.level} is past the arena's "
                f"{self._tables.n_rows} levels"
            )
        if tc.spawn.init_level_frac > 1:
            raise ValueError(
                f"terrain.spawn.init_level_frac {tc.spawn.init_level_frac} draws first levels "
                f"past the arena's {self._tables.n_rows} levels. It is at most 1."
            )
        self._feature_half = self._tables.tile_size / 2 - self._collider_reach - SPAWN_EDGE_SLACK
        if tc.spawn.mode == "feature" and self._feature_half <= 0:
            raise ValueError(
                f"feature spawns need a {self._collider_reach:.3f} m collider reach plus "
                f"{SPAWN_EDGE_SLACK} m to fit inside a {self._tables.tile_size} m tile"
            )

        self._bias_cmd = None
        if tc.command_bias.enable:
            self._bias_cmd = bias_command_config(self._config.command, tc.command_bias)
            check_pure_draw_ranges(self._bias_cmd)

        budget = HFIELD_CONTACTS_PER_PAIR * self._n_ground_colliders
        if self._naconmax_per_env is not None and self._naconmax_per_env < budget:
            print(
                f"WARNING: naconmax_per_env {self._naconmax_per_env} is under "
                f"{HFIELD_CONTACTS_PER_PAIR} contacts for each of the "
                f"{self._n_ground_colliders} ground-pairing colliders ({budget}). Warp "
                "writes up to that many per heightfield pair."
            )
        margin = self._z0 - float(self._config.fall.min_height)
        if margin < self._tables.max_step + 0.05:
            print(
                f"WARNING: the reset base height sits {margin:.3f} m above "
                f"fall.min_height, under the arena's {self._tables.max_step:.3f} m "
                "largest lookup step plus 0.05 m. With the base over a tread one "
                "step above the feet, base height reads that step low, so a "
                "standing robot can read as fallen."
            )

        self._grace_steps = round(tc.spawn.grace_sec / self.dt)
        self._early_steps = round(EARLY_SEC / self.dt)
        self._n_ground_boxes = len(self._arena.boxes) + len(scene.APRON_GEOMS)

        self._mirror_maps_wait = False
        if self._config.symmetry.enable:
            self._init_mirror_maps()

    def _init_mirror_maps(self) -> None:
        """Joystick.__init__ sizes the mirror maps by evaluating the obs
        catalog, which here reads the robot tables built after that call.
        This env's __init__ builds them last instead."""
        if not self._mirror_maps_wait:
            super()._init_mirror_maps()

    # -- scene ------------------------------------------------------------------
    def _customize_spec(self, spec: mujoco.MjSpec) -> None:
        """Replace the floor plane with the arena. The untouched spec
        compiles first: that flat model is the settle model and supplies
        the render statistics."""
        cfg = self._config
        tc = cfg.terrain
        if tc.spawn.mode not in ("pad", "feature"):
            raise ValueError(f"terrain.spawn.mode must be 'pad' or 'feature', got {tc.spawn.mode!r}")
        if tc.spawn.mode == "feature" and int(tc.spawn.feature_candidates) < 1:
            raise ValueError(
                f"terrain.spawn.feature_candidates must be at least 1 in feature mode, got "
                f"{tc.spawn.feature_candidates}. Pad spawns need terrain.spawn.mode=pad."
            )
        naccd = cfg.sim.get("naccdmax_per_env")
        nacon = cfg.sim.naconmax_per_env
        if nacon is None:
            nacon = self._robot_spec.sim_budget.get("naconmax_per_env")
        if naccd is not None and nacon is not None and naccd > nacon:
            raise ValueError(
                f"sim.naccdmax_per_env {naccd} exceeds the naconmax_per_env budget "
                f"{nacon}. MJWarp refuses a CCD pool larger than the contact pool."
            )

        self._arena = arena_for(params_from_config(tc.arena.to_dict()))
        # Refuses a crowded arena before any compile.
        self._tables = tg.tables_from_arena(self._arena)
        backend = resolve_backend(cfg.sim.backend)
        gc = scene.ground_contact(scene.floor_plane(spec))
        self._check_colliders(spec, gc)
        ground = scene.ground_pairing_geoms(spec, gc)
        box_box = any(g.type == mujoco.mjtGeom.mjGEOM_BOX for g in ground)
        self._ccd_slot_bytes = sim_budget.ccd_slot_bytes(spec.option.ccd_iterations, box_box=box_box)
        # Before any warp work: a terrain run must not fall back on the
        # flat-floor budget.
        require_terrain_budgets(backend, cfg.sim, ccd_slot_bytes=self._ccd_slot_bytes)

        self._flat_model = compile_spec(spec)
        cap = tc.jax_contacts.max_contact_points
        if cap is None:
            cap = HFIELD_CONTACTS_PER_PAIR * len(ground) + self._robot_robot_jax_slots(spec)
        self._jax_contact_cap = int(cap)
        flat = self._flat_model.stat
        scene.attach_terrain(
            spec, self._arena, stat=(flat.extent, flat.center), max_contact_points=self._jax_contact_cap
        )

    def _settle_model(self) -> mujoco.MjModel:
        # The robot's own plane. The keyframe's xy is a four-tile corner of
        # the arena, so settling there would be wrong.
        return self._flat_model

    def _check_colliders(self, spec: mujoco.MjSpec, gc: scene.GroundContact) -> None:
        """Refuse robot colliders the terrain rules cannot read, before
        anything compiles."""
        rs = self._robot_spec
        robot = [g for g in spec.geoms if g.parent != spec.worldbody and (g.contype or g.conaffinity)]
        unpaired = [
            g.name for g in robot if not _pairs(g.contype, g.conaffinity, gc.contype, gc.conaffinity)
        ]
        if unpaired:
            raise ValueError(
                f"robot colliders {unpaired} do not pair with the ground (contype "
                f"{gc.contype}, conaffinity {gc.conaffinity}). Every robot collider "
                "must, so spawns and contacts see all of them."
            )
        bad = [f"{g.name} ({g.type.name})" for g in robot if int(g.type) not in _COLLIDER_TYPES]
        if bad:
            raise ValueError(
                f"robot colliders {bad} are not a box, capsule or sphere. The terrain "
                "spawn and base-contact rules have no lowest point for them."
            )
        feet = [spec.geom(n) for n in rs.foot_geoms]
        round_types = (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_SPHERE)
        if any(g.type not in round_types for g in feet):
            raise ValueError(f"foot geoms {list(rs.foot_geoms)} must be capsules or spheres")
        term = [g for g in robot if g.parent.name in rs.termination_bodies]
        if self._config.terrain.base_contact.terminate and not term:
            raise ValueError(
                f"terrain.base_contact.terminate is on, and the termination bodies "
                f"{list(rs.termination_bodies)} carry no collision geom"
            )

    @staticmethod
    def _robot_robot_jax_slots(spec: mujoco.MjSpec) -> int:
        """jax contact slots of robot-robot pairs, 0 when no two robot
        colliders on different bodies can pair.

        With such pairs it counts the contacts jax allocates for a copy of
        the spec whose floor collides with nothing."""
        robot = [g for g in spec.geoms if g.parent != spec.worldbody and (g.contype or g.conaffinity)]
        pairs = any(
            a.parent != b.parent and _pairs(a.contype, a.conaffinity, b.contype, b.conaffinity)
            for i, a in enumerate(robot)
            for b in robot[i + 1 :]
        )
        if not pairs:
            return 0
        copy = spec.copy()
        plane = scene.floor_plane(copy)
        plane.contype = 0
        plane.conaffinity = 0
        d = mjx.make_data(compile_spec(copy), impl="jax")
        return int(d._impl.contact.dist.shape[0])

    def _build_robot_tables(self) -> None:
        """Robot tables at the reset pose with the base at xy (0, 0), host
        side: colliders, spawn sample points, foot and termination geoms."""
        m = self._mj_model
        rs = self._robot_spec
        b = self._base_qadr

        ground_ids = scene.ground_geom_ids(m)
        ground_bits = {(int(m.geom_contype[g]), int(m.geom_conaffinity[g])) for g in ground_ids}
        body = np.asarray(m.geom_bodyid)
        collides = (np.asarray(m.geom_contype) != 0) | (np.asarray(m.geom_conaffinity) != 0)
        colliders = np.flatnonzero((body != 0) & collides)

        # attach_terrain copies the floor's contact attributes onto every
        # ground geom, so the spec-level pairing check holds for all of them.
        for g in colliders:
            for ct, ca in ground_bits:
                if not _pairs(m.geom_contype[g], m.geom_conaffinity[g], ct, ca):
                    raise ValueError(
                        f"robot collider {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, int(g))} "
                        f"does not pair with a ground geom of contype {ct}, conaffinity {ca}"
                    )

        d = mujoco.MjData(m)
        qpos = np.asarray(self._reset_qpos, dtype=float).copy()
        qpos[b : b + 2] = 0.0
        d.qpos[:] = qpos
        mujoco.mj_kinematics(m, d)

        points, radii, owner = [], [], []
        for g in colliders:
            c = d.geom_xpos[g]
            R = d.geom_xmat[g].reshape(3, 3)
            size = m.geom_size[g]
            kind = _COLLIDER_TYPES[int(m.geom_type[g])]
            if kind == "capsule":
                pts = [c + t * size[1] * R[:, 2] for t in tg.SPAWN_CAPSULE_POINTS]
                r = size[0]
            elif kind == "sphere":
                pts, r = [c], size[0]
            else:
                pts = [c + R @ ((2 * np.array(s) - 1) * size) for s in np.ndindex(2, 2, 2)]
                r = 0.0
            points += pts
            radii += [r] * len(pts)
            owner += [int(g)] * len(pts)
        points = np.array(points)
        radii = np.array(radii)
        owner = np.array(owner)
        bottom = points[:, 2] - radii
        u = points[:, :2] - qpos[b : b + 2]
        reach = np.linalg.norm(u, axis=1) + radii
        sole = np.isin(owner, self._foot_geom_ids)

        self._n_ground_colliders = len(colliders)
        self._spawn_u = jp.asarray(u, dtype=jp.float32)
        self._spawn_lift = jp.asarray(bottom - bottom.min(), dtype=jp.float32)
        self._sole_idx = np.flatnonzero(sole)
        self._spawn_u_sole = self._spawn_u[self._sole_idx]
        self._spawn_owner = owner
        self._feet_reach = float(reach[sole].max())
        self._collider_reach = float(reach.max())
        self._z0 = float(qpos[b + 2])
        self._reset_quat = jp.asarray(self._reset_qpos)[b + 3 : b + 7]

        foot_types = np.asarray(m.geom_type)[self._foot_geom_ids]
        half = np.where(
            foot_types == mujoco.mjtGeom.mjGEOM_CAPSULE, m.geom_size[self._foot_geom_ids, 1], 0.0
        )
        self._foot_half = jp.asarray(half, dtype=jp.float32)

        term_bodies = [m.body(n).id for n in rs.termination_bodies]
        term = [g for g in colliders if body[g] in term_bodies]
        types = np.asarray(m.geom_type)
        sizes = np.asarray(m.geom_size)
        self._term_box = np.array([g for g in term if types[g] == mujoco.mjtGeom.mjGEOM_BOX], dtype=int)
        self._term_capsule = np.array(
            [g for g in term if types[g] == mujoco.mjtGeom.mjGEOM_CAPSULE], dtype=int
        )
        self._term_sphere = np.array([g for g in term if types[g] == mujoco.mjtGeom.mjGEOM_SPHERE], dtype=int)
        self._term_box_half = jp.asarray(sizes[self._term_box], dtype=jp.float32)
        self._term_capsule_half = jp.asarray(sizes[self._term_capsule, 1], dtype=jp.float32)
        self._term_capsule_r = jp.asarray(sizes[self._term_capsule, 0], dtype=jp.float32)
        self._term_sphere_r = jp.asarray(sizes[self._term_sphere, 0], dtype=jp.float32)

    def _check_jax_box_limit(self) -> None:
        if self._backend == "jax" and self._n_ground_boxes > JAX_BOX_LIMIT:
            raise ValueError(
                f"the arena has {self._n_ground_boxes} ground boxes, over the jax "
                f"limit of {JAX_BOX_LIMIT}. jax runs narrowphase on every one of the "
                f"{self._n_ground_colliders * self._n_ground_boxes} robot-box pairs. "
                "Use a smaller arena on jax, e.g. experiment=terrain_cpu, or warp."
            )

    # -- ground reads -----------------------------------------------------------
    def _ground_height(self, xy):
        return tg.height(self._tables, xy)

    def _base_height(self, data):
        """Base height above the ground under the base. On the flat row it
        is the free joint's z bit for bit."""
        b = self._base_qadr
        return data.qpos[b + 2] - self._ground_height(data.qpos[b : b + 2])

    def _foot_gaps(self, data):
        """(gap (G, S), centre z (G,), in band (G,)) for the foot geoms. gap
        is the geom's underside above the exact ground at each of its
        `tg.SOLE_SAMPLES`."""
        ids = self._foot_geom_ids
        centre = data.geom_xpos[ids]
        axis = data.geom_xmat[ids][:, :, 2]
        gap = tg.sole_gaps(self._tables, centre, axis, self._foot_half, self._foot_geom_radius)
        return gap, centre[:, 2], tg.in_band(self._tables, centre[:, :2])

    def _foot_contact(self, data):
        """Per-foot contact bool. A geom touches when its underside comes
        within 5 mm of the exact ground at any sole sample. On the flat row
        it is the flat floor's centre test."""
        gap, centre_z, band = self._foot_gaps(data)
        touch = jp.where(band, centre_z < self._foot_geom_radius + 0.005, jp.min(gap, axis=-1) < 0.005)
        per_geom = touch.astype(jp.float32)
        return jp.zeros(self._n_feet).at[self._foot_geom_foot_idx].max(per_geom) > 0

    def _foot_clearance(self, data):
        """Per-foot sole height above the exact ground: the smallest sole
        gap over the foot's geoms. On the flat row it is the flat floor's
        site expression.

        A planted foot on a slope of angle a reads r / cos(a) - r, under
        1 mm at 20 degrees."""
        site = self._foot_site_pos(data)
        flat = site[:, 2] - self._foot_site_sole_offset
        gap, _, _ = self._foot_gaps(data)
        terrain = jp.full(self._n_feet, jp.inf).at[self._foot_geom_foot_idx].min(jp.min(gap, axis=-1))
        return jp.where(tg.in_band(self._tables, site[:, :2]), flat, terrain)

    def _base_contact(self, data):
        """Whether any termination collider's lowest point is within
        base_contact.tol of the ground under that point.

        The ground is the dilated spawn grid. A lowest point resting on a
        box top reads on time. No stair top reads low. The rubble figure is
        in terrain_geometry's module docstring. A cell tilted over a box
        edge, with its lowest point past the edge, can read late. The grid
        holds the box top one node past the box's last node. It falls to
        the lower ground over the next cell. The read is late where that
        fall exceeds the lowest point's own drop by tol. Most late reads
        have the lowest point within two cells of the edge. The module
        docstring gives the measured rates. A chessboard corner left
        unfilled touches nothing, so contact registers when a filled cell
        reaches the ground."""
        tol = self._config.terrain.base_contact.tol

        def ground(xy):
            return tg.height(self._tables, xy, "spawn")

        hits = []
        if self._term_box.size:
            xy, z = tg.lowest_point_box(
                data.geom_xpos[self._term_box], data.geom_xmat[self._term_box], self._term_box_half
            )
            hits.append(z - ground(xy) < tol)
        if self._term_capsule.size:
            xy, z = tg.lowest_point_capsule(
                data.geom_xpos[self._term_capsule],
                data.geom_xmat[self._term_capsule],
                self._term_capsule_half,
                self._term_capsule_r,
            )
            hits.append(z - ground(xy) < tol)
        if self._term_sphere.size:
            xy, z = tg.lowest_point_sphere(data.geom_xpos[self._term_sphere], self._term_sphere_r)
            hits.append(z - ground(xy) < tol)
        if not hits:
            return jp.zeros((), bool)
        return jp.any(jp.concatenate(hits))

    def _fall(self, data, gravity):
        fall = super()._fall(data, gravity)
        if self._config.terrain.base_contact.terminate:
            fall = fall | self._base_contact(data)
        return fall

    def _on_flat(self, level):
        """Whether `level` is the flat row. Always False without one."""
        if not self._tables.flat_row:
            return jp.zeros(jp.shape(level), bool)
        return level == 0

    # -- spawns -----------------------------------------------------------------
    def _terrain_keys(self, rng) -> TerrainKeys:
        return TerrainKeys(*jax.random.split(jax.random.fold_in(rng, TERRAIN_DRAW), 5))

    def draw_spawn(self, rng, ttype, level):
        """(xy, yaw, kind) of a spawn on tile (level, ttype). kind 0 is the
        pad, 1 a feature point, 2 the pad after no candidate qualified."""
        s = self._config.terrain.spawn
        return tg.spawn_draw(
            rng,
            self._tables,
            ttype,
            level,
            feature=s.mode == "feature",
            yaw_enable=bool(s.yaw),
            pad_jitter=float(s.pad_jitter),
            k=int(s.feature_candidates),
            half_extent=self._feature_half,
            max_spread=float(s.feature_max_spread),
            u_sole=self._spawn_u_sole,
        )

    def spawn_qpos(self, qpos, xy, yaw, kind):
        """`qpos` with the base moved to `xy`, turned by `yaw` and lifted
        clear of the ground. Writes the base slots only.

        Kind 1 reads the dilated spawn grid, kinds 0 and 2 the lookup. The
        yaw composes onto the reset quaternion, never onto `qpos`'s own."""
        b = self._base_qadr
        u, lift, z0 = self._spawn_u, self._spawn_lift, self._z0
        z_plain = tg.spawn_z(self._tables, u, lift, z0, xy, yaw, "lookup")
        z_spawn = tg.spawn_z(self._tables, u, lift, z0, xy, yaw, "spawn")
        z = jp.where(kind == tg.SPAWN_FEATURE, z_spawn, z_plain)
        quat = tg.quat_mul(tg.yaw_quat(yaw), self._reset_quat)
        return qpos.at[b : b + 2].set(xy).at[b + 2].set(z).at[b + 3 : b + 7].set(quat)

    def _place_base(self, rng, qpos):
        tc = self._config.terrain
        t = self._tables
        k = self._terrain_keys(rng)
        ttype = jax.random.randint(k.type, (), 0, len(TYPES))
        level = tg.first_level(k.level, t.n_rows, float(tc.spawn.init_level_frac), int(tc.spawn.level))
        xy, yaw, kind = self.draw_spawn(k.spawn, ttype, level)
        qpos = self.spawn_qpos(qpos, xy, yaw, kind)
        origin = t.origin_xy[level, ttype]
        r0 = tg.chebyshev(xy, origin)
        info = {
            "terrain_type": ttype,
            "terrain_level": level,
            "terrain_rng": k.next,
            "spawn_xy": xy,
            "last_xy": xy,
            "spawn_kind": kind,
            "tile_origin": origin,
            "cheby_min": r0,
            "cheby_max": r0,
            "commanded_dist": jp.zeros(()),
            "since_spawn": jp.zeros((), jp.int32),
            "curriculum_strikes": jp.zeros((), jp.int32),
        }
        if self._backend == "warp":
            for key in ("nefc_peak", "nacon_peak", "ncollision_peak"):
                info[key] = jp.zeros((), jp.int32)
        return qpos, info

    # -- reset / step -------------------------------------------------------------
    def reset(self, rng: jax.Array) -> mjx_env.State:
        self._check_jax_box_limit()
        state = super().reset(rng)
        info = dict(state.info)
        metrics = {**state.metrics, **{k: jp.zeros(()) for k in TERRAIN_METRICS}}
        obs = state.obs
        if self._bias_cmd is not None:
            biased = self._draw_command(self._terrain_keys(rng).bias, self._bias_cmd)
            info["command"] = jp.where(self._on_flat(info["terrain_level"]), info["command"], biased)
            if "progress_ema" in info:
                info["progress_ema"] = self._cmd_speed(info["command"])
            obs = self._mirror_obs(self._build_obs(state.data, info), info)
        return state.replace(obs=obs, metrics=metrics, info=info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        self._check_jax_box_limit()
        # The command that drives this step. super().step may resample it.
        cmd = state.info["command"]
        nxt = super().step(state, action)
        info = dict(nxt.info)
        b = self._base_qadr
        xy = nxt.data.qpos[b : b + 2]
        info["since_spawn"] = info["since_spawn"] + 1
        info["last_xy"] = xy
        info["commanded_dist"] = info["commanded_dist"] + jp.linalg.norm(cmd[:2]) * self.dt
        r = tg.chebyshev(xy, info["tile_origin"])
        info["cheby_min"] = jp.minimum(info["cheby_min"], r)
        info["cheby_max"] = jp.maximum(info["cheby_max"], r)

        level = info["terrain_level"]
        done = nxt.done > 0
        metrics = {
            **nxt.metrics,
            "terrain/level_per_step": level.astype(jp.float32),
            "terrain/on_flat_per_step": self._on_flat(level).astype(jp.float32),
            "terrain/spawn_fallback_per_step": (info["spawn_kind"] == tg.SPAWN_FALLBACK).astype(
                jp.float32
            ),
            "terrain/base_contact_at_done": (done & self._base_contact(nxt.data)).astype(jp.float32),
            "terrain/early_end": (done & (info["since_spawn"] <= self._early_steps)).astype(jp.float32),
        }
        return nxt.replace(metrics=metrics, info=info)

    # -- overrides inside Joystick.step -------------------------------------------
    def _compute_rewards(self, data, info, action, first_contact, contact):
        rewards, fall = super()._compute_rewards(data, info, action, first_contact, contact)
        if self._grace_steps > 0:
            # Reassigning an existing key keeps the dict order, so the
            # scaled sum adds the terms in the same order.
            rewards["termination"] = jp.where(
                info["since_spawn"] < self._grace_steps, 0.0, rewards["termination"]
            )
        return rewards, fall

    def _next_command(self, rng, info):
        """Joystick's draw on the flat row, the bias draw elsewhere. Both
        draws run, and each gates on static probabilities."""
        base = super()._next_command(rng, info)
        if self._bias_cmd is None:
            return base
        biased = self._draw_command(jax.random.fold_in(rng, BIAS_DRAW), self._bias_cmd)
        return jp.where(self._on_flat(info["terrain_level"]), base, biased)

    def _no_progress_params(self, info):
        grace, p_max = super()._no_progress_params(info)
        tnp = self._config.terrain.no_progress
        off_flat = ~self._on_flat(info["terrain_level"])
        if tnp.grace_sec > 0:
            grace = jp.where(off_flat, tnp.grace_sec, grace)
        if tnp.p_max_scale != 1.0:
            p_max = p_max * jp.where(off_flat, tnp.p_max_scale, 1.0)
        return grace, p_max

    # -- observations -------------------------------------------------------------
    def _obs_catalog(self, data, info):
        catalog = super()._obs_catalog(data, info)
        catalog["height"] = self._base_height(data)[None]
        return catalog

    # -- records ----------------------------------------------------------------
    def arena_record(self) -> dict:
        """The arena block of run.json: what was built and how this env
        reads it."""
        a = self._arena
        sim = self._config.sim
        return {
            "generator_version": GENERATOR_VERSION,
            "fingerprint": fingerprint(a),
            "params": params_to_dict(a.spec.params),
            "hfield": asdict(a.spec.hfield),
            "n_boxes": len(a.boxes),
            "n_ground_geoms": len(scene.ground_geom_ids(self._mj_model)),
            "max_step": self._tables.max_step,
            "spawn_mode": str(self._config.terrain.spawn.mode),
            "jax_max_contact_points": self._jax_contact_cap,
            "ccd_scratch": ccd_scratch(sim, self._ccd_slot_bytes, self._naconmax_per_env),
        }
