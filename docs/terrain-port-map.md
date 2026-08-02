# Terrain: where to start when a flat keeper exists

Terrain training is deferred until a flat-ground keeper policy exists. The
full working system lives in machinekind/w01-tek's `training/` project; this
is the map of what to port and what to know before reading it.

- The procedural tiled arena on one shared heightfield with a lookup grid,
  in `terrain.py`.
- The legged_gym-style promote and demote curriculum with the demote
  projection fix, in `terrain_env.py:240-290`, commit `a1b8aef`.
- The teleport auto-reset wrapper, in `terrain_wrapper.py`.
- The flat recovery row, commit `a0df2a9`.
- Three arena kinds with fingerprint guards.
- Chebyshev measurement radii, commit `61da321`.
- Terrain-relative height, clearance, and contact, commits `37816dd` and
  `4e78a47`.
- The MJWarp heightfield contact cap of 50 per geom pair. Large flat
  colliders must be decomposed into small cells. A humanoid pelvis or torso
  box will hit this cap hard. Commits `01f177e` and `ff17d58`. The gate is
  fd-level printf capture, `check_terrain.py:69-101`.

Corrections to carry when reading w01-tek: `spawn_level` pins initial spawns
only, and respawns still move. The smoke grep for overflow was removed as
dead code, and `check-terrain --backend warp` is the real gate. The
run-report template to copy is the v2 first-run report. The v4 report uses
a different structure.
