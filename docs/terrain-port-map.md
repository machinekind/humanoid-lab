# Terrain: where to start when a flat keeper exists

Terrain training is deferred until a flat-ground keeper policy exists. The
full working system lives in the `training/` project of machinekind/w01-tek.
This is the map of what to port and what to know before reading it. Paths
are relative to `training/wojtek_rl/`, and line numbers match w01-tek
`e44c1a9`.

- The procedural tiled arena on one shared heightfield with a lookup grid,
  in `terrain.py`.
- The promote and demote curriculum, `curriculum_step` in
  `terrain_env.py:360-425`. It follows legged_gym. Commit `143cfc8`
  projects the demote threshold onto a full episode.
- The auto-reset wrapper, `TerrainAutoResetWrapper` in `terrain_wrapper.py`.
  When an episode ends, it teleports the robot to a tile at the level
  `curriculum_step` picks.
- The flat row, `task.env.terrain.flat_row`. It adds a flat level 0 to the
  train arena. Commit `ef8a794` introduced it.
- Four arena kinds, listed in `paths.TERRAIN_KINDS`. `load` in
  `terrain_env.py` refuses an arena built by a different generator. It also
  refuses a train or test arena whose flat row or stair geometry differs
  from the run's config. The `eval` and `eval_deep` arenas skip that check,
  so a flat-row policy still scores on them. `check_arena` in
  `terrain_scan.py` refuses a measurement arena that differs from the
  course `terrain_suite.py` defines.
- The measurement suite's crossing radii, `OUT_RADIUS` and `BACK_RADIUS` in
  `terrain_suite.py`. A crossing is one walk out to `OUT_RADIUS` or back to
  `BACK_RADIUS`. Commit `04c5ed3` measures both as Chebyshev distance from
  the tile centre. The slope and stair features are concentric squares. The
  scattered boxes stay inside a square reach.
- Base height, foot clearance, and foot contact measured from the local
  terrain surface. Commit `2d04bf6` does this in `_base_height`,
  `_foot_clearance`, and `_foot_contact` in `base.py`. Commit `7e161c9`
  does it in the eval tools.
- The MJWarp heightfield contact cap of 50 per geom pair. Large flat
  colliders must be decomposed into small cells. Commits `4a8d572` and
  `ce9a464` split the quadruped's base box. A humanoid pelvis or torso box
  lying on the heightfield can exceed the cap too.
- `check-terrain --backend warp`, the gate for that cap. MJWarp prints the
  overflow warning from device code straight to file descriptor 1.
  `capture_os_stdout` in `check_terrain.py:69-101` captures it there.
- The pin of `ppo.num_resets_per_eval` to 0 on terrain runs, in
  `train.py:123-136`, commit `64da36a`. A periodic reset redraws every
  env's level, so the curriculum never climbs. humanoid-lab's
  `src/humanoid_lab/train.py` starts from playground's
  `Go1JoystickFlatTerrain` PPO config. That config sets
  `num_resets_per_eval` to 10.
- `XLA_PYTHON_CLIENT_PREALLOC=false` on every terrain run. The terrain
  keeper's preset header calls it mandatory.

Carry these corrections when reading w01-tek. `spawn_level` pins only each
env's first spawn, and respawns still follow the curriculum. The comment on
`spawn_level` in `conf/task/joystick.yaml` says it pins every spawn. The
flat-row paragraph of w01-tek's `training/docs/configuration.md` says the
same. Both are wrong. That guide and the yaml's `arena` comment list three
arena kinds and leave out `eval_deep`. Commit `4a8d572` says `run.sh smoke`
fails on an overflow warning. Commit `64da36a` removed that check, because a
CPU smoke run cannot see a warp overflow.

## Run reports

The terrain run reports are not in w01-tek. Write each humanoid-lab terrain
run report in this layout. Open with the date, the commit, and the jax,
mujoco-mjx, and warp-lang versions. Then give numbered findings. Each
finding is a one-line claim followed by its evidence. Then describe the
run: seed, GPU, steps, wall time, steps/s, curriculum level reached, and
wandb run IDs. End with the repo changes made for the run, what the report
leaves out, and the artifacts.

## The keeper recipe

The terrain keeper's recipe is `conf/experiment/terrain_blind_v4_2.yaml`,
frozen on 2026-08-04. Its header lists the recipe's changes and the
`build-terrain` flags for its arena. No later recipe replaced it. The
candidates were v4.3 to v4.6 and the first DAgger distillation. Only v4.3
and the v4.4 variants have presets of their own in w01-tek. The
distillation trainer is `distill.py`. Its preset,
`conf/experiment/terrain_distill.yaml`, is the revised recipe written after
the first distillation failed. Its env block is the v4.6 recipe's,
unchanged. `terrain_scan_v5` is the preset that puts the height scan in the
actor. It is older than the keeper. It adds the scan to the v4.1 recipe and
did not beat v4.1. No preset puts the scan in the actor on the keeper's
recipe.

The keeper's recipe adds these items to port:

- Feature spawns, which start an episode anywhere on its tile. An episode
  that ends within 1 s of its spawn does not count for the curriculum. Its
  fall pays no termination penalty either. The grace covers the quadruped's
  knee linkage settling after a spawn on a stair edge, so measure the
  humanoid's own settle time. Promotion needs a crossing of the tile's
  feature band. Demotion takes three failed movement episodes without a
  clean one between them. Stand and spin episodes do not count. A fifth of
  the envs are pinned to fixed levels that the curriculum never moves.
  Commit `086b8f0` adds the stand and spin rule. Commit `544de5a` adds the
  rest.
- Stair treads of 0.25 to 0.45 m, per-type difficulty caps, and summit
  platforms with a 0.3 m half-width. Commits `b6d92aa`, `271b5ae`, and
  `086b8f0` add them. The tread range follows real stair geometry. Commit
  `40cbbd9` sized the caps from the quadruped's leg-length curve. The
  summit fits the quadruped's standing footprint. Re-derive the caps and the
  summit for the humanoid.
- The constraint-row cap `sim.njmax`, raised from 640 to 1024. Scans
  measured peaks of 647 to 681 rows in pile-up states. Rows past the cap
  apply no force. `TerrainAutoResetWrapper` logs the running peak as
  `nefc_peak_per_step` and warns once at 90% of `sim.njmax`, on the warp
  backend only. Commit `9b5c352` adds both. These are the quadruped's
  numbers. Asimov v1 budgets 1120 rows on flat ground, seven times its
  measured peak of 157 rows, rounded up. Measure the humanoid's peak on the
  ported arena.
- The critic's height scan, the `height_scan_clean` observation. `_scan_raw`
  in `env.py` builds it with the grid helpers in `height_scan.py`. It
  carries no sensor corruption. The keeper's actor sees no scan. Commit
  `e55a629` adds the scan.
- Deep-tread stair cells in the measurement suite, `DEEP_CELLS` in
  `terrain_suite.py`, on their own `eval_deep` arena. Commit `b0f9045` adds
  them. Their treads are 0.30 m and their risers 3 to 9 cm. The keeper's
  stair climbing gives out at 7 cm risers, and these risers bracket that
  height. Choose risers that bracket the humanoid's own limit.

Each measurement run starts at a tile centre and walks out and back without
turning. The return legs flip the sign of the forward command. A pyramid
stair tile peaks at its centre, so its cells measure climbing backward. An
inverted pyramid stair tile is a pit, so its cells measure climbing forward.
Seed variance is large. A second v4.2 run at another seed scored 34% below
the keeper on the legacy cells, the original cells on the `eval` arena.
Judge a recipe on the median of three seeds. Score the keeper again in the
same eval jobs, because one checkpoint's scores drift between jobs.
