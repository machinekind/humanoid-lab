# Terrain courses

Status: design. Nothing in this document is implemented yet.

Terrain courses extend the [course
benchmark](configuration.md#course-benchmark-coursesjson) (`eval/courses/`)
to terrain arenas. The flat benchmark asks how faithfully a policy follows a
path on a flat floor. Terrain courses ask the same question across one
terrain feature at a time. The goal is that every model gets both
benchmarks. A flat-trained run and a terrain-trained run each get
`courses.json` on flat ground and `courses_terrain.json` on terrain.

This document is for review before implementation. Each Roboto Origin
number in it says how it was measured or computed. The cost section is an
unmeasured estimate.

## At a glance

| Question | Decision |
|---|---|
| What a row measures | How well the policy follows a straight 5 m line across one feature, against the same line on flat ground in the same arena, engine, env and foot estimator. |
| Arena | A dedicated course arena per robot, built from `ArenaParams` with the generator unchanged. Roboto Origin's has 6 m tiles, a flat row and two levels. Its fingerprint is pinned. |
| Catalogue | 17 rows: `flat_ground`, then stairs up and down, slope up and down, a mirrored side-hill pair, rough ground and rubble, each at level A and level B. |
| Method | The flat benchmark's: frozen follower, physical normalizers, `min` of the scored axes, completion gate, median and worst over seeds. 32 seeds. |
| Scoring changes | One foot estimator on every row. `grip` unscored everywhere, `height` unscored on stairs and rubble. Per-axis deltas, a bootstrap SE, a feature window and a side-hill contrast are reported. |
| Execution | MJWarp on a CUDA GPU. A batch-major executor runs all 544 lanes (17 rows × 32 seeds) in one dispatch. |
| Tests | MJWarp on the CPU device, on the full course arena. They never write a canonical file. |
| Every model | `courses_terrain` joins `jobs/eval_runs.sh`'s default `EVALS` and `jobs/train.sh`'s eval stage. Without a GPU it is DEFERRED (exit 4). |
| Cost | Estimated 2-5 min per model on one GPU. Unmeasured. |

## What a row measures

Each row asks one question. Does the policy follow a straight path across
this feature as well as it follows the same path on flat ground?
`d_score_median` against `flat_ground` answers it. The absolute score says
how well, on the flat benchmark's physical scale.

| Aspect | Flat benchmark | Terrain courses |
|---|---|---|
| Baseline | each row names one | every row names `flat_ground`. It differs from each row only in `origin`, that is, in the ground. |
| Follower | pure pursuit, `vy = 0`, lookahead 0.40 m | the same constants and code |
| Frame | `anchor="start"`, the post-settle pose | `anchor="world"`, `origin` the spawn pose |
| Settle | 1 s at zero command | the same, on exactly flat ground |
| Budget | 2.5 × ideal + 2 s | the same: 27 s, 1350 steps, on every row |
| Seeds | 8 | 32, with the same keying |
| Speed | `v_nom` plus a speed family | `v_nom` only |

Terrain courses complement the [terrain scan](terrain.md). The scan asks how
often a policy gets across a cell, over 64 starts from the tile's pad. Its
docstring says it does not measure climbing onto a flight from flat ground,
or walking down into a pit from its rim. Courses measure exactly those, on
one exact profile, with the flat method's scores. The scan keeps the hard
rows, waves and discrete obstacles. The scan is human-launched. Courses run
after every training.

Rules for a row:

1. A row is one straight line across one realized profile. The row's JSON
   prints that profile.
2. The robot walks 1.5 m of exactly flat ground after the settle before
   the tile edge. It meets the feature at a steady gait.
3. Straight rows end on the feature's flat top. A stumble on the last
   riser then shows as a fall before the goal.
4. Up and down are separate rows. A toe stubs on a rising riser. An
   overspeed drop happens on a falling one.
5. The baseline has the same path, speed, budget, seeds, engine, env and
   foot estimator. Only the ground differs.
6. Two rows meant to differ in one property differ in that property alone.

## The course arena

### Why not the scan's arena

The scan suite's arena breaks rules 2, 4 and 5. These facts were read from
the generated arena with numpy:

- From a pad on its 4 m tiles, the first riser is 0.41 m ahead and the
  first rubble box 0.55 m ahead.
- A straight line into a row above the first starts on the pad of the row
  below. It crosses two features. Into the 9.8 cm stair cell, it goes
  0.36 m down and then 0.49 m up.
- Its lowest stair riser is 4.6 cm, 1.46 times the p90 toe apex of a
  current `deploy_pd` model (see Roboto-derived parameters). Every current
  Roboto model would score 0 on every stair row.
- Its slope tiles carry rough noise. It has no flat row.

### The params

The generator and `GENERATOR_VERSION` do not change.

```python
COURSE_ARENA_ROBOTO = ArenaParams(
    seed=0, ordered=True, flat_row=True,
    difficulties=(0.0, 1.0),              # arena row 1 = level A, row 2 = level B
    tile_size=6.0, border=2.0, cell_size=0.04,
    summit_floor=0.9,                     # 1.8 m stair summit and pit floor
    slope_platform_half=1.0,              # 2.0 m plateau, 2.0 m ramps
    stair_tread=0.30,
    overlay_fraction=0.0,                 # pure slopes; boxes seated on flat ground
    wave_half_periods=(3,),               # no row uses wave tiles
    stair_riser=Ramp(0.015, 0.031),       # A 1.5 cm, B 4.6 cm
    slope_angle=Ramp(0.07, 0.07),         # A 4.0 deg, B 8.0 deg
    rough_amplitude=Ramp(0.012, 0.007),   # A 1.2 cm, B 1.9 cm
    obstacle_height=Ramp(0.015, 0.015),   # rubble top A 1.5 cm, B 3.0 cm
    grid_height_fraction=(1.0, 1.0),      # every rubble box at its level's top
    wave_amplitude=Ramp(0.020, 0.010),
)
```

- `difficulties=(0, 1)` with per-type ramps sets each type's two levels
  exactly. 6 m tiles hold six risers, 2 m ramps and a 1.8 m summit.
- With the default `grid_height_fraction` of (0.3, 1.0), each rubble box
  draws its own height. The rubble a foot meets then depends on where it
  walks. With the default fraction, a foot line 6 cm off the centre met no
  level-A box, counted with numpy on the generated arena.
- Discrete-obstacle and wave tiles are still generated, because every
  arena row holds all eight types. No course uses them.

Columns run along +x in `TYPES` order, centred at x = -21 + 6c. Arena row 0
is the flat row, at y -9 to -3. Level A is arena row 1, at y -3 to 3. Level
B is arena row 2, at y 3 to 9. A 2 m flat border at z = 0 surrounds them.
Both levels therefore have flat approach ground.

Counted with numpy, the arena has 835 ground geoms: 831 boxes (stair
flights, obstacles and rubble) and 4 aprons. The scan suite's arena has
1139. The heightfield is 551 × 1301 nodes. The arena generates in 0.04 s.
The highest point, 0.2818 m, sets the park height. Every start footprint,
±0.2 m around the start, reads exactly z = 0. The fingerprint on the
current generator is
`f6e3bba7d4128fe5c75260f1779b078eac512dab3f50b569e890244fbc59737b`. The
implementation pins whatever its own generator produces.

A variant of this arena differs only in riser and box heights. On MJWarp's
CPU device it built into `TerrainJoystick`. `spawn_qpos` put the base at
exactly `z0` at every start. The jax backend refuses more than 128 ground
boxes, so it cannot build this arena.

## The rows

Every row is a `PathCourse` in the `terrain` family, with waypoints
`line(5.0)` at 0.50 m/s, `anchor="world"`, `ground` set to the arena
fingerprint and a 1350-step budget. It has no push and nominal friction.
Its baseline is `flat_ground`. Placement reads the arena's `TileSpec`s:

- A level-A row starts 1.5 m before its tile's -y edge and heads +y. A
  level-B row starts 1.5 m beyond the tile's +y edge and heads -y.
- A straight row runs on the tile's centre column and ends 0.5 m past the
  tile centre: 1.5 + 3.0 + 0.5 = 5.0 m.
- Both side-hill rows of a level run on its `pyramid_slope` tile, 2.0 m
  either side of the centre column. Each climbs 1 m of the near face, then
  follows the side face's contour at constant height. Each foot line of
  one equals the other's opposite foot line to 3e-16 m, read with numpy.
- `flat_ground` starts where `rough_1.2cm` does and heads -y onto the
  border.

| Row | Isolates | Tile | Origin (x, y, yaw) | `height` |
|---|---|---|---|---|
| `flat_ground` | nominal | flat row | (-21, -4.5, -π/2) | scored |
| `stairs_up_1.5cm` | stepping up toe-scale risers | A `pyramid_stairs` | (-3, -4.5, π/2) | unscored |
| `stairs_up_4.6cm` | stepping up past the swing lift | B `pyramid_stairs` | (-3, 10.5, -π/2) | unscored |
| `stairs_down_1.5cm` | stepping down | A `inverted_pyramid_stairs` | (3, -4.5, π/2) | unscored |
| `stairs_down_4.6cm` | stepping down a larger drop | B `inverted_pyramid_stairs` | (3, 10.5, -π/2) | unscored |
| `slope_up_4deg` | climbing a grade | A `pyramid_slope` | (-15, -4.5, π/2) | scored |
| `slope_up_8deg` | climbing a steeper grade | B `pyramid_slope` | (-15, 10.5, -π/2) | scored |
| `slope_down_4deg` | descending a grade | A `inverted_pyramid_slope` | (-9, -4.5, π/2) | scored |
| `slope_down_8deg` | descending a steeper grade | B `inverted_pyramid_slope` | (-9, 10.5, -π/2) | scored |
| `sidehill_up_right_4deg` | holding a line across a slope, uphill right | A `pyramid_slope` | (-17, -4.5, π/2) | scored |
| `sidehill_up_left_4deg` | the mirror, uphill left | A `pyramid_slope` | (-13, -4.5, π/2) | scored |
| `sidehill_up_right_8deg` | across a steeper slope, uphill right | B `pyramid_slope` | (-13, 10.5, -π/2) | scored |
| `sidehill_up_left_8deg` | the mirror, uphill left | B `pyramid_slope` | (-17, 10.5, -π/2) | scored |
| `rough_1.2cm` | relief below the swing lift | A `rough_uniform` | (-21, -4.5, π/2) | scored |
| `rough_1.9cm` | relief near the swing lift | B `rough_uniform` | (-21, 10.5, -π/2) | scored |
| `rubble_1.5cm` | toe-scale box tops | A `random_grid` | (15, -4.5, π/2) | unscored |
| `rubble_3.0cm` | box tops past the toe's apex | B `random_grid` | (15, 10.5, -π/2) | unscored |

A row's name carries its realized dimension, as a scan cell's name does.
A params change therefore renames the row and moves the fingerprint.

### Realized profiles

Read from the generated arena with numpy. `s` is metres from the start,
along the centre line unless noted. The window is the feature window (see
Scoring).

| Row | Profile | Feature span s | Window s |
|---|---|---|---|
| `stairs_up_*` | 6 risers up: 9.0 cm total at 1.5 cm, 27.6 cm at 4.6 cm | 2.07-3.62 | 2.07-4.12 |
| `stairs_down_*` | 6 risers down: 9.0 / 27.6 cm | 2.11-3.62 | 2.11-4.12 |
| `slope_up_*` | a 2.0 m ramp: +14.0 / +28.2 cm | 1.51-3.50 | 1.51-4.00 |
| `slope_down_*` | -14.0 / -28.2 cm | 1.51-3.50 | 1.51-4.00 |
| `sidehill_*` | a 1 m climb to +7.0 / +14.1 cm, then a level contour at ±4.01 / ±8.02 deg cross-slope | contour 2.58-5.0 | 2.58-4.75 |
| `rough_1.2cm` | ascent 5.7 cm; largest rise per 0.158 m foot length 1.6 cm on the centre, 1.0-1.1 cm on the foot lines | 1.52-5.0 | 1.52-4.75 |
| `rough_1.9cm` | ascent 9.0 cm; largest rise per foot length 2.7 cm on the centre, 1.2-1.4 cm on the foot lines | 1.51-5.0 | 1.51-4.75 |
| `rubble_1.5cm` | each foot strip, across ±8 cm of stance shift, meets 3-7 rises over 0.75 cm, top 1.4-1.5 cm | 1.67-3.86 | 1.67-4.36 |
| `rubble_3.0cm` | each foot strip meets 4-7 rises over 1 cm, top 2.7-3.0 cm | 1.91-3.86 | 1.91-4.36 |

The goal fires about 0.25 m before the end: about 1.1 m past the last
riser, 1.25 m past a ramp and 0.9 m past the last rubble box.

### Roboto-derived parameters

| Parameter | Value | Source |
|---|---|---|
| speed | 0.50 m/s | `v_nom` = 0.5 `vx_max`, the flat rule |
| lead-in | 1.5 m | 3 s after the settle: 3-6 gait cycles at a 1-2 Hz clock |
| end past the tile centre | 0.5 m | The base stops 0.65 m short of the summit's far edge. The feet lead it by at most Roboto's 0.15 m reach. |
| riser A / B | 1.5 / 4.6 cm | A riser stops the toe, so the lowest sole point's swing apex sets the level. On `yolo_v4`'s `straight_10m` lanes (4 seeds, CPU jax) it was 2.66 cm median and 3.15 cm p90. The foot site's 3.53 / 4.90 cm overstates what clears a lip. A is 0.56 times the median. B is the scan's lowest stair cell. |
| slope A / B | 4.0 / 8.0 deg | The scan's two easiest gated slope cells. B uses 0.7 of the ankle-pitch room `deploy_pd` leaves. |
| rough A / B | 1.2 / 1.9 cm | The scan's two easiest gated rough cells. Rough relief has no edge, so the toe rule does not apply. |
| rubble top A / B | 1.5 / 3.0 cm | A matches the riser. B is the scan's lowest obstacle row. |
| side-hill offset | 2.0 m | The ramp's midline. Both feet and the reach stay on one face from s 2.58 to the end. |
| stance half-width | 0.0725 m | The tracking normalizer, as on flat. |
| nominal height, fall line | 0.75 m, 0.45 m | The flat height axis. They leave 0.30 m of margin above a 4.6 cm step. |

These numbers assume `deploy_pd`, the preset every current Roboto model
trained on. The action window is still an [open
decision](terrain.md#open-decisions). Each document records
`action_window`, as the scan does.

### Not in the first version

| Candidate | Why not now |
|---|---|
| discrete obstacles | Counted with numpy over arena seeds 0-9, one line crosses 0-2 boxes at 12 per tile and 0-5 at 48 per tile, varying with the seed. Picking a seed to fit would tune the arena to the test. The scan covers obstacles. |
| waves | Along one line a wave tile is a 2-3 cm trough with grades under 4 deg, below slope A. |
| side-hill on the pit | It mixes descent with the traverse. |
| turns, friction and pushes | Each adds a second variable. The flat file measures them. |
| a third level | It needs a second arena, env build and compile. |

### Identity

- `ground` is the arena fingerprint, a string. Nothing needs a serializer.
  `_check_placement` accepts a string `ground` only with `anchor="world"`
  and a finite `origin`.
- `terrain_arena.course_arena(robot)` regenerates the arena with numpy. It
  raises when the fingerprint differs from the pin.
- The flat fingerprint does not move. A new `class_constants(ground_class)`
  is empty for flat. For terrain it holds every constant that changes a
  terrain number. The terrain fingerprint hashes it. Each class has its own
  catalogue version.
- `catalogue(p, "terrain")` checks that every row is terrain-class and a
  push-free `PathCourse`. It refuses a row name shared with a flat row,
  because both classes write per-lane artifacts under `<run>/courses/`.
- `catalogue(p, "flat")` never imports the course arena. A terrain pin
  mismatch, or Asimov's missing course arena, cannot break flat courses.
- Each row's description gains `ground_profile`: ascent, descent, risers,
  largest rise per foot length, cross-slope, boxes met, feature span and
  window. It is described and never hashed.
- The pin assumes numpy generates identical float32 arrays on every host,
  as the scan suite's pin already does. GPU step 1 checks it.

## Scoring

### One foot estimator

`TerrainJoystick` reads the feet two ways. On the flat row it uses the flat
floor's test on the capsule centre. Everywhere else, contact is any of 17
sole samples within 5 mm of the exact ground. `flat_ground` spends 4.5 of
its 5 m on the flat row. No level-B row enters it. Both estimators were
recomputed from the same recorded `yolo_v4` rollouts on flat ground
(`straight_10m`, CPU jax). They disagreed on 8.2% of foot-steps, at heel
strikes and toe-offs. From the recorded signals, the sole estimator read
1.25 m of slip against 0.19 m, grip 4.25 against 27.3, and a swing apex of
2.66 cm against 3.53 cm. Every terrain lane, `flat_ground` included,
therefore records `contact` and `foot_clear` with the sole rule, through a
new `course_signals(env, d)`. The policy never sees the difference, because
contact is on Roboto's privileged list only.

### The axes

| Axis | On terrain | Reason |
|---|---|---|
| `tracking` | scored | Against world-anchored waypoints. Settle drift lands in a row and its baseline alike. On `yolo_v4` (8 seeds of `straight_10m`, CPU jax) it was 0.83 cm median. |
| `speed` | scored | `env._local_linvel(d)[0]`, the frame the reward reads. An 8 deg grade costs 1% of horizontal speed. |
| `height` | unscored on stair and rubble rows | `_base_height` reads a bilinear lookup. Near a box edge it reads up to a full riser high. |
| `grip` | unscored on every row | The sole estimator counts heel-toe roll as slip. On the flat rollouts above, that alone dropped grip to 4.25, about twice the binding tracking score of 1.92. |
| `smoothness` | scored | Riser and rubble impacts add power above 5 Hz. The baseline separates that from the policy's own vibration. |

Two alternative slip measures were checked on the same rollouts. Neither
closed the gap. Measured by finite difference over one control step, slip
at the material point under the lowest sole sample read 1.36 m, against
1.49 m at the foot site. A 0.5 mm contact threshold read 0.55 m, 2.6 times
the flat-row estimator's 0.21 m by the same measure. A terrain slip measure
fit to score is future work.

The flat rules hold for the rest. A fall is the env's `done`: base height,
tilt or base contact. A non-finite lane scores 0 and is counted, with a
warning. `fell_at` and `progress_m` record where a lane stopped.
`ground_profile` maps that onto the feature. Gait KPIs read the course
signals and are never scored.

### `vs_baseline` on a terrain row

A flat row's `vs_baseline` does not change. A terrain row's holds:

| Key | Meaning |
|---|---|
| `baseline`, `d_score_median` | as on flat |
| `d_score_se`, `within_noise` | the bootstrap SE of the difference of the two medians (2000 resamples, `default_rng(0)`); `within_noise` is \|d\| < 2 SE. Row and baseline seeds are resampled independently. They correlated only 0.07-0.50 per seed on the flat benchmark's `yolo_v4` measurement. |
| `d_subscores` | per scored axis, the row's `subscore_median` minus the baseline's |
| `most_degraded` | the scored axis with the lowest row-over-baseline ratio |
| `slip_per_m_ratio` | median slip per metre walked, row over baseline |
| `feature` | the feature window's medians, row minus baseline |

`d_subscores` exists because a seed's score is its weakest axis. On the
flat `yolo_v4` measurement of `straight_10m`, tracking read 1.92, speed
4.17 and smoothness 3.39. A 50% slowdown on a climb would leave the score
unchanged.

`slip_ratio` remains the flat friction rows' key. On a terrain row it stays
null. `slip_per_m_ratio` carries the comparison instead. A seed that falls
halfway slides about half as far. A ratio of slip distances would
therefore fall as a row gets harder.

### The feature window

The whole-run axes span s 0 to 4.75 m, lead-in included. The feature's share
of that is 33% on stairs, 42% on slopes, 41-46% on rubble and 68% on rough
and side-hill rows. The same per-step degradation therefore moves a stair
row about half as much as a rough row. Each seed of a terrain row gets a
`feature` block over its row's window. The window runs from the start of the
feature span to 0.5 m past its end, capped at 4.75 m. A side-hill window is
the contour only. The block holds `xte_rms_m`, `speed_err_rms`,
`slip_per_m`, `vibration` and `covered_m`. `flat_ground`'s seeds carry one
block per distinct window, computed from `ground_profile`. At the precision
of the profiles table there are eight. Each comparison is then like with
like. The window is reported and never scored.

### Seeds and noise

On the flat benchmark's `yolo_v4` measurement at four seed bases on CPU
jax, `straight_10m`'s 32 per-seed scores had an SD of 0.48. A bootstrap
gives an SE of 0.28 for the difference of two 8-seed medians. At 8 seeds a
difference must exceed about 0.6 to clear 2 SE. That is 30% of a score
near 2. The terrain class runs 32 seeds in the same dispatch, with no
extra compile. 2 SE of a difference then falls to about 0.28.
`spec.CLASS_SEEDS = {"flat": 8, "terrain": 32}` feeds the CLI default,
`is_canonical` and the payloads. A terrain `score_worst` is a worst of 32.
It never mixes with flat's worst of 8.

The side-hill pairs are one-variable pairs. The summary's `contrasts`
holds, per level, `sidehill_up_right_*` minus `sidehill_up_left_*`, with
its SE. The report prints it as lateral asymmetry. Level A against level B
is not a one-variable pair for rough ground and rubble, because each tile
draws its own random stream.

### What to expect on the first GPU run

These are predictions to check, not results. `flat_ground` is
tracking-bound and near `straight_10m`'s 1.5-2.5 for `yolo_v4` on CPU jax.
It will not equal it: another engine, a 5 m line and a heightfield floor.
A healthy `deploy_pd` policy completes the level-A rows, with a measurable
`d_score_median` on stairs and rubble. `stairs_up_4.6cm` scores at or near
0 for every current `deploy_pd` model. On slope rows `d_subscores.speed`
shows a slow climb that the `min` hides. The side-hill contrast exceeds its
SE for a policy with a lateral asymmetry.

## Execution

### The env

`course_overrides` builds the measurement env in the shape of the scan's
`scan_overrides`. It sets the course arena, pad spawn with no yaw, jitter
or grace, base contact at 0.01 m and the command bias off. Its `sim` block
holds the backend, `num_envs` = 544 and the budgets. Obs noise is pinned to
Roboto's overlay, as on flat. The battery's measurement overrides apply as
on flat. `--set sim.*` and `--set terrain.*` stay refused. The env loads
through `load_checkpoint_policy(run_dir, measurement, flat=False)`. The
runner checks its arena fingerprint against the pin. `measured_task` comes
from the env actually built.

### Why the flat lane cannot run on warp

The obvious executor, `jit(vmap(lane))` over the flat lane, does not trace
on warp. MJWarp keeps its contact pool unbatched (`DATA_NON_VMAP`). Its
`step` rule requires each `contact__*` leaf's leading dim to equal
`naconmax`. The flat lane batches those leaves twice: through its per-lane
`while_loop` predicate and through its per-lane stop hook. On MJWarp's CPU
device both a toy model and the real `TerrainJoystick` failed with `Leaf
node leading dim (4) does not match naconmax for field contact__dim`. The
scan's pattern, one unbatched `while_loop` around `vmap(env.step)`, traced
and ran. jax data has no pool, so tests on jax would have passed.

### The batch-major executor

`eval/courses/terrain.py` holds the executor. It is the scan's
`make_batch_runner` with the course lane's per-step semantics:

- Placement calls `env.reset(key)`, then `pad_start` at the row's origin.
  `pad_start` is factored out of the scan's `scan_reset`. The joints keep
  the reset noise. A seed then means what it means on flat.
- The settle is an unbatched `scan`. The course is an unbatched
  `while_loop` that runs while any lane runs. Each iteration steps every
  world, records running lanes and advances their followers.
- Only `qpos` and `qvel` are ever selected per lane. The follower state,
  flags, step counter and masked records live outside the env state. No
  select therefore touches the pool.
- A stopped world is parked every step: the reset pose, 2 m above the
  arena's top, at rest. On MJWarp's CPU device, 8 parked worlds read
  `nacon` 0 and `ncollision` 0, against 98 and 80 for 8 standing.
- The flat runner's per-lane `finish` scores each lane's slice.

The flat lane stays as built. Its three hooks (`reset_fn`, `on_stop`,
`observe`) have no caller under this design and cannot run on warp, so
commit 2 removes them with their tests. `eval/courses/terrain.py` imports
`terrain_scan`. That module imports `train` through `check_terrain`.
`train` sets the training process's XLA variables on import. The runner
imports `eval/courses/terrain.py` first on `--ground terrain`. The flat
path never imports it.

### Batch, budgets and overflow

- One dispatch holds every lane, rows then seeds. `num_envs` equals the
  lane count. `--only` runs the full batch and writes only the requested
  rows. A subset then sees the same program, pool and neighbours.
- The per-step records stay on the device until one copy to the host. At
  about 360 B per lane-step, 544 lanes × 1350 steps come to about 264 MB.
- The budgets default to `check-terrain`'s 512 contacts and 4096 rows per
  world, twice the scan suite's untuned 256 / 2048. At 544 worlds the CCD
  scratch is 544 × 512 × 5480 B = 1.5 GB, outside the XLA pool.
  `--naccdmax-per-env` lowers it. `check-terrain --arena courses` gates
  the budgets.
- Budgets lie outside the fingerprint, as in the scan suite. A run with
  larger budgets is still canonical. The document records them.
- The dispatch runs under `fd_capture.capture_fd1`, as in the scan. On
  MJWarp's CPU device, a 2-contact budget printed overflow messages. The
  capture counted them.
- `physics_clean` comes from `terrain_scan.physics_verdict`: no gating
  MJWarp message and no pool or row fill at or above 1. Non-finite lanes do
  not count against `physics_clean`, because no budget can clear them.
  They follow the flat rule: each scores 0 and is counted.
- An unclean document is still written. The CLI exits 6. The file stays
  stale. The remedy is a rerun with larger budgets. There is no automatic
  retry.

### Determinism

Lane k's key is `PRNGKey(seed_base + k)`, independent of the batch. A row
and `flat_ground` therefore share their reset pose and noise draws.
Run-to-run bit identity on warp is not claimed. MJWarp fills one contact
pool shared by all worlds. Nothing fixes the order of a world's contacts in
it. GPU step 3 measures rerun identity. If reruns differ, their spread
joins the noise band. `is_current` keys on inputs only. Across GPUs,
agreement is distributional. `engine` records the GPU and the package
versions.

### Platform rule

The terrain class decides in this order, before it builds a model:

1. The robot has no course arena: exit 5, N/A, also under `--check`.
2. The run's task has no terrain counterpart: exit 2, refused, also under
   `--check`. A CPU audit then reports REFUSED, not DEFERRED.
3. `--check` and `--skip-if-current` are decided without a model, on any
   host.
4. `jax.default_backend()` is not `gpu`: exit 4. The message names the
   backend and the `JAX_PLATFORMS` value.
5. On a GPU, `mujoco.mjx.warp` is imported without catching. An import
   error is exit 1. `resolve_backend("auto")` would swallow it and return
   jax. A broken MJWarp, such as a `warp-lang` off its pin, would then
   defer forever.

There is no `--backend` flag. A Python switch, `warp_on_cpu`, lets tests
build warp on MJWarp's CPU device. Such a document records platform `cpu`
and is never canonical. `is_canonical` for terrain requires 32 seeds, seed
base 0, no `--set`, no `--only`, no settle override, backend warp and
platform gpu. A non-canonical request into the run's own
`courses_terrain.json` is refused (exit 2). Replicates and subsets write
under `eval/<TAG>/`. `is_current` runs the flat checks with the terrain
fingerprint and catalogue version. It also requires warp, gpu and
`physics_clean`.

### `courses_terrain.json`

The file is schema 1, like `courses.json`. It fills the slots that schema
reserves for another ground class:

| Key | Terrain value |
|---|---|
| `ground_class`, `measured_task` | `"terrain"` |
| `catalogue` | the terrain fingerprint and version, `class_constants`, and `arena`: generator version, fingerprint, params, geom counts, `ccd_scratch` |
| `engine` | warp, gpu, `executor` `"batch_major"`, `batch` (544 lanes, rows × seeds), the GPU name and versions |
| `budgets`, `action_window` (new) | the budgets used; `check-terrain`'s action window |
| `contacts`, `messages`, `physics_clean` | the pool report and the verdict |
| summary `contrasts` (new) | the side-hill contrasts |
| row `ground`, `anchor`, `origin`, `unscored` | the fingerprint, `"world"`, `[x, y, yaw]`, `["grip"]` or `["grip", "height"]` |
| row `ground_profile`, per-seed `feature` (new) | see above |

`eval/courses/report.py` renders one section per class file. The terrain
table shows completions and falls out of 32, the median and worst, the
binding axis, `most_degraded`, `d_score_median` with its SE, the windowed
`xte` delta, `slip_per_m_ratio`, where fallen seeds stopped, and the
profile. The header prints `physics_clean`, the GPU and `action_window`.
`eval/report.py` reads both files. A missing terrain file prints "terrain
courses: not measured".

## The CPU test path

The integration tests run the real runner and executor with
`warp_on_cpu=True`, on the full course arena, with a random-init `deploy_pd`
checkpoint. On a laptop CPU with a warm warp kernel cache, a variant of the
course arena with different riser and box heights built, compiled and warmed
up on warp in 5.9 s. It stepped at 4.4 ms per world-step at 17 worlds, with
zero actions on standing worlds. A test of 17 lanes with a 10-step settle
and a 25-step cap is therefore about 2.6 s of stepping. A cold kernel cache
adds a one-time compile that is not measured yet.

On warp, the test path proves the executor, placement, parking, the
counters, the masked records, equal foot reads on and off the flat row,
every schema slot for both kinds of run, and the exit codes. At
`naconmax_per_env=2` it proves the overflow path: counted messages,
`physics_clean` false, exit 6 and a stale `--check`. A parity test on CPU
jax runs the batch-major executor on flat Joystick with `straight_10m`. It
matches the flat lane in steps and outcomes, with records equal to float
tolerance. That pins the per-step semantics.

The CPU path cannot prove GPU numerics at scale, GPU memory, cost or rerun
identity. A random policy also makes every score meaningless.

## Every model gets both benchmarks

### Coverage

| Run | `courses.json` (CPU) | `courses_terrain.json` (GPU) |
|---|---|---|
| Roboto, flat-trained | as built | through `registry.terrain_counterpart` |
| Roboto, terrain-trained | its flat rebuild | on the course arena, not its training arena |
| Roboto, `sizing` task | as built | once `sizing` joins `tasks.TERRAIN_SOURCE_TASKS` |
| Asimov, any task | as built | N/A (exit 5) until Asimov has a course arena |

A flat model loads with `flat=False` and becomes task `terrain` with its
overrides kept. Its actor layout matches. Inference reads only the actor.
`sizing` is a thin Joystick subclass, so it can join
`TERRAIN_SOURCE_TASKS`. A task with no terrain counterpart is refused on
every host. It counts as a failure. It therefore cannot go unmeasured
silently.

### `jobs/eval_runs.sh`

- `EVALS` defaults to `"courses battery courses_terrain report"`. The GPU
  word runs after the battery. A slow compile then cannot cost the CPU
  words.
- The terrain call is `env -u JAX_PLATFORMS
  XLA_PYTHON_CLIENT_PREALLOCATE=false python3 -m humanoid_lab.eval.courses
  --ground terrain ...`. An exported `JAX_PLATFORMS=cpu` then cannot hide
  the GPU. The flat words keep `JAX_PLATFORMS=cpu`.
- `TERRAIN_SEEDS` (default 32) is the terrain word's seed count. A terrain
  replicate takes seed bases 32, 64 and 96 with a `TAG`.
- `TERRAIN_NACONMAX_PER_ENV`, `TERRAIN_NJMAX` and
  `TERRAIN_NACCDMAX_PER_ENV` forward budgets to the terrain word only. A
  file measured with them is canonical. They are the remedy for exit 6.
- A run counts as measured only when none of its words failed or deferred.
- `REDO=false` passes `--skip-if-current`. A current terrain file is then
  skipped, even on a CPU host.
- `CHECK=true` runs the terrain `--check` on CPU. It lists a terrain file
  that is missing, stale, not from warp on a GPU or not physics-clean. It
  also lists a REFUSED run and an `eval_report.md` older than any of its
  inputs.
- CLI code 4 prints `DEFERRED` and code 5 prints `N/A`. Codes 1, 2 and 6
  count as failed. The summary line gains `deferred` and `n/a` lists.

The script's exit codes are in the table below. Its header says how to
complete a deferred run: `RUNS=<run> EVALS="courses_terrain report"
./jobs/eval_runs.sh` on a host with a CUDA GPU.

### `jobs/train.sh` and the deadline

- The stage runs `courses battery courses_terrain report` with
  `TERRAIN_SEEDS=32`. It starts after the training exits, when the GPU is
  free. Three new `EVAL_TERRAIN_*` parameters forward the budgets.
- A failed or deferred eval after a training that exited 0 gives 75, as
  today. On exit 4 the message names the follow-up command.
- `EVAL_TIMEOUT` stays 1800 s until GPU step 5 measures the stage.
- The `EVAL=false` passages and `jobs/train_chain.sh`'s hint say that
  measuring afterwards needs a host with a CUDA GPU.

A payload killed at its deadline never reaches its stage. An independent
change lets a training stop itself in time. `early_stop.wall_s` defaults to
0, which is off. When set, it raises `EarlyStop` at the first eval past
that many seconds. Brax has already saved that eval's checkpoint.
`run.json` gains `stop_reason`: `"plateau"`, `"wall"` or null.
`jobs/train.sh` gains `TRAIN_WALL_S`. The caller derives it from its time
limit minus `EVAL_TIMEOUT`, one eval interval and a margin. The interval is
needed because the stop fires only at the first eval past `wall_s`.

### The caller duty and the audit

- After every training payload ends, by any path, the caller runs
  `RUNS=<runs> ./jobs/eval_runs.sh` on a host with a CUDA GPU. When the
  stage already measured the run, the call exits 0 in seconds.
- On a CPU host that call exits 4 when a terrain file is missing or stale.
  The caller then runs the terrain words on a GPU host. That is a GPU job.
  A human authorizes it.
- `RUNS=all CHECK=true ./jobs/eval_runs.sh` runs on any host. It lists
  every terrain gap beside the flat ones. `RUNS=all EVALS="courses_terrain
  report"` on a GPU host is the backfill, after the runs are synced there.
- `jobs/README.md` states these rules. It lists the files a pass writes,
  so a sync can bring them back.

A new `run.sh` verb, `courses-terrain`, runs the module with
`--ground terrain` and no `JAX_PLATFORMS`, like `terrain-scan`.

### Exit codes in one place

| Tool | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 75 |
|---|---|---|---|---|---|---|---|---|
| courses CLI | written or current | exception, including a broken MJWarp on a GPU | refused, including a non-canonical request into the run's own file | `--check`: stale | terrain: jax sees no GPU | terrain: no course arena | terrain: written, not physics-clean | |
| `eval_runs.sh` | all done; `CHECK`: nothing listed | a failure, refusal or skip; `CHECK`: anything listed; a bad parameter | | | nothing failed, a GPU word deferred | | | |
| `train.sh` | trained and measured | training's own codes | | | | | | trained, measurement missing |

## Cost

This section is an estimate. Nothing in it was measured on a GPU. `perf`
records each part. GPU step 2 replaces the table.

| Part | Estimate | Basis |
|---|---|---|
| arena, env build, checkpoint | 10-30 s | The arena generates in 0.04 s. It has 835 ground geoms, against 1904 on `roboto_terrain_v1`'s training arena, counted with numpy. |
| compile | 1-3 min | Unknown on a GPU. On MJWarp's CPU device at 17 worlds, a variant of the course arena built, compiled and warmed up in 5.9 s. The flat benchmark compiles two lane programs in 21-29 s on a 10-core Apple M2 Pro ([Cost](configuration.md#cost)). |
| one dispatch | 10-25 s | At most 1400 iterations of a 544-world step, at an assumed 7-18 ms per step. |
| scoring on the host | 20-40 s | numpy in a thread pool, as on flat |
| total | about 2-5 min | The parts sum to 100-275 s. |

The flat words of the stage took 117 s on a 10-core Apple M2 Pro ([When it
runs](configuration.md#when-it-runs)). Adding this GPU estimate to that
laptop figure gives about 4-7 min for the stage, inside `EVAL_TIMEOUT`. The
stage runs on the training host, where neither figure was measured. GPU
step 5 measures the stage there.

## Implementation

The commits assume the terrain path and the flat course benchmark. They
rely on the flat runner recording `measured_task` from the env it built.
Each commit passes `./run.sh test` alone. `./run.sh test-all` runs before
the merge. Python paths are under `src/humanoid_lab/`.

| Commit | Adds | Verified without a GPU |
|---|---|---|
| 1. Add the terrain course arena, catalogue and scoring | `eval/courses/terrain_arena.py` (numpy), `families/terrain.py`, `class_constants`, per-class versions, the terrain scoring keys, `check-terrain --arena courses`. The runner still refuses `--ground terrain`. | all of it |
| 2. Run terrain course lanes batch-major on warp | `eval/courses/terrain.py`: overrides, placement, `course_signals`, the executor, the physics block. `pad_start` factored out of the scan. The flat lane's hooks removed. | tracing on warp, placement, parking, counters and overflow capture, on MJWarp's CPU device |
| 3. Measure terrain courses: runner, CLI, JSON, report, docs | `GROUND_CLASSES["terrain"]`, the platform rule, exits 4-6, the budget flags, the JSON, both reports, `sizing` in `TERRAIN_SOURCE_TASKS`, the `courses-terrain` verb, the docs sections | the CLI, the JSON, the report, the refusals, the overflow path and flat invariance; no real terrain number |
| 4. Measure every model on terrain courses: the GPU eval word | `jobs/eval_runs.sh`, `jobs/train.sh`, `jobs/train_chain.sh`, `jobs/README.md`, `CLAUDE.md` | all of it, with a stub `python3` |
| 5. Stop a training before its wall-clock deadline (independent) | `early_stop.wall_s`, `stop_reason`, `TRAIN_WALL_S` | all of it |

What the tests pin:

- Commit 1, model-free. The flat fingerprints and shapes do not move. The
  arena regenerates to its pin. Each terrain row equals `flat_ground`
  except in `name`, `origin`, `unscored` and `baseline`. Each row's
  `unscored` set matches the rows table. The profiles match the table. The
  side-hill pairs mirror to 1e-12 m, with cross-slope within 0.05 deg of
  the tile angle. Each rubble foot strip meets at least 3 boxes at level A
  and 4 at level B. Every start reads z = 0. Scoring on synthetic per-seed
  results gives `slip_per_m_ratio`, `d_subscores`, `most_degraded`, a
  deterministic `d_score_se`, `within_noise`, the feature windows and
  `contrasts`. `--ground terrain` still exits 2 from the CLI and raises
  `Refused` from `run_courses`. `check-terrain --arena courses` builds the
  course arena block. Asimov's terrain catalogue raises
  `NoTerrainCatalogue`, while its flat catalogue still builds. Importing
  `families` or `terrain_arena` sets no XLA variable.
- Commit 2. The CPU test path's executor test and the parity test pass.
  The scan's tests pass unchanged. The flat lane's hook tests are removed
  with the hooks. Its bit-identity tests pass unchanged. The new test
  module must finish in under 3 min on a laptop CPU.
- Commit 3. Exits 1, 2, 4 and 5 are raised before any model loads. Exit 3
  comes from a model-free `--check`. Exit 6 comes from the integration run
  at a 2-contact budget. `is_canonical` is false off warp or gpu, below 32
  seeds or with a settle override. A non-canonical request into the run's
  own file is refused. The report renders from a synthetic
  `courses_terrain.json` without jax. `terrain_counterpart` and the scan
  accept a `sizing` run. An integration run of `run_courses(...,
  warp_on_cpu=True, seeds=1)` covers both kinds of run. A flat
  `courses.json` for one Roboto run is identical before and after, except
  in `timestamp`, `provenance` and `perf`.
- Commit 4, with a stub `python3`. The payloads' call environment, word
  order, budget forwarding, exit mapping, `CHECK` listing and the stage's
  75.

### GPU acceptance

A human launches each step.

| Step | After | What | Pass |
|---|---|---|---|
| 1 | commit 1 | `check-terrain --arena courses --backend warp --require-warp --num-envs 544` under `deploy_pd` | status pass; budgets at or under 512 / 4096; the arena matches its pin on the GPU host |
| 2 | commit 4 | `courses-terrain` on `yolo_v4`, and on a terrain-trained run if one exists | exit 0, `physics_clean`, `perf` recorded |
| 3 | step 2 | step 2 again | per-lane identity reported |
| 4 | step 2 | seed bases 32, 64 and 96 | the per-row noise band, written to the docs beside the bootstrap SE |
| 5 | commit 4 | `train.sh` on a short training | the stage's wall time against `EVAL_TIMEOUT` |
| 6 | step 2 | read the `yolo_v4` document | the predictions above; a miss is a finding about the action window |

The numbers then land in `docs/configuration.md`. The backfill on one GPU
host follows.

## Decisions

| Decision | Reason |
|---|---|
| A dedicated course arena per robot, fingerprint pinned | The scan's arena breaks the row rules. Every current model would score 0 on its stairs. |
| Level A at toe scale, level B at the scan's easiest cell | A separates today's policies. B shares a dimension with the scan. |
| Side-hill rows as a mirror pair on one tile | A pyramid line against a pit line would also differ in climb and contour height. |
| One foot estimator on every row | The two estimators read slip 6.6-fold apart on identical motion. |
| Budgets outside the fingerprint, settable from the payloads | A too-small pool makes a run invalid, not different. The remedy must be reachable. |
| No automatic retry on overflow | Overflow means the gated budgets were wrong. A silent retry would hide that. |
| `courses_terrain` in the default stage, after the battery | One payload, one audit, one caller duty. |

## Open questions for the owner

1. **Level B on stairs and rubble.** Against the toe apex in Roboto-derived
   parameters, `stairs_up_4.6cm` will likely score 0 for every current
   `deploy_pd` model. It then acts as a sentinel for the action-window
   decision. The alternative is a 3.0 cm riser at the p90 apex. That
   separates today's policies but loses the shared dimension with the
   scan. Keep the scan's values (recommended), or size B to the window?
   Either can change later, at the cost of a new pin and a re-measure.
2. **How the launch side covers the GPU pass.** A run whose stage did not
   run gets its terrain file only from a human-authorized GPU job. The
   options are a GPU eval job queued with every training submission
   (recommended), a CPU pass followed by a per-run GPU pass, or a periodic
   GPU backfill with a lag. The recommended option wins because one human
   authorization covers the training and its GPU eval, whichever way the
   training ends. Should every submission also set `TRAIN_WALL_S`
   (recommended)? The same choice covers the first backfill of the
   existing Roboto runs. This needs an answer before commit 4 merges.
3. **Asimov.** Asimov runs report N/A until Asimov has a course arena.
   Keep N/A (recommended), since Asimov terrain has no recipe yet, or build
   an Asimov course arena now?
4. **Warp on a CPU host.** A CPU host could measure terrain courses with no
   GPU job. At the laptop's 4.4 ms per world-step, 544 worlds × 1400
   iterations is about 56 min per model, extrapolated from standing worlds.
   Keep canonical files GPU-only (recommended)? The alternative compares a
   CPU-warp file with a GPU file in GPU step 3. It admits platform `cpu` if
   they agree within the noise band.
