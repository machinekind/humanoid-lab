# The terrain path

humanoid-lab has two training paths. The flat path trains `task=joystick` on
a floor plane. The terrain path trains `task=terrain` on a procedural arena.
Either path trains a policy on its own. Both share the robot models, the
actuator presets, the PPO trainer, the eval tools and the export.

Roboto Origin is the robot the terrain path is built and measured for. The
code is robot-generic. Asimov v1 on terrain is untested and untuned.

## What `task=terrain` is

`task=terrain` is the joystick task on an arena. Its env is
`TerrainJoystick` in `src/humanoid_lab/envs/terrain_joystick.py`. It keeps
every Joystick reward term, observation and command. Its config is
Joystick's plus one block, `task.env.terrain`. [Terrain
task](configuration.md#terrain-task-taskenvterrain) in the configuration
guide lists every key and its default.

The arena replaces the floor plane. It is one heightfield, the arena's
static boxes, and four apron boxes that continue the flat border outward.
The arena is a grid of 4 m tiles. Each row is one level of difficulty. Each
column is one of eight terrain types:

- rough ground
- a pyramid slope and an inverted pyramid slope
- pyramid stairs and inverted pyramid stairs
- discrete obstacles
- rubble
- waves

Every tile has a flat spawn pad of radius 0.4 m at its centre. The default
arena has 10 rows, 1900 boxes and 1101 × 901 heightfield nodes at 4 cm.
`terrain.arena.flat_row: true` adds a flat row as level 0. The flat row's
ground reads exactly 0. There, base height, foot contact and foot clearance
equal Joystick's bit for bit on the same data. The flat row's physics
differs from Joystick's, because its ground is the arena's heightfield.

The env measures the robot against the ground under it:

- Base height, the `height` observation and the fall check read the base
  above the ground under the base.
- Off the flat row, foot contact and foot clearance read each foot
  capsule's underside against the ground at 17 samples across its
  footprint.
- Base contact ends an episode when a collider of a termination body comes
  within `base_contact.tol` of the ground. Roboto Origin's termination
  colliders are its 15 base and torso cells.

## How the two paths relate

A policy trained on one path runs on the other when three things match:

- the actor list, `obs.state`
- the actuator preset
- the preset's `actuators.overrides`

The preset and its overrides set each joint's action scale, so they set the
window of targets the policy can reach. `configs/task/terrain.yaml` keeps
`configs/task/joystick.yaml`'s actor list.

The terrain critic appends the height scan, `height_scan_clean`, to the
joystick critic list. Roboto Origin's critic is 111 wide on `task=joystick`
and 209 wide on `task=terrain`. Its actor is 82 wide on both. The scan is 98
ground heights on a 14 × 7 grid with a 0.1 m pitch. The grid turns with the
base's heading and stays level. It reaches 0.4 m behind the base, 0.9 m
ahead and 0.3 m to each side. Each value is the ground height relative to
the lowest sole, clipped to ±0.5 m. The env refuses the scan on `obs.state`,
so the actor never sees it.

The mirror augmentation (`symmetry.enable`) has no map for the scan yet. With
symmetry on, the shipped terrain task stops at env construction with a
`KeyError` naming `height_scan_clean`. A terrain env whose critic does not
list the scan runs mirrored. The map needs the scan's left-right flip in
`envs/symmetry.py`, and the curriculum wrapper must flip the scan it
splices in after a respawn.

A checkpoint warm-starts across the two critic layouts, in either direction:

```bash
./run.sh train experiment=roboto_terrain_v1 actuators=<preset> \
    restore=runs/<flat run>/checkpoints/<step>
```

Shared critic columns keep their normalizer statistics and value weights.
The added scan columns get zero weights and a measured normalizer prior,
mean 0.014 m and std 0.066 m. Removed columns are dropped. A change to the
actor list refuses. `run.json`'s `restore` block records the plan.

The measurement tools rebuild a terrain run on the flat floor. `battery`,
`courses`, `report`, `eval`, `sizing-collect` and `export` drop the
`terrain` block and run the rest of the run's config as `task=joystick`. A
terrain policy therefore scores on the same flat scene as a flat one, and
`jobs/train.sh`'s eval stage measures a terrain run as it measures a flat
one. `courses.json` records both tasks, `trained_task` and
`measured_task`. The deploy contract comes from that flat rebuild. Its
`task` field still says `terrain`.
`terrain-scan` goes the other way. It rebuilds a joystick or a terrain run
on the scan suite's arena, so a flat policy gets a terrain score too.

## Training

A terrain recipe is an experiment under `configs/experiment/`.
`roboto_terrain_v1` is Roboto Origin's first one. Its arena is the default
arena with the flat row. That arena has 11 levels and 1900 boxes on 1201 ×
901 nodes. The recipe spawns on the pads. It demotes after three failed
episodes and pins a fifth of the envs to fixed rows. Every other key is the
terrain default. Every value is untuned.

The recipe leaves two things open:

- It sets no warp budget. A terrain run on warp refuses to build until the
  recipe sets `task.env.sim.naconmax_per_env` and `task.env.sim.njmax`.
  `check-terrain` measures them on a GPU.
- It pins no actuator preset. The config default, `sizing_ideal`, applies
  unless the command line names one. The action window is an open
  decision, below.

The default arena does not run on the jax backend. jax refuses an arena of
more than 128 ground boxes. This one has 1904 with its aprons. On a host
without CUDA the terrain pipeline runs on `terrain_cpu`'s small arena:

```bash
./run.sh smoke experiment=terrain_cpu
```

`terrain_cpu` has the flat row and one terrain row at difficulty 0.5. Its 33
boxes are all stairs. The smoke trains 16 envs for 20,000 steps and took
about 7 minutes on a laptop CPU. It checks the plumbing. In the measured
smoke the curriculum level stayed at 0.

`train.py` treats a terrain task differently in four ways:

- It sets `XLA_PYTHON_CLIENT_PREALLOCATE=false` before jax starts, unless
  the environment already sets it.
- It refuses `ppo.num_resets_per_eval` other than 0. A periodic reset
  redraws every env's level. `configs/task/terrain.yaml` sets it to 0.
- It refuses `early_stop.enable=true`. Eval episodes start from reset's
  first levels, so the eval reward plateaus while the curriculum climbs.
- On warp it runs the contact preflight under a capture of file descriptor
  1, where MJWarp prints its overflow messages.

## The order of work on a GPU

A human launches each GPU step. `jobs/check_terrain.sh`, `jobs/train.sh` and
`jobs/terrain_scan.sh` are the payloads for them. `jobs/README.md` states
their contract.

1. Choose the actuator preset and any `actuators.overrides`. The flat and
   terrain runs that should share a policy use the same choice.
2. Gate the recipe on its training arena:

   ```bash
   ./run.sh check-terrain experiment=roboto_terrain_v1 actuators=<preset> \
       --backend warp --require-warp
   ```

   With no budget in the recipe, the gate runs at 512 contacts and 4096 rows
   per world, and at the naconmax pool for CCD. The report's `recommend`
   block holds `naconmax_per_env`, `naccdmax_per_env` and `njmax`. A
   recommendation marked `lower_bound` comes from a run that dropped work.
   Gate again at the recommended values, given as `--naconmax-per-env`,
   `--naccdmax-per-env` and `--njmax`, until `lower_bound` clears.
3. Copy the three budgets into the recipe under `task.env.sim`. Run step 2's
   command again. The gate now reads the recipe's own budgets, and it must
   pass.
4. Size the batch on the node class the run will use, with
   `jobs/preflight_sizing.sh` and `EXPERIMENT` set. The CCD scratch grows
   with `ppo.num_envs`. The gate's `ccd_scratch` block projects it for the
   recipe's batch.
5. Train:

   ```bash
   ./run.sh train experiment=roboto_terrain_v1 actuators=<preset> run_name=<name>
   ```

6. Gate the scan suite's arena. The gate reads the recipe's budgets there
   too, and its status can fail on the harder arena. Its `recommend` block
   sizes the scan's budgets:

   ```bash
   ./run.sh check-terrain experiment=roboto_terrain_v1 actuators=<preset> \
       --arena eval --backend warp --require-warp
   ```

7. Scan the run at those budgets:

   ```bash
   ./run.sh terrain-scan --run runs/<name> \
       --naconmax-per-env N --naccdmax-per-env N --njmax N
   ```

Three checks run on any host before the GPU steps:

- `./run.sh smoke experiment=terrain_cpu`, the training pipeline.
- `./run.sh check-terrain experiment=roboto_terrain_v1 actuators=<preset>
  --engine mujoco`, the C engine's count of heightfield contacts per
  collider. Add `--arena eval` for the scan's arena.
- `./run.sh test-all`.

## What each verb reports

`train` prints a progress line per eval and per training-metrics call. On a
terrain run the line ends in `terrain_lvl`, the mean level of the episodes
the curriculum moves. Training metrics log every
`ppo.training_metrics_steps` env steps, 10,000,000 in
`configs/task/terrain.yaml`. They carry the `episode/terrain/` metrics:
`level_per_step`, `level_free_per_step`, `free_per_step`, `promoted`,
`demoted`, `on_flat_per_step`, `spawn_fallback_per_step`,
`base_contact_at_done` and `early_end`. wandb gets `curriculum/level`,
`curriculum/promoted` and `curriculum/demoted` over the envs the curriculum
moves. On warp the auto-reset wrapper also logs the batch's peak constraint
rows and pool fill. It prints one warning per counter the first time that
counter reaches 90% of its budget. `run.json` is written before training
starts. On a terrain run it adds:

- `arena`: the generator version, fingerprint, params, heightfield size,
  box count, spawn mode, jax contact cap, CCD scratch projection, the
  allocator setting and the preflight's MJWarp message counts.
- `curriculum`: the last and the highest level, with their env steps, and
  the last promotion and demotion rates.
- `restore`: the warm-start plan, when there is one.

`check-terrain` writes a report with a `status`, the measured pool and row
peaks per regime, the MJWarp message counts, the recommended budgets and the
CCD scratch projection. [Terrain
budgets](configuration.md#terrain-budgets-check-terrain) gives its regimes,
exit rules and keys.

`terrain-scan` writes `<run>/terrain_scan.json`. Each of the suite's 48
cells gets 64 runs per speed. Each cell reports its passes, falls, timeouts,
progress, tracking error, saturation and clearance. The absolute gate reads
the twelve cells that carry a bar. [Terrain
scan](configuration.md#terrain-scan-terrain_scanjson) gives the protocol and
the keys.

`check-friction --task terrain` checks foot-friction randomization on the
CPU terrain arena.

## Measured facts

**Chessboard cells.** MJWarp collects at most 50 prism hits per pair of a
geom and the heightfield. It skips every prism past that, and it writes at
most 4 contacts per pair. A face at most 10 cm across overlaps at most 32
prisms of a 4 cm heightfield. Roboto Origin's base and torso boxes overlap
up to 128. Each box therefore collides as the filled cells of a 3D
chessboard. The base has 6 cells of 9.3 × 9.0 × 7.9 cm. The torso has 9
cells of 9.6 × 8.3 × 8.6 cm. The split applies on both paths. Roboto Origin
collides with these 15 cells and 14 capsules. On the C engine's count, over
the `roboto_terrain_v1` arena under `deploy_pd` with one world per tile, no
collider reached the cap. The highest counts were 44, on a thigh and a shin
capsule while fallen.

**Exact foot ground.** The feet read the exact ground: the heightfield on
MuJoCo's own triangles, or the top of a box that contains the point. It
matches `mj_ray` within 0.08 mm on the default arena. The base height reads
a bilinear blend of the arena's lookup grid instead. The lookup grid is the
heightfield with each box top written onto the nodes the box covers. Within
one 4 cm cell of a box edge that blend reads up to a full step high outside
the box. It reads up to three quarters of a step low inside it. Read on the
lookup, a level foot resting on a stair tread showed a clearance as low as
-10.7 cm. Base contact reads a 3 × 3-node max of the lookup, the spawn grid.

**Spawn rules.** A spawn sets the base at the lowest height that keeps every
robot collider out of the ground. Roboto Origin has 190 sample points for
it. A pad spawn sits at the pad height plus the reset height. The jitter is
0.15 m per axis. Its diagonal, 0.21 m, plus the feet's 0.15 m reach fits
inside the 0.4 m pad. The env refuses a jitter that does not fit. A feature
spawn starts at a point of its tile where the ground under the soles spans
at most 2 cm. Its height reads the spawn grid, which reads high beside every
box. On the default arena above difficulty 0.3, 1000 feature spawns on
rubble and obstacle tiles put no collider into the ground. A feature spawn
can start the robot beside a riser. `spawn.grace_sec` then exempts an early
fall from the termination penalty and from the curriculum. Its value needs
Roboto Origin's measured settle time, so `roboto_terrain_v1` uses pad
spawns.

**Served-distance curriculum.** An episode's level moves when it ends.

- It promotes when the episode crossed its tile. On the flat row that is a
  walk of more than 2 m from the spawn. On every other row the base must
  reach the outer radius of the tile's features. On the default arena that
  is 1.6 m on stairs and 1.9 m on the other types.
- It fails when it served less than `demote_fraction` of its commanded
  distance. Served distance is the base velocity along the command, summed
  over the episode. A tracked command serves its full distance on any path,
  arcs included. The commanded distance is projected onto a full episode,
  so an early fall almost always fails.
- A failure adds a strike. The failure that brings the count to
  `demote_strikes` drops one level and clears the count. A promotion or a
  clean moving episode clears it too.
- An episode that commands no distance, a stand or a spin, neither strikes
  nor clears.
- A promotion from the top row lands on a uniform random row.
- A pinned env reaches its row at its first respawn and holds it for the
  rest of the run.

**Warp CCD scratch.** MJWarp allocates scratch for convex collision on every
collision call, outside the XLA pool. A model with no convex pair skips it.
Neither robot's flat model has one. Terrain adds heightfield and box-box
pairs, which are convex. Roboto Origin's slot is 5,480 bytes at its 35 CCD
iterations. The scratch is `naccdmax_per_env` × `num_envs` slots. 128 slots
per world at 8192 worlds is 5.7 GB. A preallocated XLA pool leaves that
scratch only what the pool fraction leaves over. A terrain process therefore
starts with `XLA_PYTHON_CLIENT_PREALLOCATE=false`. `sim.naccdmax_per_env`
caps it, and `None` sizes it to the naconmax pool.

**jax limits.** jax runs narrowphase on every pair of a robot collider and a
ground box, so it refuses more than 128 ground boxes. Its heightfield
collider misses the prisms under a yawed or rolled box's low-side corners. A
tilted box cell gets partial contact there. Its counts are lower bounds, and
`check-terrain` reports a jax run as `unverified`. A jax step of 16 worlds
on the CPU arena takes about 67 ms.

## Untuned values

Every terrain number below is a starting value. None was tuned by training.

- The arena generator's ramps, tapers, margins, obstacle and rubble draws.
  `src/humanoid_lab/terrain/params.py` says which were sized for the biped.
- Spawns: `pad_jitter` 0.15 m, `feature_max_spread` 0.02 m,
  `feature_candidates` 16, `init_level_frac` 0.5, `grace_sec` 0, and the
  spawn grid's dilation of 1 node.
- The curriculum: `demote_fraction` 0.5, `demote_strikes` 1, and the 0.05 m
  below which an episode counts as not moving. `roboto_terrain_v1` sets
  `demote_strikes` 3 and `pinned_frac` 0.2.
- `base_contact.tol` 0.01 m.
- The command bias probabilities.
- The height scan's grid and clip, and its normalizer prior.
- The jax box limit, 128, and the jax contact cap of 4 per ground-pairing
  collider plus the robot-robot slots.
- `check-terrain`'s headrooms of 2.0 on all three budgets, its `--max-fill`
  of 0.9, and its default gate budgets of 512 contacts and 4096 rows.
- `terrain_cpu`'s sizes: 16 envs, 20,000 steps, training metrics every
  2000.
- The scan suite's speeds, rows, bars, offsets, `budget_slack` of 1.6, and
  its warp budgets of 256 contacts and 2048 rows.
- Every terrain recipe's warp budgets. None exists until `check-terrain`
  runs on a GPU.

## Open decisions

**Roboto Origin's action window.** The first GPU run needs a preset and its
`actuators.overrides`. Under `deploy_pd` the knee target spans 0.125 to 0.54
rad, hip pitch -0.46 to 0.26 rad, and ankle pitch -0.40 to 0.00 rad. A
sagittal model of the leg that keeps the foot flat lifts the swing foot at
most 2.5 cm inside that window. Under `sizing_ideal` the ankle pitch target
spans only -0.26 to -0.14 rad. The default arena's stair risers run from 2
cm at difficulty 0 to 15 cm at difficulty 1. The choice must be the same on
the flat and the terrain path for their policies to stay interchangeable.
Until it is made, the scan tracks stairs, obstacles and rubble without a
bar.

**Flat critics and the scan.** Flat recipes do not list `height_scan_clean`
in the critic. The flat env serves the flat floor's scan if one does. Warm
starts work either way, because the restore planner adds or drops the scan
columns. The proposed default is to leave flat critics as they are.

**The allocator for every warp run.** Terrain processes start with
`XLA_PYTHON_CLIENT_PREALLOCATE=false`. Flat warp runs keep XLA's
preallocated pool. `jobs/preflight_sizing.sh` exports
`XLA_PYTHON_CLIENT_PREALLOC`, a name jaxlib 0.9.2 does not read. Its flat
slices therefore measure the preallocated pool. Whether every warp run
should turn preallocation off, with that line fixed, is open. `train.py`
also sets `XLA_PYTHON_CLIENT_MEM_FRACTION`. jaxlib 0.9.2 raises when a
launcher also sets `XLA_CLIENT_MEM_FRACTION`.

**The scan protocol.** Every scan run is a forward crossing from the pad at
0.3 or 0.6 m/s. A pyramid measures walking down, and a pit measures climbing
out. Two extensions are open:

- Rim-start stair cells. They climb onto a flight from flat ground and walk
  down into a pit from its rim.
- Backward cells at -0.3 m/s.

**EPA horizon.** MJWarp's `EPA horizon` message reports and never gates. The
pair it names gets no contact on that step. No budget enlarges the horizon.
Whether it should gate or warn in the training preflight is open.

## Later work

- A relative gate and stored baselines, keyed on the suite version, arena
  fingerprint, robot, preset and engine.
- Rim-start stair cells.
- A terrain section in `report`, with per-cell output.
- The height scan in the actor.
- A mirror map for the height scan, so terrain runs can train with symmetry.
- Terrain courses: the course benchmark on terrain arenas, for every flat
  and terrain model. [terrain-courses.md](terrain-courses.md) holds the
  design. None of it is implemented yet.
- Feature spawns for Roboto Origin, after its settle time is measured.
- A terrain video and terrain torque demand for sizing.
- TODO: Asimov v1 on terrain. It has no scan suite and no recipe. On the C
  engine's count over the default arena under `deploy_pd`, its waist capsule
  reaches the 50-contact cap in every regime. The env also warns that its
  reset height sits too close to its fall height for the default arena's
  largest step.

## Run reports

Write each terrain run report in this layout:

1. The date, the commit, and the jax, mujoco-mjx and warp-lang versions.
2. Numbered findings. Each is a one-line claim followed by its evidence.
3. The run: seed, GPU, steps, wall time, steps/s, the curriculum level
   reached and the wandb run IDs.
4. The repo changes made for the run.
5. What the report leaves out.
6. The artifacts.
