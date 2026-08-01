# humanoid-lab bootstrap plan

Written 2026-07-14, from a w01-tek session. This file is self-contained. It carries the design,
the research findings, and the build order for a fresh session working in this directory.

## Goal

Train locomotion policies for several open-source humanoid robots from one repo. Robot #1 is
Asimov v1. The first experiment is motor sizing for Asimov: train with instrumented idealized
actuators, read off per-joint torque and speed demands, then choose motors.

## Decisions already made

- The training core is Brax PPO on MuJoCo MJX, with the MJWarp backend on CUDA. Port it from the
  w01-tek `training/` project rather than building on mjlab. mjlab's config shapes are worth
  copying. Its runtime and rsl_rl dependency are not.
- Each robot is data. Everything robot-specific lives in `robots/<name>/` as config and assets.
  Env code reads a RobotSpec and never names joints directly.
- Actuator presets are a Hydra axis separate from the robot. Base MJCFs stay actuator-free.
  Actuators are injected at spec-build time via `mujoco.MjSpec`, the way asimov-mjlab does it.
- A preset carries kp/kd, effort and velocity limits, friction, and armature. Injection overrides
  the XML's per-joint armature. This matters because the ENCOS motors are a reference point, not
  the target: the end goal is choosing our own motors, and a different motor/gearbox pair changes
  reflected inertia, not just torque limits.
- Motor sizing is a task (`task=sizing`), not a one-off script.
- Robot #2 is RoboParty's roboto_origin (`robots/roboto_origin/`), decided by Marcin on
  2026-07-15. The G1 default existed to validate the pipeline against known-good baselines;
  Asimov already did that, and roboto_origin ships MJCF plus hardware-proven gains upstream.
- Keeper policies publish to a private HF repo under <hf-org> with the established flat layout, plus a
  frozen Hydra preset in-repo. This carries the w01-tek convention.
- Ops rules carry over from w01-tek: resolve config before spending GPU time, smoke run before a
  bounded train, wandb on every real run, generated XML is never hand-edited, reward changes gate
  on a fixed eval battery.

## Repo shape

```
humanoid-lab/
├── robots/                          # one self-contained dir per robot
│   ├── _template/                   # skeleton + adding-a-robot checklist
│   └── asimov_v1/
│       ├── source/                  # upstream MJCF + meshes, vendored verbatim
│       ├── PROVENANCE.md            # upstream repo, pinned commit, license, local diffs
│       ├── mjx/                     # generated training XMLs (build output)
│       ├── robot.yaml               # RobotSpec: joint order, actuated groups, keyframes,
│       │                            #   foot sites/geoms, symmetry map, obs layout,
│       │                            #   termination bodies, passive-joint spring params
│       └── actuators/               # named presets: sizing_ideal.yaml,
│                                    #   encos_datasheet.yaml, deploy_pd.yaml
├── motors/                          # motor catalog (encos.yaml, mab.yaml, ...): tau_peak,
│                                    #   tau_rated, omega_max, Kt, J_rotor; feeds presets and
│                                    #   reports. See "MAB Robotics actuator survey" below.
├── src/<pkg>/
│   ├── robot/                       # RobotSpec loader + actuator injection + action scale
│   ├── actuators/                   # position PD, DC motor with speed saturation,
│   │                                #   delayed, ideal-torque
│   ├── envs/                        # robot-agnostic tasks: joystick/velocity, sizing
│   ├── rewards/                     # term library, composed per task config
│   ├── dr/                          # domain randomization switches
│   ├── sizing/                      # per-joint tau/omega logging, percentiles,
│   │                                #   torque-speed scatter with motor overlays
│   ├── eval/                        # fixed evaluation battery + report
│   └── export/                      # checkpoint/ONNX + obs-layout metadata
├── configs/                         # Hydra axes: robot / task / actuators / dr / ppo /
│                                    #   experiment (with keepers/ frozen presets)
├── run.sh                           # verbs: build, check, smoke, train, eval, report,
│                                    #   export, sizing-report
├── jobs/                            # remote training payloads parametrized by env vars
├── docs/                            # configuration.md, adding-a-robot.md, lessons/<robot>.md
└── tests/                           # per-robot: compile, obs dims, short NaN smoke (CPU CI)
```

Deliberately out of scope: ROS or deployment workspaces. Export artifacts plus obs-layout metadata
are the interface to each robot's own runtime. w01-tek stays in w01-tek.

## Research findings (verified 2026-07-14)

### Asimov v1 sources

- Model: `sim-model/xmls/asimov.xml` in https://github.com/asimovinc/asimov-1 (plus 28 STL meshes).
  25 actuated DOF, CAD-derived inertias, capsule-only collisions, IMU site with
  gyro/velocimeter/accelerometer/framequat sensors, foot sites, contact excludes, passive spring
  toes. MJX-friendly. License CERN-OHL-S 2.0 (hardware) / GPL-2.0 (software).
- The XML has no actuator block and no keyframe. Its comment says actuators come from
  `BuiltinPositionActuatorCfg`, which is an mjlab config class.
- The v1 arm joints have zero damping and no actuators. Sizing runs must make arms passive
  (spring/damper like the XML's `passive_upper` neck class) or weakly actuated.
- Actuator specs live in https://github.com/asimovinc/asimov-mjlab (their public training repo,
  an mjlab fork, last pushed 2025-12-18, describes the 12-DOF legs-only v0 robot):
  - `src/mjlab/asset_zoo/robots/asimov/asimov_constants.py`: actuator configs, keyframes,
    collision configs, action scale.
  - `motor_parameters.md`: ENCOS datasheets, CAN protocol limits, firmware PD gains,
    reflected-inertia derivations.
  - Also useful: `walking_reference.csv` (1.25 Hz gait imitation data at 50 Hz) and `deploy.py`
    (sim2sim path).
- Their docs (conceptual only, no code): https://docs.menlo.ai/asimov/v1/locomotion and the
  sysid chapter at
  https://docs.menlo.ai/guides/locomotion-training/reinforcement-learning-deep-dive-system-identification.
  Design numbers published there: 200 Hz physics, 50 Hz policy (decimation 4), command envelope
  x ±0.8 m/s, y ±0.6 m/s, yaw ±0.6 rad/s, obs noise (gyro ±0.01, qpos ±0.01 rad, qvel ±0.1 rad/s).

### v0 leg motors (ENCOS, from motor_parameters.md)

| Joint | Motor | Gear | Peak τ (Nm) | Rated τ (Nm) | CAN τ clamp | Peak speed (rad/s) | Kt (Nm/A) | J_rotor (kg·mm²) |
|---|---|---|---|---|---|---|---|---|
| hip pitch | EC-A6416-P2-25 | 25:1 planetary | 120 | 40 | ±120 | 12.57 | 2.74 | 104.395 |
| hip roll | EC-A5013-H17-100 | 100:1 harmonic | 90 | 30 | ±90 | 3.98 | 5.9 | 10 |
| hip yaw | EC-A3814-H14-107 | 107:1 harmonic | 60 | 20 | ±60 | 5.45 | 4.2 | 3 |
| knee | EC-A4315-P2-36 | 36:1 planetary | 75 | 25 | ±70 | 12.25 | 2.8 | 25.5 |
| ankle | EC-A4310-P2-36 | 36:1 planetary | 36 | 12 | ±30 | 9.32 | 1.4 | 18.2 |

The CAN command clamp is the real deployable limit. It is tighter than the datasheet peak for knee
and ankle.

### Gain schemes

- v0 sim scheme: KP = armature × (2π·10 Hz)², KD = 5.0 (the CAN hardware max), effort = datasheet
  peak, action_scale = 0.3 × effort / KP.
- Their docs report the physics-calculated KPs caused vibration on hardware.
- v1 sysid example row (hip pitch, DC-motor model): stiffness 65, damping 5, effort limit 39.4,
  saturation 120, velocity limit 12.57 rad/s, static friction 1.30, dynamic friction 0.10, action
  delay 0–1 control steps, soft joint limits at 0.9 of range.
- v0 firmware gains after gain identification: hip_pitch 12.8/0.8, hip_roll 328/5, hip_yaw
  212.2/5, knee 64.2/2.7, ankle_pitch 19.3/3.3, ankle_roll 18.1/0.9 (KP/KD).

### v1 armatures (XML defaults; a motor choice overrides them via preset)

hip_pitch 0.095625, hip_roll 0.11, hip_yaw 0.038, knee 0.0339552, ankle 0.0565056 (doubled: two
motors drive a parallel ankle, modeled as serial pitch/roll joints), waist_yaw 0.095625,
shoulder_pitch 0.11, shoulder_roll 0.0339552, shoulder_yaw 0.038, elbow 0.0282528, wrist
0.0282528, neck 0.0565056 (passive), toe 0.001 (passive spring). The v1 ankle motor model is not
published. Treat the v0 ankle effort limit as approximate for v1. v1 arm and waist motors are not
published either.

### MAB Robotics actuator survey (2026-07-14)

MAB Robotics (Poznań, mabrobotics.pl) sells integrated actuators: brushless motor, planetary or
harmonic reducer, and an MD-series controller with an absolute encoder in one housing. All models
run CAN-FD or CANopen at 12–48 VDC, with a 60 V option. Marcin already runs MAB MD controllers
and mdtool on four_bar_bot, so the tooling and CAN ecosystem are familiar. Prices below are
single-unit list prices from their shop; the shop states volume pricing is significantly lower.
Customization is offered on all models.

| Model | Torque rated/peak (Nm) | Max speed @48V | Gear | Mass | List price |
|---|---|---|---|---|---|
| MA-p-100-30 | 50 / 150 | 96 RPM (10.1 rad/s) | 30:1 planetary | 1.1 kg | EUR 898 |
| MA-p-100-IP66 KV60 | 18 / 48 | 228 RPM (23.9 rad/s) | 9:1 planetary | 1.1 kg | EUR 1559 |
| MA-p-100-IP66 KV100 | 15 / 38 | 421 RPM (44.1 rad/s) | 9:1 planetary | 1.1 kg | EUR 1559 |
| MA-p-80-IP66 (6:1) | 6 / 12 | 603 RPM | 6:1 planetary | 0.5 kg | EUR 1059 |
| MA-p-80-IP66 (9:1) | 9 / 18 | 390 RPM | 9:1 planetary | 0.5 kg | EUR 1059 |
| MA-p-45-36 | 8 / 24 | 80 RPM (8.4 rad/s) | 36:1 planetary | 0.34 kg | EUR 357 |
| MA-H (sizes 14/17/20/25/32) | 4–151 across sizes | 35–100 RPM | 50/80/100:1 harmonic | 0.54 kg (small size) | from EUR 1609 |

Joint mapping against the ENCOS anchor:

- Hip pitch and knee: MA-p-100-30 covers both (150 Nm peak vs 120/70 needed; 10.1 rad/s vs the
  ~7.4 rad/s demand peak in the mockup, though below the ENCOS 12.6 rad/s spec).
- Hip roll: MA-p-100-30, or an MA-H size 25/32 if the harmonic-drive properties matter.
- Hip yaw: MA-p-100-IP66 KV60 (48 Nm peak vs 60 spec, but the demand estimate is ~12 Nm).
- Ankle: 2x MA-p-45-36 in the parallel-ankle arrangement, ~48 Nm combined peak in pitch. A
  single unit is marginal against the ~22 Nm P99 estimate.

Open gaps before a `mab.yaml` preset can be written:

- MAB publishes no rotor inertia, so armature cannot be computed from the shop specs. Ask MAB
  for J_rotor, or identify it on the bench with the fbb MD/mdtool setup. Lead: the MA-p-100-30
  product URL slug is "ak10-30-cubemars-t-motor", so the base motor is likely a CubeMars AK10
  and its rotor inertia may be in CubeMars datasheets. MAB also resells CubeMars actuators
  directly (shop category "Cubemars Actuators").
- MAB publishes no torque-speed curves, only the peak-torque and max-speed points. The DC-motor
  model needs the corner speed. Ask or measure.
- MA-H specs are given as ranges across sizes. Per-size numbers require a quote.

Sources:

- MA series overview: https://www.mabrobotics.pl/ma-actuators
- Shop category (21 products, prices): https://www.mabrobotics.pl/category/ma-actuators
- MA-p-100-30: https://www.mabrobotics.pl/product-page/ak10-30-cubemars-t-motor
- MA-p-100-IP66: https://www.mabrobotics.pl/product-page/ma-p-100-ip66
- MA-p-80-IP66: https://www.mabrobotics.pl/product-page/ma-p-80-ip66
- MA-p-45-36: https://www.mabrobotics.pl/product-page/ma-p-45-36
- MA-H: https://www.mabrobotics.pl/product-page/ma-h

## First experiments (after bootstrap)

1. Quasi-static bounds, no training: `mj_inverse` over standing, deep squat, and single-leg
   stance on the v1 model. Gives the gravity-load floor per joint in an afternoon.
2. Sizing loop: `robot=asimov_v1 task=sizing actuators=sizing_ideal`. Torque and energy penalties
   on, DR off, generous effort caps. Train, read per-joint τ/ω percentiles and torque-speed
   scatter, tighten caps, retrain. Expect 3–5 runs to find where the task degrades. That knee of
   the curve plus margin is the motor requirement.
3. Sanity anchor: the v0 ENCOS table above is an existence proof for this mass class (35 kg,
   1.2 m). Sizing results far outside it mean the setup is wrong.
4. Motor selection closes the loop: shortlist candidates from the torque-speed scatter, write a
   new actuator preset with their real armature, torque/velocity limits, and friction, retrain,
   and confirm the gait survives with margin. A sizing result is not final until it holds under
   the chosen motors' reflected inertia.

## Build order for the next session

Each step has a gate. Do not start the next step before the gate passes.

1. `git init`, scaffold the tree above, pyproject, package name (ask Marcin; default `hlab`).
   Gate: `pip install -e .` and an empty test pass.
2. Port the PPO/train/checkpoint core and run.sh verb pattern from w01-tek `training/`
   (repo: ~/git/machinekind/w01-tek). Strip w01-tek-specific env code. Gate: `run.sh train --cfg job
   --resolve` works on a placeholder config.
3. Implement RobotSpec loader + actuator injection + the position-PD and ideal-torque actuator
   models. Gate: unit test injects actuators into a toy MJCF and compiles.
4. Vendor asimov-1 `sim-model/` at a pinned commit into `robots/asimov_v1/source/` with
   PROVENANCE.md. Write `robot.yaml` (12 actuated leg joints; arms, neck, toes passive; foot
   sites `left_foot`/`right_foot` exist in the XML). Add a standing keyframe (base ~0.75 m,
   knees-bent variant per asimov_constants.py). Gate: `run.sh build && run.sh check` compiles on
   CPU and steps without NaN.
5. Write the three actuator presets from the tables above. Gate: resolved config shows per-joint
   kp/kd/forcerange matching the tables.
6. Port the joystick/velocity task with a generic reward set. Gate: CPU smoke run completes and
   checkpoints.
7. Implement `task=sizing` (torque/energy penalties, τ/ω logging) and `run.sh sizing-report`.
   Gate: report renders percentile table + scatter from a smoke run.
8. GPU model check (`check --gpu --backend warp`) and a bounded train on the GPU box. Gate:
   tracking reward improves and videos render.
9. Remote training payloads under `jobs/`, parametrized by environment variables. `warp-lang==1.13.0`
   is mandatory on the training hosts. Do not submit jobs without Marcin's explicit go.
10. Fixed eval battery + report, then the sizing experiments above.

## Open questions for Marcin

- Package name and license for the new repo.
- Use asimov-mjlab's walking reference for imitation-style training, or velocity rewards only?
- Sizing first, or a plain walking baseline with the ENCOS datasheet preset first?
