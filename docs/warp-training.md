# Native Windows training: torch + MJWarp (`wtrain/`)

A torch port of the Brax PPO trainer and of the Joystick task, running on MJWarp
(vendored in `mujoco-mjx` 3.10.0) with `warp-lang==1.13.0`. Built for hosts with
no WSL2 / JAX-CUDA (Windows VM, RTX A4500). It warm-starts from a Brax checkpoint
and writes Brax/orbax-format checkpoints, so `eval/battery` and `eval/video`
work on its output unchanged. The JAX env (CPU) is the parity oracle.

    python wtrain/train.py --src runs/<src> --src-ckpt <step dir> --name <run> \
        --steps 3e8 --ckpt-minutes 10 --sym-coef 1.0 --warp-reward '<json>'

Other tools: `measure.py` (gait + stability metrics), `video_watch.sh`
(clip per checkpoint), `long_video.py` (scripted 40 s command sequence),
`compare_video.py`, `parity.py`, `bench_warp.py`.

## Roboto Origin gait, v1-v40 (about 3.1 G env steps, chained warm starts)

Recipe at the best point: gait clock on (0.8-1.4 Hz), `heading_err` actor+critic
obs, `command.deadband=0.15` plus oversampled slow commands (start-up without
shuffling), PPO mirror-symmetry loss, pair-min landing rewards
(`step_length`, `foot_reach`, `knee_extend`), `forward_lean` target 0 deg.
Measured at 0.3/0.6/0.9 m/s: speed 100-107 %, heading drift <= 0.3 deg,
torso pitch 0.5/0.7/1.3 deg, no flight phase.

Lessons that cost time:
- Per-landing rewards get harvested by one leg; use the min of both feet and
  always read a left|right table.
- A strong per-event bonus invites tapping exploits; gate on real steps.
- exp kernels narrower than the current error give no gradient; use linear ones.
- A constant side-dependent actor input (frozen clock phase) lets the policy
  specialise one leg; a frozen clock now emits zeros.
- Turning the gait clock off caused shuffling; stepping penalties and command
  ramps collapsed slow walking into standing (use the deadband instead).
- Measure achieved speed on every video; it went unnoticed at 48 % for a while.

Open: roll sway (std about 2.4 deg, about 20 deg/s) and yaw wobble;
backward foot kick larger than the forward reach.
