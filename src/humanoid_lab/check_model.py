"""CLI: gate-check a robot's built mjx model for NaN divergence.

    JAX_PLATFORMS=cpu python -m humanoid_lab.check_model --robot asimov_v1 --preset sizing_ideal

Loads robots/<robot>/mjx/<preset>.xml (building it in-memory via build_spec,
without writing to disk, if the file doesn't exist yet), then for EVERY
keyframe in robot.yaml: resets to it, holds ctrl at the keyframe's
actuated-joint targets (knees_bent has nonzero targets, so the ctrl wiring
is actually exercised, not just the all-zero home pose), and runs `--steps`
(default 200) plain-MuJoCo steps. Unless --skip-mjx, also runs a short mjx
rollout (~50 steps) from `home` on the CPU jax impl. mujoco 3.10 vendors
MJWarp, but this check always uses mjx.put_model(impl="jax") -- it is a CPU
gate, not a GPU benchmark; see envs/backend.py's resolve_backend for the
jax/warp convention this repo uses elsewhere.

Fails on NaN in qpos/qvel OR on |qvel| exceeding --max-qvel (default 100
rad/s): MuJoCo's bad-qacc auto-reset keeps a diverging simulation
huge-but-finite, so an isfinite check alone waves through gross
instability.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np

from humanoid_lab import paths
from humanoid_lab.build_model import ensure_training_scene
from humanoid_lab.robot.build import build_spec, compile_spec
from humanoid_lab.robot.presets import parse_set_overrides
from humanoid_lab.robot.spec import load_robot_spec

MJX_ROLLOUT_STEPS = 50


def _free_joint_qpos_addr(model: mujoco.MjModel) -> int:
    for i in range(model.njnt):
        if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE:
            return int(model.jnt_qposadr[i])
    raise ValueError("model has no free joint; cannot read base height")


def _load_model(
    robot: str, preset: str, xml_override: Path | None, actuator_overrides: dict | None = None
) -> tuple[mujoco.MjModel, str]:
    if xml_override is not None:
        return mujoco.MjModel.from_xml_path(str(xml_override)), str(xml_override)

    robot_dir = paths.ROBOTS_DIR / robot
    xml_path = robot_dir / "mjx" / f"{preset}.xml"
    # actuator_overrides forces the in-memory build_spec branch even if a
    # prebuilt XML exists: the prebuilt file was compiled from the bare
    # preset, so the fast path would silently ignore --set.
    if not actuator_overrides and xml_path.exists():
        return mujoco.MjModel.from_xml_path(str(xml_path)), str(xml_path)

    reason = "--set given" if actuator_overrides else f"{xml_path} not found"
    print(f"{reason}; building in-memory (not writing to disk)")
    spec = build_spec(robot_dir, preset, actuator_overrides)
    ensure_training_scene(spec)
    return compile_spec(spec), f"{xml_path} (in-memory, not built yet)"


def _keyframe_ctrl(robot_spec, kf_name: str, model: mujoco.MjModel) -> np.ndarray:
    kf = robot_spec.keyframes[kf_name]
    ctrl = np.zeros(model.nu)
    for i, joint_name in enumerate(robot_spec.actuated_joints):
        ctrl[i] = kf.joints.get(joint_name, 0.0)
    return ctrl


def check(
    robot: str,
    preset: str,
    steps: int,
    xml_override: Path | None,
    skip_mjx: bool,
    max_qvel: float,
    actuator_overrides: dict | None = None,
) -> bool:
    robot_dir = paths.ROBOTS_DIR / robot
    robot_spec = load_robot_spec(robot_dir)
    model, xml_label = _load_model(robot, preset, xml_override, actuator_overrides)
    base_addr = _free_joint_qpos_addr(model)

    print(f"model: {xml_label}")
    print(f"nq={model.nq} nv={model.nv} nu={model.nu}")
    print(f"total mass: {model.body_mass.sum():.4f} kg")

    ok = True

    # --- Plain MuJoCo pass, once per keyframe ---
    for kf_name in robot_spec.keyframes:
        key = model.key(kf_name)
        ctrl = _keyframe_ctrl(robot_spec, kf_name, model)

        data = mujoco.MjData(model)
        data.qpos[:] = key.qpos
        data.ctrl[:] = ctrl
        mujoco.mj_forward(model, data)
        z0 = float(data.qpos[base_addr + 2])

        qvel_min, qvel_max = np.inf, -np.inf
        for _ in range(steps):
            mujoco.mj_step(model, data)
            qvel_min = min(qvel_min, float(data.qvel.min()))
            qvel_max = max(qvel_max, float(data.qvel.max()))

        z1 = float(data.qpos[base_addr + 2])
        nan_plain = not (
            np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel))
        )
        blown_up = max(abs(qvel_min), abs(qvel_max)) > max_qvel
        print(
            f"plain mujoco [{kf_name}]: {steps} steps, base z {z0:.4f} -> {z1:.4f}, "
            f"qvel [{qvel_min:.4f}, {qvel_max:.4f}], NaN={'YES' if nan_plain else 'no'}"
            + (f", |qvel| > {max_qvel} (UNSTABLE)" if blown_up else "")
        )
        if nan_plain or blown_up:
            ok = False

    # --- MJX pass (CPU jax impl), home keyframe only to bound runtime ---
    if not skip_mjx:
        from mujoco import mjx

        mjx_seed_data = mujoco.MjData(model)
        mjx_seed_data.qpos[:] = model.key("home").qpos
        mjx_seed_data.ctrl[:] = _keyframe_ctrl(robot_spec, "home", model)
        mujoco.mj_forward(model, mjx_seed_data)

        mjx_model = mjx.put_model(model, impl="jax")
        mjx_data = mjx.put_data(model, mjx_seed_data, impl="jax")
        z0_mjx = float(mjx_data.qpos[base_addr + 2])

        for _ in range(MJX_ROLLOUT_STEPS):
            mjx_data = mjx.step(mjx_model, mjx_data)

        qpos_mjx = np.asarray(mjx_data.qpos)
        qvel_mjx = np.asarray(mjx_data.qvel)
        z1_mjx = float(qpos_mjx[base_addr + 2])
        nan_mjx = not (np.all(np.isfinite(qpos_mjx)) and np.all(np.isfinite(qvel_mjx)))
        blown_up_mjx = float(np.abs(qvel_mjx).max()) > max_qvel
        print(
            f"mjx (jax/cpu): {MJX_ROLLOUT_STEPS} steps, base z {z0_mjx:.4f} -> {z1_mjx:.4f}, "
            f"qvel [{qvel_mjx.min():.4f}, {qvel_mjx.max():.4f}], NaN={'YES' if nan_mjx else 'no'}"
            + (f", |qvel| > {max_qvel} (UNSTABLE)" if blown_up_mjx else "")
        )
        if nan_mjx or blown_up_mjx:
            ok = False
    else:
        print("mjx: skipped (--skip-mjx)")

    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", required=True, help="robot dir name under robots/")
    parser.add_argument("--preset", required=True, help="actuator preset name")
    parser.add_argument("--steps", type=int, default=200, help="plain-MuJoCo step count")
    parser.add_argument(
        "--xml", type=Path, default=None, help="override: check this built XML path directly"
    )
    parser.add_argument(
        "--skip-mjx", action="store_true", help="skip the MJX rollout, plain MuJoCo only"
    )
    parser.add_argument(
        "--max-qvel",
        type=float,
        default=100.0,
        help="fail if any |qvel| exceeds this (rad/s); catches huge-but-finite divergence",
    )
    parser.add_argument(
        "--set",
        dest="set_",
        action="append",
        default=None,
        metavar="PATH=VALUE",
        help="override a preset value (see robot/presets.py's parse_set_overrides); repeatable",
    )
    args = parser.parse_args()

    if args.set_ and args.xml is not None:
        parser.error("--set and --xml are mutually exclusive: --xml already names a fixed built model")

    actuator_overrides = parse_set_overrides(args.set_) if args.set_ else None
    ok = check(
        args.robot, args.preset, args.steps, args.xml, args.skip_mjx, args.max_qvel, actuator_overrides
    )
    print("GATE PASS" if ok else "GATE FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
