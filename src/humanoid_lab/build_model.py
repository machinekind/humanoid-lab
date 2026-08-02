"""CLI: build a robot's training-scene MJCF for a given actuator preset.

    ./run.sh build --robot roboto_origin --preset deploy_pd [--out PATH]

Loads robot.yaml + the named actuator preset (robot/build.py's build_spec:
actuator injection, armature/frictionloss override, passive-joint springs,
keyframes), composes the training scene (adds a ground plane + light only
if the vendored source XML doesn't already carry one), and writes the
resulting MJCF to robots/<robot>/mjx/<preset>.xml (or --out). Mesh
references in the written file are rewritten so they resolve from the mjx/
output location back to the vendored source/assets/ directory. The write is
verified by recompiling the WRITTEN file straight from disk.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import mujoco

from humanoid_lab import paths
from humanoid_lab.robot.build import build_spec
from humanoid_lab.robot.presets import parse_set_overrides


def _has_ground_plane(spec: mujoco.MjSpec) -> bool:
    return any(g.type == mujoco.mjtGeom.mjGEOM_PLANE for g in spec.geoms)


def _has_light(spec: mujoco.MjSpec) -> bool:
    return len(list(spec.lights)) > 0


def ensure_training_scene(spec: mujoco.MjSpec) -> list[str]:
    """Add a ground plane / light only if the vendored XML doesn't have one.

    Returns human-readable notes on what was found vs. added, for the CLI
    summary.
    """
    notes = []
    if _has_ground_plane(spec):
        notes.append("floor: already present in source XML (not added)")
    else:
        spec.worldbody.add_geom(
            name="floor",
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=[0.0, 0.0, 0.05],
            rgba=[0.7, 0.7, 0.7, 1.0],
        )
        notes.append("floor: added (source XML had none)")

    if _has_light(spec):
        notes.append("light: already present in source XML (not added)")
    else:
        spec.worldbody.add_light(
            name="training_light",
            pos=[1.0, 0.0, 3.5],
            dir=[0.0, 0.0, -1.0],
            directional=True,
        )
        notes.append("light: added (source XML had none)")
    return notes


def _rewrite_meshdir_for_output(xml_text: str, robot_dir: Path, out_path: Path) -> str:
    """Rewrite the compiler meshdir so mesh refs resolve from out_path's dir.

    Setting spec.meshdir directly on a live MjSpec and re-serializing makes
    mujoco re-resolve every mesh file relative to the process's cwd (not the
    model's original file dir) as soon as the value changes, which breaks
    here since the caller's cwd is arbitrary (confirmed empirically: it is
    not modelfiledir-relative once meshdir is mutated). Instead, keep the
    spec's own working meshdir for compilation, then string-substitute the
    compiler tag's meshdir in the already-serialized XML text before writing
    it out -- the mesh binary data going into to_xml() was already resolved
    at MjSpec.from_file time, so this substitution is purely textual.
    """
    match = re.search(r'meshdir="([^"]*)"', xml_text)
    if not match:
        raise ValueError("compiled spec's XML has no compiler meshdir to rewrite")
    original_meshdir = match.group(1)
    source_xml_dir = robot_dir / "source" / "xmls"
    assets_abs = (source_xml_dir / original_meshdir).resolve()
    new_meshdir = os.path.relpath(assets_abs, start=out_path.parent.resolve())
    return xml_text.replace(f'meshdir="{original_meshdir}"', f'meshdir="{new_meshdir}"', 1)


def build(
    robot: str, preset: str, out: Path | None = None, actuator_overrides: dict | None = None
) -> Path:
    robot_dir = paths.ROBOTS_DIR / robot
    spec = build_spec(robot_dir, preset, actuator_overrides)
    scene_notes = ensure_training_scene(spec)

    out_path = out if out is not None else robot_dir / "mjx" / f"{preset}.xml"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    xml_text = spec.to_xml()
    xml_text = _rewrite_meshdir_for_output(xml_text, robot_dir, out_path)
    out_path.write_text(xml_text)

    # Verify by compiling the WRITTEN file straight from disk, exactly as
    # any downstream consumer (check_model.py, envs) will load it.
    model = mujoco.MjModel.from_xml_path(str(out_path))

    print(f"wrote {out_path}")
    for note in scene_notes:
        print(f"  {note}")
    print(f"  nq={model.nq} nv={model.nv} nu={model.nu}")
    print(f"  total mass: {model.body_mass.sum():.4f} kg")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", required=True, help="robot dir name under robots/")
    parser.add_argument("--preset", required=True, help="actuator preset name")
    parser.add_argument(
        "--out", type=Path, default=None, help="output XML path (default: robots/<robot>/mjx/<preset>.xml)"
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

    if args.set_ and args.out is None:
        parser.error(
            "--set requires --out: the default robots/<robot>/mjx/<preset>.xml is the canonical "
            "preset build and must not be overwritten by an ad-hoc --set variant"
        )

    actuator_overrides = parse_set_overrides(args.set_) if args.set_ else None
    build(args.robot, args.preset, args.out, actuator_overrides)


if __name__ == "__main__":
    main()
