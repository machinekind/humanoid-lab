# Provenance

Upstream repo: https://github.com/Roboparty/roboto_origin
Pinned commit: de10e7a624805b79b547abf3eec846618136f52f

Vendored path: upstream `modules/rpo_description/` -> `source/` in this
directory (`source/mjcf/*.xml`, `source/meshes/*.STL`, `source/urdf/rpo.urdf`,
`source/terrain_assets/`, `source/README.md`, `source/rpo_urdf.png`), copied
verbatim, byte-for-byte, at the pinned commit.

License (per upstream): GPL-3.0 for the whole upstream repo, `rpo_description`
included. The upstream top-level `LICENSE` is copied to `LICENSE` next to this
file. The upstream training module that the actuator presets cite for gains
and limits (`modules/roboparty_train/robolab/`) carries its own BSD-3-Clause
license at that path; nothing from it is vendored.

Vendoring date: 2026-07-15.

Local diffs: none. `source/` is untouched upstream content. All robot-specific
additions (actuator presets, keyframes, injected foot sites and collision
geoms, solver overrides) live in `robot.yaml` and `actuators/*.yaml` next to
this file, applied at build time via `mujoco.MjSpec`
(`humanoid_lab.robot.build`), never by hand-editing files under `source/`.
