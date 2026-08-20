# Provenance

Upstream repo: https://github.com/asimovinc/asimov-1
Pinned commit: 41b13c7404a71c394104f355bc4e7d73b42f6519

Vendored path: upstream `sim-model/` -> `source/` in this directory
(`source/xmls/asimov.xml`, `source/assets/meshes/*.STL`, `source/README.md`),
copied verbatim, byte-for-byte, at the pinned commit.

License (per upstream): CERN-OHL-S 2.0 for hardware, GPL-2.0 for software.
See upstream's `HARDWARE-LICENSE.txt` and `SOFTWARE-LICENSE.txt` at the
pinned commit, copied verbatim next to this file as `HARDWARE-LICENSE.txt`
and `SOFTWARE-LICENSE.txt`.

Vendoring date: 2026-07-14.

Local diffs: none. `source/` is untouched upstream content. All
robot-specific additions (actuators, keyframes, passive-joint spring
params, symmetry map, foot sites/geoms) live in `robot.yaml` and
`actuators/*.yaml` next to this file, applied at build time via
`mujoco.MjSpec` injection (`humanoid_lab.robot.build`), never by
hand-editing files under `source/`.
