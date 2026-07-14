# Adding a robot

Checklist for onboarding a new robot into `robots/<name>/`:

1. `source/` — vendor the upstream MJCF + meshes verbatim, pinned to a commit.
2. `PROVENANCE.md` — upstream repo, pinned commit, license, local diffs.
3. `robot.yaml` — RobotSpec: joint order, actuated groups, keyframes, foot sites/geoms,
   symmetry map, obs layout, termination bodies, passive-joint spring params.
4. `actuators/` — named presets (e.g. `sizing_ideal.yaml`, `deploy_pd.yaml`).
5. `mjx/` — generated training XMLs (build output, do not hand-edit).
