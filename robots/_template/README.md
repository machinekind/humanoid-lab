# `_template`: skeleton for a new robot

Full checklist: `docs/adding-a-robot.md`. Short version:

1. `source/` — vendor the upstream MJCF + meshes verbatim, pinned to a commit.
2. `PROVENANCE.md` — upstream repo, pinned commit, license, local diffs.
3. `robot.yaml` — copy `robot.yaml.template`, fill in every field. Canonical
   joint order in `actuated_joints` is the action/obs contract. Never
   reorder it once training starts. Measure keyframe `base_pos` z against
   the compiled model's foot geoms. Do not copy another robot's height.
4. `actuators/` — copy `actuators/preset.yaml.template` into at least one
   named preset (e.g. `sizing_ideal.yaml`).
5. `mjx/` — generated training XMLs (build output of `run.sh build`, do not
   hand-edit).
6. `./run.sh build --robot <name> --preset <preset>` and `./run.sh check
   --robot <name> --preset <preset>` must both pass.
7. Add `tests/integration/test_<name>.py`, mirroring `tests/integration/test_asimov_v1.py`.
8. Add `configs/robot/<name>.yaml` (a `name`/`dir` pointer, see
   `configs/robot/asimov_v1.yaml`).
