# Port plan: w01-tek learnings → humanoid-lab

Written 2026-07-31. The source is machinekind/w01-tek, cloned locally at
`~/git/machinekind/w01-tek`. The repo was renamed from w01-tek. We mined its commits
from 2026-07-14 to 2026-07-30 and verified every claim below against its code.

## TLDR

w01-tek spent July learning why trained policies refuse to walk. Additive
tracking rewards pay a robot that stands still. Narrow tracking kernels leave
fast commands with no reward gradient. Gait-shaping rewards pay for lifting a
leg while stationary. Command corners that training never samples become
learned refusals. w01-tek built a fix for each of these. We port that set
before the first serious GPU run. We port the deploy contract before we
publish the first policy. Terrain work is deferred. Every ported mechanism is
off by default and changes nothing while off.

## Process

- Strict TDD. Write the failing test first and commit it. Then implement
  until it passes. A mechanism merges only with its behavior tests and a test
  that proves the env is unchanged while the mechanism is off.
- One item is one branch and one PR. The implementing agent reads its brief
  in [docs/port-details.md](docs/port-details.md). Do phase 0 first. Do
  phases 1 and 2 next. Phases 3 to 6 can interleave.
- Write commit messages as plain statements of what changed and why.
- w01-tek is a quadruped. Copy its mechanisms and methodology. Re-derive every
  joint map and tuned number for our robots. Each brief lists what to
  re-derive.

**When do we train?** This file plans code changes and does not schedule
training runs. The `asimov_gentle_penalties` A/B runs on current code and can
launch now. The first serious walking runs launch once phases 1 and 2 land.
Phases 3 to 6 proceed in parallel with those runs. Phase 0 takes about half a
day and makes the TDD loop fast.

## The list

**Phase 0. Test infrastructure.**
- [x] 0.1 Split the tests into a fast unit suite and a slow integration suite, enforced by a guard test. [details](docs/port-details.md#01-test-split)
  - Landed with the pre-port goldens the off-switch gate below needs:
    `tests/integration/test_golden_baseline.py` against
    `tests/data/golden/*.npz`, recorded before any mechanism.

**Phase 1. Reward and command mechanics.**
- [x] 1.1 `tracking_product` multiplies the linear and angular tracking kernels. [details](docs/port-details.md#11-tracking_product)
- [x] 1.2 `tracking_relative` normalizes tracking error by the commanded magnitude. [details](docs/port-details.md#12-tracking_relative)
- [x] 1.3 `tracking_far` blends in a wide kernel so far-off tracking still has gradient. [details](docs/port-details.md#13-tracking_far)
- [x] 1.4 `shaping_tracking_gate` makes gait-shaping rewards pay only while tracking. [details](docs/port-details.md#14-shaping_tracking_gate)
- [x] 1.5 `no_progress` terminates episodes that ignore their command. [details](docs/port-details.md#15-no_progress)
- [x] 1.6 Pure command draws sample clean wz, vy, slow, fast, and backward commands. [details](docs/port-details.md#16-pure-command-draws)
- [x] 1.7 `feet_apex` pays swing peaks and `feet_landing` taxes hard touchdowns. [details](docs/port-details.md#17-feet_apex-and-feet_landing)
- [x] 1.8 `orientation_tol_deg` adds a tolerance cone to the tilt penalty. [details](docs/port-details.md#18-orientation_tol_deg)
- [ ] 1.9 `real_pose_ref` anchors pose rewards on the settled pose. [details](docs/port-details.md#19-real_pose_ref)

**Phase 2. Trainer and MJWarp runtime. This phase gates the first GPU run.**
- [ ] 2.1 Early stopping ends runs whose eval reward has plateaued. [details](docs/port-details.md#21-early-stopping)
- [ ] 2.2 MJWarp preflight measures contact budgets and adds the DR vmap composition test. [details](docs/port-details.md#22-mjwarp-preflight)

**Phase 3. Symmetry.**
- [ ] 3.1 Mirror augmentation, plus the rule that eval always runs in the deployment frame. [details](docs/port-details.md#31-symmetry)

**Phase 4. Eval additions.**
- [ ] 4.1 `spin_left` and `spin_right` battery probes. [details](docs/port-details.md#41-spin-probes)
- [ ] 4.2 Gait KPIs report swing apex and touchdown softness. [details](docs/port-details.md#42-gait-kpis)
- [ ] 4.3 `tracking_error` reports servo error, and run.json records effective gains. [details](docs/port-details.md#43-tracking_error)
- [ ] 4.4 A robustness grid sweeps Kt error, torque lag, and a torque-speed envelope. [details](docs/port-details.md#44-robustness-grid)
- [ ] 4.5 Video improvements: per-joint grid, `--plots` selection, push-free rollouts. [details](docs/port-details.md#45-video-qol)

**Phase 5. Deploy contract. Land before the first published policy.**
- [ ] 5.1 A fail-closed deploy contract with a key ledger. [details](docs/port-details.md#51-deploy-contract)
- [ ] 5.2 An exporter that validates round-trips before writing artifacts. [details](docs/port-details.md#52-exporter)

**Phase 6. Ops quick wins. No TDD needed.**
- [ ] 6.1 Audit the Makefile `--export` lists. [details](docs/port-details.md#61-makefile-export)
- [ ] 6.2 Harden the batch templates. [details](docs/port-details.md#62-batch-hardening)
- [ ] 6.3 Add the `hpc.sh` wrapper. [details](docs/port-details.md#63-hpch)
- [ ] 6.4 Add a GPU preflight sizing job. Optional. [details](docs/port-details.md#64-preflight-sizing)

**Deferred.** The terrain system waits until a flat keeper exists.
[details](docs/port-details.md#deferred-terrain)

## CR checklist

Run this against the full diff at the end of the port.

### Off-switch integrity
- [ ] Every new mechanism is off by default in every existing config.
- [ ] With all new features off, an env rollout is bit-exact against the pre-port commit. `tests/integration/test_golden_baseline.py` proves it, for four robot/preset pairs over 260 steps.
- [ ] Disabled features consume no RNG keys. Every new `jax.random.split` or `fold_in` sits behind its feature flag.
- [ ] The resolved config of every pre-existing experiment yaml is unchanged.

### TDD evidence
- [ ] Every mechanism's test landed before or with its implementation.
- [ ] Unit tests build no model, create no env, and touch no device. The guard test enforces this. The unit suite runs in seconds.
- [ ] Integration tests cover each mechanism through a real env step.

### Reward mechanics
- [ ] `tracking_product` reassigns both kernels from the pre-product values.
- [ ] The `tracking_far` blend applies in the absolute branch and in the relative branch. The ported regression test proves it.
- [ ] `shaping_tracking_gate` uses the post-product kernel. Stand-still penalties stay gated on the command.
- [ ] `no_progress` arms only when demand exceeds the threshold and the grace period has elapsed.
- [ ] `no_progress` reseeds its EMA on every command resample.
- [ ] `no_progress` is a true termination with no reward term attached.
- [ ] Motion against the command scores negative progress. A test proves it.
- [ ] `real_pose_ref` settles against the runtime target bounds, keeps only the strictly increasing height prefix, and raises on a degenerate table.
- [ ] The settled anchor is identical under two different gain sets. A test proves it.
- [ ] The legacy anchor path still works and remains the default.
- [ ] `orientation_tol_deg=0` reproduces the legacy penalty bit-exact.
- [ ] `feet_apex` pays once per swing at first contact and resets its tracker on contact.
- [ ] Each pure command draw has its own `fold_in` index. Enabling one draw leaves the other draws' samples unchanged. A test proves it.
- [ ] The zero-command overwrite still runs last.

### Symmetry
- [ ] The mirror map derives from `robot.yaml`'s symmetry map.
- [ ] The env validates the mirror map against its actual observation layout at construction.
- [ ] Actions are un-mirrored before physics. Physics, rewards, and termination run in the real frame.
- [ ] Battery, eval, and video reconstruction force `symmetry.enable=false`.
- [ ] The invisibility test is ported. No doc claims the mirror fixes asymmetric behavior.

### Eval integrity
- [ ] Gait KPIs are raw metrics. No existing score, gate, or JSON field changes meaning.
- [ ] Continuous metrics average only over runs that measured something. Sample counts are visible in the output.
- [ ] Every new KPI excludes the reset transient with a settle window.
- [ ] Every velocity in a new metric is body-frame.
- [ ] Spin probes report each direction separately.
- [ ] Battery JSON changes are additive.
- [ ] The robustness grid's baseline cell is documented as the native path.
- [ ] The explicit-PD path reproduces the native pipeline within a stated tolerance as lag goes to zero. A test proves it.
- [ ] Grid outputs never overwrite the canonical battery.json.

### Deploy contract
- [ ] Every env config key is classified as consumed or training-only.
- [ ] An unclassified key makes export raise. A test proves it.
- [ ] Both round-trip validations run before any artifact is written to its destination.
- [ ] The runtime check loads the artifacts it will ship.
- [ ] Torque caps and tables travel in the policy metadata.
- [ ] Deploy docs state facts and give no imperatives.

### Ops and MJWarp
- [ ] Contact budgets come from measured peaks on the biped model, including fallen states, with stated headroom.
- [ ] Jobs record their `nacon` and `nefc` peaks and flag overflow. njmax overflow is silent, so nothing waits for a warning.
- [ ] A composition test calls `jax.vmap` with the DR wrapper's contract.
- [ ] Every static model field is patched before `in_axes` derivation.
- [ ] The Makefile builds `--export` lists with a comma variable.
- [ ] Every env var a batch script consumes appears in the export list. An audit checks this.
- [ ] `run_main` wrappers capture and return the real exit code.
- [ ] Cache and log directories are probed for writability and fall back to TMPDIR.

### Hygiene
- [ ] Every comment states the numbers the code uses. w01-tek's settle docstring said kp 2000 while the code used 400.
- [ ] configuration.md documents every new config key.
- [ ] No doc table is stale. Count the files each table claims to list.
- [ ] run.json records each mechanism's enable state and any effective values a reader needs to reproduce the run.
- [ ] Commit messages are plain statements. Each commit is self-contained.
- [ ] docs/lessons/ records anything non-obvious learned during the port.
