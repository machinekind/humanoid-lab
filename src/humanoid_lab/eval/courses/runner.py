"""Run the course catalogue against a run's newest checkpoint: courses.json.

    ./run.sh courses --run runs/<name>

Flow. Read run.json and pick the newest checkpoint, the battery's pick. A
checkpoint whose save is incomplete is refused. Build the run's measurement
env through battery.load_checkpoint_policy, on the jax backend and under the
robot's pinned observation noise. Group the rows by arena and floor
friction. Flat rows share one arena. For each group, set the friction, build
and compile a fresh lane, and run every (row, seed) lane on a thread pool.
Each lane is scored in the worker that ran it. Then restore the friction,
add each path row's perfect-unicycle numbers, aggregate, and write the JSON
atomically.

The run's own file (<run>/courses.json for the flat class) holds only the
full measurement: every row, 8 seeds from seed 0, the pinned noise and no
--set. Anything else needs another --out. The report reads the run's own
file. Any two runs' own files for one robot and catalogue therefore compare.

Exit codes. 0: written, or current under --skip-if-current or --check.
2: refused. That covers a bad flag, the canonical guard, an unknown row, a
`sim` or `terrain` --set, a command.resample_steps below 2, a ground class
other than flat, a jax backend other than CPU and an incomplete newest
checkpoint. 3: --check found the output missing or not current. 1: any other
error. A crash leaves the previous file in place.

Importing this module imports jax but starts no backend. The lane module,
and mujoco with it, loads inside run_courses. So --list, --check and a
--skip-if-current that finds its output current build no model and start no
backend.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

import numpy as np

from humanoid_lab.eval import battery
from humanoid_lab.eval.courses import families, report, scoring, spec
from humanoid_lab.eval.courses.spec import Course, CourseParams, PathCourse, SpinCourse
from humanoid_lab.eval.render import DEFAULT_SIZE, frame_size
from humanoid_lab.paths import REPO_ROOT

EXIT_REFUSED = 2
EXIT_STALE = 3

# The default thread count's ceiling. On a 10-core CPU, 8 threads ran 16
# lanes of a trained Roboto policy faster than 1, 4, 6 or 10 did. More than
# 10 cores was not measured. The count never changes a number.
MAX_DEFAULT_WORKERS = 8

# The measurement env's sim block. A run trained on warp still measures on
# jax: flat scores and their seed noise bands are defined on CPU jax, where
# lanes reproduce bit for bit.
FORCED_SIM = {"backend": "jax"}

# --set blocks that are refused: `sim` is forced above, and `terrain` would
# change the ground under a flat measurement.
REFUSED_SET_BLOCKS = ("sim", "terrain")

# The env resamples its command once info["steps_since_cmd"] reaches
# command.resample_steps. A lane zeroes the counter before every step and
# the step raises it to 1, so at 2 or more the env never resamples. At 1 or
# 0 it resamples on every step, and the policy acts on an observation that
# carries the random command.
MIN_RESAMPLE_STEPS = 2

# Version keys in `provenance.versions` -> the distribution that reports them.
DISTRIBUTIONS = {
    "jax": "jax",
    "jaxlib": "jaxlib",
    "mujoco": "mujoco",
    "mujoco_mjx": "mujoco-mjx",
    "warp_lang": "warp-lang",
    "brax": "brax",
    "playground": "playground",
}


class Refused(Exception):
    """A request the runner does not measure (exit 2)."""


# -- the run and its checkpoint ------------------------------------------------


def run_robot(run: dict) -> str:
    """The robot a run trained: hydra_config.robot.name, else the robot dir's name."""
    robot = run["hydra_config"]["robot"]
    return str(robot.get("name") or Path(robot["dir"]).name)


def _require_complete(ckpt: Path) -> Path:
    if not (ckpt / "ppo_network_config.json").exists():
        raise Refused(
            f"{ckpt} has no ppo_network_config.json, which brax writes last in a save: "
            "the newest checkpoint is incomplete"
        )
    return ckpt


def newest_checkpoint(run_dir: Path, run: dict) -> Path:
    """The step dir the battery measures (the largest numeric one), refused
    when brax has not finished writing it. No run status gates it: a run
    with a run.json is measured on its newest complete checkpoint, whether
    training finished, stopped early or was killed and its run.json written
    afterwards."""
    return _require_complete(battery._find_latest_checkpoint(run, Path(run_dir)))


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def checkpoint_sha256(ckpt_dir: Path) -> str:
    """sha256 over every file of a checkpoint dir: one line per file, its
    path relative to the dir and its own sha256, sorted by path."""
    ckpt_dir = Path(ckpt_dir)
    files = sorted(
        (p.relative_to(ckpt_dir).as_posix(), p) for p in ckpt_dir.rglob("*") if p.is_file()
    )
    h = hashlib.sha256()
    for rel, p in files:
        h.update(f"{rel}\0{_file_sha256(p)}\n".encode())
    return h.hexdigest()


def default_workers() -> int:
    """min(MAX_DEFAULT_WORKERS, the CPUs this process may run on). macOS has
    no os.sched_getaffinity, so the CPU count stands in there."""
    affinity = getattr(os, "sched_getaffinity", None)
    n = len(affinity(0)) if affinity is not None else (os.cpu_count() or 1)
    return max(1, min(MAX_DEFAULT_WORKERS, n))


def require_cpu() -> None:
    """Refuse a default jax backend other than CPU. Starts the backend."""
    import jax

    backend = jax.default_backend()
    if backend != "cpu":
        raise Refused(
            f"the default jax backend is {backend!r}. Flat course scores are defined on "
            "CPU jax, where lanes reproduce bit for bit. Run with JAX_PLATFORMS=cpu "
            "(./run.sh courses sets it)"
        )


# -- requests ------------------------------------------------------------------


def check_env_set(extra: dict | None) -> None:
    """Refuse a --set the measurement cannot honour."""
    for block in REFUSED_SET_BLOCKS:
        if block in (extra or {}):
            raise Refused(f"--set {block}.* is refused: the flat measurement owns that block")
    resample = ((extra or {}).get("command") or {}).get("resample_steps")
    if resample is not None:
        try:
            ok = float(resample) >= MIN_RESAMPLE_STEPS
        except (TypeError, ValueError):
            ok = False
        if not ok:
            raise Refused(
                f"--set command.resample_steps={resample!r} is refused: below "
                f"{MIN_RESAMPLE_STEPS} the env replaces the held command on every step"
            )


def merge_overrides(base: dict, extra: dict | None) -> dict:
    """`extra` merged one level deep over `base`, as
    battery.merged_env_overrides merges blocks."""
    out = {k: dict(v) for k, v in base.items()}
    for block, values in (extra or {}).items():
        out[block] = {**out.get(block, {}), **values} if isinstance(values, dict) else values
    return out


def select_rows(cat: dict[str, Course], only) -> list[str]:
    """The row names to run, in catalogue order. An unknown name is refused
    and the refusal lists the catalogue."""
    if not only:
        return list(cat)
    unknown = [n for n in only if n not in cat]
    if unknown:
        raise Refused(
            f"unknown course(s) {unknown}. The catalogue:\n" + "\n".join(f"  {n}" for n in cat)
        )
    wanted = set(only)
    return [n for n in cat if n in wanted]


def requested_rows(run_dir: Path, ground: str, only) -> list[str] | None:
    """The rows a request runs, read from run.json without loading a model.
    None for a robot with no pinned inputs: its catalogue needs the model."""
    params = spec.params_for(run_robot(battery._load_run(Path(run_dir))))
    if params is None:
        return None
    return select_rows(families.catalogue(params, ground), only)


def is_canonical(*, seeds: int, seed_base: int, only, extra_env_overrides,
                 budget_cap: int | None) -> bool:
    """Whether a request is the full measurement the run's own file holds."""
    return (
        not only and not extra_env_overrides and seeds == spec.SEEDS
        and seed_base == 0 and budget_cap is None
    )


def same_file(a: Path, b: Path) -> bool:
    # On a case-insensitive filesystem (the Mac default) resolve() keeps the
    # given case, so a differently cased spelling compares unequal. samefile
    # catches it only when both files exist. Comparing the parent directories
    # and the case-folded names catches it while the run's own file is
    # missing. On a case-sensitive filesystem a name that differs from the
    # run's own only in case is then refused too.
    a, b = Path(a), Path(b)
    if a.resolve() == b.resolve() or (a.exists() and b.exists() and a.samefile(b)):
        return True
    return (
        a.parent.exists() and b.parent.exists() and a.parent.samefile(b.parent)
        and a.name.casefold() == b.name.casefold()
    )


def is_current(out: Path, *, run_dir: Path, ground: str, seeds: int, seed_base: int,
               rows: list[str] | None, env_overrides: dict | None) -> tuple[bool, str]:
    """Whether `out` already holds this request's measurement of the run's
    newest checkpoint, and why not. Loads no model.

    Current means the same schema, ground class, catalogue version and
    fingerprint, robot, pinned params, checkpoint name and sha256, seeds and
    seed base, row set and user --set, and no budget_cap. A file measured
    from inputs read off the model is never current.
    """
    out, run_dir = Path(out), Path(run_dir)
    if not out.exists():
        return False, "it does not exist"
    try:
        doc = json.loads(out.read_text())
    except (OSError, ValueError) as exc:
        return False, f"it is not readable JSON: {exc}"
    if not isinstance(doc, dict):
        return False, "it does not hold a JSON object"
    run = battery._load_run(run_dir)
    robot = run_robot(run)
    params = spec.params_for(robot)
    if params is None:
        return False, (
            f"{robot} has no pinned inputs, and a catalogue measured from the model "
            "is never current"
        )
    try:
        ckpt = newest_checkpoint(run_dir, run)
    except (Refused, FileNotFoundError) as exc:
        return False, str(exc)

    cat = doc.get("catalogue") or {}
    checks = (
        ("schema", doc.get("schema"), spec.SCHEMA_VERSION),
        ("ground class", doc.get("ground_class"), ground),
        ("catalogue version", cat.get("version"), families.CATALOGUE_VERSION),
        ("catalogue fingerprint", cat.get("fingerprint"),
         families.catalogue_fingerprint(params, ground)),
        ("params source", cat.get("params_source"), "pinned"),
        ("robot", doc.get("robot"), robot),
        ("checkpoint", doc.get("checkpoint"), ckpt.name),
        ("seeds", doc.get("seeds"), seeds),
        ("seed base", doc.get("seed_base"), seed_base),
        ("env_overrides", doc.get("env_overrides") or None, env_overrides or None),
        ("budget_cap", doc.get("budget_cap"), None),
    )
    for what, have, want in checks:
        if have != want:
            return False, f"its {what} is {have!r}; this request's is {want!r}"
    have_rows, want_rows = set(doc.get("courses") or {}), set(rows or [])
    if have_rows != want_rows:
        return False, (
            f"its row set differs: missing {sorted(want_rows - have_rows)}, "
            f"extra {sorted(have_rows - want_rows)}"
        )
    if doc.get("checkpoint_sha256") != checkpoint_sha256(ckpt):
        return False, f"checkpoint {ckpt.name}'s files changed since it was measured"
    return True, f"{out} is current for checkpoint {ckpt.name}"


# -- the command box -------------------------------------------------------------


def box_warnings(courses: list[Course], params: CourseParams, command) -> list[str]:
    """One warning per row and axis where a row commands outside the run's
    resolved command box. Rows are measured as asked, not clipped."""
    box = {axis: tuple(float(v) for v in command[axis]) for axis in ("vx", "vy", "wz")}
    out = []
    for c in courses:
        if isinstance(c, SpinCourse):
            asks = {"vx": (0.0, 0.0, "0 m/s"), "vy": (0.0, 0.0, "0 m/s"),
                    "wz": (float(c.wz), float(c.wz), f"{float(c.wz):+.3f} rad/s")}
        else:
            v = c.segment_speeds
            vx = f"{v.max():.2f} m/s" if v.min() == v.max() else f"{v.min():.2f}-{v.max():.2f} m/s"
            asks = {"vx": (float(v.min()), float(v.max()), vx), "vy": (0.0, 0.0, "0 m/s"),
                    # The follower's wz clip: a lagging robot gets up to it.
                    "wz": (-params.yaw_cap, params.yaw_cap, f"up to +-{params.yaw_cap:.3f} rad/s")}
        for axis, (lo, hi, said) in asks.items():
            b_lo, b_hi = box[axis]
            if lo < b_lo - 1e-9 or hi > b_hi + 1e-9:
                out.append(
                    f"{c.name}: {axis} {said} outside the run's {axis} box [{b_lo:g}, {b_hi:g}]"
                )
    return out


# -- artifacts -------------------------------------------------------------------


def course_frame(xy: np.ndarray, anchor) -> np.ndarray:
    """World points (N, 2) in the course frame laid out at `anchor` (x, y, yaw)."""
    x0, y0, yaw0 = (float(v) for v in anchor)
    c, s = math.cos(yaw0), math.sin(yaw0)
    d = np.asarray(xy, dtype=float) - (x0, y0)
    return np.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]], axis=-1)


def path_figure(course: PathCourse, trails: list, seed_numbers: list[int]):
    """The overhead plot of a path row: the course and every seed's base
    trail, in the course frame, labelled by the seeds' own numbers."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = np.asarray(course.waypoints)
    box = np.concatenate([path] + [t for t in trails if len(t)])
    span_x = max(float(np.ptp(box[:, 0])), 1.0)
    span_y = max(float(np.ptp(box[:, 1])), 1.0)
    # Equal aspect keeps the geometry honest, so the figure takes the data's
    # shape: a 10 x 1 m course in a square figure is mostly margin.
    width = 7.0
    height = float(np.clip(width * span_y / span_x + 1.4, 3.0, 9.0))
    fig, ax = plt.subplots(figsize=(width, height))
    ax.plot(path[:, 0], path[:, 1], "k--", lw=1.2, label="course")
    for seed, xy in zip(seed_numbers, trails):
        if len(xy):
            ax.plot(xy[:, 0], xy[:, 1], lw=1.0, alpha=0.8, label=f"seed {seed}")
    ax.set_aspect("equal")
    ax.set_xlabel("x (m), along the settled heading")
    ax.set_ylabel("y (m)")
    ax.set_title(f"{course.name}: {course.isolates}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    return fig


def write_path_plot(out_png: Path, course: PathCourse, trails: list, seed_numbers: list[int]):
    import matplotlib.pyplot as plt

    fig = path_figure(course, trails, seed_numbers)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def write_videos(env, clips: dict, out_dir: Path, size, overlay_torque: bool) -> None:
    """One MP4 per row from its recorded qpos (and tau for the overlay), on
    the calling thread. Frames stream into the writer, so one frame is held
    at a time. A Roboto lane that times out on straight_slow records 6350
    steps. A renderer that fails to start or write warns and leaves the
    numbers alone. A video it stopped partway is deleted."""
    from humanoid_lab.eval.render import SceneView
    from humanoid_lab.eval.video import _pick_camera
    from humanoid_lab.eval.writer import write_video

    every = max(1, round(1 / (30 * env.dt)))
    camera = _pick_camera(env.mj_model, env.robot_spec.eval_camera)
    path = None
    try:
        with SceneView(env, size=size, camera=camera, torque=overlay_torque) as view:
            for name, (qpos, tau) in clips.items():
                if not len(qpos):
                    print(f"courses: no video for {name}: its first seed recorded no course step")
                    continue
                frames = (
                    view.frame(qpos[i], torque=tau[i] if overlay_torque else None)
                    for i in range(0, len(qpos), every)
                )
                path = out_dir / f"{name}.mp4"
                write_video(path, frames, 1.0 / (env.dt * every))
                path = None
    except Exception as exc:  # noqa: BLE001 -- any GL or writer failure, same fallback
        if path is not None:
            path.unlink(missing_ok=True)
        print(f"courses: warning: videos stopped, renderer unavailable ({exc})", file=sys.stderr)


# -- provenance ------------------------------------------------------------------


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _versions() -> dict:
    out = {}
    for key, dist in DISTRIBUTIONS.items():
        try:
            out[key] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            out[key] = None
    return out


def _git(*args: str) -> str | None:
    try:
        done = subprocess.run(["git", "-C", str(REPO_ROOT), *args],
                              capture_output=True, text=True, check=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout


# The /proc/cpuinfo fields that name an arm64 core, in the order recorded.
ARM_CPUINFO_FIELDS = (
    ("CPU implementer", "implementer"),
    ("CPU variant", "variant"),
    ("CPU part", "part"),
    ("CPU revision", "revision"),
)


def _cpuinfo_model(text: str) -> str | None:
    """The CPU model in a /proc/cpuinfo text, read from its first processor
    block. x86 kernels print a `model name`. arm64 kernels print no model
    name; the implementer, variant, part and revision identify the core.
    LLVM picks the CPU that XLA:CPU compiles for from the implementer and
    the part."""
    first: dict[str, str] = {}
    for line in text.splitlines():
        if not line.strip():
            if first:
                break
            continue
        key, sep, value = line.partition(":")
        if sep:
            first.setdefault(key.strip(), value.strip())
    if first.get("model name"):
        return first["model name"]
    arm = [(label, first[key]) for key, label in ARM_CPUINFO_FIELDS if key in first]
    if not arm:
        return None
    return "arm64 " + " ".join(f"{label} {value}" for label, value in arm)


def _cpu_model() -> str | None:
    """The CPU's model string. XLA:CPU compiles for the host's instruction
    set, so bit-identical reruns are expected on the same model only."""
    try:
        if sys.platform == "darwin":
            done = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                  capture_output=True, text=True, check=True, timeout=10)
            return done.stdout.strip() or None
        with open("/proc/cpuinfo") as f:
            model = _cpuinfo_model(f.read())
        if model is not None:
            return model
    except (OSError, subprocess.SubprocessError):
        pass
    return platform.processor() or None


def provenance(run: dict, started_at: str) -> dict:
    """What produced the measurement: this checkout's git commit and whether
    its tree was dirty, the package versions, the jax device (backend, kind,
    count), the start time, and the seed in run.json's hydra_config."""
    import jax

    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain", "--untracked-files=no") if commit else None
    devices = jax.devices()
    return {
        "git_commit": commit.strip() if commit else None,
        "git_dirty": None if status is None else bool(status.strip()),
        "versions": _versions(),
        "device": {
            "backend": jax.default_backend(),
            "kind": devices[0].device_kind if devices else None,
            "count": len(devices),
        },
        "started_at": started_at,
        "run_json_seed": (run.get("hydra_config") or {}).get("seed"),
    }


# -- the measurement -------------------------------------------------------------


def _measured_inputs(env, lane) -> spec.RobotInputs:
    """A robot with no pinned entry: the box, push and noise from the run's
    resolved env config, the stance and height from its model."""
    command = env._config.command
    return spec.RobotInputs(
        vx_max=float(command.vx[1]),
        wz_max=min(-float(command.wz[0]), float(command.wz[1])),
        push_vel=float(env._config.push.vel),
        obs_noise={k: float(v) for k, v in env._config.obs_noise.items()},
        **lane.measure_robot_inputs(env),
    )


def run_courses(run_dir: Path, *, ground: str = "flat", seeds: int = spec.SEEDS,
                seed_base: int = 0, only=None, extra_env_overrides: dict | None = None,
                workers: int | None = None, video: bool = False, video_size=DEFAULT_SIZE,
                overlay_torque: bool = False, paths: bool = False,
                artifacts_dir: Path | None = None, budget_cap: int | None = None) -> dict:
    """Measure the catalogue on `run_dir`'s newest checkpoint and return the
    courses.json document (schema 1).

    `extra_env_overrides` is the user's --set, merged one level deep over
    the measurement env and the pinned noise; the document records it under
    `env_overrides`. Seed k of every row uses PRNGKey(seed_base + k).
    `--video` keeps each row's first seed for an MP4, `--paths` every seed's
    trail for a PNG, both under `artifacts_dir` (default <run>/courses/).
    `budget_cap` cuts every lane's budget to that many steps, for tests; the
    lane program and its shapes do not change.
    """
    wall0 = time.perf_counter()
    started_at = _utc_now()
    run_dir = Path(run_dir)
    if ground not in spec.GROUND_CLASSES:
        raise Refused(f"ground class {ground!r}: only {sorted(spec.GROUND_CLASSES)} exist")
    if seeds < 1:
        raise Refused(f"seeds must be at least 1, got {seeds}")
    check_env_set(extra_env_overrides)
    require_cpu()

    import jax
    import jax.numpy as jp

    from humanoid_lab.eval.courses import lane

    run = battery._load_run(run_dir)
    robot = run_robot(run)
    newest_checkpoint(run_dir, run)
    params = spec.params_for(robot)
    if params is not None:
        select_rows(families.catalogue(params, ground), only)
    workers = default_workers() if workers is None else max(1, int(workers))

    pinned = {} if params is None else {"obs_noise": dict(params.obs_noise)}
    measurement = merge_overrides({"sim": dict(FORCED_SIM), **pinned}, extra_env_overrides)
    t0 = time.perf_counter()
    run, env, ckpt, inf = battery.load_checkpoint_policy(run_dir, measurement)
    env_build_s = time.perf_counter() - t0
    # The loader picks the newest checkpoint again; a save may have landed
    # in between.
    _require_complete(ckpt)

    warnings = []
    if params is None:
        inputs = _measured_inputs(env, lane)
        params = spec.derive(robot, inputs, source="measured")
        warnings.append(
            f"{robot} has no pinned inputs: the catalogue derives from inputs measured on "
            "this run's model and config, and compares only with runs measured the same way"
        )
    else:
        inputs = spec.ROBOT_INPUTS[robot]
    cat = families.catalogue(params, ground)
    rows = select_rows(cat, only)
    warnings += box_warnings([cat[n] for n in rows], params, env._config.command)

    dt = env.dt
    settle = battery.settle_steps(dt)
    max_steps, n_points = lane.lane_shapes(params, dt, ground)
    ids = lane.friction_geoms(env)
    base_friction = env.mj_model.geom_friction[ids, 0].copy()
    # Equal-priority contacts combine by element-wise max, so this is the
    # friction the nominal rows walk on.
    nominal_friction = float(base_friction.max())

    def finish(job, out):
        name, k = job.tag
        out = jax.device_get(out._replace(state=None))
        seed = scoring.seed_out(out, seed_base + k)
        n = seed["steps"]
        kept = {
            "result": scoring.seed_result(out.rec, seed, dt, cat[name], params),
            "env_steps": settle + n,
        }
        if paths and isinstance(cat[name], PathCourse):
            xy = np.stack([out.rec["x"][:n], out.rec["y"][:n]], axis=-1)
            kept["trail"] = course_frame(xy, out.anchor)
        if video and k == 0:
            kept["clip"] = (out.rec["qpos"][:n].copy(), out.rec["tau"][:n].copy())
        return kept

    # Lanes group by arena (`ground`) and friction, and a group's rows run on
    # one env. Flat rows all have ground None, so the groups are the
    # frictions. Groups sort by ground first, so one arena's rows run
    # together. The nominal group runs first, on the model as loaded.
    groups: dict = {}
    for name in rows:
        groups.setdefault((cat[name].ground, cat[name].friction), []).append(name)
    kept: dict[str, list] = {name: [None] * seeds for name in rows}
    compile_s = lanes_s = 0.0
    current = None
    try:
        for key in sorted(groups, key=lambda g: (g[0] is not None, str(g[0]),
                                                 g[1] is not None, g[1] or 0.0)):
            _, mu = key
            if mu != current:
                lane.set_friction(env, ids, base_friction if mu is None else mu)
                current = mu
            # A fresh closure per group: env.step reads env._mjx_model when
            # it is traced.
            fn = lane.make_lane_fn(env, inf, max_steps=max_steps, settle_steps=settle)
            jobs = []
            for name in groups[key]:
                data = lane.lane_data(cat[name], params, n_points, dt)
                if budget_cap is not None:
                    data = data._replace(budget=jp.minimum(data.budget, jp.int32(budget_cap)))
                jobs += [lane.LaneJob(data, jax.random.PRNGKey(seed_base + k), (name, k))
                         for k in range(seeds)]
            t0 = time.perf_counter()
            compiled = lane.compile_lane(fn, jobs[0].data)
            compile_s += time.perf_counter() - t0
            t0 = time.perf_counter()
            done = lane.run_lanes(compiled, jobs, workers=workers, finish=finish)
            group_s = time.perf_counter() - t0
            lanes_s += group_s
            for job, result in zip(jobs, done):
                kept[job.tag[0]][job.tag[1]] = result
            label = "nominal friction" if mu is None else f"friction {mu:g}"
            print(f"courses: {len(jobs)} lanes at {label} in {group_s:.1f} s", flush=True)
    finally:
        if current is not None:
            lane.set_friction(env, ids, base_friction)

    t0 = time.perf_counter()
    unicycle = {
        name: scoring.perfect_unicycle_entry(cat[name], params, dt, n_points)
        for name in rows if isinstance(cat[name], PathCourse)
    }
    unicycle_s = time.perf_counter() - t0

    courses = {}
    for name in rows:
        course = cat[name]
        row = families.describe(course, dt)
        row["friction"] = nominal_friction if course.friction is None else float(course.friction)
        row["perfect_unicycle"] = unicycle.get(name)
        row.update(scoring.aggregate([k["result"] for k in kept[name]]))
        courses[name] = row
    totals = scoring.summary(courses, cat)

    artifacts_dir = Path(artifacts_dir) if artifacts_dir is not None else run_dir / "courses"
    if paths:
        for name in rows:
            if isinstance(cat[name], PathCourse):
                trails = [k["trail"] for k in kept[name]]
                write_path_plot(artifacts_dir / f"{name}_path.png", cat[name], trails,
                                [seed_base + k for k in range(seeds)])
    if video:
        write_videos(env, {name: kept[name][0]["clip"] for name in rows}, artifacts_dir,
                     video_size, overlay_torque)

    constants = families.frozen_constants()
    env_steps = sum(k["env_steps"] for name in rows for k in kept[name])
    wall_s = time.perf_counter() - wall0
    return {
        "schema": spec.SCHEMA_VERSION,
        "ground_class": ground,
        "run": run["run_name"],
        "run_status": run.get("status"),
        "checkpoint": ckpt.name,
        "checkpoint_step": int(ckpt.name),
        "checkpoint_sha256": checkpoint_sha256(ckpt),
        "robot": robot,
        "preset": run["hydra_config"]["actuators"]["name"],
        "trained_task": run["task"],
        # The loader measures a terrain run on its flat rebuild.
        "measured_task": battery.measurement_env_args(run, measurement)[0],
        "seeds": seeds,
        "seed_base": seed_base,
        "canonical": is_canonical(seeds=seeds, seed_base=seed_base, only=only,
                                  extra_env_overrides=extra_env_overrides,
                                  budget_cap=budget_cap),
        "env_overrides": extra_env_overrides or None,
        "budget_cap": budget_cap,
        "catalogue": {
            "version": families.CATALOGUE_VERSION,
            "fingerprint": families.catalogue_fingerprint(params, ground),
            "params_source": params.source,
            "inputs": {
                "vx_max": inputs.vx_max,
                "wz_max": inputs.wz_max,
                "stance_halfwidth_m": inputs.stance_halfwidth_m,
                "nominal_height_m": inputs.nominal_height_m,
                "push_vel": inputs.push_vel,
                "obs_noise": dict(inputs.obs_noise),
            },
            "params": families.params_record(params),
            "follower": {**constants["follower"], "yaw_cap": params.yaw_cap},
            "protocol": {
                **constants["protocol"],
                "ctrl_dt": dt,
                "nominal_friction": nominal_friction,
                # The noise the lanes ran under: the pinned one, unless a
                # --set changed it.
                "obs_noise": {k: float(v) for k, v in env._config.obs_noise.items()},
            },
            "derivation": constants["derivation"],
        },
        "engine": {
            "backend": env._backend,
            "platform": jax.default_backend(),
            "executor": "lane_pool",
            "workers": workers,
            "machine": platform.machine(),
            "cpu": _cpu_model(),
            "xla_flags": os.environ.get("XLA_FLAGS"),
            **{k: v for k, v in _versions().items()
               if k in ("jax", "jaxlib", "mujoco", "mujoco_mjx")},
        },
        "warnings": warnings,
        "summary": totals,
        "courses": courses,
        # Warp's contact and solver counters. The jax backend has none.
        "contacts": None,
        "messages": None,
        "physics_clean": totals["lanes_nonfinite"] == 0,
        "nonfinite_lanes": totals["lanes_nonfinite"],
        "perf": {
            "workers": workers,
            "lanes": totals["lanes"],
            "env_steps": env_steps,
            "env_build_s": round(env_build_s, 2),
            "compile_s": round(compile_s, 2),
            "lanes_s": round(lanes_s, 2),
            "unicycle_s": round(unicycle_s, 2),
            "wall_s": round(wall_s, 2),
            "env_steps_per_s": round(env_steps / lanes_s, 1) if lanes_s > 0 else None,
            "ms_per_env_step": round(1e3 * lanes_s / env_steps, 4) if env_steps else None,
        },
        "provenance": provenance(run, started_at),
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
    }


# -- output ----------------------------------------------------------------------


def nonfinite_fields(obj, where: str = "") -> list[str]:
    """The paths of every NaN or infinite float in a JSON-shaped object."""
    if isinstance(obj, float):
        return [] if math.isfinite(obj) else [where or "<root>"]
    if isinstance(obj, dict):
        return [
            p for k, v in obj.items()
            for p in nonfinite_fields(v, f"{where}.{k}" if where else str(k))
        ]
    if isinstance(obj, (list, tuple)):
        return [p for i, v in enumerate(obj) for p in nonfinite_fields(v, f"{where}[{i}]")]
    return []


def write_result(out: Path, doc: dict) -> None:
    """Write `doc` atomically (scoring.write_json). A non-finite value is an
    error that names where it sits; nothing is written and a previous file
    stays."""
    bad = nonfinite_fields(doc)
    if bad:
        more = f" and {len(bad) - 8} more" if len(bad) > 8 else ""
        raise ValueError(f"non-finite values, {out} not written: {', '.join(bad[:8])}{more}")
    scoring.write_json(out, doc)


def list_catalogue(robot: str) -> str:
    """The derived params and the catalogue of a pinned robot, as text."""
    p = spec.params_for(robot)
    lines = [(
        f"{robot}: v_nom {p.v_nom:.3f}, v_slow {p.v_slow:.3f}, v_fast {p.v_fast:.3f} m/s; "
        f"yaw cap {p.yaw_cap:.3f} rad/s; spins {p.spin_nom:.3f}, {p.spin_slow:.3f}, "
        f"{p.spin_fast:.3f} rad/s; r_tight {p.r_tight:.3f} m; slalom wavelength "
        f"{p.slalom_wavelength:.3f} m; push {p.push_vel:.2f} m/s"
    )]
    head = (
        f"{'course':<21} {'family':<11} {'isolates':<34} {'baseline':<14} {'size':>9} "
        f"{'command':>26} {'budget_s':>8} {'friction':>8}  push"
    )
    lines += [head, "-" * len(head)]
    for name, c in families.catalogue(p).items():
        if isinstance(c, SpinCourse):
            size, cmd, push = f"{c.turns:g} turn", f"wz {c.wz:+.3f} rad/s", "-"
        else:
            size = f"{c.length_m:.2f} m"
            cmd = "vx " + "/".join(f"{v:.2f}" for v in c.speeds) + " m/s"
            push = "-" if c.push_at_m is None else f"{c.push_vel:g} m/s at {c.push_at_m:g} m"
        friction = "-" if c.friction is None else f"{c.friction:g}"
        lines.append(
            f"{name:<21} {c.family:<11} {c.isolates:<34} {c.baseline or '-':<14} {size:>9} "
            f"{cmd:>26} {spec.budget_sec(c):>8.2f} {friction:>8}  {push}"
        )
    return "\n".join(lines)


# -- the CLI -----------------------------------------------------------------------


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m humanoid_lab.eval.courses",
        description="Path-following course benchmark: runs/<name>/courses.json. "
        "See humanoid_lab.eval.courses and docs/configuration.md.",
    )
    ap.add_argument("--run", type=Path, help="the run directory (not with --list)")
    ap.add_argument(
        "--ground", default="flat", choices=sorted(spec.GROUND_CLASSES),
        help="the catalogue partition, and with it the output name (default flat: courses.json)",
    )
    ap.add_argument(
        "--out", type=Path, default=None,
        help="JSON path (default <run>/courses.json). Artifacts go to <out dir>/courses/",
    )
    ap.add_argument("--seeds", type=int, default=spec.SEEDS,
                    help=f"rollouts per row (default {spec.SEEDS}); median and worst reported")
    ap.add_argument(
        "--seed-base", type=int, default=0,
        help="first seed (default 0). Lanes reproduce bit for bit, so a replicate "
        "needs a disjoint base",
    )
    ap.add_argument("--only", nargs="+", metavar="NAME", default=None,
                    help="run only these rows")
    ap.add_argument(
        "--set", dest="set_", action="append", default=None, metavar="BLOCK.KEY=VALUE",
        help="a task.env override merged over the measurement env, e.g. "
        "obs_noise.joint_vel=0.2; repeatable; recorded under env_overrides",
    )
    ap.add_argument("--workers", type=int, default=None,
                    help="lane threads (default min(8, CPUs)); never changes a number")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--skip-if-current", action="store_true",
                      help="exit 0 without loading a model when --out is current")
    mode.add_argument("--check", action="store_true",
                      help="exit 0 when --out is current, 3 when not; loads no model")
    ap.add_argument("--video", action="store_true",
                    help="one MP4 per row from its first seed, into <out dir>/courses/")
    ap.add_argument("--video-size", type=frame_size, default=DEFAULT_SIZE, metavar="WxH",
                    help="rendered frame size (default 640x480)")
    ap.add_argument("--overlay-torque", action="store_true",
                    help="torque bars in the --video frames, from the recorded torques")
    ap.add_argument("--paths", action="store_true",
                    help="one overhead PNG per path row with every seed's trail")
    ap.add_argument("--list", action="store_true",
                    help="print the derived params and the catalogue; loads no model")
    ap.add_argument("--robot", default=None, help="with --list: one robot (default all)")
    return ap


def main(argv=None) -> int:
    ap = parser()
    args = ap.parse_args(argv)

    if args.list:
        if args.run is not None:
            ap.error("--list loads no run; name a robot with --robot")
        robots = [args.robot] if args.robot else list(spec.ROBOT_INPUTS)
        unknown = [r for r in robots if r not in spec.ROBOT_INPUTS]
        if unknown:
            ap.error(f"no pinned inputs for {unknown}; known: {list(spec.ROBOT_INPUTS)}")
        print("\n\n".join(list_catalogue(r) for r in robots))
        return 0
    if args.robot is not None:
        ap.error("--robot goes with --list")
    if args.run is None:
        ap.error("--run is required unless --list")
    if args.seeds < 1:
        ap.error(f"--seeds must be at least 1, got {args.seeds}")
    if args.seed_base < 0:
        ap.error(f"--seed-base must be at least 0, got {args.seed_base}")
    if args.workers is not None and args.workers < 1:
        ap.error(f"--workers must be at least 1, got {args.workers}")
    if args.overlay_torque and not args.video:
        ap.error("--overlay-torque draws into the --video frames; pass --video")
    extra = None
    if args.set_:
        try:
            extra = battery.parse_env_set(args.set_)
            check_env_set(extra)
        except (ValueError, Refused) as exc:
            ap.error(str(exc))

    own = args.run / spec.GROUND_CLASSES[args.ground]
    out = args.out if args.out is not None else own
    noncanonical = [
        flag for flag, used in (
            ("--only", bool(args.only)),
            ("--set", bool(extra)),
            (f"--seeds {args.seeds}", args.seeds != spec.SEEDS),
            (f"--seed-base {args.seed_base}", args.seed_base != 0),
        ) if used
    ]
    if noncanonical and same_file(out, own):
        verb = "needs" if len(noncanonical) == 1 else "need"
        ap.error(
            f"{', '.join(noncanonical)} {verb} an --out other than {own}: that file is the "
            "run's own full measurement, the one the report reads"
        )

    try:
        if args.check or args.skip_if_current:
            rows = requested_rows(args.run, args.ground, args.only)
            current, why = is_current(
                out, run_dir=args.run, ground=args.ground, seeds=args.seeds,
                seed_base=args.seed_base, rows=rows, env_overrides=extra,
            )
            if args.check:
                print(f"current: {why}" if current else f"not current: {out}: {why}")
                return 0 if current else EXIT_STALE
            if current:
                print(f"skipped: {why}")
                return 0
        # run_courses makes these refusals too. Making them here keeps the
        # CLI's exit 2 independent of it.
        requested_rows(args.run, args.ground, args.only)
        require_cpu()
        newest_checkpoint(args.run, battery._load_run(args.run))
        doc = run_courses(
            args.run, ground=args.ground, seeds=args.seeds, seed_base=args.seed_base,
            only=args.only, extra_env_overrides=extra, workers=args.workers,
            video=args.video, video_size=args.video_size,
            overlay_torque=args.overlay_torque, paths=args.paths,
            artifacts_dir=out.parent / "courses",
        )
    except Refused as exc:
        print(f"courses: refused: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    write_result(out, doc)
    print(report.console_table(doc))
    print(f"wrote {out}")
    return 0
