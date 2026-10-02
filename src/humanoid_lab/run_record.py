"""run.json: provenance, the curriculum trace, and the record's life cycle.

train.py writes run.json before training starts, rewrites its `progress`
block at every eval, and rewrites it whole at the end. A run that is
killed still leaves its configs, provenance and arena on disk, and its
checkpoints stay loadable. Every write is atomic: a reader sees the old
record or the new one, never a partial file.
"""

from __future__ import annotations

import json
import os
import secrets
import subprocess
import time
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

# Version keys in run.json -> the distribution that reports them.
DISTRIBUTIONS = {
    "jax": "jax",
    "jaxlib": "jaxlib",
    "mujoco": "mujoco",
    "mujoco_mjx": "mujoco-mjx",
    "warp_lang": "warp-lang",
    "brax": "brax",
    "playground": "playground",
}

# Training metrics, as brax's episode logger reports them: means over the
# last 100 episodes that ended. LEVEL_FREE_METRIC is the level of an
# episode the curriculum moves and 0 for a pinned one. FREE_METRIC is 1
# and 0 for the same episodes. Promotions and demotions are counts per
# episode, always 0 for a pinned one. Each is divided by FREE_METRIC's
# mean, so pinned episodes drop out (curriculum_rates). The env's
# terrain/level_per_step is the level of every training episode, pinned
# ones included. It reaches wandb unchanged.
LEVEL_FREE_METRIC = "episode/terrain/level_free_per_step"
FREE_METRIC = "episode/terrain/free_per_step"
PROMOTED_METRIC = "episode/terrain/promoted"
DEMOTED_METRIC = "episode/terrain/demoted"

RUNNING = "running"
FINISHED = "finished"
EARLY_STOPPED = "early_stopped"
FAILED = "failed"


def versions() -> dict[str, str | None]:
    """Installed version of each DISTRIBUTIONS entry, None when missing."""
    out = {}
    for key, dist in DISTRIBUTIONS.items():
        try:
            out[key] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            out[key] = None
    return out


def _git(repo: Path, *args: str) -> str | None:
    try:
        done = subprocess.run(
            ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout


def git_state(repo: Path) -> tuple[str | None, bool | None]:
    """(commit, dirty) of the checkout at `repo`. Dirty means a tracked file
    differs from the commit. Both None outside a git checkout."""
    commit = _git(repo, "rev-parse", "HEAD")
    if commit is None:
        return None, None
    status = _git(repo, "status", "--porcelain", "--untracked-files=no")
    return commit.strip(), None if status is None else bool(status.strip())


def device_info() -> dict:
    """The jax backend and its devices. Initializes the backend."""
    import jax

    devices = jax.devices()
    return {
        "backend": jax.default_backend(),
        "kind": devices[0].device_kind if devices else None,
        "count": len(devices),
    }


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def provenance(repo: Path, *, wandb_run_id: str | None = None, started_at: str | None = None) -> dict:
    """What produced a run: commit, package versions, device, wandb run."""
    commit, dirty = git_state(repo)
    return {
        "git_commit": commit,
        "git_dirty": dirty,
        "versions": versions(),
        "device": device_info(),
        "wandb_run_id": wandb_run_id,
        "started_at": started_at or utc_now(),
    }


def write_json_atomic(path: Path, obj) -> None:
    """Write `obj` as JSON to `path` through a temp file in the same
    directory and a rename. Values json cannot encode are written as
    str. The file mode follows the umask, as a plain open's does."""
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, indent=2, default=str)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def curriculum_rates(metrics: dict) -> dict | None:
    """{level, promoted, demoted} over the episodes the curriculum moves,
    from one progress call's training metrics.

    None when the call carries no curriculum metrics, or when none of the
    episodes it averages was free. promoted and demoted are None when the
    call lacks their metric."""
    if LEVEL_FREE_METRIC not in metrics or FREE_METRIC not in metrics:
        return None
    free = float(metrics[FREE_METRIC])
    if not free > 0.0:
        return None
    out = {"level": float(metrics[LEVEL_FREE_METRIC]) / free}
    for key, name in (("promoted", PROMOTED_METRIC), ("demoted", DEMOTED_METRIC)):
        out[key] = float(metrics[name]) / free if name in metrics else None
    return out


class CurriculumTrace:
    """The curriculum level as training metrics report it, over the
    episodes the curriculum moves.

    Keeps the last level and the highest, with the env step of each, and
    the last promotion and demotion rates (per episode)."""

    def __init__(self):
        self.level_last = None
        self.level_last_steps = None
        self.level_max = None
        self.level_max_steps = None
        self.promoted_last = None
        self.demoted_last = None

    def update(self, num_steps: int, metrics: dict) -> bool:
        """Record a progress call. False when it carries no level."""
        rates = curriculum_rates(metrics)
        if rates is None:
            return False
        level = rates["level"]
        self.level_last, self.level_last_steps = level, int(num_steps)
        if self.level_max is None or level > self.level_max:
            self.level_max, self.level_max_steps = level, int(num_steps)
        if rates["promoted"] is not None:
            self.promoted_last = rates["promoted"]
        if rates["demoted"] is not None:
            self.demoted_last = rates["demoted"]
        return True

    def record(self) -> dict | None:
        """The run.json `curriculum` block, None before any level."""
        if self.level_last is None:
            return None
        return {
            "level_last": self.level_last,
            "level_last_steps": self.level_last_steps,
            "level_max": self.level_max,
            "level_max_steps": self.level_max_steps,
            "promoted_last": self.promoted_last,
            "demoted_last": self.demoted_last,
        }


class RunRecord:
    """run.json across one training run.

    `fields` are the pre-training fields: names, configs, contacts, gains.
    The record starts `running` with the final fields null, and is written
    at once."""

    def __init__(self, path: Path, fields: dict, *, provenance: dict, arena: dict | None,
                 restore: dict | None):
        self.path = Path(path)
        self.data = {
            **fields,
            "status": RUNNING,
            "early_stopped": None,
            "stopped_at_steps": None,
            "final_reward": None,
            "provenance": provenance,
            "arena": arena,
            "restore": restore,
            "progress": None,
            "wall_s": None,
            "steps_per_s": None,
            "curriculum": None,
        }
        self._t0 = None
        self.write()

    def write(self) -> None:
        write_json_atomic(self.path, self.data)

    def start_clock(self) -> None:
        """Mark the start of training. wall_s counts from here."""
        self._t0 = time.monotonic()

    def wall_s(self) -> float | None:
        return None if self._t0 is None else time.monotonic() - self._t0

    def progress(self, steps: int, eval_reward: float | None, curriculum: dict | None) -> None:
        """Rewrite the `progress` block after an eval."""
        wall = self.wall_s()
        self.data["progress"] = {
            "steps": int(steps),
            "wall_s": wall,
            "steps_per_s": None if not wall else steps / wall,
            "eval_reward": eval_reward,
            "curriculum": curriculum,
        }
        self.write()

    def finish(self, *, status: str, early_stopped: bool | None, stopped_at_steps: int | None,
               final_reward: float | None, steps: int | None, curriculum: dict | None,
               error: str | None = None) -> None:
        """Rewrite the record at the end of training. steps_per_s covers
        the whole training call, compiles and evals included."""
        wall = self.wall_s()
        self.data.update(
            status=status,
            early_stopped=early_stopped,
            stopped_at_steps=stopped_at_steps,
            final_reward=final_reward,
            wall_s=wall,
            steps_per_s=None if not wall or steps is None else steps / wall,
            curriculum=curriculum,
        )
        if error is not None:
            self.data["error"] = error
        self.write()
