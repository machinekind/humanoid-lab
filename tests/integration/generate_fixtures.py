"""Record tests/data/fixtures/*.npz and ppo_params.json from origin/main.

    generate_fixtures.py [NAME ...]

NAME is a case id from reference_rollouts.CASE_IDS, or `ppo_params`, or
`brax_train`. With no names, every fixture is recorded.

The fixtures pin what the JAX path of origin/main (042f018) computes. They
must only ever be recorded from that code, never from the refactored tree.
Procedure:

1. git worktree add <tmp>/hl-base 042f018
2. On the owner's Mac, the golden platform, create the venv with the pinned
   stack: python3.12 -m venv .venv && .venv/bin/pip install -e '.[dev]'
3. JAX_PLATFORMS=cpu PYTHONPATH=<tmp>/hl-base/src \
       .venv/bin/python tests/integration/generate_fixtures.py

PYTHONPATH overrides the editable install, so humanoid_lab comes from the
base worktree. `python -I` drops PYTHONPATH and silently runs this tree's
code, so never use -I. The script prints humanoid_lab.__file__ first. It
refuses unless the backend is cpu, the code is not this worktree's own, the
code's git HEAD is 042f018, and its src/ has no uncommitted change.

A reviewer checks the committed fixtures without touching them by running
the fixture test on the base code:

    JAX_PLATFORMS=cpu PYTHONPATH=<tmp>/hl-base/src \
        .venv/bin/python -m pytest tests/integration/test_reference_fixtures.py

It compares what that code computes with the committed arrays. Re-running
this generator instead overwrites the committed files. git then reports them
changed even when every array is equal, because zip members carry
timestamps. The tests skip on any platform other than the one recorded.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import jax
import numpy as np

import humanoid_lab
from humanoid_lab import paths
from reference_rollouts import (
    ALL_NAMES,
    BASE_SHA,
    BRAX_TRAIN_PATH,
    CASES,
    FIXTURE_DIR,
    PPO_PARAMS_PATH,
    brax_train,
    case_path,
    host_meta,
    ppo_params,
    rollout,
    source_sha,
)


def _refuse(msg: str) -> int:
    print(f"refusing to record: {msg}", file=sys.stderr)
    return 1


def _uncommitted_src() -> str:
    return subprocess.run(
        ["git", "-C", str(paths.REPO_ROOT), "status", "--porcelain", "--", "src"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _same(a: dict, b: dict) -> bool:
    return a.keys() == b.keys() and all(np.array_equal(a[k], b[k]) for k in a)


def main(argv: list[str]) -> int:
    print(f"humanoid_lab from {humanoid_lab.__file__}")
    if jax.default_backend() != "cpu":
        return _refuse(
            f"backend {jax.default_backend()!r}; the fixtures are CPU bit patterns. "
            "Re-run with JAX_PLATFORMS=cpu."
        )
    if Path(paths.REPO_ROOT).resolve() == Path(__file__).resolve().parents[2]:
        return _refuse(
            "refusing to record from this worktree's own code. Run with "
            "PYTHONPATH=<base worktree>/src (see the module docstring)."
        )
    sha = source_sha()
    if sha != BASE_SHA:
        return _refuse(f"source HEAD is {sha}, not {BASE_SHA}.")
    dirty = _uncommitted_src()
    if dirty:
        return _refuse(f"the source tree's src/ has uncommitted changes:\n{dirty}")
    unknown = [n for n in argv if n not in ALL_NAMES]
    if unknown:
        return _refuse(f"unknown name(s) {unknown}; known: {list(ALL_NAMES)}")

    selected = list(argv) or list(ALL_NAMES)
    by_name = {c.name: c for c in CASES}
    meta = host_meta()
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    for name in selected:
        if name == "ppo_params":
            data = {**ppo_params(), "_meta": meta}
            PPO_PARAMS_PATH.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
            print(f"wrote {PPO_PARAMS_PATH.name}")
            continue
        if name == "brax_train":
            first, second = brax_train(), brax_train()
            if not _same(first, second):
                return _refuse("brax_train is not repeatable; nothing written for it.")
            arrays, path = first, BRAX_TRAIN_PATH
        else:
            arrays, path = rollout(by_name[name]), case_path(name)
        np.savez_compressed(path, **arrays, **{k: np.array(v) for k, v in meta.items()})
        print(f"wrote {path.name} ({path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
