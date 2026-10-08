"""Task-name facts shared by the trainer and the tools. Imports nothing
heavy, so it can run before jax touches a device."""

from __future__ import annotations

# Tasks that train on a terrain arena.
TERRAIN_TASKS = frozenset({"terrain"})

# The flat task each terrain task rebuilds as. A terrain task's env config
# is its flat task's plus one `terrain` block.
FLAT_COUNTERPART = {"terrain": "joystick"}

# The terrain task, and the tasks whose runs rebuild on it. A joystick
# override applies to the terrain task's config unchanged.
TERRAIN_TASK = "terrain"
TERRAIN_SOURCE_TASKS = ("joystick", "terrain")

# Allocator settings a terrain run starts with, unless the environment
# already sets them. MJWarp allocates its CCD scratch outside the XLA pool,
# on every collision call of a model with convex pairs, and a terrain scene
# has them. A preallocated XLA pool leaves that scratch whatever the pool
# fraction leaves over. jaxlib reads XLA_PYTHON_CLIENT_PREALLOCATE when it
# creates the GPU client. It reads no variable named
# XLA_PYTHON_CLIENT_PREALLOC.
TERRAIN_XLA_DEFAULTS = {"XLA_PYTHON_CLIENT_PREALLOCATE": "false"}


def is_terrain(task: str) -> bool:
    return task in TERRAIN_TASKS


def xla_env_defaults(task: str) -> dict[str, str]:
    """The environment variables `task` sets with setdefault before any
    device use. Empty for a flat task."""
    return dict(TERRAIN_XLA_DEFAULTS) if is_terrain(task) else {}
