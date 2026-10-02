"""The terrain scan suite: a fixed arena, its cells, and the course every
cell is scored on. Numpy and the arena generator only, so the suite loads
without jax, a model or a checkpoint.

A suite belongs to one robot. `SUITES` holds the robots that have one, and
`suite_for` refuses any other.

Each run is one forward crossing from a tile's pad:
- The robot spawns on the pad at the tile centre, moved by the run's offset
  along its heading, and faces that heading.
- It stands for `settle_steps` at zero command. It then walks at the
  constant command `[v, 0, 0]`.
- It passes when its base reaches `r_out` from the tile centre after the
  settle, before its deadline and without a fall.

Every start is on a flat pad. On a `pyramid_*` tile a run walks down the
feature. On an `inverted_*` tile it climbs out of the pit. Climbing onto a
flight from flat ground and walking down into a pit from its rim are not
measured.

Distances are Chebyshev, max(|dx|, |dy|) from the tile centre. Stair
flights, pits and slope plateaus are concentric squares about it, and
`TileSpec.feature_radius` bounds a tile's boxes in Chebyshev distance. A
Chebyshev radius therefore crosses the same feature on every heading. A
diagonal heading walks sqrt(2) times as far to reach it. Each run's
deadline is therefore sized on its own distance.

`r_out(cell) = min(feature_r + footprint_reach, tile/2 - footprint_reach)`.
For Roboto on the eval arena it is 1.75 m on stair tiles and 1.85 m on
every other tile:
- A stair flight ends at 1.6 m. At 1.75 m the whole standing footprint
  lies past it.
- On every other tile `feature_r` is 1.9 m, and the tile edge binds. At
  1.85 m the footprint reaches the 2 m half-tile and no further.

No start is sampled. The arena, the starts and the deadlines are fixed.
The policy acts deterministically. Each run's key drives only the env's
observation noise. That noise is all that separates two draws of one start.

The rows, bars, speeds, offsets, `budget_slack` and the warp budgets are
untuned.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from humanoid_lab.terrain import (
    TYPES,
    Arena,
    ArenaParams,
    TerrainSpec,
    TileSpec,
)
from humanoid_lab.terrain.config import arena_for

# The eval arena's rows, in difficulty space, 0.2 apart. Each row holds one
# tile of every type, and every tile is a cell. Row 1.2 lies past
# difficulty 1, the default arena's top row. Its risers are 17.6 cm. The
# robot's own stair scene, robots/roboto_origin/source/mjcf/rpo_stairs.xml,
# has 10 and 15 cm risers.
DIFFICULTIES = (0.2, 0.4, 0.6, 0.8, 1.0, 1.2)

# The dimension a cell is named by: the type's ramp on ArenaParams and its
# unit. Rubble shares the obstacle height ramp.
_DIMENSIONS = {
    "rough_uniform": ("rough_amplitude", "cm"),
    "pyramid_slope": ("slope_angle", "deg"),
    "inverted_pyramid_slope": ("slope_angle", "deg"),
    "pyramid_stairs": ("stair_riser", "cm"),
    "inverted_pyramid_stairs": ("stair_riser", "cm"),
    "discrete_obstacles": ("obstacle_height", "cm"),
    "random_grid": ("obstacle_height", "cm"),
    "wave": ("wave_amplitude", "cm"),
}

# Pass-rate bars by realized difficulty. Untuned. Rough ground, both slopes
# and waves take them on the three easiest rows. Stairs, obstacles and
# rubble carry none. How high Roboto steps depends on its actuator preset's
# action window. A bar on them would presume that choice. Cells without a
# bar are tracked, never gated.
GATED_TYPES = ("rough_uniform", "pyramid_slope", "inverted_pyramid_slope", "wave")
LADDER = {0.2: 0.95, 0.4: 0.80, 0.6: 0.60}
# Every bar is untuned, so a gate reports it as provisional.
PROVENANCE = "provisional"


@dataclass(frozen=True)
class Cell:
    """One measured tile: a terrain type on one arena row.

    `value` is the realized dimension in `unit`, rounded to one decimal. It
    is the number in `name`, e.g. `pyramid_stairs_9.8cm`. Names are the
    stable keys of a scan's results. `difficulty` is the tile's realized
    difficulty, the row's times the type's cap. `bar` is the pass rate a
    gate asks for, or None for a tracked cell."""

    name: str
    terrain_type: str
    row: int
    difficulty: float
    value: float
    unit: str
    bar: float | None

    @property
    def tracked(self) -> bool:
        return self.bar is None


@dataclass(frozen=True)
class Run:
    """One start of the course, scored on every cell.

    The robot starts `offset` metres along its heading from the tile centre
    and faces `yaw`. `draw` changes only the run's key, which drives the
    env's observation noise. Inference is deterministic, so runs of two
    draws differ only in that noise."""

    index: int
    heading_index: int
    yaw: float
    offset: float
    draw: int


@dataclass(frozen=True)
class Suite:
    """A robot's terrain scan: the arena, its cells, the course and the
    protocol constants.

    `version` covers the arena, the cells, the course and the protocol
    constants below. Scans of different versions do not compare, so a change
    to any of them bumps it. The warp budgets lie outside it. A pool that is
    too small makes a scan invalid, not different. `fingerprint` is
    `terrain.fingerprint` of the arena `arena` builds. The scan generates
    the arena from `arena` in memory and refuses one whose fingerprint
    differs.

    The course is `headings` evenly spaced headings times `offsets` times
    `draws`, the same runs on every cell at every speed. `speeds` are the
    commanded forward speeds, m/s.

    `footprint_reach` bounds how far the standing feet reach horizontally
    from the base, m. It sets `r_out` and keeps every start on the pad.

    `settle_steps` is the zero-command stand before the walk. Each run's
    deadline is the settle plus `budget_slack` times its distance over the
    commanded speed. A run that averages under 1 / `budget_slack` of its
    commanded speed times out.

    `saturation_frac` is the fraction of a torque cap that counts as
    saturated. `base_contact_tol` is the clearance, m, under which a
    termination collider ends a run.

    `naconmax_per_env`, `njmax` and `naccdmax_per_env` are the scan's warp
    contact, constraint and CCD budgets. None for `naccdmax_per_env` means
    the naconmax pool."""

    robot: str
    version: int
    arena: ArenaParams
    fingerprint: str
    cells: tuple[Cell, ...]
    # Untuned. Both lie inside Roboto's trained command box.
    speeds: tuple[float, ...] = (0.3, 0.6)
    headings: int = 8
    # Untuned. The starts lie 0.06 m apart. Their 0.18 m span is 60% of a
    # 0.30 m tread on an axis heading and 42% on a diagonal.
    offsets: tuple[float, ...] = (-0.09, -0.03, 0.03, 0.09)
    draws: int = 2
    # Bounds the 0.1498 m that Roboto's foot capsules reach from the base at
    # the reset pose. It moves `r_out`, so a change bumps `version`.
    footprint_reach: float = 0.15
    # 1 s at Roboto's 0.02 s control step.
    settle_steps: int = 50
    # Untuned.
    budget_slack: float = 1.6
    # The flat battery's saturation fraction, so the two compare.
    saturation_frac: float = 0.95
    # The terrain task's default base contact tolerance.
    base_contact_tol: float = 0.01
    # Untuned. check-terrain --arena eval measures what the eval arena needs
    # on a GPU.
    naconmax_per_env: int = 256
    njmax: int = 2048
    naccdmax_per_env: int | None = None

    def __post_init__(self):
        names = [c.name for c in self.cells]
        if len(set(names)) != len(names):
            dupes = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(
                f"suite cell names must be unique, got {dupes} more than once"
            )
        if not self.speeds or min(self.speeds) <= 0:
            raise ValueError(
                f"suite speeds must be positive, got {list(self.speeds)}. A run "
                "is a forward crossing."
            )
        if self.headings < 1 or self.draws < 1 or not self.offsets:
            raise ValueError(
                "the course needs a heading, an offset and a draw, got "
                f"{self.headings} headings, {len(self.offsets)} offsets and "
                f"{self.draws} draws"
            )
        # The lookup reads a pit's first riser one cell early, so the
        # footprint stays a cell inside the pad, as the generator's spawns do.
        room = self.arena.pad_radius - self.arena.cell_size
        reach = max(abs(o) for o in self.offsets) + self.footprint_reach
        if reach > room + 1e-9:
            raise ValueError(
                f"an offset of {max(abs(o) for o in self.offsets)} m puts the "
                f"{self.footprint_reach} m footprint {reach:.3f} m from the tile "
                f"centre, past the {room:.3f} m of flat pad a cell inside its "
                f"{self.arena.pad_radius} m radius"
            )

    @property
    def runs_per_cell(self) -> int:
        """Runs per (cell, speed): headings x offsets x draws."""
        return self.headings * len(self.offsets) * self.draws


def eval_arena_params() -> ArenaParams:
    """The eval arena: one row per `DIFFICULTIES` entry, columns in `TYPES`
    order, no flat row, and one 0.30 m stair tread for every stair tile.
    Every other field keeps its default: 4 m tiles, a 2 m border, 0.04 m
    cells and 0.4 m pads.

    It has 1135 boxes on 701 x 901 nodes."""
    return ArenaParams(
        seed=0,
        ordered=True,
        difficulties=DIFFICULTIES,
        flat_row=False,
        stair_tread=0.30,
    )


def eval_arena(suite: Suite) -> Arena:
    """The arena `suite.arena` builds, from the cache `arena_for` keeps.
    Its arrays are read-only."""
    return arena_for(suite.arena)


def cell_value(
    params: ArenaParams, terrain_type: str, difficulty: float
) -> tuple[float, str]:
    """(value, unit) of the dimension that names a cell of `terrain_type` at
    realized `difficulty`, unrounded: centimetres of roughness, riser,
    obstacle height or wave amplitude, or degrees of slope."""
    ramp_name, unit = _DIMENSIONS[terrain_type]
    raw = getattr(params, ramp_name)(difficulty)
    return (math.degrees(raw) if unit == "deg" else raw * 100.0), unit


def build_cells(params: ArenaParams) -> tuple[Cell, ...]:
    """One cell per terrain tile of the arena `params` builds, rows in order
    and types in `TYPES` order within a row.

    A flat row has no feature and gives no cell. Rows keep their arena
    index, so on an arena with a flat row the first cell row is 1. The rows
    must be explicit, `params.difficulties`. Explicit rows are never
    jittered, so each cell's difficulty is exact without generating the
    arena."""
    if params.difficulties is None:
        raise ValueError(
            "suite cells need explicit row difficulties, and these params set "
            "none. Implicit rows may be jittered."
        )
    first = int(params.flat_row)
    cells = []
    for i, d_row in enumerate(params.difficulties):
        for terrain_type in TYPES:
            d = d_row * params.cap(terrain_type)
            raw, unit = cell_value(params, terrain_type, d)
            value = round(raw, 1)
            bar = LADDER.get(d) if terrain_type in GATED_TYPES else None
            cells.append(
                Cell(
                    name=f"{terrain_type}_{value:g}{unit}",
                    terrain_type=terrain_type,
                    row=first + i,
                    difficulty=d,
                    value=value,
                    unit=unit,
                    bar=bar,
                )
            )
    return tuple(cells)


def cell_tile(spec: TerrainSpec, cell: Cell) -> TileSpec:
    """The tile `cell` measures on the arena `spec` describes. Raises when
    the arena has no such tile, or its tile has another difficulty."""
    for tile in spec.tiles:
        if tile.row == cell.row and tile.terrain_type == cell.terrain_type:
            if tile.difficulty != cell.difficulty:
                raise ValueError(
                    f"cell {cell.name} is difficulty {cell.difficulty}, but row "
                    f"{cell.row}'s {cell.terrain_type} tile is {tile.difficulty}. "
                    "The cell belongs to another arena."
                )
            return tile
    raise ValueError(f"the arena has no {cell.terrain_type} tile on row {cell.row}")


def course(suite: Suite) -> tuple[Run, ...]:
    """The runs every cell is scored on, in a fixed order: draw outermost,
    then heading, then offset. Heading h faces yaw 2 pi h / headings."""
    runs = []
    for draw in range(suite.draws):
        for h in range(suite.headings):
            yaw = 2.0 * math.pi * h / suite.headings
            for offset in suite.offsets:
                runs.append(
                    Run(
                        index=len(runs),
                        heading_index=h,
                        yaw=yaw,
                        offset=offset,
                        draw=draw,
                    )
                )
    return tuple(runs)


def r_out(feature_r: float, tile_size: float, reach: float) -> float:
    """Chebyshev distance from the tile centre at which a run has crossed.

    The base walks until a footprint of `reach` lies past the tile's
    features, unless that would carry the feet off the tile. A stair tile's
    flight ends inside that limit, so a run crosses it whole."""
    return min(feature_r + reach, tile_size / 2 - reach)


def heading_stretch(yaw: float) -> float:
    """Metres walked along `yaw` per metre of Chebyshev distance: 1 on an
    axis, sqrt(2) on a diagonal."""
    return 1.0 / max(abs(math.cos(yaw)), abs(math.sin(yaw)))


def run_distance(r_out: float, run: Run) -> float:
    """Metres `run` walks from its start to Chebyshev `r_out`. A positive
    offset starts the robot ahead along its heading, so it walks less."""
    return r_out * heading_stretch(run.yaw) - run.offset


def episode_budget(speed: float, ctrl_dt: float, distance: float, suite: Suite) -> int:
    """Control steps a run of `distance` metres gets at commanded `speed`:
    the settle, then `budget_slack` times the walking time. Raises for a
    speed that is not positive, which never crosses."""
    if speed <= 0:
        raise ValueError(
            f"a commanded speed of {speed} m/s has no step budget. A run is a "
            "forward crossing, and a robot that does not walk forward never "
            "reaches r_out."
        )
    walk_s = suite.budget_slack * distance / speed
    return suite.settle_steps + math.ceil(walk_s / ctrl_dt)


def run_deadlines(
    suite: Suite, r_out: float, speed: float, ctrl_dt: float
) -> tuple[int, ...]:
    """Each run's own deadline on a cell of Chebyshev `r_out`, in `course`
    order. The longest is the cell's step budget at `speed`."""
    return tuple(
        episode_budget(speed, ctrl_dt, run_distance(r_out, run), suite)
        for run in course(suite)
    )


def threshold(cell: Cell, n_runs: int) -> tuple[int | None, str]:
    """(passes a gate asks for out of `n_runs`, provenance). A tracked cell
    gives (None, "tracked").

    The count is the bar's fraction rounded up: 61, 52 and 39 of 64 for the
    0.95, 0.80 and 0.60 bars. The epsilon keeps a product that lands just
    above a whole count, such as 0.55 * 100, on that count."""
    if cell.bar is None:
        return None, "tracked"
    return math.ceil(cell.bar * n_runs - 1e-9), PROVENANCE


ROBOTO_SUITE = Suite(
    robot="roboto_origin",
    version=1,
    arena=eval_arena_params(),
    fingerprint="7691ccb7a24c5ac0a09b492aec0bd206eaf5d601cfeee153d294aa0c5504ba8f",
    cells=build_cells(eval_arena_params()),
)

SUITES = {ROBOTO_SUITE.robot: ROBOTO_SUITE}


def suite_for(robot: str) -> Suite:
    """The terrain suite of `robot`. Raises for a robot without one."""
    try:
        return SUITES[robot]
    except KeyError:
        known = ", ".join(sorted(SUITES))
        raise ValueError(
            f"no terrain suite for {robot}. Suites exist for: {known}"
        ) from None
