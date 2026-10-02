"""Arena parameters: every number that shapes a terrain arena, in one place.

`ArenaParams` is frozen and complete. The generator reads no other number.
Equal params build equal arenas. Fields also accept the lists and dicts a
yaml config carries. Construction normalizes them to tuples and `Ramp`s.
`ArenaParams(**cfg)` and `params_from_dict` take a config block unchanged.

The rules that derive stair and slope geometry from the params live here
too. The generator and any check of an arena against its params compute
them the same way.
"""

from __future__ import annotations

import math
import numbers
import operator
from dataclasses import dataclass, fields

import numpy as np

# Column order of an ordered arena, one column per terrain type.
TYPES: tuple[str, ...] = (
    "rough_uniform",
    "pyramid_slope",
    "inverted_pyramid_slope",
    "pyramid_stairs",
    "inverted_pyramid_stairs",
    "discrete_obstacles",
    "random_grid",
    "wave",
)
STAIR_TYPES: tuple[str, ...] = ("pyramid_stairs", "inverted_pyramid_stairs")

# Construction normalizes every field by these groups. Floats and ints
# accept numpy's types too and store the Python type. A yaml `4` and `4.0`
# then name the same arena. Ramps come from {base, gain} or [base, gain].
# Ranges come from [lo, hi]. difficulties, type_caps, stair_tread and
# wave_half_periods have rules of their own.
_FLOATS = (
    "tile_size",
    "border",
    "cell_size",
    "row_jitter",
    "pad_radius",
    "pad_taper",
    "edge_taper",
    "coarse_step",
    "overlay_fraction",
    "slope_platform_half",
    "summit_floor",
    "rim_margin",
    "edge_margin",
    "pad_clearance",
    "grid_pitch",
    "grid_fill_prob",
)
_INTS = ("seed", "n_rows", "discrete_count", "stair_min_steps", "stair_max_steps")
_BOOLS = ("ordered", "flat_row")
_RAMPS = (
    "rough_amplitude",
    "slope_angle",
    "stair_riser",
    "obstacle_height",
    "wave_amplitude",
)
_RANGES = (
    "discrete_half_range",
    "discrete_height_fraction",
    "grid_half_range",
    "grid_height_fraction",
)
# Lengths that must be positive. The generator divides by the cell, the
# two tapers, the noise step and the grid pitch. The pad and the summit
# floor size the stair summit box.
_POSITIVE = (
    "cell_size",
    "pad_radius",
    "pad_taper",
    "edge_taper",
    "coarse_step",
    "summit_floor",
    "grid_pitch",
)

# Slack for float comparisons against geometry that is exact on paper. A
# flight that ends exactly at the rim fits. A tread or rubble pitch that
# divides the room exactly counts in full, even when the quotient comes out
# one ulp low.
_EPS = 1e-9


@dataclass(frozen=True)
class Ramp:
    """A linear difficulty ramp, `base + gain * d`, and its exact inverse."""

    base: float
    gain: float

    def __post_init__(self):
        object.__setattr__(self, "base", float(self.base))
        object.__setattr__(self, "gain", float(self.gain))
        if not self.gain > 0:
            raise ValueError(f"ramp gain must be positive, got {self.gain}")

    def __call__(self, d: float) -> float:
        return self.base + self.gain * d

    def difficulty(self, value: float) -> float:
        """The difficulty at which this ramp reaches `value`."""
        return (value - self.base) / self.gain


@dataclass(frozen=True)
class ArenaParams:
    """Everything that defines one arena.

    Which arena: `seed` drives every random draw. `n_rows` terrain rows ramp
    difficulty from 0 to 1. `difficulties` gives the rows explicitly and
    then sets `n_rows`. Explicit rows get no jitter and no clamp at 1.
    `ordered` keeps the `TYPES` column order and the exact ramp. Otherwise
    each row shuffles its columns. Each interior row also jitters its
    difficulty by up to `row_jitter` of a row gap. `flat_row` prepends one
    flat row as level 0. `type_caps` scales a type's difficulty, so its
    column spans [0, cap].

    No default has been tuned by training. Each group's comment says which
    numbers were sized for the biped and which are untuned. The biped has
    0.6 m legs, a 0.158 x 0.058 m foot and a standing footprint that
    reaches 0.15 m from the base.
    """

    # Which arena.
    seed: int = 0
    n_rows: int = 10
    ordered: bool = False
    difficulties: tuple[float, ...] | None = None
    flat_row: bool = False
    # Under half a row gap keeps realized difficulty strictly increasing.
    row_jitter: float = 0.4
    type_caps: tuple[tuple[str, float], ...] = ()

    # Grid. The 4 m tile is sized for the biped. It fits four 0.30 m treads
    # (1.2 m of run) between the 0.4 m summit and the rim margin. The
    # 0.04 m cell is finer than the 0.058 m sole width. The 2 m border is
    # untuned. tile_size and border must be whole multiples of cell_size,
    # so every tile edge is a node line. A zero border puts the outer tile
    # edges on the heightfield edge. Any run-off margin the env needs is
    # then the env's choice.
    tile_size: float = 4.0
    border: float = 2.0
    cell_size: float = 0.04

    # Spawn pad: a flat disc at every tile centre. 0.4 m holds the 0.15 m
    # standing footprint under up to 0.21 m of spawn jitter (pad_radius -
    # footprint - cell_size). The lookup reads a pit's first riser one cell
    # early, so a spawn's reach stays a cell inside the pad. Noise ramps in
    # over pad_taper past the pad. It ramps out over edge_taper inside the
    # tile edge. Neither the pad rim nor the tile border is a step. The two
    # tapers are untuned and must be positive.
    pad_radius: float = 0.4
    pad_taper: float = 0.25
    edge_taper: float = 0.25

    # Difficulty ramps. At d = 1 the riser is 15 cm and obstacles are 11 cm,
    # sized for the 0.6 m leg. The 20 degree slope (0.35 rad), 4 cm rough
    # noise and 6 cm wave are untuned. The riser and obstacle bases must be
    # positive, so every box has a positive height. The slope angle must
    # stay in [0, pi/2) up to the arena's steepest slope tile.
    rough_amplitude: Ramp = Ramp(0.005, 0.035)
    slope_angle: Ramp = Ramp(0.0, 0.35)
    stair_riser: Ramp = Ramp(0.02, 0.13)
    obstacle_height: Ramp = Ramp(0.01, 0.10)
    wave_amplitude: Ramp = Ramp(0.01, 0.05)

    # Rough noise: uniform noise on a coarse_step lattice, bilinearly
    # upsampled. The ground rolls at about one foot length (0.158 m) and
    # does not spike per cell. Slope and box tiles carry the same noise at
    # overlay_fraction of the rough amplitude, untuned.
    coarse_step: float = 0.15
    overlay_fraction: float = 0.3

    # Slopes: a flat square plateau of this half-width, then a ramp to 0 at
    # the tile edge. The plateau must hold the spawn pad and leave at least
    # one cell of ramp. It is otherwise untuned.
    slope_platform_half: float = 0.6

    # Stairs: a summit platform, then concentric square treads down (or, in
    # the pit, up) to the tile ground. stair_tread is one tread for every
    # stair tile, or a (lo, hi) range drawn per tile. stair_tread (its lo,
    # for a range) must be at least two cell_size, 0.08 m at the default
    # cell. 0.30 m is a common building stair tread, about two foot
    # lengths. The summit is the spawn pad's half-width, never below
    # summit_floor. The 0.3 m floor holds the 0.15 m footprint under up to
    # 0.11 m of jitter, a cell inside the summit edge like the pad.
    # rim_margin of flat ground separates the outermost riser from the tile
    # edge. It is at least one cell_size. A flight has as many risers as
    # fit, from stair_min_steps to stair_max_steps. Three risers lay two
    # treads, the fewest that make a flight. The margin and the step bounds
    # are untuned.
    stair_tread: float | tuple[float, float] = 0.30
    summit_floor: float = 0.3
    rim_margin: float = 0.25
    stair_min_steps: int = 3
    stair_max_steps: int = 6

    # Discrete obstacles, untuned: discrete_count yawed boxes per tile, with
    # half-sizes in discrete_half_range. A box's height is a
    # discrete_height_fraction of the ramp. Boxes and rubble stay
    # edge_margin inside the tile edge, at least one cell_size. An obstacle
    # turned 45 degrees must still fit there. Boxes clear the pad by
    # pad_clearance (zero or more) beyond their corner radius. Every
    # obstacle and rubble half-size must reach cell_size / sqrt(2), so each
    # box covers a node.
    discrete_count: int = 12
    discrete_half_range: tuple[float, float] = (0.10, 0.25)
    discrete_height_fraction: tuple[float, float] = (0.5, 1.0)
    edge_margin: float = 0.1
    pad_clearance: float = 0.05

    # Rubble, untuned: a grid_pitch cell grid, each cell raising one small
    # yawed box with grid_fill_prob. The boxes, 0.08 to 0.22 m across, are
    # on the scale of the 0.158 m foot. A box too wide for its cell shrinks
    # to fit (see rubble_shrink).
    grid_pitch: float = 0.25
    grid_fill_prob: float = 0.7
    grid_half_range: tuple[float, float] = (0.04, 0.11)
    grid_height_fraction: tuple[float, float] = (0.3, 1.0)

    # Wave, untuned: whole half-periods per axis, drawn per tile from this
    # set. The surface is then exactly 0 on all four tile edges.
    wave_half_periods: tuple[int, ...] = (2, 3)

    def __post_init__(self):
        put = object.__setattr__
        for name in _FLOATS:
            put(self, name, float(getattr(self, name)))
        for name in _INTS:
            put(self, name, operator.index(getattr(self, name)))
        for name in _BOOLS:
            value = getattr(self, name)
            if not isinstance(value, (bool, np.bool_)):
                raise TypeError(f"{name} must be a bool, got {value!r}")
            put(self, name, bool(value))
        for name in _RAMPS:
            put(self, name, _ramp(getattr(self, name)))
        for name in _RANGES:
            put(self, name, _pair(getattr(self, name)))
        put(
            self,
            "wave_half_periods",
            tuple(operator.index(k) for k in self.wave_half_periods),
        )
        tread = self.stair_tread
        put(
            self,
            "stair_tread",
            float(tread) if isinstance(tread, numbers.Real) else _pair(tread),
        )
        caps = self.type_caps
        caps = caps.items() if hasattr(caps, "items") else caps
        put(self, "type_caps", tuple(sorted((str(t), float(c)) for t, c in caps)))
        if self.difficulties is not None:
            put(self, "difficulties", tuple(float(d) for d in self.difficulties))
            put(self, "n_rows", len(self.difficulties))
        self._validate()

    def _validate(self):
        unknown = sorted({t for t, _ in self.type_caps} - set(TYPES))
        if unknown:
            raise ValueError(f"type_caps names unknown terrain types {unknown}")
        if any(c <= 0 for _, c in self.type_caps):
            raise ValueError(f"type_caps must be positive, got {dict(self.type_caps)}")
        min_rows = 1 if self.difficulties is not None else 2
        if self.n_rows < min_rows:
            raise ValueError(
                f"an arena needs at least {min_rows} terrain rows, got {self.n_rows}"
            )
        if self.difficulties is not None and min(self.difficulties) < 0:
            raise ValueError(
                f"difficulties must not be negative, got {list(self.difficulties)}"
            )
        if not 0 <= self.row_jitter < 0.5:
            raise ValueError(
                f"row_jitter {self.row_jitter} must lie in [0, 0.5) to keep "
                "row difficulty strictly increasing"
            )
        for name in _POSITIVE:
            if not getattr(self, name) > 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")
        for name in _RANGES:
            lo, hi = getattr(self, name)
            if not 0 < lo <= hi:
                raise ValueError(f"{name} ({lo}, {hi}) must satisfy 0 < lo <= hi")
        for name in ("stair_riser", "obstacle_height"):
            if not getattr(self, name).base > 0:
                raise ValueError(
                    f"{name} base must be positive, got {getattr(self, name).base}. "
                    "Every box needs a positive height."
                )
        if self.discrete_count < 0:
            raise ValueError(
                f"discrete_count must not be negative, got {self.discrete_count}"
            )
        if not self.wave_half_periods or min(self.wave_half_periods) < 1:
            raise ValueError(
                "wave_half_periods must hold at least one count, each at least "
                f"1. Got {list(self.wave_half_periods)}"
            )
        node_count(self.tile_size, self.cell_size)
        if self.border < 0:
            raise ValueError(f"border must not be negative, got {self.border}")
        node_count(self.border, self.cell_size, minimum=0)
        for name in ("rim_margin", "edge_margin"):
            if getattr(self, name) < self.cell_size:
                raise ValueError(
                    f"{name} {getattr(self, name)} must be at least one "
                    f"cell_size ({self.cell_size}). Boxes stay that far inside "
                    "the tile edge, so neighbouring tiles share flat perimeter "
                    "nodes."
                )
        reach = self.tile_size / 2 - self.edge_margin
        widest = self.discrete_half_range[1]
        if math.sqrt(2) * widest > reach + _EPS:
            raise ValueError(
                f"discrete_half_range upper {widest} m reaches "
                f"{math.sqrt(2) * widest:.3f} m at 45 degrees of yaw, past the "
                f"{reach} m inside edge_margin"
            )
        # The lookup sees a box only through the nodes it covers. Every point
        # lies within cell_size / sqrt(2) of a node, so a box whose shorter
        # half-size reaches that far covers one at any offset and yaw.
        # Rubble is checked at its smallest size after the cell shrink, the
        # shortest half-size drawn beside the longest.
        need = self.cell_size / math.sqrt(2)
        lo, hi = self.grid_half_range
        smallest = {
            "discrete_half_range": self.discrete_half_range[0],
            "grid_half_range": lo * rubble_shrink(lo, hi, self),
        }
        for name, half in smallest.items():
            if half < need - _EPS:
                raise ValueError(
                    f"{name} {getattr(self, name)} gives boxes with a half-size "
                    f"of {half:.4f} m, under cell_size {self.cell_size} / sqrt(2) "
                    f"= {need:.4f} m. Such a box can cover no node and the lookup "
                    "never sees it. Use larger boxes or a finer cell_size."
                )
        if self.pad_clearance < 0:
            raise ValueError(
                f"pad_clearance must not be negative, got {self.pad_clearance}"
            )
        if self.pad_radius > self.slope_platform_half:
            raise ValueError(
                f"pad_radius {self.pad_radius} overhangs the "
                f"{self.slope_platform_half} m slope plateau"
            )
        if self.slope_platform_half > self.tile_size / 2 - self.cell_size + _EPS:
            raise ValueError(
                f"slope_platform_half {self.slope_platform_half} leaves the "
                f"{self.tile_size} m tile less than one cell_size of ramp"
            )
        if self.slope_angle.base < 0:
            raise ValueError(
                f"slope_angle base must not be negative, got {self.slope_angle.base}"
            )
        # Jitter moves interior rows only, so without explicit difficulties
        # the top row sits at exactly 1.
        steepest = max(self.difficulties) if self.difficulties is not None else 1.0
        steepest *= max(self.cap("pyramid_slope"), self.cap("inverted_pyramid_slope"))
        if not self.slope_angle(steepest) < math.pi / 2:
            raise ValueError(
                f"slope_angle reaches {self.slope_angle(steepest):.3f} rad at "
                f"difficulty {steepest:g}. A slope must stay under pi/2."
            )
        lo, hi = stair_tread_bounds(self)
        if not 0 < lo <= hi:
            raise ValueError(
                f"stair_tread {self.stair_tread} must be positive, and a range "
                "must be ordered as (lo, hi) with lo <= hi"
            )
        # The pit is carved out to the last node at least one cell inside the
        # flight's edge, so that node lies under two cells in. A tread of two
        # cells keeps it under the outermost ring, and the pit wall stays one
        # riser. It also gives every ring band a node line to cover.
        if lo < 2 * self.cell_size - _EPS:
            raise ValueError(
                f"stair_tread {self.stair_tread} must be at least two cell_size "
                f"({2 * self.cell_size:g} m). The pit's last carved node must lie "
                "under the outermost ring, so its wall stays one riser. Every "
                "ring band must span a node line, so the lookup sees its tread."
            )
        if not 2 <= self.stair_min_steps <= self.stair_max_steps:
            raise ValueError(
                f"stair steps [{self.stair_min_steps}, {self.stair_max_steps}] "
                "must be an ordered range of at least two risers"
            )

    def cap(self, terrain_type: str) -> float:
        """The difficulty multiplier for `terrain_type` (1 when uncapped)."""
        return dict(self.type_caps).get(terrain_type, 1.0)


def _ramp(value) -> Ramp:
    if isinstance(value, Ramp):
        return value
    if hasattr(value, "keys"):
        return Ramp(value["base"], value["gain"])
    base, gain = value
    return Ramp(base, gain)


def _pair(value) -> tuple[float, float]:
    lo, hi = value
    return float(lo), float(hi)


def node_count(length: float, cell_size: float, minimum: int = 1) -> int:
    """Cells of `cell_size` in `length`, which must be a whole number of
    them and at least `minimum`."""
    n = round(length / cell_size)
    if abs(n * cell_size - length) > 1e-6 * max(1.0, length):
        raise ValueError(f"{length} m is not a whole number of {cell_size} m cells")
    if n < minimum:
        raise ValueError(
            f"{length} m spans {n} cells of {cell_size} m, under the minimum "
            f"of {minimum}"
        )
    return n


def params_to_dict(params: ArenaParams) -> dict:
    """JSON-ready view of `params`. `params_from_dict` inverts it exactly."""
    out = {}
    for f in fields(params):
        value = getattr(params, f.name)
        if isinstance(value, Ramp):
            value = {"base": value.base, "gain": value.gain}
        elif f.name == "type_caps":
            value = dict(value)
        elif isinstance(value, tuple):
            value = list(value)
        out[f.name] = value
    return out


def params_from_dict(data: dict) -> ArenaParams:
    """`ArenaParams` from `params_to_dict` output or a config block.

    An unknown key raises. Params written by a different generator then
    fail loudly instead of building another arena."""
    return ArenaParams(**data)


def stair_tread_bounds(params: ArenaParams) -> tuple[float, float]:
    """(lo, hi) of the stair tread. lo equals hi for a fixed tread."""
    tread = params.stair_tread
    return tread if isinstance(tread, tuple) else (tread, tread)


def summit_platform_half(params: ArenaParams) -> float:
    """Half-width of the stair summit: the spawn pad's radius, never below
    `summit_floor`. A shrunken pad still leaves a standable summit."""
    return max(params.pad_radius, params.summit_floor)


def stair_steps(tread: float, params: ArenaParams) -> int:
    """Risers of a flight on `tread`.

    It takes as many as fit between the summit and the rim, clamped to
    [stair_min_steps, stair_max_steps]. A flight of n risers lays n - 1
    treads. Raises ValueError when stair_min_steps risers reach past the
    rim."""
    platform = summit_platform_half(params)
    rim = params.tile_size / 2 - params.rim_margin
    n_treads = math.floor((rim - platform) / tread + _EPS)
    n_steps = min(max(n_treads + 1, params.stair_min_steps), params.stair_max_steps)
    edge = stair_flight_half(tread, n_steps, params)
    if edge > rim + _EPS:
        raise ValueError(
            f"stair flight breaches the tile rim: a {platform} m summit plus "
            f"{n_steps - 1} treads of {tread} m reaches {edge:.3f} m of the "
            f"{rim:.3f} m available. Use a smaller tread or pad."
        )
    return n_steps


def rubble_cells(params: ArenaParams) -> int:
    """Rubble cells along each axis of a tile: as many grid_pitch cells as
    fit edge_margin inside the tile edge."""
    room = params.tile_size - 2 * params.edge_margin
    return math.floor(room / params.grid_pitch + _EPS)


def rubble_shrink(hx: float, hy: float, params: ArenaParams) -> float:
    """Scale for a rubble box of half-sizes (hx, hy). A box whose corner
    radius passes half a grid_pitch shrinks to 0.98 of it, so it stays
    inside its cell at any yaw with room left to jitter. A smaller box keeps
    its size."""
    corner = math.hypot(hx, hy)
    half_cell = params.grid_pitch / 2
    return 0.98 * half_cell / corner if corner > half_cell else 1.0


def stair_flight_half(tread: float, n_steps: int, params: ArenaParams) -> float:
    """Outer Chebyshev half-size of a stair flight. The treads are concentric
    squares. The inverted pit is carved to this square."""
    return summit_platform_half(params) + (n_steps - 1) * tread


def slope_plateau_height(d: float, params: ArenaParams) -> float:
    """Height of a pyramid slope's plateau above its tile edge (z = 0)."""
    run = params.tile_size / 2 - params.slope_platform_half
    return math.tan(params.slope_angle(d)) * run
