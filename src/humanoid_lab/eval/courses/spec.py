"""Course dataclasses, the pinned robot inputs and the frozen protocol constants.

Every per-robot number in the benchmark comes from six pinned inputs
(`ROBOT_INPUTS`) through one frozen set of fractions (`derive`). The protocol
constants and those fractions live below. The follower's constants live in
follower.py. The lead-in lives in geometry.py, and the shape sizes live in
the family modules. The shape sizes reach the fingerprint through the rows'
waypoints. `families.frozen_constants` lists every other named constant a
score depends on. `families.catalogue_fingerprint` hashes both, and
tests/unit/test_courses.py pins the hash per robot. SCHEMA_VERSION,
GROUND_CLASSES, the axis names, the kind ids and MIN_PERFECT_TRACKING are not
score inputs and are not hashed.

numpy plus envs/progress (for its two commanded-speed constants). Nothing
here builds a model or touches a jax backend.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np

from humanoid_lab.envs import progress

SCHEMA_VERSION = 1

# The catalogue is partitioned by ground class. Each class has its own rows,
# fingerprint, lane shapes and output file, so a class added later cannot move
# the numbers or the compiled program of an existing one.
GROUND_CLASSES = {"flat": "courses.json"}

# -- protocol ----------------------------------------------------------------

# Budget: TIME_FACTOR x the ideal time plus SLACK_SEC, never above
# MAX_COURSE_SEC. The ceiling sits above every row on both robots (the
# largest, Asimov straight_slow, is 158.25 s) and only guards rows added later.
TIME_FACTOR = 2.5
SLACK_SEC = 2.0
MAX_COURSE_SEC = 160.0

# A record shorter than this scores 0 with no sub-scores: the RMS and FFT
# metrics need about a second of data to mean anything.
MIN_SCORE_SEC = 1.0
# The speed axis is scored over steps where motion was commanded
# (cmd vx > progress.SPEED_DEADBAND), and only when they add up to this much.
MIN_MOVING_SEC = 0.2

# Sub-scores are ratios, so a near-zero error sends one toward infinity and
# out of JSON. Anything this good is indistinguishable from perfect.
SUBSCORE_CAP = 1000.0

# battery.vibration_index's default cutoff, passed explicitly so a change to
# that default cannot move a course score. A gait faster than 1.67 Hz puts its
# third harmonic above it, where it counts as vibration.
VIBRATION_CUTOFF_HZ = 5.0

# Floor rows: sliding friction set on the floor and on every foot geom.
# Every Roboto preset trains with dr.foot_friction, which multiplies the
# feet's 0.9 by [0.333, 1.778], so its contacts train on 0.30-1.60. 0.25 sits
# just below that floor. Asimov trains without foot friction DR at 1.0.
SLIPPERY_MU = 0.25

# Disturbance rows: one planar kick of the robot's training push.vel, to the
# left of the current heading, once the follower has made this much progress.
PUSH_AT_M = 5.0

SEEDS = 8

# -- derivation fractions ----------------------------------------------------

# Speeds and rates are fractions of each robot's command box, below 1.0 with
# headroom, as eval/battery.py's scenarios are: a row pinned to the corner of
# the box would confound "cannot do this" with "was never trained this close
# to the edge".
V_NOM_FRAC = 0.5
V_SLOW_FRAC = 0.2
V_FAST_FRAC = 0.9
# One block per 2.5 m. Two up-steps and a down-step, none below v_nom: the
# slow gait is isolated in straight_slow, and a slow first block would repeat
# that row's failure here.
SPEED_STEP_FRACS = (0.5, 0.9, 0.5, 0.7)
# The follower's yaw clip, of wz_max (eval/battery.py's spin probes use the
# same fraction).
YAW_CAP_FRAC = 0.8
# Spin rows: nominal, slow and fast, of the yaw cap.
SPIN_FRACS = (0.8, 0.4, 1.2)
# A spin command at least twice the stand threshold. Commanded speed is
# norm(vx, vy) + YAW_SPEED_WEIGHT |wz|, and below SPEED_DEADBAND the gait
# clock freezes. 0.333 rad/s; it binds on Asimov's spin_slow.
SPIN_MIN_RAD_S = 2 * progress.SPEED_DEADBAND / progress.YAW_SPEED_WEIGHT
# One radius for every tight turn (circle_tight, the u-turn, the square's
# corners): the one that asks for R_TIGHT_DEMAND of the yaw cap at v_nom,
# floored at R_TIGHT_MIN_M. The floor binds on Roboto (0.53 of its cap) and
# keeps the follower's own cross-track error on every tight row under 1.3 cm
# on both robots.
R_TIGHT_DEMAND = 2 / 3
R_TIGHT_MIN_M = 0.75
# The slalom's wavelength puts its peak yaw demand at v_nom at this fraction
# of the cap: a sine's peak curvature is A (2 pi / lambda)^2.
SLALOM_AMPLITUDE_M = 0.5
SLALOM_DEMAND = 0.5
SLALOM_WAVELENGTHS = 3

# Test threshold only: on every path row a unicycle that executes the
# follower's own commands exactly must score at least this on tracking, on
# every robot. Pure pursuit cuts curves to the inside and a robot whose yaw
# lags pushes back out, so where the follower's own error is the size of a
# policy's, a lagging robot outscores a perfect one. With the follower's
# error at least 4x below the normalizer, the rows measure the robot.
MIN_PERFECT_TRACKING = 4.0

# Named sub-score axes per course kind. `unscored` may only name these.
PATH_AXES = ("tracking", "speed", "height", "grip", "smoothness")
SPIN_AXES = ("rotation", "drift", "height", "smoothness")

# Course kinds. The lane carries the id as data, so path and spin rows share
# one compiled program.
PATH, SPIN = 0, 1

# -- robot inputs ------------------------------------------------------------


@dataclass(frozen=True)
class RobotInputs:
    """The six per-robot numbers the catalogue is derived from.

    `vx_max` is the overlay's `task.env.command.vx[1]`, `wz_max` its
    `min(-wz[0], wz[1])`. `stance_halfwidth_m` is half the lateral distance
    between the two foot sites at the home keyframe, `nominal_height_m` the
    base z there. `push_vel` is the planar kick the robot trains against.
    `obs_noise` (gyro, joint_pos, joint_vel) is the overlay's
    `task.env.obs_noise` over configs/task/joystick.yaml's: every run of a
    robot is measured under it, whatever noise the run trained with.
    """

    vx_max: float
    wz_max: float
    stance_halfwidth_m: float
    nominal_height_m: float
    push_vel: float
    obs_noise: Mapping[str, float]


# Overlay values, not a run's resolved ones, so every run of one robot shares
# one catalogue and one sensor model. tests/unit/test_courses.py checks the
# box, push and noise against the configs; tests/integration checks the two
# keyframe numbers against the models (within 0.5 mm).
ROBOT_INPUTS: dict[str, RobotInputs] = {
    "roboto_origin": RobotInputs(
        vx_max=1.0,
        wz_max=1.57,
        stance_halfwidth_m=0.0725,
        nominal_height_m=0.750,
        push_vel=0.5,
        obs_noise=MappingProxyType({"gyro": 0.01, "joint_pos": 0.03, "joint_vel": 1.75}),
    ),
    # nominal_height_m is the home keyframe's 0.636 m, at which the feet just
    # touch the floor; the overlay says the robot stands at about 0.72-0.75 m.
    # Height is a diagnostic axis and cannot bind above the fall line, so no
    # score moves on it.
    "asimov_v1": RobotInputs(
        vx_max=0.8,
        wz_max=0.6,
        stance_halfwidth_m=0.1075,
        nominal_height_m=0.636,
        push_vel=0.4,
        obs_noise=MappingProxyType({"gyro": 0.01, "joint_pos": 0.01, "joint_vel": 0.1}),
    ),
}


@dataclass(frozen=True)
class CourseParams:
    """Everything the families and the scoring read for one robot.

    `source` is "pinned" for a robot in ROBOT_INPUTS and "measured" when the
    runner had to read the inputs from the model and the run's config.
    """

    robot: str
    source: str
    stance_halfwidth_m: float
    nominal_height_m: float
    v_nom: float
    v_slow: float
    v_fast: float
    speed_steps: tuple[float, float, float, float]
    yaw_cap: float
    spin_nom: float
    spin_slow: float
    spin_fast: float
    r_tight: float
    slalom_amplitude: float
    slalom_wavelength: float
    push_vel: float
    obs_noise: Mapping[str, float]


def derive(robot: str, inputs: RobotInputs, source: str = "pinned") -> CourseParams:
    """The frozen fractions applied to one robot's inputs."""
    vx_max, wz_max = float(inputs.vx_max), float(inputs.wz_max)
    yaw_cap = YAW_CAP_FRAC * wz_max
    v_nom = V_NOM_FRAC * vx_max
    spin_nom, spin_slow, spin_fast = (max(f * yaw_cap, SPIN_MIN_RAD_S) for f in SPIN_FRACS)
    # Peak curvature of the sine A sin(2 pi x / lambda) is A (2 pi / lambda)^2;
    # at v_nom it asks for SLALOM_DEMAND of the cap.
    peak_curvature = SLALOM_DEMAND * yaw_cap / v_nom
    wavelength = 2 * math.pi * math.sqrt(SLALOM_AMPLITUDE_M / peak_curvature)
    return CourseParams(
        robot=robot,
        source=source,
        stance_halfwidth_m=float(inputs.stance_halfwidth_m),
        nominal_height_m=float(inputs.nominal_height_m),
        v_nom=v_nom,
        v_slow=V_SLOW_FRAC * vx_max,
        v_fast=V_FAST_FRAC * vx_max,
        speed_steps=tuple(f * vx_max for f in SPEED_STEP_FRACS),
        yaw_cap=yaw_cap,
        spin_nom=spin_nom,
        spin_slow=spin_slow,
        spin_fast=spin_fast,
        r_tight=max(v_nom / (R_TIGHT_DEMAND * yaw_cap), R_TIGHT_MIN_M),
        slalom_amplitude=SLALOM_AMPLITUDE_M,
        slalom_wavelength=wavelength,
        push_vel=float(inputs.push_vel),
        obs_noise=MappingProxyType({k: float(v) for k, v in inputs.obs_noise.items()}),
    )


def params_for(robot: str) -> CourseParams | None:
    """The pinned params of `robot`, or None for a robot with no entry."""
    inputs = ROBOT_INPUTS.get(robot)
    return None if inputs is None else derive(robot, inputs)


# -- courses -----------------------------------------------------------------


def _check_placement(name: str, ground, anchor: str, origin) -> None:
    # Flat rows are laid out from the post-settle pose. `ground` (an arena
    # identity), anchor="world" and `origin` (x, y, yaw) place a row on a
    # terrain arena; no ground class uses them yet, so they are refused
    # rather than half-supported.
    if ground is not None:
        raise ValueError(f"{name}: ground {ground!r}: only the flat floor (None) exists")
    if anchor != "start":
        raise ValueError(f"{name}: anchor {anchor!r}: only 'start' exists")
    if origin is not None:
        raise ValueError(f"{name}: origin {origin!r} needs anchor='world'")


def _check_unscored(name: str, unscored, axes: tuple[str, ...]) -> frozenset[str]:
    unscored = frozenset(unscored)
    unknown = sorted(unscored - set(axes))
    if unknown:
        raise ValueError(f"{name}: unscored {unknown} are not axes of this kind {list(axes)}")
    if unscored >= set(axes):
        raise ValueError(f"{name}: every axis is unscored, so nothing is left to score")
    return unscored


@dataclass(frozen=True, eq=False)
class PathCourse:
    """One path-following row.

    `waypoints` are (K, 2) in the course frame: origin at the post-settle base
    position, +x along its heading. `speeds` is one commanded speed for every
    segment, or one per segment (K - 1). `friction` is the sliding friction
    set on the floor and every foot (None keeps the model's own). `push_at_m`
    places one kick of `push_vel` m/s, to the left of the heading, at that
    much follower progress. `baseline` names the row this one differs from in
    `isolates` alone. `geometry` holds the shape parameters for the JSON.
    `unscored` names axes left out of the score's min.
    """

    name: str
    family: str
    isolates: str
    waypoints: np.ndarray
    speeds: tuple[float, ...]
    baseline: str | None = None
    friction: float | None = None
    push_at_m: float | None = None
    push_vel: float | None = None
    geometry: Mapping[str, float] = field(default_factory=lambda: MappingProxyType({}))
    ground: Hashable | None = None
    anchor: str = "start"
    origin: tuple[float, float, float] | None = None
    unscored: frozenset[str] = frozenset()

    def __post_init__(self):
        wp = np.array(self.waypoints, dtype=float)
        if wp.ndim != 2 or wp.shape[1] != 2 or len(wp) < 2:
            raise ValueError(f"{self.name}: waypoints must be (K >= 2, 2), got {wp.shape}")
        wp.flags.writeable = False
        object.__setattr__(self, "waypoints", wp)
        object.__setattr__(self, "speeds", tuple(float(v) for v in self.speeds))
        object.__setattr__(self, "geometry", MappingProxyType(dict(self.geometry)))
        if np.any(self.segment_speeds <= 0.0):
            raise ValueError(f"{self.name}: every commanded speed must be positive")
        if (self.push_at_m is None) != (self.push_vel is None):
            raise ValueError(f"{self.name}: push_at_m and push_vel come together")
        _check_placement(self.name, self.ground, self.anchor, self.origin)
        object.__setattr__(self, "unscored", _check_unscored(self.name, self.unscored, PATH_AXES))

    @property
    def segment_speeds(self) -> np.ndarray:
        """One commanded speed per segment, broadcast from a single value."""
        n = len(self.waypoints) - 1
        if len(self.speeds) == 1:
            return np.full(n, self.speeds[0], dtype=float)
        if len(self.speeds) != n:
            raise ValueError(f"{self.name}: {len(self.speeds)} speeds for {n} segments")
        return np.asarray(self.speeds, dtype=float)

    @property
    def segment_lengths(self) -> np.ndarray:
        return np.linalg.norm(np.diff(self.waypoints, axis=0), axis=1)

    @property
    def length_m(self) -> float:
        return float(self.segment_lengths.sum())

    @property
    def ideal_sec(self) -> float:
        """The time the course takes at exactly its commanded speeds."""
        return float((self.segment_lengths / self.segment_speeds).sum())


@dataclass(frozen=True, eq=False)
class SpinCourse:
    """One rotate-in-place row: a held [0, 0, wz] command.

    Completion is `turns` full rotations of world yaw in the sign of `wz`
    (positive is left, CCW). There is no path for the follower.
    """

    name: str
    family: str
    isolates: str
    wz: float
    turns: float = 1.0
    baseline: str | None = None
    friction: float | None = None
    ground: Hashable | None = None
    anchor: str = "start"
    origin: tuple[float, float, float] | None = None
    unscored: frozenset[str] = frozenset()

    def __post_init__(self):
        if not self.wz:
            raise ValueError(f"{self.name}: a spin needs a nonzero wz")
        if not self.turns > 0:
            raise ValueError(f"{self.name}: turns must be positive")
        _check_placement(self.name, self.ground, self.anchor, self.origin)
        object.__setattr__(self, "unscored", _check_unscored(self.name, self.unscored, SPIN_AXES))

    @property
    def turn_rad(self) -> float:
        return 2 * math.pi * float(self.turns)

    @property
    def ideal_sec(self) -> float:
        return self.turn_rad / abs(float(self.wz))


Course = PathCourse | SpinCourse


def kind(course: Course) -> str:
    """"path" or "spin", the name the JSON uses."""
    return "spin" if isinstance(course, SpinCourse) else "path"


def kind_id(course: Course) -> int:
    """PATH or SPIN, the id the lane carries."""
    return SPIN if isinstance(course, SpinCourse) else PATH


def ground_class(course: Course) -> str:
    """The catalogue partition a course belongs to."""
    return "flat" if course.ground is None else "terrain"


def budget_sec(course: Course) -> float:
    return min(MAX_COURSE_SEC, TIME_FACTOR * course.ideal_sec + SLACK_SEC)


def budget_steps(course: Course, dt: float) -> int:
    return round(budget_sec(course) / dt)


def seconds_to_steps(seconds: float, dt: float) -> int:
    """A protocol duration in control steps at `dt`, at least one."""
    return max(1, round(seconds / dt))
