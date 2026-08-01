"""Joystick velocity-tracking task, robot-agnostic (biped).

The gait clock is a fixed antiphase (offsets [0, pi]) two-foot clock, there
is no commanded stand height (asimov has one static standing height, not an
adjustable crouch), and the quadruped-only reward terms (contact_match,
high_step, height_tracking, stand_feet_down) do not exist here -- see
rewards/terms.py's module docstring and PLAN.md step 6.

Actor observations use only signals the real robot has (IMU + joint
encoders + own history + the commanded velocity + the env's own gait
clock); everything else (world-frame linvel, base height, contacts,
actuator force) is privileged (critic-only).

Deferred (not built yet):
- No action_delay/latency/encoder-offset/action-filter machinery. Actions
  apply on the control step they are computed, with no delay stage in
  between. PLAN.md's v1 sysid row (action delay 0-1 control steps) means
  this machinery has to exist before sim2real-fidelity training.
"""

from __future__ import annotations

import jax
import jax.numpy as jp
import numpy as np
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env

from humanoid_lab.envs import progress
from humanoid_lab.envs.base import HumanoidEnv
from humanoid_lab.rewards import terms

# Per-foot gait clock offset, robot_spec.foot_sites order: antiphase (one
# foot swings while the other stances), the only sensible clock for a biped:
# a quadruped's walk/trot blend has no two-foot analogue.
_PHASE_OFFSETS = (0.0, np.pi)


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        # asimov docs (PLAN.md "Research findings"): 200 Hz physics, 50 Hz
        # policy (decimation 4).
        ctrl_dt=0.02,
        sim_dt=0.005,
        sim=config_dict.create(
            # Physics backend. auto picks warp on a CUDA host and jax
            # elsewhere.
            #
            # naconmax_per_env and njmax are the warp contact/constraint
            # buffers; only warp reads them (the jax branch of
            # envs/backend.py takes no kwargs). None defers to the robot's
            # own measured `sim_budget` block in robot.yaml, and a warp run
            # with neither refuses at construction -- overflow is silent in
            # both directions: contacts past naconmax are dropped and rows
            # past njmax apply no force, with no warning anywhere.
            # ./run.sh check-contacts measures a robot's peaks and
            # recommends budgets; tests/integration/test_check_contacts.py
            # fails when new collision geometry outgrows a robot's recorded
            # numbers. Warp allocates naconmax_per_env * num_envs contacts
            # at make_data time, so the budget is a device-memory line item
            # a big-batch job trades against num_envs (run.json's
            # `contacts` block records the peaks).
            backend="auto",
            naconmax_per_env=None,
            njmax=None,
            num_envs=1,
        ),
        episode_length=1000,
        # robot.yaml keyframe used for reset qpos and the PD default_pose
        # anchor (see envs/base.py). Validated to exist in the model at
        # construct time.
        reset_keyframe="home",
        # Reset pose noise: uniform +-reset_noise (rad) added
        # to every actuated joint's qpos at reset, so the policy never sees
        # the exact same start pose twice. 0.0 disables it.
        reset_noise=0.05,
        # Anchor `pose`, `stand_still` and the reset pose on the pose the
        # robot actually settles into, instead of on the keyframe's COMMANDED
        # joint values. Off = the legacy keyframe anchor, bit-exact.
        #
        # Both terms measure a deviation from `_default_pose`, which is what
        # the keyframe TELLS the joints to do. Under gravity and finite gains
        # the robot comes to rest below that command, so the deviation never
        # reaches zero and both terms charge a floor no policy can remove.
        # Measured on the quadruped predecessor: 0.343 rad of summed sag,
        # about 97% of its whole standing residual. roboto_origin's home
        # keyframe settles 0.065 rad off its command (0.015 rad at the knee),
        # which at the stock scales.stand_still of -0.5 is 0.032 of standing
        # penalty per step that exists only because the anchor is wrong.
        # Each actuator preset sags differently, so the floor also moves with
        # a config axis that has nothing to do with the task.
        #
        # On, the env settles a QUASI-RIGID copy of the model once at
        # construction (see HumanoidEnv._settle_pose) and anchors on the
        # result: a function of the geometry and the soft-limit envelope,
        # identical for every gain set and every actuator model at a given
        # soft_limit_factor. Compensating the real plant's sag to reach that
        # pose is the policy's job.
        #
        # The settle must end standing and at rest or construction raises
        # (HumanoidEnv._check_settled). asimov_v1's keyframes do not: held
        # rigid, `home` topples backward within a second and `knees_bent`
        # goes over by four. Only turn this on for a robot whose reset
        # keyframe is a standing equilibrium.
        #
        # The ctrl anchor does NOT move: `_default_pose` stays what a zero
        # action commands and what the joint_pos observation subtracts.
        real_pose_ref=False,
        # asimov docs (PLAN.md): gyro +-0.01, joint_pos +-0.01 rad,
        # joint_vel +-0.1 rad/s. No gravity noise figure was published;
        # unlisted components default to noise-free (see base.py's
        # _build_obs).
        obs_noise=config_dict.create(gyro=0.01, joint_pos=0.01, joint_vel=0.1),
        # Declarative observation spec: ordered lists of catalog names (see
        # HumanoidEnv._obs_catalog + this env's command/phase additions).
        # `state` is the actor (only signals a real robot has -- no
        # world-frame linvel), `privileged` the critic.
        obs=config_dict.create(
            state=(
                "gyro",
                "gravity",
                "joint_pos",
                "joint_vel",
                "last_action",
                "command",
                "phase",
            ),
            privileged=(
                "gyro",
                "gravity",
                "joint_pos",
                "joint_vel",
                "last_action",
                "command",
                "phase",
                "linvel",
                "height",
                "contacts",
                "actuator_force",
            ),
        ),
        # asimov docs (PLAN.md): command envelope x +-0.8 m/s, y +-0.6 m/s,
        # yaw +-0.6 rad/s.
        command=config_dict.create(
            vx=(-0.8, 0.8),
            vy=(-0.6, 0.6),
            wz=(-0.6, 0.6),
            resample_steps=250,  # 5 s at ctrl_dt=0.02
            # Velocity zeroes with this prob (stand training). An untuned
            # starting value for a biped.
            zero_prob=0.15,
            # Pure command draws (see _sample_command). Each redraws the
            # base uniform sample into a CLEAN single-axis command with the
            # given probability, all off by default.
            #
            # Why they exist at all: a uniform box over (vx, vy, wz) almost
            # never draws a clean corner -- a backward command arrives with
            # random lateral and yaw contamination attached. Under
            # tracking_product/tracking_relative a contaminated corner pays
            # ~0 no matter how well the robot serves it, so the skill is
            # never profitable to learn and the policy settles on refusing
            # it. A quadruped policy trained this way held 0.000 m/s under a
            # commanded -0.4 backward, and five isolating probes confirmed
            # the refusal was learned, not mechanical.
            #
            # Every range below is a STARTING VALUE to re-derive for asimov,
            # derived here from our own envelope (vx +-0.8, vy +-0.6,
            # wz +-0.6).
            #
            # Keep wz, zero the linear part: spin-in-place training.
            pure_wz_prob=0.0,
            # Keep vy, zero vx and wz: pure-strafe training.
            pure_vy_prob=0.0,
            # Redraw vx from slow_vx, zero vy and wz: clean slow straight
            # walking, so the gait learns to scale down instead of having
            # one speed. slow_vx sits inside our vx range.
            pure_slow_prob=0.0,
            slow_vx=(0.1, 0.35),
            # Redraw vx from fast_vx, zero vy and wz: clean fast straight
            # walking. Ours tops out at 0.8, the top of our commanded vx
            # box. Setting fast_vx ABOVE the commanded box is a known way to
            # pull a policy past a speed it deadlocks at; our
            # envelope is capped pending sysid, so commanding past it is a
            # decision for later, not a default.
            pure_fast_prob=0.0,
            fast_vx=(0.5, 0.8),
            # Redraw vx from back_vx, zero vy and wz: clean backward
            # walking, the refusal described above. Sits inside
            # asimov_v1's negative vx range -- but NOT inside
            # roboto_origin's, whose overlay narrows vx to [-0.6, 1.0].
            # Arming pure_back_prob there without narrowing back_vx is
            # refused by check_pure_draw_ranges below.
            pure_back_prob=0.0,
            back_vx=(-0.8, -0.2),
        ),
        # Untuned starting values for asimov's mass and leg length.
        push=config_dict.create(enable=True, interval_steps=200, vel=0.4),
        # No-progress termination, CaT-style (arXiv 2403.18765): an env whose
        # measured progress keeps falling short of its command is cut
        # probabilistically. Forfeiting the rest of the episode is the whole
        # penalty -- no reward term is attached and rewards["termination"]
        # stays fall-only. The hazard ramps linearly from 0 where the smoothed
        # progress reaches risk_below of the commanded speed to p_max per
        # control step at zero progress, so expected survival at a dead stop
        # is 1/p_max steps (1 s at the defaults and ctrl_dt=0.02). Zero
        # commands never arm the cut, and a command change re-arms it only
        # after grace_sec. The math is in envs/progress.py.
        #
        # Every number here is an untuned starting value from a 0.21 m
        # four-bar quadruped leg. grace_sec and risk_below are the two to
        # re-derive for a biped: turning a two-legged gait around takes
        # longer than turning a four-legged one, so 2 s of grace may be short
        # and 50% of demand may be a lot to ask inside it.
        no_progress=config_dict.create(
            enable=False,
            grace_sec=2.0,   # no hazard this long after a reset or a resample
            ema_sec=1.0,     # smoothing horizon of the progress measure
            risk_below=0.5,  # hazard starts below this fraction of demand
            p_max=0.02,      # per-step hazard at zero progress
        ),
        # asimov stands ~0.72-0.75 m. robot.yaml's home keyframe base_pos z
        # (0.636 m) is not this: it's the floating-base placeholder measured
        # so the feet just touch the floor, not the robot's standing height.
        # max_tilt_gz is the gravity_body z component (near -1 upright, rising
        # toward 0 and positive as the robot tips over); -0.4 is an untuned
        # starting value.
        fall=config_dict.create(min_height=0.45, max_tilt_gz=-0.4),
        # Two-foot antiphase gait clock (see _PHASE_OFFSETS). freq/
        # swing_height/duty are untuned starting values: only the SHAPE
        # (speed-scaled clock frequency, sinusoidal swing profile, fixed
        # stance duty) carries over from a 0.21 m four-bar quadruped leg. The
        # numbers have to be re-derived for a human-scale biped leg.
        gait=config_dict.create(
            freq=(1.0, 2.0),  # Hz, speed-scaled between these two bounds
            swing_height=0.08,  # target peak foot clearance, m
            duty=0.5,  # fixed stance fraction (biped: no walk/trot blend)
            air_time_cap=0.0,  # 0 = uncapped
        ),
        reward=config_dict.create(
            tracking_sigma=0.25,
            # Multiplicative velocity tracking (off = legacy additive).
            # Additive tracking pays the easy half of a command: a robot
            # that deadlocks under a pure-spin command still earns the FULL
            # tracking_lin_vel, because its commanded linear velocity is 0
            # and standing still "tracks" it perfectly. That payout has been
            # measured at about 63% of an ideal spin's. With tracking_product
            # both terms are gated by the product of the two kernels: full
            # pay only when the WHOLE command is tracked, and a deadlocked
            # spin earns ~0.
            tracking_product=False,
            # Command-relative tracking error (off = legacy absolute). The
            # absolute kernel pays only within ~sqrt(tracking_sigma) of the
            # target regardless of the target's size, so a fast command puts
            # the whole reward cliff out of exploration's reach: one measured
            # policy reached 0.70 m/s under a 0.8 command and 0.00 m/s under
            # a 1.0 one. Relative mode divides the squared error by the
            # squared commanded magnitude, so 80% of target pays the same at
            # every speed and tracking_rel_sigma is dimensionless.
            tracking_relative=False,
            tracking_rel_sigma=0.25,
            # Floors on the relative denominator, so a near-zero command
            # divides by the floor and never by zero. Both are quadruped
            # starting points, and a narrow kernel rounds partial tracking to
            # zero: on terrain, tracking_rel_sigma had to widen to 0.5 and
            # tracking_rel_floor_ang to 0.7.
            # Re-derive for asimov rather than trusting these.
            tracking_rel_floor_lin=0.3,  # m/s
            tracking_rel_floor_ang=0.4,  # rad/s
            # Far-field mix-in for both tracking kernels (weight 0 = off,
            # exact legacy kernel): (1-w)*kernel + w*exp(-err^2/far_sigma),
            # applied in the absolute and the relative branch alike, and the
            # far kernel stays absolute in both. exp(-err^2/sigma) is
            # gradient-free once the error is a few sigma out, so a
            # capability the policy never explored gets no pull toward the
            # command at all; the wider second exponential keeps a usable
            # gradient at range while the optimum and the [0, 1] bound stay
            # unchanged.
            #
            # This term ALONE creates a stable standing deadlock: at a yaw
            # rate error of 0.8 rad/s it pays 0.25*exp(-0.64/2.5), about 19%
            # of the maximum angular reward, for standing still, and that
            # gradient is weaker than the penalties a pivot attempt incurs.
            # Only turn it on together with tracking_product or
            # tracking_relative.
            tracking_far_weight=0.0,
            tracking_far_sigma=2.5,
            # Gate the positive gait-shaping terms by the linear tracking
            # kernel, post-product when tracking_product is on. Those terms
            # pay on a commanded env whether or not it translates, which has
            # made stand-and-lift the top income under a command on a
            # quadruped run: standing with one leg raised earned ~1.8 reward
            # per step against honest walking's ~0.25. Gated, a stride pays
            # in proportion to how well the command is being served and
            # standing pays nothing for lifting legs. Gated set:
            # feet_air_time and feet_apex. Not feet_phase, and not the
            # feet_landing penalty -- see _compute_rewards.
            shaping_tracking_gate=False,
            phase_sigma=0.002,
            # Tolerance cone around upright for the orientation penalty,
            # half-angle in degrees (0 = the legacy penalty, bit-exact). The
            # penalty is sin^2 of the base's tilt from vertical; with a cone
            # it becomes max(sin^2(tilt) - sin^2(tol), 0) -- free inside,
            # rising continuously from the edge. Tilt here is measured
            # against gravity, not against the local surface, so a
            # flat-referenced penalty taxes the body pitch that locomotion
            # needs (leaning into an acceleration) while a real nosedive
            # stays far outside any cone worth setting. 20 degrees is a
            # workable cone; 10 was measured too tight for that reason.
            orientation_tol_deg=0.0,
            # Per-swing apex target for feet_apex, m. The duration-averaged
            # clearance terms tolerate a 1.5-2 cm skim that collects nearly
            # as much as a crisp arc, so the optimizer skims; paying the
            # swing's PEAK instead measured 3-5 cm swings and 30-70%
            # better grip. Height band of the feet_landing penalty, m:
            # downward foot speed is priced in proportion to how far INSIDE
            # this band the foot is (1 at the floor, 0 at glide_height and
            # above), so the gradient reads "decelerate as you approach". It
            # is measured BEFORE contact because a penalty read AT contact
            # under-reads impacts: the solver has already absorbed the hit
            # within the control step it becomes visible. The physical
            # touchdown reference is free fall over the band,
            # sqrt(2*9.81*0.03) ~ 0.77 m/s.
            #
            # Both numbers are untuned starting values for a 0.21 m four-bar
            # quadruped leg.
            # RE-DERIVE for asimov's leg: our own gait.swing_height asks for
            # 0.08 m of swing, so a 0.05 m apex target is not this robot's.
            # When re-deriving, read docs/lessons/foot-clearance.md first:
            # _foot_clearance is referenced to the reset keyframe, which
            # floats the feet, so both numbers sit about 5 mm (asimov) or
            # 3 mm (roboto) below the physical height they name.
            apex_target=0.05,
            glide_height=0.03,
            # torque_limit hinge fires above this fraction of each
            # actuator's forcerange cap.
            torque_limit_frac=0.85,
            # UNTUNED starting values, carried from a quadruped joystick task
            # for every term that has a biped analogue
            # (rewards/terms.py's module docstring lists what has none).
            # Do not read these as tuned for asimov: they are the
            # PLAN.md-mandated starting point for the first CPU smoke run,
            # nothing more.
            scales=config_dict.create(
                tracking_lin_vel=1.5,
                tracking_ang_vel=0.8,
                lin_vel_z=-2.0,
                ang_vel_xy=-0.05,
                orientation=-5.0,
                torques=-2e-4,
                torque_rate=0.0,
                action_rate=-0.25,
                action_accel=-0.1,
                energy=-2e-3,
                pose=-0.5,
                feet_air_time=2.0,
                feet_slip=-0.25,
                feet_phase=1.0,
                stand_still=-0.5,
                termination=-1.0,
                torque_limit=0.0,
                # Per-swing apex shaping (see apex_target above), 0 = off.
                feet_apex=0.0,
                # Soft-landing penalty (see glide_height above), 0 = off. A
                # policy trained with it glides its feet into stance instead
                # of striking the floor at swing free-fall speed.
                feet_landing=0.0,
            ),
        ),
    )


# (probability key, range key, command box axis) for each pure draw that
# REDRAWS an axis. pure_wz and pure_vy are absent on purpose: they keep the
# value the base uniform already drew inside the box and only zero the other
# axes, so they cannot leave it.
PURE_DRAW_RANGES = (
    ("pure_slow_prob", "slow_vx", "vx"),
    ("pure_fast_prob", "fast_vx", "vx"),
    ("pure_back_prob", "back_vx", "vx"),
)


def check_pure_draw_ranges(command) -> None:
    """Refuse an armed pure draw that redraws outside the command box.

    A pure draw's range is a re-draw of one axis, not a widening of it: the
    box is what the policy is told it was trained for, and every commanded
    velocity the env can produce has to sit inside it. Setting a fast range
    ABOVE the box to pull a policy past a speed it deadlocks at is exactly
    the state this refuses.

    Export gets the same guarantee for free: loading a checkpoint rebuilds
    its env through this constructor, so a run that could never ship fails
    here, before the GPU hours.

    Only draws with a nonzero probability are checked, so the shipped
    all-off defaults validate nothing and no preset changes meaning.
    """
    for prob_key, range_key, axis in PURE_DRAW_RANGES:
        if float(command.get(prob_key, 0.0)) <= 0.0:
            continue
        if range_key not in command or axis not in command:
            continue
        lo, hi = (float(v) for v in command[range_key])
        box_lo, box_hi = (float(v) for v in command[axis])
        if lo < box_lo or hi > box_hi:
            raise ValueError(
                f"command.{prob_key} is armed but command.{range_key} = "
                f"({lo}, {hi}) draws outside the command box command.{axis} = "
                f"({box_lo}, {box_hi}) -- the policy would train on commands "
                f"the exported box says it never saw. Widen command.{axis} or "
                f"narrow command.{range_key}"
            )


class Joystick(HumanoidEnv):
    def __init__(self, robot_dir, preset_name, config=None, config_overrides=None, actuator_overrides=None):
        super().__init__(
            robot_dir, preset_name, config or default_config(), config_overrides, actuator_overrides
        )

        # Checked against the COMPOSED config: a robot overlay moves the box
        # (roboto_origin's vx is [-0.6, 1.0], narrower on the backward side
        # than the shipped back_vx range), so the defaults alone do not
        # settle it.
        check_pure_draw_ranges(self._config.command)

        self._torque_cap = jp.array(self._mj_model.actuator_forcerange[:, 1])
        # Orientation tolerance cone as sin^2 of its half-angle, so it
        # subtracts directly from the penalty's own sum(square(gravity_xy))
        # (see reward.orientation_tol_deg). Resolved here, once, like the
        # other reward constants: 0.0 is falsy, so the default never enters
        # _compute_rewards' expression at all.
        self._orientation_tol = float(
            np.square(np.sin(np.radians(self._config.reward.get("orientation_tol_deg", 0.0))))
        )
        # Pose-deviation weight: uniform. A quadruped's abduction/leg-joint
        # split (full weight on the ab/adduction joint, lighter on the other
        # two) doesn't map cleanly onto asimov's 6-joint leg (hip_pitch,
        # hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll); an explicit
        # deviation, left uniform rather than guessing a split.
        self._pose_weight = jp.ones(self.action_size)

        c = self._config.command
        self._cmd_vmax = self._cmd_speed(jp.array([max(abs(c.vx[0]), abs(c.vx[1])), 0.0, 0.0]))

        # Neutral ctrl (zero action): default_pose for a PD-style actuator
        # model, zero torque for ideal-torque -- whatever the actuator
        # model's own ctrl_from_action says "zero action" means.
        self._neutral_ctrl = self._actuator_model.ctrl_from_action(
            jp.zeros(self.action_size), self._default_pose, self._action_scale
        )

    # -- command / gait clock ------------------------------------------------
    def _sample_command(self, rng):
        r1, r2, r3, r4 = jax.random.split(rng, 4)
        c = self._config.command
        vel = jp.array(
            [
                jax.random.uniform(r1, minval=c.vx[0], maxval=c.vx[1]),
                jax.random.uniform(r2, minval=c.vy[0], maxval=c.vy[1]),
                jax.random.uniform(r3, minval=c.wz[0], maxval=c.wz[1]),
            ]
        )
        # Pure command draws, in order: wz, vy, slow, fast, back. Each
        # rewrites the base sample into a clean single-axis command with its
        # own probability, and a later draw overwrites an earlier one. What
        # they are for is in default_config's command block.
        #
        # Two rules make them free while off. Each block is gated on a static
        # config probability, so `if p:` is a Python branch and a preset with
        # the prob at 0 does not put the block in the trace at all. Each block
        # keys off `fold_in(rng, 0x100 + idx)` with an index of its own rather
        # than taking keys from the split above, so every draw owns an
        # independent stream: enabling one draw moves no other draw's samples,
        # and enabling none leaves this sampler bit-identical to the pre-1.6
        # draw (tests/integration/test_golden_baseline.py is the gate).
        #
        # 0x100 is why that holds. `fold_in(key, i)` is bit-identical to
        # `split(key, n)[i]` for every i < n, so a draw folding in a small
        # index would key off one of the base split keys above -- index 1
        # would BE r2, the vy uniform key. Offsetting the domain puts every
        # draw's key out of reach of any split of `rng`, however wide that
        # split later grows. Re-splitting the folded key (which every block
        # below does) is not enough on its own: a block that sampled straight
        # off an un-offset folded key would get a selector that is a
        # deterministic function of another axis's value.
        # tests/integration/test_pure_command_draws.py pins this.
        #
        # The index table is fixed: 1 wz, 2 vy, 3 slow, 4 fast, 5 back. 0 is
        # reserved and 6 up are free for future draws. A new draw takes the
        # next free index and never reuses one.
        #
        # Every draw is fold_in-gated, pure_wz and pure_vy included. A draw
        # that took its key out of the base split unconditionally would shift
        # every other sample whenever it was switched off.
        wz_p = c.get("pure_wz_prob", 0.0)
        if wz_p:
            # r_b is unused: pure_wz keeps the wz the base draw already made
            # and only zeroes the linear part. The split still runs so all
            # five draws have the same shape -- giving wz a dedicated range
            # later can take r_b without moving r_a's bernoulli stream.
            r_a, r_b = jax.random.split(jax.random.fold_in(rng, 0x100 + 1))
            vel = jp.where(jax.random.bernoulli(r_a, wz_p), vel.at[:2].set(0.0), vel)
        vy_p = c.get("pure_vy_prob", 0.0)
        if vy_p:
            r_a, r_b = jax.random.split(jax.random.fold_in(rng, 0x100 + 2))  # r_b unused, see above
            vel = jp.where(
                jax.random.bernoulli(r_a, vy_p), jp.array([0.0, vel[1], 0.0]), vel
            )
        slow_p = c.get("pure_slow_prob", 0.0)
        if slow_p:
            r_a, r_b = jax.random.split(jax.random.fold_in(rng, 0x100 + 3))
            vx = jax.random.uniform(r_b, minval=c.slow_vx[0], maxval=c.slow_vx[1])
            vel = jp.where(jax.random.bernoulli(r_a, slow_p), jp.array([vx, 0.0, 0.0]), vel)
        fast_p = c.get("pure_fast_prob", 0.0)
        if fast_p:
            r_a, r_b = jax.random.split(jax.random.fold_in(rng, 0x100 + 4))
            vx = jax.random.uniform(r_b, minval=c.fast_vx[0], maxval=c.fast_vx[1])
            vel = jp.where(jax.random.bernoulli(r_a, fast_p), jp.array([vx, 0.0, 0.0]), vel)
        back_p = c.get("pure_back_prob", 0.0)
        if back_p:
            r_a, r_b = jax.random.split(jax.random.fold_in(rng, 0x100 + 5))
            vx = jax.random.uniform(r_b, minval=c.back_vx[0], maxval=c.back_vx[1])
            vel = jp.where(jax.random.bernoulli(r_a, back_p), jp.array([vx, 0.0, 0.0]), vel)

        # Last word: standing still overrides every draw above.
        zero = jax.random.bernoulli(r4, c.zero_prob)
        return jp.where(zero, jp.zeros(3), vel)

    def _cmd_speed(self, command):
        """Planar speed the gait clock should serve; turning counts too."""
        return jp.linalg.norm(command[:2]) + 0.3 * jp.abs(command[2])

    def _leg_phases(self, info):
        phase = info["phase"] + jp.array(_PHASE_OFFSETS)
        return jp.fmod(phase + jp.pi, 2 * jp.pi) - jp.pi

    def _gait_targets(self, info):
        """(target foot clearance, stance mask) from the duty-aware clock."""
        g = self._config.gait
        duty = g.duty
        theta = jp.fmod(self._leg_phases(info) + 2 * jp.pi, 2 * jp.pi) / (2 * jp.pi)
        swing_frac = 1.0 - duty
        in_swing = theta < swing_frac
        rz = g.swing_height * jp.sin(jp.pi * theta / swing_frac) * in_swing
        return rz, ~in_swing

    def _phase_dt(self, command):
        """Clock increment: speed-scaled; frozen when told to stand."""
        g = self._config.gait
        speed = self._cmd_speed(command)
        frac = jp.clip(speed / self._cmd_vmax, 0.0, 1.0)
        freq = g.freq[0] + (g.freq[1] - g.freq[0]) * frac
        return jp.where(speed > 0.05, 2 * jp.pi * self.dt * freq, 0.0)

    # -- reset / step -------------------------------------------------------
    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, r_cmd, r_pose = jax.random.split(rng, 3)
        command = self._sample_command(r_cmd)

        # Reset pose noise: uniform +-reset_noise (rad) on
        # every actuated joint's qpos. reset_noise=0.0 degenerates to the
        # old exact-pose reset (still draws/consumes r_pose either way, so
        # the rng split discipline doesn't depend on the config value).
        #
        # The noise rides on _reset_qpos, which is _home_qpos itself unless
        # real_pose_ref settled one (see default_config): under the settled
        # pose the episode starts at rest, base height and residual tilt
        # included, instead of dropping into its own sag over the first
        # control steps. ctrl still holds the ctrl-space neutral, which the
        # anchor never moves: the pose anchor and the ctrl anchor are
        # separate by design (see envs/base.py).
        reset_noise = self._config.get("reset_noise", 0.0)
        pose_noise = jax.random.uniform(r_pose, (self.action_size,), minval=-1.0, maxval=1.0)
        qpos = self._reset_qpos.at[self._qadr].add(pose_noise * reset_noise)

        data = self._make_data()
        data = data.replace(qpos=qpos, qvel=jp.zeros(self._mj_model.nv), ctrl=self._neutral_ctrl)
        data = mjx.forward(self._mjx_model, data)

        info = {
            "rng": rng,
            "command": command,
            "last_action": jp.zeros(self.action_size),
            "last_last_action": jp.zeros(self.action_size),
            "last_torque": jp.zeros(self.action_size),
            "feet_air_time": jp.zeros(self._n_feet),
            # Per-swing peak clearance, for feet_apex (see step()). Seeded
            # unconditionally, like feet_air_time: the info pytree must not
            # change shape with the reward scales.
            "swing_apex": jp.zeros(self._n_feet),
            "last_contact": jp.zeros(self._n_feet, dtype=bool),
            "phase": jp.array(0.0),
            "step_count": jp.array(0),
            "steps_since_cmd": jp.array(0),
        }
        # CRITICAL for scan-carry parity: every reward/* key present here
        # must also be present after every step() (see step()'s metric
        # merge below), or brax's training scan chokes on a changing
        # metrics pytree structure across steps.
        metrics = {f"reward/{k}": jp.zeros(()) for k in self._config.reward.scales}
        if self._config.no_progress.enable:
            # Optimistic seed: a fresh episode starts at progress ratio 1, so
            # the hazard can only come from measured shortfall, never from the
            # seed (see no_progress in default_config). Both metrics are
            # seeded here for the scan-carry parity the comment above
            # demands: step() writes them on every step when the cut is on.
            info["progress_ema"] = self._cmd_speed(command)
            metrics["no_progress_cut"] = jp.zeros(())
            metrics["progress_ratio_per_step"] = jp.zeros(())
        obs = self._build_obs(data, info)
        return mjx_env.State(data, obs, jp.zeros(()), jp.zeros(()), metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        info = dict(state.info)
        if self._config.no_progress.enable:
            # The extra key is split off only when the cut is on, so the
            # default trajectory keeps the stock 4-way split and the whole
            # RNG stream is unchanged while the feature is off
            # (tests/integration/test_golden_baseline.py is the gate).
            rng, r_noise, r_cmd, r_push, r_term = jax.random.split(info["rng"], 5)
        else:
            rng, r_noise, r_cmd, r_push = jax.random.split(info["rng"], 4)
            r_term = None
        info["rng"] = rng

        motor_targets = jp.clip(
            self._actuator_model.ctrl_from_action(action, self._default_pose, self._action_scale),
            self._ctrl_lo,
            self._ctrl_hi,
        )

        data = state.data
        if self._config.push.enable:
            push_now = (info["step_count"] % self._config.push.interval_steps) == (
                self._config.push.interval_steps - 1
            )
            push = jax.random.uniform(r_push, (2,), minval=-1.0, maxval=1.0)
            push = push / (jp.linalg.norm(push) + 1e-6) * self._config.push.vel
            qvel = data.qvel.at[self._base_vadr : self._base_vadr + 2].add(
                jp.where(push_now, push, jp.zeros(2))
            )
            data = data.replace(qvel=qvel)

        data = mjx_env.step(self._mjx_model, data, motor_targets, self.n_substeps)

        contact = self._foot_contact(data)
        contact_filt = contact | info["last_contact"]
        first_contact = (info["feet_air_time"] > 0) & contact_filt

        # Swing-apex tracker for feet_apex, bracketing the reward call. On
        # the way up it takes the running maximum of each airborne foot's
        # clearance; on the step a foot lands, contact_filt is already true,
        # so this update skips and the term reads the peak the completed
        # swing actually reached. The zeroing after the reward arms the next
        # swing. Both halves run whatever the scale is: the tracker is info
        # state, not a reward, and its shape must not depend on the config.
        info["swing_apex"] = jp.where(
            ~contact_filt,
            jp.maximum(info["swing_apex"], self._foot_clearance(data)),
            info["swing_apex"],
        )

        rewards, fall = self._compute_rewards(data, info, action, first_contact, contact)

        info["swing_apex"] = jp.where(contact_filt, 0.0, info["swing_apex"])
        info["feet_air_time"] = jp.where(contact_filt, 0.0, info["feet_air_time"] + self.dt)
        info["last_contact"] = contact
        info["last_last_action"] = info["last_action"]
        info["last_action"] = action
        info["last_torque"] = data.actuator_force
        info["step_count"] = info["step_count"] + 1

        phase = info["phase"] + self._phase_dt(info["command"])
        info["phase"] = jp.fmod(phase + jp.pi, 2 * jp.pi) - jp.pi

        info["steps_since_cmd"] = info["steps_since_cmd"] + 1

        # No-progress cut (see no_progress in default_config): an env that
        # keeps ignoring its command dies with a probability that grows with
        # the shortfall. Evaluated here, after the steps_since_cmd increment
        # and before the resample below, so it scores the command that
        # actually drove this step. A true termination: it only flips `done`,
        # it attaches no reward term, and rewards["termination"] stays
        # fall-only. Losing the rest of the episode is the whole penalty.
        done = fall
        if self._config.no_progress.enable:
            npg = self._config.no_progress
            cmd = info["command"]
            demand = self._cmd_speed(cmd)
            served = progress.served(self._local_linvel(data)[:2], self._gyro(data)[2], cmd)
            alpha = self.dt / npg.ema_sec
            info["progress_ema"] = (1.0 - alpha) * info["progress_ema"] + alpha * served
            progress_ratio = info["progress_ema"] / jp.maximum(demand, 1e-6)
            hazard = progress.hazard(progress_ratio, npg.risk_below, npg.p_max)
            armed = progress.armed(demand, info["steps_since_cmd"], self.dt, npg.grace_sec)
            no_progress_cut = jax.random.bernoulli(r_term, jp.where(armed, hazard, 0.0))
            done = done | no_progress_cut

        resample = info["steps_since_cmd"] >= self._config.command.resample_steps
        info["command"] = jp.where(resample, self._sample_command(r_cmd), info["command"])
        info["steps_since_cmd"] = jp.where(resample, 0, info["steps_since_cmd"])
        if self._config.no_progress.enable:
            # A fresh command restarts the meter at ratio 1: the EMA is
            # reseeded to the new demand, and steps_since_cmd going back to 0
            # keeps the hazard at zero for grace_sec while the robot turns
            # the gait around.
            #
            # This is the only reseed site INSIDE the env. It is not the only
            # one that has to exist: `info` survives every respawn, because
            # the trainer's wrap_for_brax_training ends in
            # BraxAutoResetWrapper(full_reset=False), which restores data and
            # obs from the cached first state and leaves info alone. A cut
            # env would otherwise come back with the dead episode's shortfall
            # and a steps_since_cmd well past grace, and die again inside
            # what should have been its grace window. envs/wrappers.py's
            # ProgressReseedWrapper does that reseed, and train.py layers it
            # on whenever this flag is set. Any other wrapper that restarts an
            # episode in place owns the same reseed.
            info["progress_ema"] = jp.where(
                resample, self._cmd_speed(info["command"]), info["progress_ema"]
            )

        reward = sum(rewards[k] * self._config.reward.scales[k] for k in rewards)
        reward = jp.clip(reward * self.dt, -100.0, 100.0)

        # Merge over the incoming metrics rather than replacing them: brax's
        # EvalWrapper injects extra keys (e.g. "reward") into state.metrics
        # that must survive every step for scan-carry parity (the pytree
        # structure fed into jax.lax.scan cannot change shape step to step).
        metrics = {
            **state.metrics,
            **{f"reward/{k}": v for k, v in rewards.items()},
        }
        if self._config.no_progress.enable:
            # No `_per_step` suffix on the first: brax reports its episode
            # sum, which for this indicator is 1 exactly when the episode
            # ended on the cut. The ratio does carry the suffix, so brax
            # reports its mean, and it is clipped so over-delivery on a slow
            # command cannot hide shortfall elsewhere in the average.
            metrics["no_progress_cut"] = no_progress_cut.astype(jp.float32)
            metrics["progress_ratio_per_step"] = jp.clip(progress_ratio, 0.0, 2.0)

        obs = self._build_obs(data, info, r_noise)
        return mjx_env.State(data, obs, reward, done.astype(jp.float32), metrics, info)

    # -- observations -------------------------------------------------------
    def _obs_catalog(self, data, info):
        catalog = super()._obs_catalog(data, info)
        catalog["command"] = info["command"]
        leg_phase = self._leg_phases(info)
        catalog["phase"] = jp.concatenate([jp.cos(leg_phase), jp.sin(leg_phase)])
        return catalog

    def _catalog_probe_info(self):
        """This task's own `info` inputs to the catalog: the command and the
        gait clock (see _obs_catalog). Shapes only -- the values are zeros."""
        return {**super()._catalog_probe_info(), "command": jp.zeros(3), "phase": jp.zeros(())}

    # -- rewards --------------------------------------------------------------
    def _compute_rewards(self, data, info, action, first_contact, contact):
        cmd = info["command"]
        linvel = self._local_linvel(data)
        gyro = self._gyro(data)
        gravity = self._gravity_body(data)
        cfg = self._config.reward
        moving = self._cmd_speed(cmd) > 0.05

        qpos_act = data.qpos[self._qadr]
        qvel_act = data.qvel[self._vadr]

        foot_clearance = self._foot_clearance(data)
        target_clearance, _stance_mask = self._gait_targets(info)
        foot_vel = self._foot_linvel(data)

        base_height = data.qpos[self._base_qadr + 2]
        fall = (base_height < self._config.fall.min_height) | (gravity[2] > self._config.fall.max_tilt_gz)

        # Velocity tracking kernels: absolute or command-relative, optionally
        # blended with a wider far-field exponential, optionally gating each
        # other. Every switch below is static config read here at trace time
        # under a plain Python if, so all-off compiles to exactly the two
        # legacy absolute kernels.
        err_lin = terms.tracking_err_lin(cmd[:2], linvel[:2])
        err_ang = terms.tracking_err_ang(cmd[2], gyro[2])

        far_w = cfg.get("tracking_far_weight", 0.0)

        def _far_blend(kernel, err_sq):
            """The far-field mix-in (see tracking_far_weight in
            default_config). far_w is static config, so weight 0 returns the
            bare kernel untouched rather than a numerically-equal blend."""
            if not far_w:
                return kernel
            return terms.tracking_far_blend(kernel, err_sq, far_w, cfg.tracking_far_sigma)

        if cfg.get("tracking_relative", False):
            k_lin = terms.tracking_kernel(
                err_lin,
                terms.tracking_rel_sigma(
                    jp.linalg.norm(cmd[:2]), cfg.tracking_rel_sigma, cfg.tracking_rel_floor_lin
                ),
            )
            k_ang = terms.tracking_kernel(
                err_ang,
                terms.tracking_rel_sigma(
                    jp.abs(cmd[2]), cfg.tracking_rel_sigma, cfg.tracking_rel_floor_ang
                ),
            )
        else:
            k_lin = terms.tracking_kernel(err_lin, cfg.tracking_sigma)
            k_ang = terms.tracking_kernel(err_ang, cfg.tracking_sigma)
        # Both branches blend, and both pass the raw squared error: the far
        # kernel is absolute either way.
        k_lin = _far_blend(k_lin, err_lin)
        k_ang = _far_blend(k_ang, err_ang)
        if cfg.get("tracking_product", False):
            # Gate each term by the other's kernel: tracking pays only for
            # tracking the whole command (see tracking_product in
            # default_config). The reassignment is simultaneous, so both
            # sides read the pre-product kernels and both come out equal to
            # the same product.
            k_lin, k_ang = k_lin * k_ang, k_ang * k_lin

        # Gait-shaping gate (see shaping_tracking_gate in default_config):
        # the positive gait terms follow the linear tracking kernel, after
        # the product gate when that is on. Gated set: feet_air_time and
        # feet_apex. feet_phase stays ungated on purpose -- it is the
        # clock-following gradient, and it has to survive at zero tracking
        # because stepping is how tracking starts.
        shape_gate = k_lin if cfg.get("shaping_tracking_gate", False) else 1.0

        # sin^2 of the tilt from vertical, less the tolerance cone (see
        # reward.orientation_tol_deg). The cone is a construct-time float, so
        # the default 0 leaves the legacy expression exactly as it was.
        orientation = terms.orientation(gravity[:2])
        if self._orientation_tol:
            orientation = jp.maximum(orientation - self._orientation_tol, 0.0)

        rewards = {
            "tracking_lin_vel": k_lin,
            "tracking_ang_vel": k_ang,
            "lin_vel_z": terms.lin_vel_z(linvel[2]),
            "ang_vel_xy": terms.ang_vel_xy(gyro[:2]),
            "orientation": orientation,
            "torques": terms.torques(data.actuator_force),
            "torque_rate": terms.torque_rate(data.actuator_force, info["last_torque"]),
            "action_rate": terms.action_rate(action, info["last_action"]),
            "action_accel": terms.action_accel(action, info["last_action"], info["last_last_action"]),
            "energy": terms.energy(qvel_act, data.actuator_force),
            # Both pose terms score the deviation from _pose_anchor, which is
            # _default_pose itself unless real_pose_ref settled one (see
            # default_config). The anchor is a REWARD reference only: the
            # action still centers on _default_pose.
            "pose": terms.pose(qpos_act, self._pose_anchor, self._pose_weight),
            "feet_air_time": terms.feet_air_time(info["feet_air_time"], first_contact, self._config.gait.air_time_cap)
            * moving
            * shape_gate,
            "feet_slip": terms.feet_slip(foot_vel[:, :2], contact) * moving,
            "feet_phase": terms.feet_phase(foot_clearance, target_clearance, cfg.phase_sigma) * moving,
            "stand_still": terms.stand_still(qpos_act, self._pose_anchor, qvel_act) * (~moving),
            "termination": terms.termination(fall),
            "torque_limit": terms.torque_limit(data.actuator_force, self._torque_cap, cfg.torque_limit_frac),
            # Appended at the END on purpose: the scaled sum below adds the
            # terms in dict order, so a key inserted mid-dict shifts every
            # later float addition and breaks the pre-port golden.
            #
            # feet_apex is in the shape_gate's set (see above): a tall swing
            # while the command goes unserved is the same stand-and-lift
            # income the gate exists to remove. feet_landing is NOT -- it is
            # a penalty, and gating a penalty on the tracking kernel would
            # relax it exactly when tracking is failing, which is when feet
            # are being slammed into the floor.
            "feet_apex": terms.feet_apex(info["swing_apex"], first_contact, cfg.apex_target)
            * moving
            * shape_gate,
            "feet_landing": terms.feet_landing(foot_vel[:, 2], foot_clearance, cfg.glide_height)
            * moving,
        }
        return rewards, fall
