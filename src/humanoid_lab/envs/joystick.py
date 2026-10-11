"""Joystick velocity-tracking task, robot-agnostic (biped).

The gait clock is a fixed antiphase (offsets [0, pi]) two-foot clock,
there is no commanded stand height, and the quadruped-only reward terms
(contact_match, high_step, height_tracking, stand_feet_down) do not exist
here -- see rewards/terms.py's module docstring.

Actor observations use only signals the real robot has (IMU + joint
encoders + own history + the commanded velocity + the env's own gait
clock); everything else (world-frame linvel, base height, contacts,
actuator force) is privileged (critic-only).

Deferred (not built yet):
- No action_delay/latency/encoder-offset/action-filter machinery. Actions
  apply on the control step they are computed, with no delay stage in
  between. Real hardware is expected to add 0-1 control steps of action
  delay, so this machinery has to exist before sim2real-fidelity training.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jp
import numpy as np
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env

from humanoid_lab.envs import gait_state, progress, push, signals, symmetry
from humanoid_lab.envs.base import HumanoidEnv
from humanoid_lab.envs.constants import TaskConstants
from humanoid_lab.envs.gait_state import gait_dur_ema_on  # also re-exported
from humanoid_lab.rewards import compose


def jax_command_draws(rng, command_cfg):
    """The JAX random draws behind one command, per env (vmap me).

    progress.command_from_draws turns these into the command. The key scheme
    is fixed, because the golden files and the reference fixtures pin it: the
    base vx, vy, wz and zero draws take r1..r4 of `split(rng, 4)`. Each
    enabled pure draw keys off `fold_in(rng, 0x100 + idx)`, splits it into
    r_a (the bernoulli) and r_b (the range draw, if it has one). An off draw
    makes no call, so enabling one moves no other draw's samples.
    (tests/integration/test_golden_baseline.py is the gate.)

    0x100 is why that holds. `fold_in(key, i)` is bit-identical to
    `split(key, n)[i]` for every i < n, so a draw folding in a small
    index would key off one of the base split keys above -- index 1
    would BE r2, the vy uniform key. Offsetting the domain puts every
    draw's key out of reach of any split of `rng`, however wide that
    split later grows. Re-splitting the folded key is not enough on its
    own: a draw that sampled straight off an un-offset folded key would
    get a selector that is a deterministic function of another axis's
    value. tests/integration/test_pure_command_draws.py pins this.

    r_b is unused by pure_wz and pure_vy: they keep the wz or vy the base draw
    already made. The split still runs so all five draws have the same shape.
    Giving them a range later can take r_b without moving r_a's stream.
    """
    c = command_cfg
    r1, r2, r3, r4 = jax.random.split(rng, 4)
    draws = {
        "vx": jax.random.uniform(r1, minval=c.vx[0], maxval=c.vx[1]),
        "vy": jax.random.uniform(r2, minval=c.vy[0], maxval=c.vy[1]),
        "wz": jax.random.uniform(r3, minval=c.wz[0], maxval=c.wz[1]),
    }
    for name, idx, prob_key, range_key in progress.COMMAND_DRAWS:
        p = c.get(prob_key, 0.0)
        if p:
            r_a, r_b = jax.random.split(jax.random.fold_in(rng, 0x100 + idx))
            draws[f"{name}_on"] = jax.random.bernoulli(r_a, p)
            if range_key:
                draws[f"{name}_vx"] = jax.random.uniform(
                    r_b, minval=c[range_key][0], maxval=c[range_key][1]
                )
    draws["zero_on"] = jax.random.bernoulli(r4, c.zero_prob)
    return draws


def jax_push_draws(r_push, push_cfg):
    """The JAX random draws behind one push, per env (vmap me).

    The direction uses r_push itself. The other draws use
    `fold_in(r_push, 0x100 + idx)` with the indices in push.PUSH_DRAWS, for
    the reason jax_command_draws gives. An off knob makes no call. The ranges
    come from push.push_ranges, and push.push_kick turns the draws into the
    kick.
    """
    ranges = push.push_ranges(push_cfg)
    draws = {"dir": jax.random.uniform(r_push, (2,), minval=-1.0, maxval=1.0)}
    for name, idx, shape, _ in push.PUSH_DRAWS:
        if name == "dir" or name not in ranges:
            continue
        key = jax.random.fold_in(r_push, 0x100 + idx)
        lo, hi = ranges[name]
        if name == "next_countdown":
            draws[name] = jax.random.randint(key, shape, lo, hi + 1)
        elif name == "ang":
            draws[name] = jax.random.uniform(key, shape, minval=jp.array(lo), maxval=jp.array(hi))
        else:
            draws[name] = jax.random.uniform(key, shape, minval=lo, maxval=hi)
    return draws


def jax_reset_draws(rng, cfg, action_size):
    """The JAX random draws behind one reset, per env (vmap me).

    Keys: `rng, r_cmd, r_pose = split(rng, 3)`, then one more split for the
    first push gap when the schedule is ranged, then one for the mirror flag
    when symmetry is on. 'rng' is the key left for the step.
    """
    rng, r_cmd, r_pose = jax.random.split(rng, 3)
    draws = {
        "command": jax_command_draws(r_cmd, cfg.command),
        "pose_noise": jax.random.uniform(r_pose, (action_size,), minval=-1.0, maxval=1.0),
    }
    if cfg.push.enable and cfg.push.get("interval_steps_range", None):
        lo, hi = (int(v) for v in cfg.push.interval_steps_range)
        rng, r_cd = jax.random.split(rng)
        draws["push_countdown"] = jax.random.randint(r_cd, (), lo, hi + 1)
    if cfg.symmetry.enable:
        rng, r_mirror = jax.random.split(rng)
        draws["mirror_u"] = jax.random.uniform(r_mirror)
    draws["rng"] = rng
    return draws


def jax_step_draws(rng, cfg):
    """The JAX random draws behind one step, per env (vmap me).

    A 4-way split, or a 5-way split when the no-progress cut is on. 'rng' is
    the key left for the next step.
    """
    if cfg.no_progress.enable:
        rng, r_noise, r_cmd, r_push, r_term = jax.random.split(rng, 5)
    else:
        rng, r_noise, r_cmd, r_push = jax.random.split(rng, 4)
        r_term = None
    return {
        "rng": rng,
        "noise_key": r_noise,
        "command": jax_command_draws(r_cmd, cfg.command),
        "push": jax_push_draws(r_push, cfg.push) if cfg.push.enable else None,
        "u_term": jax.random.uniform(r_term) if r_term is not None else None,
    }


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        # 200 Hz physics, 50 Hz policy (decimation 4).
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
        # roboto_origin's home
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
        # Fallback noise scales; robot overlays pin their measured ones
        # (configs/robot/*.yaml). Unlisted components default to noise-free
        # (see base.py's _build_obs).
        obs_noise=config_dict.create(gyro=0.01, joint_pos=0.01, joint_vel=0.1),
        # Declarative observation spec: ordered lists of catalog names (see
        # envs/signals.catalog).
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
        # Fallback command envelope; robot overlays pin their own
        # (configs/robot/*.yaml).
        command=config_dict.create(
            vx=(-0.8, 0.8),
            vy=(-0.6, 0.6),
            wz=(-0.6, 0.6),
            resample_steps=250,  # 5 s at ctrl_dt=0.02
            # Velocity zeroes with this prob (stand training). An untuned
            # starting value for a biped.
            zero_prob=0.15,
            # Commanded speed at or below which a command is a stand command (see Joystick.__init__).
            deadband=progress.SPEED_DEADBAND,
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
            # it. The quadruped predecessor measured exactly that refusal
            # (0.000 m/s held under a clean backward command), learned, not
            # mechanical.
            #
            # Every range below is a STARTING VALUE derived from the
            # fallback envelope above; re-derive per robot.
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
        # Untuned starting values; re-derive per robot. Base kick: planar,
        # uniform direction, magnitude `vel`, every `interval_steps`.
        # Optional knobs, each off by default. Off = the legacy trace and
        # RNG stream, bit-identical. interval_steps_range draws the gap to
        # the next push (steps). vel_z adds a vertical kick (+-m/s).
        # ang_vel_rp / ang_vel_yaw add angular kicks (+-rad/s). Kicks add
        # to the base velocity; upstream's push overwrites it.
        push=config_dict.create(
            enable=True,
            interval_steps=200,
            vel=0.4,
            interval_steps_range=None,
            vel_z=0.0,
            ang_vel_rp=0.0,
            ang_vel_yaw=0.0,
        ),
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
        # Every number here is an untuned starting value carried from the
        # quadruped predecessor. grace_sec and risk_below are the two to
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
        # Left/right mirror augmentation. With mirror_prob, an env presents
        # the policy a world mirrored about the body xz-plane: observations
        # are mirrored on the way out and the action is mirrored back on the
        # way in, so physics, rewards and termination stay in the real
        # frame. One policy then has to walk both chiralities. The flag is
        # drawn at reset; under the trainer's auto-reset wrapper `info`
        # survives respawns, so it holds per env for the whole run. Off adds
        # no info key and splits no RNG key. Training-only: the battery,
        # eval and export envs force it off (eval/battery.py), and the
        # deployed policy always sees real observations. The maps are in
        # envs/symmetry.py, which holds joint signs for roboto_origin only.
        symmetry=config_dict.create(enable=False, mirror_prob=0.5),
        # Fallback fall thresholds; robot overlays pin their own (a
        # workable min_height is ~60% of the robot's standing height).
        # max_tilt_gz is the gravity_body z component: near -1 upright,
        # rising toward 0 and positive as the robot tips over.
        fall=config_dict.create(min_height=0.45, max_tilt_gz=-0.4),
        # Two-foot antiphase gait clock (see progress.PHASE_OFFSETS). freq/
        # swing_height/duty are untuned starting values: only the SHAPE
        # (speed-scaled clock frequency, sinusoidal swing profile, fixed
        # stance duty) carries over from the quadruped predecessor. The
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
            # starting points, and a narrow kernel rounds partial tracking
            # to zero; re-derive rather than trusting these.
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
            # feet_air_time, feet_air_time_biped, feet_apex, feet_apex_min
            # and gait_symmetry_income. Not feet_phase, and not the
            # feet_landing penalty -- see rewards/compose.py.
            shaping_tracking_gate=False,
            phase_sigma=0.002,
            # feet_air_time_biped clamp, s (upstream feet_air_time_positive_
            # biped's threshold): single-stance dwell beyond this earns no
            # extra per-step pay.
            biped_air_time_threshold=0.4,
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
            # Both numbers are untuned quadruped starting values, in
            # physical metres (_foot_clearance reads sole height above the
            # floor). RE-DERIVE for this leg: the env's own
            # gait.swing_height asks for 0.08 m of swing, so a 0.05 m apex
            # target is not this robot's.
            apex_target=0.05,
            glide_height=0.03,
            # torque_limit hinge fires above this fraction of each
            # actuator's forcerange cap.
            torque_limit_frac=0.85,
            # knee_stance: stance-leg knee flexion tolerance, rad. Flexion
            # inside the cone is free (touchdown shock absorption lives
            # there); beyond it the stance leg's knee is charged
            # quadratically. roboto_origin's home keyframe holds the knee
            # at 0.3, so 0.15 asks the stance leg to carry the body
            # straighter than the crouch it resets into.
            knee_stance_tol=0.15,
            # gait_symmetry and gait_symmetry_income (shared): per-event
            # EMA weight folding each completed swing/stance duration into
            # its foot's running average, the denominator floor (s) of the
            # relative-difference kernel, and the cap on the summed relative
            # asymmetry. The income pays 0 at or over the cap, and needs
            # cap > 0. For the penalty, the cap bounds the per-step charge
            # at scale*cap: an uncapped charge taxes the near-maximal
            # asymmetry of the first clumsy steps hard enough that the
            # optimizer takes standing instead (see terms.gait_symmetry).
            gait_symmetry_alpha=0.25,
            gait_symmetry_floor=0.1,
            gait_symmetry_cap=1.0,
            # Velocity-damping share of stand_still (position share is 1;
            # only the ratio matters, scales.stand_still prices the sum).
            stand_still_vel_weight=0.2,
            # pose_l1 weights: a map joint_group -> weight. None = 1.0 for
            # every joint. A group not in the map gets 0. An unknown group
            # fails at construction. scales.pose_l1 prices the sum, so only
            # the ratios matter here.
            pose_l1_weights=None,
            # Separation bands for feet_distance / knee_distance, m, base
            # frame. isaaclab defaults; robot overlays pin measured ones.
            feet_distance_range=(0.2, 0.5),
            knee_distance_range=(0.2, 0.5),
            # UNTUNED starting values, carried from the quadruped
            # predecessor for every term that has a biped analogue
            # (rewards/terms.py's module docstring lists what has none).
            # Do not read these as tuned for either robot.
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
                # Two-foot variant of feet_apex: each landing pays the lower
                # of its own peak and the other foot's last completed peak
                # (see terms.feet_apex_min). 0 = off; a style term (see
                # _STYLE_TERMS). Needs exactly two feet.
                feet_apex_min=0.0,
                # Soft-landing penalty (see glide_height above), 0 = off. A
                # policy trained with it glides its feet into stance instead
                # of striking the floor at swing free-fall speed.
                feet_landing=0.0,
                # Ported robolab terms (rewards/terms.py), 0 = off.
                # roboto_origin's overlay pins the upstream weights.
                pose_l1=0.0,
                joint_pos_limits=0.0,
                joint_vel=0.0,
                joint_acc=0.0,
                upward=0.0,
                feet_distance=0.0,
                knee_distance=0.0,
                feet_contact_without_cmd=0.0,
                # Per-step single-stance dwell (upstream
                # feet_air_time_positive_biped), 0 = off. The landing-event
                # feet_air_time above never fires on a policy with zero
                # swings; this one pays from the first instant of a lift.
                feet_air_time_biped=0.0,
                # Style terms, 0 = off. Config-gated harder than the rest:
                # at scale 0 they add no info keys, no reward/* metric keys
                # and nothing to the trace, so the recorded goldens (which
                # compare the full metric-name set) stay bit-exact.
                # knee_stance charges stance-leg knee flexion beyond
                # knee_stance_tol; gait_symmetry charges the relative
                # left-right difference of the completed swing/stance
                # duration EMAs (the limp signature); gait_symmetry_income
                # pays for the absence of that difference on single-stance
                # steps (see terms.gait_symmetry_income). The two share one
                # set of EMAs, kept whenever either scale is nonzero.
                knee_stance=0.0,
                gait_symmetry=0.0,
                gait_symmetry_income=0.0,
            ),
        ),
    )


# Scales keys whose reward is config-gated on a nonzero scale (see the
# scales comment above): reset() seeds their reward/* metrics only when
# armed, matching _compute_rewards, which computes them only when armed.
# feet_apex_min's input, info["last_apex"], is carried at every scale.
_STYLE_TERMS = ("knee_stance", "gait_symmetry", "gait_symmetry_income", "feet_apex_min")


def _reward_metric_names(scales) -> list[str]:
    """The reward/* metric keys reset() seeds for these reward scales: every
    scales key, less the style terms whose scale is 0. _compute_rewards
    emits exactly this set, which the scan-carry parity in reset() needs."""
    return [f"reward/{k}" for k in scales if k not in _STYLE_TERMS or scales[k]]


def check_gait_symmetry_cap(reward) -> None:
    """Raise unless reward.gait_symmetry_cap > 0 whenever the income is on.
    gait_symmetry_income divides by the cap, so cap 0 would be NaN on every
    armed single-stance step. The penalty reads cap 0 as 'charge nothing'
    and needs no check."""
    cap = reward.get("gait_symmetry_cap", 1.0)
    if reward.scales.get("gait_symmetry_income", 0.0) and not cap > 0.0:
        raise ValueError(
            f"reward.scales.gait_symmetry_income needs reward.gait_symmetry_cap > 0, got {cap}"
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
        self._pose_l1_weight = self._resolve_group_weights(
            self._config.reward.get("pose_l1_weights", None)
        )
        self._soft_lo_j = jp.array(self._soft_lo)
        self._soft_hi_j = jp.array(self._soft_hi)
        self._ankle_pair = self._group_body_pair("ankle_roll")
        self._knee_pair = self._group_body_pair("knee")
        scales = self._config.reward.scales
        if self._ankle_pair is None and scales.feet_distance:
            raise ValueError(
                "reward.scales.feet_distance needs an 'ankle_roll' joint group "
                "with one joint per side"
            )
        if self._knee_pair is None and scales.knee_distance:
            raise ValueError(
                "reward.scales.knee_distance needs a 'knee' joint group "
                "with one joint per side"
            )

        # Style-term gates (see _STYLE_TERMS): static Python booleans, so an
        # off term never enters the reward sum or the metric set.
        self._knee_stance_on = bool(scales.knee_stance)
        self._gait_symmetry_on = bool(scales.gait_symmetry)
        self._gait_symmetry_income_on = bool(scales.get("gait_symmetry_income", 0.0))
        self._feet_apex_min_on = bool(scales.get("feet_apex_min", 0.0))
        if self._feet_apex_min_on and self._n_feet != 2:
            raise ValueError(
                f"reward.scales.feet_apex_min needs exactly two feet; this robot has {self._n_feet}"
            )
        # The swing/stance duration EMAs both symmetry terms read.
        self._gait_dur_ema_on = gait_dur_ema_on(scales)
        check_gait_symmetry_cap(self._config.reward)
        if self._knee_stance_on:
            self._knee_qidx = self._foot_ordered_group_qidx("knee")

        c = self._config.command
        self._cmd_vmax = self._cmd_speed(jp.array([max(abs(c.vx[0]), abs(c.vx[1])), 0.0, 0.0]))

        # Neutral ctrl (zero action): default_pose for a PD-style actuator
        # model, zero torque for ideal-torque -- whatever the actuator
        # model's own ctrl_from_action says "zero action" means.
        self._neutral_ctrl = self._actuator_model.ctrl_from_action(
            jp.zeros(self.action_size), self._default_pose, self._action_scale
        )

        # A command is a stand command when its speed, |v_xy| + YAW_SPEED_WEIGHT * |wz|,
        # is at or below command.deadband. Then the gait clock freezes, stand_still pays
        # and the stepping rewards switch off. The no-progress cut uses the same threshold.
        # A larger value such as 0.15 stops the robot stepping in place while a walk
        # command is still near zero.
        self._deadband = float(self._config.command.deadband)
        # The policy observes only the yaw rate, so heading drift goes unseen.
        # With heading_err on an obs list, info['yaw_ref'] holds the heading to keep.
        # yaw_ref follows the measured yaw while |commanded wz| > progress.TURN_WZ_THRESHOLD
        # and holds otherwise. The observation is wrap(yaw_ref - yaw), clipped to +-1 rad.
        self._heading_obs = "heading_err" in (
            list(self._config.obs.state) + list(self._config.obs.privileged)
        )

        # Built before the mirror maps: _init_mirror_maps evaluates the obs
        # catalog, which reads the constants.
        self._consts = self._task_constants()
        if self._config.symmetry.enable:
            self._init_mirror_maps()
            self._consts = dataclasses.replace(
                self._consts,
                act_perm=self._act_perm,
                act_sign=self._act_sign,
                state_perm=self._state_perm,
                state_sign=self._state_sign,
                priv_perm=self._priv_perm,
                priv_sign=self._priv_sign,
            )

    def _task_constants(self) -> TaskConstants:
        """The resolved constants as one record. The private attributes stay
        as aliases of the same objects."""
        return TaskConstants(
            dt=float(self.dt),
            action_size=self.action_size,
            n_feet=self._n_feet,
            qadr=self._qadr,
            vadr=self._vadr,
            base_qadr=self._base_qadr,
            base_vadr=self._base_vadr,
            sensor_adr=self._sensor_adr,
            default_pose=self._default_pose,
            pose_anchor=self._pose_anchor,
            reset_qpos=self._reset_qpos,
            pose_weight=self._pose_weight,
            pose_l1_weight=self._pose_l1_weight,
            action_scale=self._action_scale,
            actuator_model=self._actuator_model,
            neutral_ctrl=self._neutral_ctrl,
            ctrl_lo=self._ctrl_lo,
            ctrl_hi=self._ctrl_hi,
            soft_lo=self._soft_lo_j,
            soft_hi=self._soft_hi_j,
            torque_cap=self._torque_cap,
            foot_site_ids=self._foot_site_ids,
            foot_site_sole_offset=self._foot_site_sole_offset,
            foot_root_body=self._foot_root_body,
            foot_dof_mask=self._foot_dof_mask,
            foot_geom_ids=self._foot_geom_ids,
            foot_geom_radius=self._foot_geom_radius,
            foot_geom_member=self._foot_geom_member,
            ankle_pair=self._ankle_pair,
            knee_pair=self._knee_pair,
            knee_qidx=getattr(self, "_knee_qidx", None),
            cmd_vmax=self._cmd_vmax,
            deadband=self._deadband,
            orientation_tol=self._orientation_tol,
            heading_obs=self._heading_obs,
            gait_dur_ema_on=self._gait_dur_ema_on,
            knee_stance_on=self._knee_stance_on,
            gait_symmetry_on=self._gait_symmetry_on,
            gait_symmetry_income_on=self._gait_symmetry_income_on,
            feet_apex_min_on=self._feet_apex_min_on,
            naconmax_per_env=self._naconmax_per_env,
            njmax=self._njmax,
        )

    def _init_mirror_maps(self):
        """Assemble the symmetry augmentation's (perm, sign) maps from the
        resolved obs lists and the robot's joint and foot names, and check
        each against the real vector size it will index."""
        rs = self._robot_spec
        joints = list(rs.actuated_joints)
        feet = list(rs.foot_sites)
        state_perm, state_sign, act_perm, act_sign = symmetry.policy_mirror_maps(
            rs, self.actor_obs_names
        )
        priv_perm, priv_sign = symmetry.obs_mirror(self._config.obs.privileged, joints, feet, rs.name)
        # Shapes only: the catalog evaluated abstractly on a fresh jax-impl
        # data, so nothing is allocated on the training backend.
        obs = jax.eval_shape(
            self._build_obs, mjx.make_data(self._mj_model, impl="jax"), self._catalog_probe_info()
        )
        for label, perm, size in (
            ("action", act_perm, self.action_size),
            ("obs.state", state_perm, obs["state"].shape[0]),
            ("obs.privileged", priv_perm, obs["privileged_state"].shape[0]),
        ):
            if len(perm) != size:
                raise ValueError(
                    f"symmetry: the {label} mirror map has {len(perm)} entries but the "
                    f"env's {label} vector has {size}"
                )
        self._act_perm = jp.array(act_perm)
        self._act_sign = jp.array(act_sign)
        self._state_perm = jp.array(state_perm)
        self._state_sign = jp.array(state_sign)
        self._priv_perm = jp.array(priv_perm)
        self._priv_sign = jp.array(priv_sign)

    def _mirror_obs(self, obs, info):
        """The observation as the env's world presents it: mirrored when
        info["mirror"] is set. Identity, and absent from the trace, when
        symmetry is off."""
        if not self._config.symmetry.enable:
            return obs
        m = info["mirror"]
        return {
            "state": symmetry.mirror(obs["state"], m, self._state_perm, self._state_sign),
            "privileged_state": symmetry.mirror(
                obs["privileged_state"], m, self._priv_perm, self._priv_sign
            ),
        }

    def _resolve_group_weights(self, group_map):
        """Per-joint weights from a joint_group -> weight map."""
        if group_map is None:
            return jp.ones(self.action_size)
        rs = self._robot_spec
        weights = np.zeros(self.action_size)
        joint_index = {name: i for i, name in enumerate(rs.actuated_joints)}
        for group, weight in dict(group_map).items():
            if group not in rs.joint_groups:
                raise ValueError(
                    f"reward.pose_l1_weights names unknown joint group '{group}'; "
                    f"robot defines groups: {sorted(rs.joint_groups)}"
                )
            for name in rs.joint_groups[group]:
                weights[joint_index[name]] = float(weight)
        return jp.array(weights)

    def _foot_ordered_group_qidx(self, group: str):
        """Indices into the actuated-joint arrays for `group`'s joints,
        reordered so entry i belongs to the same leg as foot_sites[i].

        knee_stance multiplies per-knee flexion by per-foot contact, so the
        two arrays must agree on which side is which. Group order is not
        guaranteed to match foot_sites order, so both are matched on their
        "left"/"right" name token; a robot whose names carry no side token
        refuses here, at construction, instead of silently gating the wrong
        leg."""
        rs = self._robot_spec
        joints = rs.joint_groups.get(group, ())
        if len(joints) != len(rs.foot_sites):
            raise ValueError(
                f"reward.scales.knee_stance needs a '{group}' joint group "
                f"with one joint per foot; got {len(joints)} joints for "
                f"{len(rs.foot_sites)} feet"
            )

        def side(name):
            tokens = [t for t in ("left", "right") if t in name]
            if len(tokens) != 1:
                raise ValueError(
                    f"cannot infer a side from '{name}': reward.scales."
                    f"knee_stance pairs {group} joints with feet by their "
                    "'left'/'right' name token"
                )
            return tokens[0]

        by_side = {side(j): j for j in joints}
        joint_index = {name: i for i, name in enumerate(rs.actuated_joints)}
        return jp.array([joint_index[by_side[side(f)]] for f in rs.foot_sites])

    def _group_body_pair(self, group: str) -> tuple[int, int] | None:
        """The two body ids that carry `group`'s joints. None when the
        robot has no such pair; the constructor refuses an armed distance
        term in that case, and an off term reads 0."""
        joints = self._robot_spec.joint_groups.get(group, ())
        if len(joints) != 2:
            return None
        m = self._mj_model
        a, b = (int(m.jnt_bodyid[m.joint(n).id]) for n in joints)
        return a, b

    def _body_y_separation(self, data, pair):
        """|y| distance between two bodies, base frame."""
        return signals.body_y_sep(data.xpos, pair, self._quat(data))

    # -- command / gait clock ------------------------------------------------
    def _sample_command(self, rng):
        c = self._config.command
        return progress.command_from_draws(jax_command_draws(rng, c), c)

    def _yaw(self, data):
        return signals.base_yaw(self._quat(data))

    def _heading_err(self, data, info):
        return signals.heading_err(info["yaw_ref"], self._yaw(data))

    def _cmd_speed(self, command):
        """Planar speed the gait clock should serve; turning counts too."""
        return progress.cmd_speed(command)

    def _leg_phases(self, info):
        return progress.leg_phases(info["phase"])

    def _gait_targets(self, info):
        """(target foot clearance, stance mask) from the duty-aware clock."""
        g = self._config.gait
        return progress.gait_targets(info["phase"], g.duty, g.swing_height)

    def _phase_dt(self, command):
        """Clock increment: speed-scaled; frozen when told to stand."""
        return progress.phase_dt(
            command, self._cmd_vmax, self._config.gait.freq, self.dt, self._deadband
        )

    # -- reset / step -------------------------------------------------------
    def reset(self, rng: jax.Array) -> mjx_env.State:
        d = jax_reset_draws(rng, self._config, self.action_size)
        command = progress.command_from_draws(d["command"], self._config.command)

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
        qpos = self._reset_qpos.at[self._qadr].add(
            gait_state.reset_offset(d["pose_noise"], self._config.get("reset_noise", 0.0))
        )

        data = self._make_data()
        data = data.replace(qpos=qpos, qvel=jp.zeros(self._mj_model.nv), ctrl=self._neutral_ctrl)
        data = mjx.forward(self._mjx_model, data)

        # The gait trackers and the config-gated keys (push_countdown, mirror,
        # air_dur_ema, progress_ema) are seeded by gait_state.reset_state. The
        # trackers are seeded unconditionally, like feet_air_time: the info
        # pytree must not change shape with the reward scales, and
        # envs/wrappers.py's GaitReseedWrapper restarts them on every
        # in-place respawn. The EMAs, the push countdown and the mirror flag
        # are config-gated: off, no key is split and the pytree is unchanged.
        # progress_ema starts at ratio 1, so the hazard can only come from
        # measured shortfall, never from the seed (see no_progress in
        # default_config).
        info = {
            "rng": d["rng"],
            **gait_state.reset_state(
                command,
                self._consts,
                self._config,
                yaw=self._yaw(data) if self._heading_obs else None,
                push_countdown=d.get("push_countdown"),
                mirror_u=d.get("mirror_u"),
            ),
        }
        # CRITICAL for scan-carry parity: every reward/* key present here
        # must also be present after every step() (see step()'s metric
        # merge below), or brax's training scan chokes on a changing
        # metrics pytree structure across steps. Off style terms are absent
        # on BOTH sides: _compute_rewards skips them, so seeding them here
        # would break that parity (and the goldens' metric-name set).
        metrics = {k: jp.zeros(()) for k in _reward_metric_names(self._config.reward.scales)}
        if self._config.no_progress.enable:
            # Optimistic seed: a fresh episode starts at progress ratio 1, so
            # the hazard can only come from measured shortfall, never from the
            # seed (see no_progress in default_config). Both metrics are
            # seeded here for the scan-carry parity the comment above
            # demands: step() writes them on every step when the cut is on.
            metrics["no_progress_cut"] = jp.zeros(())
            metrics["progress_ratio_per_step"] = jp.zeros(())
        obs = self._mirror_obs(self._build_obs(data, info), info)
        return mjx_env.State(data, obs, jp.zeros(()), jp.zeros(()), metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        info = dict(state.info)
        # jax_step_draws splits off the cut's key only when the cut is on, so
        # the default trajectory keeps the stock 4-way split and the whole RNG
        # stream is unchanged while the feature is off
        # (tests/integration/test_golden_baseline.py is the gate).
        draws = jax_step_draws(info["rng"], self._config)
        info["rng"] = draws["rng"]

        # Symmetry augmentation: a mirrored env's policy acted on mirrored
        # observations, so its action is in the mirrored frame.
        # pre_physics maps it back before anything reads it (motor targets,
        # the action rewards, last_action): everything below stays in the
        # real frame. The stored real-frame last_action reaches the policy
        # through the observation mirror (_mirror_obs), which maps it back to the
        # action the policy emitted, since the mirror is an involution.
        info, action, motor_targets, kick = gait_state.pre_physics(
            info, action, self._consts, self._config, draws["push"]
        )

        data = state.data
        if kick is not None:
            # Each push knob is config-gated and draws off fold_in(r_push,
            # 0x100 + i), the offset rule of the command draws. An off knob
            # does not change the trace.
            bv = self._base_vadr
            qvel = data.qvel.at[bv : bv + 2].add(kick["planar"])
            if kick["z"] is not None:
                qvel = qvel.at[bv + 2].add(kick["z"])
            if kick["ang"] is not None:
                qvel = qvel.at[bv + 3 : bv + 6].add(kick["ang"])
            data = data.replace(qvel=qvel)

        data = mjx_env.step(self._mjx_model, data, motor_targets, self.n_substeps)
        r = signals.readings(data, self._consts)
        # The step order is stated once, in gait_state.update's docstring.
        info, rewards, done, aux = gait_state.update(info, r, action, self._consts, self._config, draws)
        reward = compose.total_reward(rewards, self._config.reward.scales, self.dt)

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
            metrics["no_progress_cut"] = aux["cut"].astype(jp.float32)
            metrics["progress_ratio_per_step"] = jp.clip(aux["progress_ratio"], 0.0, 2.0)

        obs = self._mirror_obs(self._observe(signals.catalog(r, info, self._consts), draws["noise_key"]), info)
        return mjx_env.State(data, obs, reward, done.astype(jp.float32), metrics, info)

    # -- observations -------------------------------------------------------
    def _obs_catalog(self, data, info):
        return signals.catalog(signals.readings(data, self._consts), info, self._consts)

    def _catalog_probe_info(self):
        """This task's own `info` inputs to the catalog: the command and the
        gait clock (see _obs_catalog). Shapes only -- the values are zeros."""
        return {**super()._catalog_probe_info(), "command": jp.zeros(3), "phase": jp.zeros(())}

    # -- rewards --------------------------------------------------------------
    def _compute_rewards(self, data, info, action, first_contact, contact):
        return compose.compute_rewards(
            signals.readings(data, self._consts), info, action, first_contact, contact, self._consts, self._config
        )
