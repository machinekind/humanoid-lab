"""Path-following course benchmark: one interpretable score per named row.

The battery measures gait quality under open-loop commands. Courses measure
whether the robot can walk a path. Each row is a geometric course (or a held
spin) plus a commanded speed. A frozen pure-pursuit follower turns the
robot's pose into the [vx, 0, wz] command the policy tracks, and the row
scores how faithfully the base followed.

Method. The geometry rows and spin_left run at the nominal: flat floor, the
model's friction, v_nom (spin_nom for a spin), no kick. The geometry rows
differ from one another only in shape. Every other row changes one thing
from the row it names as `baseline`, so a bad row has one interpretation.
The families:

    geometry      straight_10m, arc_r3_90deg, circle_r2, circle_tight,
                  figure_eight_r15, square_3m, slalom_05m, u_turn
    speed         straight_slow, straight_fast, circle_r2_fast,
                  speed_steps_straight
    floor         straight_slippery, circle_r2_slippery
    disturbance   straight_push, straight_push_fast
    spin          spin_left, spin_right, spin_slow, spin_fast

Speeds, the yaw cap and spin rates are frozen fractions of each robot's
command box (spec.derive). Spin rates have a floor, spec.SPIN_MIN_RAD_S. The
tight-turn radius is the one that asks for spec.R_TIGHT_DEMAND of the yaw cap
at v_nom, floored at spec.R_TIGHT_MIN_M. No path row asks for the full yaw
cap, so the follower's own error stays at least 4x below the tracking
normalizer on every row.

Score. Each axis divides a physical reference by a measured error. 1.0
means the error equals the reference, and higher is better (scoring.py has
the table). A seed's score is its weakest axis if it completed the course,
else 0. A row reports the median and the worst of 8 seeds. Scores compare
across policies on one row of one robot, never across rows.

Protocol. Reset, settle 1 s at zero command, and lay the course out from the
settled pose. Then follow until the goal, a fall, non-finite physics or the
time budget (2.5x the ideal time plus 2 s). Every run of a robot is measured
under that robot's pinned observation noise.

Modules: spec (dataclasses, inputs, constants), geometry (polylines),
families (the catalogue and its fingerprint), follower (the jax follower),
scoring (per-seed results and aggregates), model_ids (friction geoms), report
(the courses section). The output is `<run>/courses.json`, schema 1.

This module imports nothing, so `courses.report` loads without jax.
"""
