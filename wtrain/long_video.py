"""One long clip of a run's latest checkpoint under a scripted command sequence
(stand, ramp up to 0.6 and 0.9 m/s, slow down, turn left, turn right, straight,
stop), rendered with the original JAX env.

    python wtrain/long_video.py --run runs/best_v40_46M --out clip.mp4
"""
import argparse
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
from pathlib import Path

import numpy as np

from humanoid_lab.eval import video

# (t_start, t_end, vx0, vx1, wz): vx is interpolated linearly inside a segment
SEGMENTS = [
    (0.0, 1.5, 0.0, 0.0, 0.0),
    (1.5, 5.5, 0.0, 0.6, 0.0),
    (5.5, 10.0, 0.6, 0.6, 0.0),
    (10.0, 13.0, 0.6, 0.9, 0.0),
    (13.0, 17.0, 0.9, 0.9, 0.0),
    (17.0, 20.0, 0.9, 0.3, 0.0),
    (20.0, 25.0, 0.4, 0.4, 0.5),
    (25.0, 30.0, 0.4, 0.4, -0.5),
    (30.0, 34.0, 0.6, 0.6, 0.0),
    (34.0, 37.0, 0.6, 0.0, 0.0),
    (37.0, 40.0, 0.0, 0.0, 0.0),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    def scenarios(dt, command):
        def cmd_at(i):
            t = i * dt
            for t0, t1, a, b, wz in SEGMENTS:
                if t0 <= t < t1:
                    return np.array([a + (b - a) * (t - t0) / (t1 - t0), 0.0, wz], dtype=np.float32)
            return np.zeros(3, dtype=np.float32)

        return {"long": (cmd_at, int(round(SEGMENTS[-1][1] / dt)))}

    video.battery_scenarios = scenarios
    video.render_video(args.run, "long", out=args.out)


if __name__ == "__main__":
    main()
