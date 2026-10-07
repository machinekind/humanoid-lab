"""Side-by-side clip of a run's latest checkpoint: open-loop (the watcher's
clip) next to the same rollout with the outer heading loop closed.

    python wtrain/compare_video.py --run runs/loco_warp_v34 --step 172769280
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw


def label(frame, text):
    im = Image.fromarray(frame)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 330, 32], fill=(0, 0, 0))
    d.text((10, 8), text, fill=(255, 255, 255))
    return np.asarray(im)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--step", required=True, type=int)
    ap.add_argument("--gain", type=float, default=1.5)
    args = ap.parse_args()

    vids = args.run / "videos"
    open_loop = vids / f"step_{args.step:012d}.mp4"
    closed = vids / f"heading_{args.step:012d}.mp4"
    env = dict(os.environ, JAX_PLATFORMS="cpu")
    subprocess.run([sys.executable, "-m", "humanoid_lab.eval.video", "--run", str(args.run), "--scenario", "walk_ramp",
                    "--heading-gain", str(args.gain), "--out", str(closed)], check=True, env=env,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    a, b = iio.imread(open_loop), iio.imread(closed)
    n = min(len(a), len(b))
    out = np.stack([np.concatenate([label(a[i], "bez regulatora kierunku"),
                                    label(b[i], "z regulatorem kierunku")], axis=1) for i in range(n)])
    dst = vids / f"compare_{args.step:012d}.mp4"
    iio.imwrite(dst, out, fps=30)
    print(dst)


if __name__ == "__main__":
    main()
