"""python -m humanoid_lab.eval.courses: the course benchmark CLI (runner.main).

A --video run on linux renders through EGL, the headless GPU hosts' GL.
MUJOCO_GL must be set before mujoco is first imported, so it is set here,
before any other import, with setdefault so an exported value wins. Darwin
keeps its default GL (CGL), as eval/video.py does.
"""

import os
import sys

if sys.platform == "linux" and "--video" in sys.argv:
    os.environ.setdefault("MUJOCO_GL", "egl")

from humanoid_lab.eval.courses import runner

if __name__ == "__main__":
    sys.exit(runner.main())
