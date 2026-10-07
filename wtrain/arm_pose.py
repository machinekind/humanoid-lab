"""Find and render the straight-arms-along-the-torso pose for roboto_origin.

Prints the upper-arm and forearm directions for a sweep of elbow angles and
writes a side/front render strip to the given png.
"""
import sys

import imageio.v3 as iio
import mujoco
import numpy as np

from humanoid_lab import paths
from humanoid_lab.robot.build import build_spec, compile_spec

m = compile_spec(build_spec(paths.REPO_ROOT / "robots/roboto_origin", "deploy_pd", {}))
d = mujoco.MjData(m)


def setj(vals):
    d.qpos[:] = m.key("home").qpos
    for n, v in vals.items():
        d.qpos[m.joint(n).qposadr[0]] = v
    mujoco.mj_forward(m, d)


def forearm_tip(side):
    """Farthest point of the elbow_yaw link's mesh geoms from the elbow."""
    b = m.body(f"{side}_elbow_yaw_link").id
    best, tip = -1.0, None
    for g in range(m.ngeom):
        if m.geom_bodyid[g] != b or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mid = m.geom_dataid[g]
        v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
        w = d.geom_xpos[g] + v @ d.geom_xmat[g].reshape(3, 3).T
        dist = np.linalg.norm(w - d.xpos[b], axis=1)
        i = int(dist.argmax())
        if dist[i] > best:
            best, tip = dist[i], w[i]
    return tip


def report(vals, label):
    setj(vals)
    sh = d.xpos[m.body("left_arm_pitch_link").id]
    el = d.xpos[m.body("left_elbow_pitch_link").id]
    tip = forearm_tip("left")
    up, fa = el - sh, tip - el
    ang = np.degrees(np.arccos(np.dot(up, fa) / np.linalg.norm(up) / np.linalg.norm(fa)))
    tilt = np.degrees(np.arccos(-(tip - sh)[2] / np.linalg.norm(tip - sh)))
    print(f"{label:34s} upper {np.round(up, 3)}  forearm {np.round(fa, 3)}  bend {ang:5.1f} deg  "
          f"shoulder->tip from vertical {tilt:5.1f} deg")


for el in (0.0, 0.78, 1.2, 1.4, 1.57, 1.75, 1.9):
    report({"left_elbow_pitch_joint": el, "left_arm_pitch_joint": 0.0}, f"arm_pitch 0.00 elbow {el:.2f}")
for ap in (-0.2, 0.0, 0.18):
    report({"left_elbow_pitch_joint": 1.57, "left_arm_pitch_joint": ap}, f"arm_pitch {ap:+.2f} elbow 1.57")

if len(sys.argv) > 2:
    poses = [dict(zip(("left_arm_pitch_joint", "left_elbow_pitch_joint", "right_arm_pitch_joint",
                       "right_elbow_pitch_joint"), [float(x) for x in a.split(",")] * 2)) for a in sys.argv[2:]]
    r = mujoco.Renderer(m, 480, 360)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0, 0, 0.75]
    cam.distance = 1.8
    cam.elevation = -5
    imgs = []
    for p in poses:
        for az in (180, 90):
            setj(p)
            cam.azimuth = az
            r.update_scene(d, cam)
            imgs.append(r.render().copy())
    iio.imwrite(sys.argv[1], np.concatenate(imgs, 1))
