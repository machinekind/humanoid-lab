"""Offscreen rollout rendering: the frame-size knob and the one SceneView.

`SceneView` owns the camera, the offscreen buffer sizing and the in-frame
overlays, so every video consumer renders through one code path instead of
each carrying its own Renderer wiring.
"""

import argparse
import copy

import numpy as np

from humanoid_lab.eval import overlays

DEFAULT_SIZE = (640, 480)


def frame_size(text):
    """Frame size written as WxH."""
    parts = str(text).lower().split("x")
    try:
        w, h = (int(p) for p in parts)
    except ValueError:
        raise argparse.ArgumentTypeError(f"frame size must be WxH, got {text!r}") from None
    if w <= 0 or h <= 0:
        raise argparse.ArgumentTypeError(f"frame size must be positive, got {text!r}")
    return w, h


class SceneView:
    """Offscreen renderer for one env's scene, with optional in-frame overlays.

    `torque=True` arms the per-actuator bar strip: `frame()` then draws
    `overlays.draw_torques` whenever the caller hands it the step's
    actuator forces. The caps and the joint grouping are resolved from the
    env's model and RobotSpec once, here, so the overlay stays a pure
    function of arrays.
    """

    def __init__(self, env, *, size=DEFAULT_SIZE, camera=None, torque=False):
        import mujoco

        self.size = tuple(size)
        self.camera = camera  # name, id or MjvCamera; None leaves mujoco's default
        model = env.mj_model
        if (
            model.vis.global_.offwidth < self.size[0]
            or model.vis.global_.offheight < self.size[1]
        ):
            # The offscreen buffer is sized in the model. Widen a private
            # copy rather than the env's own model, so a video tool never
            # mutates the model an env is simulating with.
            model = copy.deepcopy(model)
            model.vis.global_.offwidth = max(model.vis.global_.offwidth, self.size[0])
            model.vis.global_.offheight = max(model.vis.global_.offheight, self.size[1])
        self.model = model
        self.data = mujoco.MjData(model)
        self._view = mujoco.Renderer(model, height=self.size[1], width=self.size[0])
        self._caps = np.asarray(model.actuator_forcerange[:, 1]) if torque else None
        if torque:
            self._joint_names = list(env.robot_spec.actuated_joints)
            self._joint_groups = dict(env.robot_spec.joint_groups)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def close(self):
        self._view.close()

    def frame(self, qpos, *, torque=None):
        """The scene at `qpos`, with the overlays this view was built for."""
        import mujoco

        self.data.qpos[:] = np.asarray(qpos)
        mujoco.mj_forward(self.model, self.data)
        if self.camera is None:
            self._view.update_scene(self.data)
        else:
            self._view.update_scene(self.data, camera=self.camera)
        frame = self._view.render()
        if self._caps is not None and torque is not None:
            frame = overlays.draw_torques(
                frame, torque, self._caps, self._joint_names, self._joint_groups
            )
        return frame
