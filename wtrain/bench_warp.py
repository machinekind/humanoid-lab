"""mujoco_warp throughput for roboto_origin/deploy_pd at the training sim settings."""
import sys, time
import mujoco, numpy as np, warp as wp
from mujoco.mjx.third_party import mujoco_warp as mjw
from humanoid_lab import paths
from humanoid_lab.robot.build import build_spec, compile_spec

wp.config.quiet = True
robot_dir = paths.REPO_ROOT / "robots/roboto_origin"
m = compile_spec(build_spec(robot_dir, "deploy_pd", {}))
m.opt.timestep = 0.005
print("solver", m.opt.solver, "iters", m.opt.iterations, "ls", m.opt.ls_iterations, "nq nv nu", m.nq, m.nv, m.nu, "ngeom", m.ngeom, flush=True)
d = mujoco.MjData(m); d.qpos[:] = m.key("home").qpos; mujoco.mj_forward(m, d)
for nworld in [int(x) for x in sys.argv[1:]] or [1024, 4096]:
    mm = mjw.put_model(m)
    dd = mjw.put_data(m, d, nworld=nworld, naconmax=96 * nworld, njmax=896)
    mjw.step(mm, dd)  # compile
    with wp.ScopedCapture() as cap:
        for _ in range(4):
            mjw.step(mm, dd)
    wp.synchronize()
    n = 50
    t = time.perf_counter()
    for _ in range(n):
        wp.capture_launch(cap.graph)
    wp.synchronize()
    dt = time.perf_counter() - t
    print(f"nworld {nworld:6d}: {nworld*n/dt:,.0f} control steps/s ({nworld*n*4/dt:,.0f} physics steps/s)", flush=True)
