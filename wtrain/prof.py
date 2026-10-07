import os, sys, time
os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.dirname(__file__))
import torch
from wenv import WarpJoystick, load_run
run = load_run("runs/roboto-keepers/roboto-locomotion-v1")
env = WarpJoystick(run, num_envs=4096)
S, P = env.reset()
a = torch.zeros(4096, env.nu, device=env.dev)
for _ in range(3): env.step(a)
torch.cuda.synchronize()
t = time.perf_counter()
for _ in range(20): env._physics()
torch.cuda.synchronize(); tp = (time.perf_counter() - t) / 20
t = time.perf_counter()
for _ in range(20): env.step(a)
torch.cuda.synchronize(); ts = (time.perf_counter() - t) / 20
print(f"physics {tp*1e3:.1f} ms/step, full env.step {ts*1e3:.1f} ms/step -> {4096/ts:,.0f} steps/s")
from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    for _ in range(5): env.step(a)
    torch.cuda.synchronize()
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=15))
