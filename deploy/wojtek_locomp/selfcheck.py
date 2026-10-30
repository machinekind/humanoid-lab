"""selfcheck.py POLICY_DIR [--make]

Runs the exported policy through the robot's own interpreter (wojtek_policy.policy.WojtekPolicy, patched for
ELU / no output tanh) on a canned observation sequence and compares the motor targets with the values the
training-side numpy policy produced (expected.json next to the policy).  --make writes expected.json using a
standalone re-implementation of the runtime formula (used on the export side, where wojtek_policy is absent).
"""
import json, os, sys
import numpy as np

d = sys.argv[1]
meta = json.load(open(os.path.join(d, "policy_meta.json")))
npz = np.load(os.path.join(d, "policy.npz"))
home = np.array(meta["home_ctrl"], np.float32)
rng = np.random.default_rng(123)
STEPS = 6
seq = []
for k in range(STEPS):
    seq.append(dict(gyro=(0.3 * rng.normal(size=3)).tolist(), gravity=[0.05 * k, -0.03, -0.998],
                    joint_pos=(home + 0.1 * rng.normal(size=12)).tolist(), joint_vel=(1.0 * rng.normal(size=12)).tolist(),
                    command=[0.4, 0.0, 0.2]))

if "--make" in sys.argv:
    def mlp(o):
        x = (o - npz["norm_mean"]) / npz["norm_std"]
        i = 0
        while f"hidden_{i}_kernel" in npz:
            x = x @ npz[f"hidden_{i}_kernel"] + npz[f"hidden_{i}_bias"]
            if f"hidden_{i+1}_kernel" in npz:
                x = np.where(x > 0, x, np.expm1(np.minimum(x, 0)))
            i += 1
        return x
    last = np.zeros(12, np.float32)
    exp = []
    for s in seq:
        o = np.concatenate([s["gyro"], s["gravity"], s["command"], np.array(s["joint_pos"]) - home, s["joint_vel"], last]).astype(np.float32)
        a = mlp(o).astype(np.float32)
        last = a
        t = np.clip(home + a * np.array(meta["action_scale"], np.float32), meta["target_low"], meta["target_high"])
        exp.append(t.tolist())
    json.dump(exp, open(os.path.join(d, "expected.json"), "w"))
    print("wrote expected.json", np.round(exp[-1], 3).tolist())
else:
    from wojtek_policy.policy import WojtekPolicy
    pol = WojtekPolicy(os.path.join(d, "policy.npz"), os.path.join(d, "policy_meta.json"), robot="wojtek_v2")
    exp = json.load(open(os.path.join(d, "expected.json")))
    worst = 0.0
    for s, e in zip(seq, exp):
        t = pol.step(s["gyro"], s["gravity"], s["joint_pos"], s["joint_vel"], s["command"])
        worst = max(worst, float(np.max(np.abs(np.asarray(t) - np.asarray(e)))))
    print("SELFCHECK max |target - expected| =", worst, "rad ->", "OK" if worst < 1e-4 else "MISMATCH")
