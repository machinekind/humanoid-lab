"""export_robot.py POLICY.pkl OUTDIR NAME [kp_factor]

locomp.py policy.pkl -> Wojtek runtime policy (policy.npz + policy_meta.json, schema 2).

Only a policy trained with deploy_obs=True (battery obs constant, carrot obs zero) and a fixed kp
(var_kp False) can run on the robot: the constant obs entries are folded into the first layer bias.
Runtime layout: gyro(3) gravity(3) command(3) joint_pos(12) joint_vel(12) last_act(12).
Needs the patched wojtek_policy/policy.py (meta "activation": "elu", "output_tanh": false).
"""
import json, pickle, sys
import numpy as np

src, out, name = sys.argv[1:4]
kpf = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
b = pickle.load(open(src, "rb"))
cfg = b["cfg"]
NA = 12
assert not cfg["var_kp"] and cfg["deploy_obs"], "export needs var_kp False and deploy_obs True"
layers = b["actor"]
obs_dim = layers[0]["W"].shape[0]
base = 3 + 3 + 3 + 2 * NA + NA
nextra = obs_dim - base
assert layers[-1]["W"].shape[1] == NA, layers[-1]["W"].shape

mean, std = b["mean"].astype(np.float64), np.sqrt(b["var"].astype(np.float64) + 1e-8)
W0, b0 = layers[0]["W"].astype(np.float64), layers[0]["b"].astype(np.float64)
# constant extra obs (what deploy_obs feeds in training): battery [0.5, 0.5], carrot [0, 0, 0]
const = []
if cfg["battery"]:
    const += [0.5, 0.5]
if cfg["dots"]:
    const += [0.0, 0.0, 0.0]
assert len(const) == nextra, (len(const), nextra)
ce = (np.array(const) - mean[base:]) / std[base:]
b0 = b0 + ce @ W0[base:]
W0 = W0[:base]

npz = dict(norm_mean=mean[:base].astype(np.float32), norm_std=std[:base].astype(np.float32),
           hidden_0_kernel=W0.astype(np.float32), hidden_0_bias=b0.astype(np.float32))
for i, l in enumerate(layers[1:], 1):
    npz[f"hidden_{i}_kernel"] = np.asarray(l["W"], np.float32)
    npz[f"hidden_{i}_bias"] = np.asarray(l["b"], np.float32)

LEGS = ["rear_left", "rear_right", "front_right", "front_left"]
names = [f"{l}_{j}_joint" for l in LEGS for j in ("first", "second", "third")]
home = [float(x) for x in b["home"]]
meta = dict(schema_version=2, run_name=name, checkpoint=src, task="locomp_dots", robot="legs_v627",
            obs_size=base, action_size=NA,
            obs_layout=["gyro:3", "gravity:3", "command:3", "joint_pos:12", "joint_vel:12", "last_act:12"],
            actuator_names=names, home_ctrl=home, anchor_ctrl=home,
            action_scale=[float(x) for x in b["act_scale"]],
            target_low=[float(x) for x in b["lo"]], target_high=[float(x) for x in b["hi"]],
            command_low=[cfg["vx"][0], cfg["vy"][0], cfg["wz"][0]], command_high=[cfg["vx"][1], cfg["vy"][1], cfg["wz"][1]],
            command_fill=[], action_filter=0.0, ctrl_dt=cfg["ctrl_dt"], knee_singularity=None,
            activation="elu", output_tanh=False, obs_clip=10.0,
            pd=dict(kp=60.0 * kpf, kp_per_actuator=[k * kpf for k in [80.0, 60.0, 60.0] * 4], kd=2.0 * float(np.sqrt(kpf)), max_torque=9.8),
            tau_ff=dict(enable=False, scale=0.0),
            notes="trained on the calibrated twin (gains x2.09 in sim == robot gains); kp factor %.2f" % kpf)
import os
os.makedirs(out, exist_ok=True)
np.savez(os.path.join(out, "policy.npz"), **npz)
json.dump(meta, open(os.path.join(out, "policy_meta.json"), "w"), indent=1)


# ---- self-check: runtime formula (patched policy.py) == training numpy policy on random obs
def runtime(o):
    x = np.clip((o - npz["norm_mean"]) / npz["norm_std"], -10, 10)
    i = 0
    while f"hidden_{i}_kernel" in npz:
        x = x @ npz[f"hidden_{i}_kernel"] + npz[f"hidden_{i}_bias"]
        if f"hidden_{i+1}_kernel" in npz:
            x = np.where(x > 0, x, np.expm1(np.minimum(x, 0)))
        i += 1
    return x


def train_policy(o):
    x = np.clip((o - b["mean"]) / np.sqrt(b["var"] + 1e-8), -10, 10)
    for i, l in enumerate(b["actor"]):
        x = x @ l["W"] + l["b"]
        if i < len(b["actor"]) - 1:
            x = np.where(x > 0, x, np.expm1(np.minimum(x, 0)))
    return x


rng = np.random.default_rng(0)
err = 0.0
for _ in range(20):
    o = (b["mean"] + np.sqrt(b["var"]) * rng.normal(size=obs_dim) * 0.5).astype(np.float32)
    o[base:] = const
    err = max(err, float(np.max(np.abs(runtime(o[:base]) - train_policy(o)))))
print("max |runtime - training| action diff:", err)
print("wrote", out, "obs", base, "act", NA, "kp", meta["pd"]["kp"], "kd", round(meta["pd"]["kd"], 3))
