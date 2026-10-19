import os, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

P = os.path.expanduser(
    "~/ped-cv/code/PedestrianActionBenchmark/data/models/jaad/Static/"
    "15Oct2026-10h35m32s/test_output.pkl")
OUT = os.path.expanduser("~/ped-cv/ped-crossing-repro/figures")

d = pickle.load(open(P, "rb"))
tte = np.asarray(d["tte"]).reshape(-1)
gt = np.asarray(d["gt"]).reshape(-1).astype(int)
prob = np.asarray(d["y"]).reshape(-1)
pid = np.asarray(d["pid"]).reshape(-1)
pred = (prob >= 0.5).astype(int)

vals = np.unique(tte)
base = 1 - gt.mean()
print("samples", len(gt), "crossing share", round(gt.mean(), 4),
      "always-not-crossing accuracy", round(base, 4))
print("unique pedestrian ids", len(np.unique(pid)), "tte values", vals)

def metrics(idx):
    rows = []
    for t in vals:
        m = idx[tte[idx] == t]
        g, p, pr = gt[m], prob[m], pred[m]
        acc = (pr == g).mean()
        rec = (pr[g == 1] == 1).mean() if (g == 1).any() else np.nan
        auc = roc_auc_score(g, p) if len(np.unique(g)) == 2 else np.nan
        rows.append((len(m), acc, rec, auc))
    return np.array(rows)

point = metrics(np.arange(len(gt)))
ids = np.unique(pid)
groups = {p: np.where(pid == p)[0] for p in ids}
rng = np.random.default_rng(0)
boots = []
for _ in range(500):
    sel = rng.choice(ids, len(ids))
    boots.append(metrics(np.concatenate([groups[p] for p in sel])))
boots = np.array(boots)
lo = np.nanpercentile(boots[:, :, 1:], 2.5, axis=0)
hi = np.nanpercentile(boots[:, :, 1:], 97.5, axis=0)

print("\n tte     n    acc    recall   auc    auc 95% CI")
for i, t in enumerate(vals):
    print(f"{t:4d} {int(point[i,0]):6d}  {point[i,1]:.3f}  {point[i,2]:.3f}  {point[i,3]:.3f}  [{lo[i,2]:.3f}, {hi[i,2]:.3f}]")

names = ["accuracy", "recall (crossing)", "AUC"]
fig, ax = plt.subplots(1, 3, figsize=(14, 4))
for k in range(3):
    ax[k].plot(vals, point[:, k + 1], marker="o")
    ax[k].fill_between(vals, lo[:, k], hi[:, k], alpha=0.25)
    ax[k].set_xlabel("time to event (frames)")
    ax[k].set_title(names[k])
ax[0].axhline(base, ls="--", lw=1, color="gray")
ax[0].text(vals[0], base + 0.003, "always 'not crossing'", fontsize=8)
ax[2].axhline(0.5, ls="--", lw=1, color="gray")
fig.suptitle("Static VGG16, JAADall test set: metrics by time to event (95% CI, resampling pedestrians)")
fig.tight_layout()
fig.savefig(f"{OUT}/metrics_vs_tte.png", dpi=150)
print("saved", f"{OUT}/metrics_vs_tte.png")
