import os, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
ok = (pred == gt)

print("samples", len(gt), "crossing", int(gt.sum()))
print("accuracy at 0.5 threshold:", round(ok.mean(), 4))
print("tte min, max:", tte.min(), tte.max())

lo = int(tte.min() // 10 * 10)
edges = np.arange(lo, int(tte.max()) + 11, 10)
nb = len(edges) - 1
b = np.digitize(tte, edges) - 1

ids, pidx = np.unique(pid, return_inverse=True)
C = np.zeros((len(ids), nb)); N = np.zeros((len(ids), nb))
for i in range(len(gt)):
    if 0 <= b[i] < nb:
        N[pidx[i], b[i]] += 1
        C[pidx[i], b[i]] += ok[i]

acc = C.sum(0) / np.maximum(N.sum(0), 1)
rng = np.random.default_rng(0)
boots = []
for _ in range(1000):
    sel = rng.integers(0, len(ids), len(ids))
    n = N[sel].sum(0)
    boots.append(np.where(n > 0, C[sel].sum(0) / np.maximum(n, 1), np.nan))
lo_ci, hi_ci = np.nanpercentile(boots, [2.5, 97.5], axis=0)

labels = [f"{edges[i]}-{edges[i+1]-1}" for i in range(nb)]
print("\nbin (frames)  n   accuracy  95% CI (resampling pedestrians)")
for i in range(nb):
    print(f"{labels[i]:>10} {int(N.sum(0)[i]):5d}   {acc[i]:.3f}   [{lo_ci[i]:.3f}, {hi_ci[i]:.3f}]")

fig, ax = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [3, 2]})
x = np.arange(nb)
ax[0].plot(x, acc, marker="o")
ax[0].fill_between(x, lo_ci, hi_ci, alpha=0.25)
ax[0].axhline(ok.mean(), ls="--", lw=1)
ax[0].set_xticks(x); ax[0].set_xticklabels(labels)
ax[0].set_xlabel("time to event (frames)"); ax[0].set_ylabel("accuracy")
ax[0].set_title("Static VGG16, JAADall: accuracy by time to event")
ax[1].bar(x, N.sum(0))
ax[1].set_xticks(x); ax[1].set_xticklabels(labels)
ax[1].set_xlabel("time to event (frames)"); ax[1].set_ylabel("test samples")
fig.tight_layout()
fig.savefig(f"{OUT}/accuracy_vs_tte.png", dpi=150)
print("saved", f"{OUT}/accuracy_vs_tte.png")
