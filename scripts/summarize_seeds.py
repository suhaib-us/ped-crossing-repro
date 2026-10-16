import glob, os, yaml
import numpy as np

OUT = os.path.expanduser("~/ped-cv/runs/seeds")
for setup in ["jaad_all", "jaad_beh"]:
    files = sorted(glob.glob(f"{OUT}/seed_*/{setup}_results.yaml"))
    rows = [yaml.safe_load(open(f))["results"] for f in files]
    print(setup, len(rows), "runs")
    for m in ["acc", "auc", "f1", "precision", "recall"]:
        v = np.array([r[m] for r in rows])
        print(f"  {m}: {v.mean():.3f} +- {v.std(ddof=1):.3f}")
