import os, subprocess, shutil, sys

BASE = "data/models/jaad/Static"
OUT = os.path.expanduser("~/ped-cv/runs/seeds")
os.makedirs(OUT, exist_ok=True)

for s in [1, 2, 3, 4, 5]:
    if os.path.exists(f"{OUT}/seed_{s}/jaad_beh_results.yaml"):
        print("skip seed", s)
        continue
    before = set(os.listdir(BASE))
    env = dict(os.environ, SEED=str(s))
    with open(f"{OUT}/seed_{s}.log", "w") as log:
        r = subprocess.run(
            [sys.executable, "seed_run.py", "config_files/Static_vgg16_jaad.yaml"],
            env=env, stdout=log, stderr=subprocess.STDOUT)
    new = [d for d in set(os.listdir(BASE)) - before
           if os.path.exists(f"{BASE}/{d}/results.yaml")]
    new.sort(key=lambda d: os.path.getmtime(f"{BASE}/{d}/results.yaml"))
    if r.returncode != 0 or len(new) != 2:
        print("problem with seed", s, "returncode", r.returncode, "new folders", new)
        sys.exit(1)
    os.makedirs(f"{OUT}/seed_{s}", exist_ok=True)
    for name, d in zip(["jaad_all", "jaad_beh"], new):
        shutil.copy(f"{BASE}/{d}/results.yaml", f"{OUT}/seed_{s}/{name}_results.yaml")
        with open(f"{OUT}/seed_{s}/{name}_folder.txt", "w") as f:
            f.write(d)
    print("seed", s, "done", new, flush=True)
