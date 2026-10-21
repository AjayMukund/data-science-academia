"""
experiment.py - "Never grade on the homework" (reel, 2026-11-02).

A memoriser (1-nearest-neighbour: copy the weight of the most similar penguin
it has already seen) against a plain straight line, both predicting a
penguin's weight from bill length, bill depth and flipper length. Each is
scored on the penguins it learned from (the homework) and on penguins it
never saw (the test). Real Palmer Penguins data (CC0), plain numpy.

    python experiment.py          -> out/results.json
"""
import os, sys, csv, json, urllib.request
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)
SRC = os.path.join(HERE, "download", "penguins.csv")
if not os.path.exists(SRC):
    os.makedirs(os.path.dirname(SRC), exist_ok=True)
    urllib.request.urlretrieve("https://raw.githubusercontent.com/allisonhorst/palmerpenguins/"
                               "main/inst/extdata/penguins.csv", SRC)

R = [r for r in csv.DictReader(open(SRC, encoding="utf-8")) if "NA" not in (r["bill_length_mm"], r["body_mass_g"])]
F = np.array([[float(r[k]) for k in ("bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g")] for r in R])
X3, M = F[:, :3], F[:, 3]
N_TRAIN = 240                       # 240 to learn from, the other 102 held back


def split(seed):
    idx = np.random.default_rng(seed).permutation(len(R))
    return idx[:N_TRAIN], idx[N_TRAIN:]


def scores(seed):
    tr, te = split(seed)
    Z = (X3 - X3[tr].mean(0)) / X3[tr].std(0)                 # scale, so "similar" weighs all three
    nearest = lambda A: M[tr][((A[:, None] - Z[tr][None]) ** 2).sum(-1).argmin(1)]
    A = np.c_[X3, np.ones(len(R))]
    w = np.linalg.lstsq(A[tr], M[tr], rcond=None)[0]
    mae = lambda p, idx: float(np.abs(p - M[idx]).mean())
    return {"memoriser_homework_g": mae(nearest(Z[tr]), tr), "memoriser_test_g": mae(nearest(Z[te]), te),
            "line_homework_g": mae(A[tr] @ w, tr), "line_test_g": mae(A[te] @ w, te),
            "n_train": len(tr), "n_test": len(te)}


runs = [scores(s) for s in range(5)]
json.dump({"n_penguins": len(R), "splits": runs}, open(os.path.join(OUT, "results.json"), "w"), indent=1)
print("%d penguins, %d to learn from, %d held back\n" % (len(R), runs[0]["n_train"], runs[0]["n_test"]))
print("split  memoriser: homework  test   |  line: homework  test")
for s, r in enumerate(runs):
    print("  %d        %4.0f g     %4.0f g   |        %4.0f g   %4.0f g"
          % (s, r["memoriser_homework_g"], r["memoriser_test_g"], r["line_homework_g"], r["line_test_g"]))
