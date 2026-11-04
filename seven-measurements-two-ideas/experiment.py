"""PCA - UCI Seeds: 210 wheat kernels (Kama, Rosa, Canadian; 70 each), 7 measurements
from soft X-ray images (Institute of Agrophysics, Lublin). CC BY 4.0.

Measures: the 7 columns' correlations, PCA on standardised columns (numpy eigh),
the share of the spread each direction holds, what the first two directions are
made of, and how well the varieties separate: leave-one-out nearest-centre
accuracy using all 7 columns, the best 2 raw columns, PC1+PC2, and PC1 alone.

    python experiment.py
"""
import itertools
import json
import os
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(HERE, "download"), exist_ok=True)
os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
TXT = os.path.join(HERE, "download", "seeds_dataset.txt")
if not os.path.exists(TXT):
    urllib.request.urlretrieve("https://archive.ics.uci.edu/ml/machine-learning-databases/00236/seeds_dataset.txt", TXT)
# some rows carry doubled tabs: split on any whitespace
rows = [l.split() for l in open(TXT) if l.strip()]
A = np.array(rows, dtype=float)
X, y = A[:, :7], A[:, 7].astype(int)
NAMES = ["area", "perimeter", "compactness", "length", "width", "asymmetry", "groove length"]
VARIETY = {1: "Kama", 2: "Rosa", 3: "Canadian"}

Z = (X - X.mean(0)) / X.std(0)
vals, vecs = np.linalg.eigh(np.cov(Z.T))
order = vals.argsort()[::-1]
vals, vecs = vals[order], vecs[:, order]
vecs *= np.sign(vecs[0])                       # PC signs: area loads positive
P = Z @ vecs


def loo_nearest_centre(F):
    ok = 0
    for i in range(len(F)):
        m = np.ones(len(F), bool)
        m[i] = False
        cents = {c: F[m & (y == c)].mean(0) for c in VARIETY}
        guess = min(cents, key=lambda c: np.sum((F[i] - cents[c]) ** 2))
        ok += guess == y[i]
    return ok / len(F)


out = {"n": len(y), "per_variety": {VARIETY[c]: int((y == c).sum()) for c in VARIETY},
       "share": [round(float(v / vals.sum()), 4) for v in vals],
       "pc1": dict(zip(NAMES, np.round(vecs[:, 0], 2).tolist())),
       "pc2": dict(zip(NAMES, np.round(vecs[:, 1], 2).tolist())),
       "corr_area_perimeter": round(float(np.corrcoef(X[:, 0], X[:, 1])[0, 1]), 3),
       "corr_area_length": round(float(np.corrcoef(X[:, 0], X[:, 3])[0, 1]), 3),
       "corr_area_width": round(float(np.corrcoef(X[:, 0], X[:, 4])[0, 1]), 3)}
out["acc_all7"] = loo_nearest_centre(Z)
out["acc_pc12"] = loo_nearest_centre(P[:, :2])
out["acc_pc1"] = loo_nearest_centre(P[:, :1])
pairs = {(NAMES[a], NAMES[b]): loo_nearest_centre(Z[:, [a, b]]) for a, b in itertools.combinations(range(7), 2)}
best = max(pairs, key=pairs.get)
worst = min(pairs, key=pairs.get)
out["best_raw_pair"] = dict(cols=best, acc=pairs[best])
out["worst_raw_pair"] = dict(cols=worst, acc=pairs[worst])
out["acc_area_alone"] = loo_nearest_centre(Z[:, [0]])
out["means_by_variety"] = {VARIETY[c]: dict(zip(NAMES, np.round(X[y == c].mean(0), 3).tolist())) for c in VARIETY}
out["pc_means_by_variety"] = {VARIETY[c]: np.round(P[y == c, :2].mean(0), 2).tolist() for c in VARIETY}
print(json.dumps(out, indent=1, default=str))
json.dump(out, open(os.path.join(HERE, "out", "results.json"), "w"), indent=1, default=str)
np.save(os.path.join(HERE, "out", "pc12.npy"), np.column_stack([P[:, :2], y]))
