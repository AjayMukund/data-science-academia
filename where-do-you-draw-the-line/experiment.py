"""Thresholds - Wisconsin breast biopsies (UCI WDBC, CC BY 4.0, 569 samples).

Each sample is a digitised fine-needle biopsy: 30 measurements of its cell nuclei,
labelled M (malignant, 212) or B (benign, 357). A logistic model on three of them
(largest nucleus area, roughest outline - "worst concave points" -, worst texture)
gives every biopsy a score from 0 to 1. Scores are out-of-fold (10-fold, seed 0):
each biopsy is scored by a model that never saw it.

Then the dial: for each threshold, cancers caught (of 212) and healthy flagged
(of 357). Also the single-feature version (largest nucleus area alone) and AUC.

    python experiment.py
"""
import csv
import json
import os
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(HERE, "download"), exist_ok=True)
os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
F = os.path.join(HERE, "download", "wdbc.data")
if not os.path.exists(F):
    urllib.request.urlretrieve("https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data", F)
rows = list(csv.reader(open(F)))
y = np.array([r[1] == "M" for r in rows], dtype=float)
X30 = np.array([[float(v) for v in r[2:]] for r in rows])
# column order (wdbc.names): 10 means, 10 standard errors, 10 "worst"; within each:
# radius texture perimeter area smoothness compactness concavity concave_points symmetry fractal
WORST = 20
cols = {"worst_area": WORST + 3, "worst_concave_points": WORST + 7, "worst_texture": WORST + 1}
X = X30[:, list(cols.values())]


def logistic(X, y, steps=4000, lr=0.1):
    mu, sd = X.mean(0), X.std(0)
    Z = np.column_stack([np.ones(len(X)), (X - mu) / sd])
    w = np.zeros(Z.shape[1])
    for _ in range(steps):
        p = 1 / (1 + np.exp(-Z @ w))
        w -= lr * Z.T @ (p - y) / len(y)
    return lambda Xn: 1 / (1 + np.exp(-np.column_stack([np.ones(len(Xn)), (Xn - mu) / sd]) @ w))


def oof(X, y, k=10, seed=0):
    idx = np.random.default_rng(seed).permutation(len(y))
    s = np.zeros(len(y))
    for f in range(k):
        te = idx[f::k]
        tr = np.setdiff1d(idx, te)
        s[te] = logistic(X[tr], y[tr])(X[te])
    return s


def table(s):
    out = []
    for t in (0.9, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01):
        flag = s >= t
        out.append(dict(t=t, caught=int((flag & (y == 1)).sum()), missed=int((~flag & (y == 1)).sum()),
                        healthy_flagged=int((flag & (y == 0)).sum())))
    # the lowest threshold that still catches every cancer
    t_all = float(s[y == 1].min())
    out_all = dict(t=t_all, healthy_flagged=int(((s >= t_all) & (y == 0)).sum()))
    # AUC: chance a random cancer outscores a random benign
    pos, neg = s[y == 1], s[y == 0]
    auc = float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())
    return out, out_all, auc


res = {"n": len(y), "malignant": int(y.sum()), "benign": int((1 - y).sum())}
s = oof(X, y)
res["three"] = dict(zip(("thresholds", "catch_all", "auc"), table(s)))
s1 = oof(X30[:, [cols["worst_area"]]], y)
res["area_only"] = dict(zip(("thresholds", "catch_all", "auc"), table(s1)))
# every biopsy's out-of-fold score and label, for ranking every biopsy
res["scores"] = [round(float(v), 6) for v in s]
res["labels"] = y.astype(int).tolist()
# score histogram (10 bins) per class, for the two hills
res["hist"] = {c: np.histogram(s[y == v], bins=10, range=(0, 1))[0].tolist() for c, v in (("M", 1), ("B", 0))}
print(json.dumps({k: v for k, v in res.items() if k not in ("scores", "labels")}, indent=1))
json.dump(res, open(os.path.join(HERE, "out", "results.json"), "w"), indent=1)
