"""
experiment.py - "Two questions to name a penguin" (reel, 2026-11-11).

How many yes/no questions does it take to name a
penguin's species? A plain-numpy CART tree (Gini impurity, axis-aligned
splits at midpoints) on the Palmer Penguins (bill length, bill depth,
flipper length, body mass), 240 to learn from and 102 held back (seed 0, the
same split as the other penguin reels). Depth 1, 2, 3 and unlimited.

    python experiment.py          -> out/results.json  (downloads the data on first run)
"""
import csv, json, os, sys, urllib.request
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)
SRC = os.path.join(HERE, "download", "penguins.csv")
if not os.path.exists(SRC):
    os.makedirs(os.path.dirname(SRC), exist_ok=True)
    urllib.request.urlretrieve("https://raw.githubusercontent.com/allisonhorst/palmerpenguins/"
                               "main/inst/extdata/penguins.csv", SRC)
FEAT = ["bill length (mm)", "bill depth (mm)", "flipper (mm)", "body mass (g)"]
COLS = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]


def gini(y, k=3):
    if len(y) == 0:
        return 0.0
    p = np.bincount(y, minlength=k) / len(y)
    return 1 - (p * p).sum()


def best_split(X, y):
    best = (None, None, gini(y))
    for j in range(X.shape[1]):
        vals = np.unique(X[:, j])
        for a, b in zip(vals[:-1], vals[1:]):
            t = (a + b) / 2
            L = X[:, j] <= t
            g = (L.sum() * gini(y[L]) + (~L).sum() * gini(y[~L])) / len(y)
            if g < best[2] - 1e-12:
                best = (j, float(t), g)
    return best


def grow(X, y, depth, max_depth):
    counts = np.bincount(y, minlength=3).tolist()
    node = {"n": len(y), "counts": counts, "label": int(np.argmax(counts))}
    if (max_depth is not None and depth >= max_depth) or gini(y) == 0:
        return node
    j, t, g = best_split(X, y)
    if j is None:
        return node
    L = X[:, j] <= t
    node.update(feature=j, threshold=t, left=grow(X[L], y[L], depth + 1, max_depth),
                right=grow(X[~L], y[~L], depth + 1, max_depth))
    return node


def predict(node, x):
    while "feature" in node:
        node = node["left"] if x[node["feature"]] <= node["threshold"] else node["right"]
    return node["label"]


def stats(node, depth=0):
    if "feature" not in node:
        return 1, depth
    a, da = stats(node["left"], depth + 1)
    b, db = stats(node["right"], depth + 1)
    return a + b, max(da, db)


def main(src=SRC):
    R = [r for r in csv.DictReader(open(src, encoding="utf-8")) if all(r[c] != "NA" for c in COLS)]
    SP = ["Adelie", "Chinstrap", "Gentoo"]
    X = np.array([[float(r[c]) for c in COLS] for r in R])
    y = np.array([SP.index(r["species"]) for r in R])
    idx = np.random.default_rng(0).permutation(len(R))
    tr, te = idx[:240], idx[240:]
    out = {"n": len(R), "n_train": len(tr), "n_test": len(te), "species": SP, "features": FEAT,
           "counts_train": np.bincount(y[tr], minlength=3).tolist(), "trees": {}}
    for d in (1, 2, 3, None):
        T = grow(X[tr], y[tr], 0, d)
        acc = lambda ix: float(np.mean([predict(T, X[i]) == y[i] for i in ix]))
        leaves, depth = stats(T)
        key = "unlimited" if d is None else str(d)
        out["trees"][key] = {"tree": T, "train_acc": acc(tr), "test_acc": acc(te), "leaves": leaves, "depth": depth,
                             "test_right": int(round(acc(te) * len(te)))}
        print("depth %-9s leaves %2d  homework %.1f%%  new penguins %.1f%% (%d of %d)"
              % (key, leaves, 100 * acc(tr), 100 * acc(te), out["trees"][key]["test_right"], len(te)))
    def show(n, ind=""):
        if "feature" not in n:
            print(ind + "-> %s %s" % (SP[n["label"]], n["counts"])); return
        print(ind + "%s <= %.1f ?  %s" % (FEAT[n["feature"]], n["threshold"], n["counts"]))
        show(n["left"], ind + "  yes "); show(n["right"], ind + "  no  ")
    show(out["trees"]["3"]["tree"])
    json.dump(out, open(os.path.join(OUT, "results.json"), "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:2])
