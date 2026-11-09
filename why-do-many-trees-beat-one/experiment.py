"""
experiment.py - why do many trees beat one? A random forest, measured on the Titanic.

Plain numpy (no scikit-learn): a CART tree (Gini, binary splits) and a forest of them, each tree grown on a
bootstrap sample of the passengers and choosing each question from a random 3 of the 7 features.
Data: Titanic passengers, 891 (Kaggle/Stanford train.csv, public), features: class, sex, age (missing -> -1),
siblings/spouses, parents/children, fare, port. Target: survived.

  python experiment.py --data titanic.csv --out out
"""
import argparse, csv, json, os
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"))
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

FEATS = ["class", "sex", "age", "sib_sp", "par_ch", "fare", "port"]
rows = list(csv.DictReader(open(a.data, encoding="utf-8")))
X = np.array([[float(r["Pclass"]), 1.0 if r["Sex"] == "female" else 0.0, float(r["Age"]) if r["Age"] else -1.0,
               float(r["SibSp"]), float(r["Parch"]), float(r["Fare"]), {"S": 0, "C": 1, "Q": 2}.get(r["Embarked"], 0)]
              for r in rows])
y = np.array([int(r["Survived"]) for r in rows])
names = [r["Name"] for r in rows]
N, F = X.shape


def gini(c1, n):
    p = c1 / np.maximum(n, 1)
    return 2 * p * (1 - p)


def best_split(Xs, ys, feats):
    best = (None, None, gini(ys.sum(), len(ys)))
    n = len(ys)
    for f in feats:
        order = np.argsort(Xs[:, f], kind="stable")
        xv, yv = Xs[order, f], ys[order]
        c1 = np.cumsum(yv)[:-1]
        nl = np.arange(1, n)
        ok = xv[1:] != xv[:-1]                       # only cut between different values
        if not ok.any():
            continue
        g = (nl * gini(c1, nl) + (n - nl) * gini(yv.sum() - c1, n - nl)) / n
        g = np.where(ok, g, np.inf)
        i = int(np.argmin(g))
        if g[i] < best[2] - 1e-12:
            best = (f, (xv[i] + xv[i + 1]) / 2, g[i])
    return best


def grow(Xs, ys, rng=None, mtry=None, depth=0, max_depth=None):
    if ys.min() == ys.max() or (max_depth is not None and depth >= max_depth) or len(ys) < 2:
        return float(ys.mean())
    feats = range(F) if mtry is None else rng.choice(F, mtry, replace=False)
    f, t, _ = best_split(Xs, ys, feats)
    if f is None:
        return float(ys.mean())
    m = Xs[:, f] <= t
    return (f, t, grow(Xs[m], ys[m], rng, mtry, depth + 1, max_depth), grow(Xs[~m], ys[~m], rng, mtry, depth + 1, max_depth))


def prob(node, x):
    while isinstance(node, tuple):
        f, t, l, r = node
        node = l if x[f] <= t else r
    return node


def leaves(node):
    return 1 if not isinstance(node, tuple) else leaves(node[2]) + leaves(node[3])


def tree_acc(node, idx):
    return float(np.mean([(prob(node, X[i]) >= 0.5) == y[i] for i in idx]))


def forest(idx, n_trees, rng, mtry=3):
    out = []
    for _ in range(n_trees):
        b = rng.choice(idx, len(idx), replace=True)
        out.append(grow(X[b], y[b], rng, mtry))
    return out


def forest_votes(trees, i):
    return np.array([prob(t, X[i]) >= 0.5 for t in trees])


res = {"passengers": N, "features": FEATS, "survived": int(y.sum())}
rng = np.random.default_rng(0)
perm = rng.permutation(N)
test, train = perm[:179], perm[179:]                    # an 80/20 split: 712 to learn from, 179 kept back
res["split"] = {"train": len(train), "test": len(test)}

# 1. one tree, grown until every leaf is pure
t1 = grow(X[train], y[train])
res["one_tree"] = {"leaves": leaves(t1), "train_acc": tree_acc(t1, train), "test_acc": tree_acc(t1, test)}


def top(node, idx, depth):
    """The tree's first questions, as asked of the training passengers."""
    if not isinstance(node, tuple) or depth == 0:
        return {"n": int(len(idx)), "survived": int(y[idx].sum())}
    f, t, l, r = node
    m = X[idx, f] <= t
    return {"feature": FEATS[f], "threshold": float(t), "n": int(len(idx)), "survived": int(y[idx].sum()),
            "yes": top(l, idx[m], depth - 1), "no": top(r, idx[~m], depth - 1)}


def path(node, x):
    steps = []
    while isinstance(node, tuple):
        f, t, l, r = node
        steps.append({"feature": FEATS[f], "threshold": float(t), "value": float(x[f]), "le": bool(x[f] <= t)})
        node = l if x[f] <= t else r
    return steps, node


res["one_tree"]["top"] = top(t1, train, 3)
ANNA = next(i for i, n in enumerate(names) if n.startswith("Salkjelsvik"))
steps, leaf = path(t1, X[ANNA])
res["anna"] = {"i": ANNA, "name": names[ANNA], "in_test": bool(ANNA in set(test.tolist())), "survived": int(y[ANNA]),
               "deep_tree_path": steps, "deep_tree_leaf": leaf, "questions_asked": len(steps)}


def follows(i, steps):
    return all((X[i, FEATS.index(s["feature"])] <= s["threshold"]) == s["le"] for s in steps)


in_leaf = [int(i) for i in train if follows(i, steps)]
in_group = [int(i) for i in train if follows(i, steps[:3])]          # female, third class, fare under the cut
res["anna"]["leaf_train_passengers"] = len(in_leaf)
res["anna"]["leaf_train_survived"] = int(y[in_leaf].sum())
res["anna"]["leaf_train_ages"] = sorted(float(X[i, 2]) for i in in_leaf)
res["anna"]["group_train_passengers"] = len(in_group)
res["anna"]["group_train_survived"] = int(y[in_group].sum())
print("one deep tree: %d leaves, train %.3f, test %.3f" % (leaves(t1), res["one_tree"]["train_acc"], res["one_tree"]["test_acc"]))
for d in (1, 2, 3, 4, 6, 8):
    td = grow(X[train], y[train], max_depth=d)
    res.setdefault("by_depth", {})[d] = {"train": tree_acc(td, train), "test": tree_acc(td, test)}

# 2. ten trees, each from a different resample of the same 712: how much do they disagree?
rng = np.random.default_rng(1)
ten = []
for _ in range(10):
    b = rng.choice(train, len(train), replace=True)
    ten.append(grow(X[b], y[b]))
res["ten_trees_test_acc"] = [tree_acc(t, test) for t in ten]
votes10 = np.array([[prob(t, X[i]) >= 0.5 for t in ten] for i in test])
split_votes = [(int(test[k]), int(v.sum())) for k, v in enumerate(votes10) if 3 <= v.sum() <= 7]
res["ten_trees_split_passengers"] = len(split_votes)
print("ten resampled trees: test acc", [round(v, 3) for v in res["ten_trees_test_acc"]], "| split 3-7 on", len(split_votes), "of", len(test))

# 3. a forest of 100 (bootstrap + 3 random features per question)
rng = np.random.default_rng(2)
trees = forest(train, 200, rng)
curve = {}
for k in (1, 3, 5, 10, 25, 50, 100, 200):
    acc = np.mean([(forest_votes(trees[:k], i).mean() >= 0.5) == y[i] for i in test])
    curve[k] = float(acc)
res["forest_curve_test_acc"] = curve
f100 = trees[:100]
res["forest100"] = {"test_acc": curve[100], "train_acc": float(np.mean([(forest_votes(f100, i).mean() >= 0.5) == y[i] for i in train]))}
print("forest test acc by size:", {k: round(v, 3) for k, v in curve.items()})

# the running example: a passenger the single trees argued about, and what the forest said
ex = []
for i, s in split_votes:
    v = forest_votes(f100, i)
    ex.append({"i": i, "name": names[i], "class": int(X[i, 0]), "sex": "female" if X[i, 1] else "male",
               "age": None if X[i, 2] < 0 else X[i, 2], "fare": X[i, 5], "survived": int(y[i]),
               "ten_trees_say_survive": s, "forest_say_survive": int(v.sum()), "forest_right": bool((v.mean() >= 0.5) == y[i]),
               "deep_tree_right": bool((prob(t1, X[i]) >= 0.5) == y[i])})
res["argued_passengers"] = ex

# 4. twenty different splits: how much does the score itself move?
one, many = [], []
for s in range(20):
    r = np.random.default_rng(100 + s)
    p = r.permutation(N); te, tr = p[:179], p[179:]
    one.append(tree_acc(grow(X[tr], y[tr]), te))
    fr = forest(tr, 100, r)
    many.append(float(np.mean([(forest_votes(fr, i).mean() >= 0.5) == y[i] for i in te])))
    print("split %2d: one tree %.3f, forest %.3f" % (s, one[-1], many[-1]), flush=True)
res["twenty_splits"] = {"one_tree": one, "forest": many,
                        "one_mean": float(np.mean(one)), "one_min": float(min(one)), "one_max": float(max(one)),
                        "forest_mean": float(np.mean(many)), "forest_min": float(min(many)), "forest_max": float(max(many)),
                        "forest_better": int(sum(m > o for o, m in zip(one, many)))}

# 5. which questions the forest leans on (how often each feature is the first question)
firsts = np.bincount([t[0] for t in f100 if isinstance(t, tuple)], minlength=F)
res["first_question_counts"] = {FEATS[k]: int(firsts[k]) for k in range(F)}
json.dump(res, open(os.path.join(a.out, "results.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("DONE", res["twenty_splits"]["one_mean"], res["twenty_splits"]["forest_mean"])
