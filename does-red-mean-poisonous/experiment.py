"""
experiment.py - how does a decision tree choose its question? UCI Mushroom data, plain numpy.

Data: UCI Mushroom (Schlimmer 1987, from The Audubon Society Field Guide to North American
Mushrooms, 1981), 8,124 mushrooms x 22 described features, edible or poisonous. CC BY 4.0.
https://archive.ics.uci.edu/dataset/73/mushroom

A plain CART tree: every question has the form "is <feature> <value>?"; the tree tries every one,
and keeps the one whose two groups are least MIXED. Mixed = pick two mushrooms from a group at
random: how often is one edible and the other poisonous? (Gini impurity, 2p(1-p) for two classes.)

    python experiment.py --out out
"""
import os, json, argparse, urllib.request
import numpy as np

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"
FEAT = ["cap-shape", "cap-surface", "cap-color", "bruises", "odor", "gill-attachment", "gill-spacing",
        "gill-size", "gill-color", "stalk-shape", "stalk-root", "stalk-surface-above-ring",
        "stalk-surface-below-ring", "stalk-color-above-ring", "stalk-color-below-ring", "veil-type",
        "veil-color", "ring-number", "ring-type", "spore-print-color", "population", "habitat"]
NAMES = {
    "odor": dict(a="almond", l="anise", c="creosote", y="fishy", f="foul", m="musty", n="none", p="pungent", s="spicy"),
    "spore-print-color": dict(k="black", n="brown", b="buff", h="chocolate", r="green", o="orange", u="purple", w="white", y="yellow"),
    "cap-shape": dict(b="bell", c="conical", x="convex", f="flat", k="knobbed", s="sunken"),
    "cap-color": dict(n="brown", b="buff", c="cinnamon", g="gray", r="green", p="pink", u="purple", e="red", w="white", y="yellow"),
    "gill-size": dict(b="broad", n="narrow"), "bruises": dict(t="bruises", f="no"),
    "gill-color": dict(k="black", n="brown", b="buff", h="chocolate", g="gray", r="green", o="orange", p="pink", u="purple", e="red", w="white", y="yellow"),
    "habitat": dict(g="grasses", l="leaves", m="meadows", p="paths", u="urban", w="waste", d="woods"),
    "population": dict(a="abundant", c="clustered", n="numerous", s="scattered", v="several", y="solitary"),
    "stalk-surface-below-ring": dict(f="fibrous", y="scaly", k="silky", s="smooth"),
    "stalk-surface-above-ring": dict(f="fibrous", y="scaly", k="silky", s="smooth"),
    "ring-type": dict(c="cobwebby", e="evanescent", f="flaring", l="large", n="none", p="pendant", s="sheathing", z="zone"),
    "stalk-root": {"b": "bulbous", "c": "club", "u": "cup", "e": "equal", "z": "rhizomorphs", "r": "rooted", "?": "missing"},
}

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="download/agaricus-lepiota.data")
ap.add_argument("--out", default="out")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
if not os.path.exists(a.data):
    os.makedirs(os.path.dirname(a.data) or ".", exist_ok=True)
    urllib.request.urlretrieve(URL, a.data)

rows = [l.strip().split(",") for l in open(a.data) if l.strip()]
y = np.array([r[0] == "p" for r in rows], int)          # 1 = poisonous
X = np.array([r[1:] for r in rows])
N = len(y)


def name(f, v):
    return NAMES.get(FEAT[f], {}).get(v, v)


def mixed(yy):
    """chance two mushrooms drawn at random (with replacement) disagree"""
    if len(yy) == 0:
        return 0.0
    p = yy.mean()
    return 2 * p * (1 - p)


def split_score(idx, f, v):
    m = X[idx, f] == v
    a_, b_ = idx[m], idx[~m]
    return (len(a_) * mixed(y[a_]) + len(b_) * mixed(y[b_])) / len(idx), a_, b_


def best_question(idx):
    best = None
    for f in range(X.shape[1]):
        for v in np.unique(X[idx, f]):
            s, a_, b_ = split_score(idx, f, v)
            if len(a_) and len(b_) and (best is None or s < best[0] - 1e-15):
                best = (s, f, v, a_, b_)
    return best


def counts(idx):
    return {"edible": int((y[idx] == 0).sum()), "poisonous": int((y[idx] == 1).sum())}


R = {"mushrooms": N, **counts(np.arange(N)), "features": len(FEAT),
     "start_mixed": round(mixed(y), 4)}

# --- 1. every question on all 8,124: how mixed are the two groups it makes? -------------------
allq = []
for f in range(X.shape[1]):
    for v in np.unique(X[:, f]):
        s, a_, b_ = split_score(np.arange(N), f, v)
        if len(a_) and len(b_):
            allq.append((s, FEAT[f], name(f, v), counts(a_), counts(b_)))
allq.sort(key=lambda t: t[0])
R["questions_tried"] = len(allq)
R["best_questions"] = [{"q": "%s = %s" % (q[1], q[2]), "mixed_after": round(q[0], 4), "yes": q[3], "no": q[4]}
                       for q in allq[:8]]
for want in [("cap-shape", "flat"), ("cap-color", "red"), ("habitat", "woods"), ("bruises", "bruises"),
             ("gill-size", "broad")]:
    for q in allq:
        if (q[1], q[2]) == want:
            R.setdefault("other_questions", []).append(
                {"q": "%s = %s" % want, "mixed_after": round(q[0], 4), "yes": q[3], "no": q[4]})
R["worst_question"] = {"q": "%s = %s" % (allq[-1][1], allq[-1][2]), "mixed_after": round(allq[-1][0], 4)}

# odor, value by value (the whole feature at a glance)
R["odor_groups"] = {name(4, v): counts(np.where(X[:, 4] == v)[0]) for v in np.unique(X[:, 4])}

# --- 2. learn on 80%, test on 20% -----------------------------------------------------------
rng = np.random.default_rng(a.seed)
p = rng.permutation(N)
n_tr = int(0.8 * N)
tr, te = p[:n_tr], p[n_tr:]


def grow(idx, depth, max_depth, path, out):
    c = counts(idx)
    leaf = {"depth": depth, "path": path, **c, "says": "poisonous" if c["poisonous"] > c["edible"] else "edible"}
    if c["edible"] == 0 or c["poisonous"] == 0 or depth == max_depth:
        out.append(leaf)
        return ("leaf", int(c["poisonous"] > c["edible"]))
    s, f, v, a_, b_ = best_question(idx)
    q = "%s = %s" % (FEAT[f], name(f, v))
    return ("q", f, v, grow(a_, depth + 1, max_depth, path + [q + "? yes"], out),
            grow(b_, depth + 1, max_depth, path + [q + "? no"], out))


def predict(node, x):
    while node[0] == "q":
        node = node[3] if x[node[1]] == node[2] else node[4]
    return node[1]


def acc(node, idx):
    pr = np.array([predict(node, X[i]) for i in idx])
    return int((pr == y[idx]).sum())


R["split"] = {"seed": a.seed, "learn": len(tr), "test": len(te), "test_counts": counts(te)}
by_depth = []
for d in [0, 1, 2, 3, 4, 5, 6, 99]:
    leaves = []
    t = grow(tr, 0, d, [], leaves)
    wrong_te = len(te) - acc(t, te)
    # the costly error: poisonous called edible
    pe = sum(1 for i in te if y[i] == 1 and predict(t, X[i]) == 0)
    by_depth.append({"questions_deep": d if d < 99 else "until pure", "end_points": len(leaves),
                     "learn_right": acc(t, tr), "test_right": len(te) - wrong_te, "test_wrong": wrong_te,
                     "poison_called_edible": pe})
    if d == 99:
        R["full_tree"] = {"end_points": len(leaves), "deepest": max(l["depth"] for l in leaves),
                          "leaves": leaves}
R["by_depth"] = by_depth

# the first three questions the tree learned, with the counts at each node (on the learning 80%)
first = best_question(tr)
R["q1"] = {"q": "%s = %s" % (FEAT[first[1]], name(first[1], first[2])), "yes": counts(first[3]), "no": counts(first[4]),
           "mixed_yes": round(mixed(y[first[3]]), 4), "mixed_no": round(mixed(y[first[4]]), 4)}

# the same questions, asked of the learning pile only (what the tree actually sees)
lq = []
for f in range(X.shape[1]):
    for v in np.unique(X[tr, f]):
        s_, a_, b_ = split_score(tr, f, v)
        if len(a_) and len(b_):
            lq.append((s_, "%s = %s" % (FEAT[f], name(f, v)), counts(a_), counts(b_)))
lq.sort(key=lambda t: t[0])
R["learn_questions"] = {"tried": len(lq), "start_mixed": round(mixed(y[tr]), 4), "learn_counts": counts(tr),
                        "top": [{"q": q[1], "mixed_after": round(q[0], 4), "yes": q[2], "no": q[3]} for q in lq[:6]],
                        "rank_of": {q[1]: i + 1 for i, q in enumerate(lq)},
                        "picked": {q[1]: {"mixed_after": round(q[0], 4), "yes": q[2], "no": q[3]} for q in lq
                                   if q[1] in ("cap-color = red", "habitat = woods", "bruises = bruises",
                                               "odor = foul", "odor = none", "cap-shape = flat", "gill-size = broad")}}
leaves3 = []
grow(tr, 0, 3, [], leaves3)
R["tree3_leaves"] = leaves3

# --- 3. the honest turn: are the test mushrooms new? -----------------------------------------
trset = {tuple(r) for r in X[tr]}
exact = sum(1 for i in te if tuple(X[i]) in trset)
Xtr = X[tr]
diff = np.array([(Xtr != X[i]).sum(1).min() for i in te])
R["test_vs_learn"] = {"test": len(te), "identical_twin_in_learning": int(exact),
                      "nearest_differs_by_hist": {int(k): int((diff == k).sum()) for k in np.unique(diff)},
                      "distinct_rows_all": len({tuple(r) for r in X}),
                      "species_in_book": 23}

json.dump(R, open(os.path.join(a.out, "results.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in R.items() if k != "full_tree"}, indent=1))
print("full tree:", R["full_tree"]["end_points"], "end points, deepest", R["full_tree"]["deepest"])
for l in R["full_tree"]["leaves"]:
    print("  ", " / ".join(l["path"]), "->", l["edible"], l["poisonous"])
