"""
experiment.py - how does a model learn from its own mistakes? Gradient boosting on California house prices.

Data: California Housing (Pace & Barry 1997, from the 1990 US census; StatLib, public): 20,640 districts, each with
its median house value and 8 facts (income, age of houses, rooms, people, location). Mirror used by scikit-learn:
https://ndownloader.figshare.com/files/5976036 -> data/CaliforniaHousing/cal_housing.data
Values are capped at $500,001 in the data itself.

Everything in plain numpy: small regression trees on binned features (64 bins per feature).
  1. Guess the average for every district: how far off, on average (mean absolute error, in dollars).
  2. One small tree (3 questions deep).
  3. Boosting: each new small tree is fitted to the mistakes (residuals) of all the trees so far, and added at a
     tenth of its size (learning rate 0.1). Error after 1, 2, 3, 5, 10, 30, 100, 300, 1000 trees, on districts it
     trained on and on 4,128 it never saw.
  4. One district followed tree by tree.
  5. Contrast: one deep tree; a random forest (100 deep trees, each on a bootstrap sample, averaged).
  6. Too eager: learning rate 1.0 (each tree added at full size).
Seeded (numpy 0). Writes out/results.json.   python experiment.py
"""
import json, os, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
raw = np.loadtxt(os.path.join(HERE, "data", "CaliforniaHousing", "cal_housing.data"), delimiter=",")
lon, lat, age, rooms, beds, pop, hh, inc, val = raw.T
NAMES = ["income", "house age", "rooms per home", "bedrooms per room", "people per home", "longitude", "latitude",
         "homes in district"]
X = np.column_stack([inc, age, rooms / hh, beds / rooms, pop / hh, lon, lat, hh])
y = val
rng = np.random.default_rng(0)
perm = rng.permutation(len(y))
te, tr = perm[:4128], perm[4128:]

# bin every feature into 64 quantile bins (on the training rows)
B = 64
edges = [np.unique(np.quantile(X[tr, f], np.linspace(0, 1, B + 1)[1:-1])) for f in range(X.shape[1])]
Xb = np.column_stack([np.searchsorted(edges[f], X[:, f], side="right") for f in range(X.shape[1])]).astype(np.int16)


def fit_tree(rows, target, depth, min_leaf=20, feats=None):
    """greedy least-squares tree on binned features; returns nested dict"""
    t = target[rows]
    node = {"value": float(t.mean()), "n": int(len(rows))}
    if depth == 0 or len(rows) < 2 * min_leaf:
        return node
    best = None
    fs = range(X.shape[1]) if feats is None else feats(rng)
    for f in fs:
        b = Xb[rows, f]
        cnt = np.bincount(b, minlength=B)
        sm = np.bincount(b, weights=t, minlength=B)
        cl, sl = np.cumsum(cnt)[:-1], np.cumsum(sm)[:-1]
        cr, sr = len(t) - cl, t.sum() - sl
        ok = (cl >= min_leaf) & (cr >= min_leaf)
        if not ok.any():
            continue
        gain = np.where(ok, sl ** 2 / np.maximum(cl, 1) + sr ** 2 / np.maximum(cr, 1), -np.inf)
        k = int(np.argmax(gain))
        if best is None or gain[k] > best[0]:
            best = (gain[k], f, k)
    if best is None:
        return node
    _, f, k = best
    left = rows[Xb[rows, f] <= k]
    right = rows[Xb[rows, f] > k]
    node.update(feature=int(f), bin=int(k),
                threshold=float(edges[f][k]) if k < len(edges[f]) else float("inf"),
                left=fit_tree(left, target, depth - 1, min_leaf, feats), right=fit_tree(right, target, depth - 1, min_leaf, feats))
    return node


def predict(node, rows):
    out = np.empty(len(rows))
    stack = [(node, np.arange(len(rows)))]
    while stack:
        n, ix = stack.pop()
        if "feature" not in n:
            out[ix] = n["value"]
            continue
        go = Xb[rows[ix], n["feature"]] <= n["bin"]
        stack += [(n["left"], ix[go]), (n["right"], ix[~go])]
    return out


def mae(p, rows):
    return float(np.abs(p - y[rows]).mean())


def describe(node, depth=0):
    if "feature" not in node:
        return {"value": round(node["value"]), "n": node["n"]}
    return {"question": "%s <= %.3g" % (NAMES[node["feature"]], node["threshold"]), "n": node["n"],
            "yes": describe(node["left"], depth + 1), "no": describe(node["right"], depth + 1)}


R = {"districts": int(len(y)), "train": int(len(tr)), "test": int(len(te)), "capped_at_500001": int((y >= 500001).sum())}
avg = y[tr].mean()
R["average"] = dict(guess=float(avg), test_mae=mae(np.full(len(te), avg), te))
print("average guess $%.0f, off by $%.0f" % (avg, R["average"]["test_mae"]))

# boosting
LR, KEEP = 0.1, (1, 2, 3, 5, 10, 30, 100, 300, 1000)
pred_tr, pred_te = np.full(len(tr), avg), np.full(len(te), avg)
allp = np.full(len(y), avg)
curve, trees = [], []
t0 = time.time()
resid_all = np.zeros(len(y))
for m in range(1, 1001):
    resid_all[tr] = y[tr] - pred_tr
    tree = fit_tree(tr, resid_all, depth=3)
    pred_tr += LR * predict(tree, tr)
    pred_te += LR * predict(tree, te)
    if m <= 5:
        trees.append(tree)
    if m in KEEP:
        curve.append(dict(trees=m, train_mae=mae(pred_tr, tr), test_mae=mae(pred_te, te)))
        print("%4d trees: train $%.0f  test $%.0f  (%.0fs)" % (m, curve[-1]["train_mae"], curve[-1]["test_mae"], time.time() - t0))
R["boost"] = dict(learning_rate=LR, depth=3, curve=curve, secs=time.time() - t0)
R["first_trees"] = [describe(t) for t in trees[:3]]
R["one_small_tree_full_size"] = dict(test_mae=mae(avg + predict(fit_tree(tr, y - avg, 3), te), te))
print("one small tree at full size: $%.0f" % R["one_small_tree_full_size"]["test_mae"])
json.dump(R, open(os.path.join(HERE, "out", "results.json"), "w"), indent=1)

# 4. districts followed tree by tree: the test district nearest Santa Monica's pier (the running example) and the
#    one nearest Pasadena's city hall (one of the 17% that boosting makes worse)
def nearest(a, b):
    return int(te[np.argmin((lat[te] - a) ** 2 + ((lon[te] - b) * 0.83) ** 2)])
EX = nearest(34.009, -118.497)
PX = nearest(34.148, -118.144)
p, path = avg, [float(avg)]
resid_all = np.zeros(len(y))
pred_tr = np.full(len(tr), avg)
rows_ex = np.array([EX, PX])
pp, ppath = avg, [float(avg)]
for m in range(1, 1001):
    resid_all[tr] = y[tr] - pred_tr
    tree = fit_tree(tr, resid_all, depth=3)
    pred_tr += LR * predict(tree, tr)
    q = predict(tree, rows_ex)
    p += LR * float(q[0]); pp += LR * float(q[1])
    path.append(p); ppath.append(pp)
R["example"] = dict(row=EX, lat=float(lat[EX]), lon=float(lon[EX]), true=float(y[EX]), income=float(inc[EX] * 10000),
                    house_age=float(age[EX]), homes=float(hh[EX]),
                    path={str(k): path[k] for k in (0, 1, 2, 3, 5, 10, 30, 100, 300, 1000)},
                    every_tree=[round(v) for v in path[:101]])
R["pasadena"] = dict(row=PX, true=float(y[PX]), path={str(k): ppath[k] for k in (0, 1, 10, 100, 1000)})
e0, e1 = np.abs(avg - y[te]), np.abs(pred_te - y[te])
R["test_improved_share"] = float(np.mean(e1 < e0))
R["test_median_error"] = float(np.median(e1))
print("example district", R["example"])

# 5. contrast: one deep tree; a random forest of 100 deep trees
deep = fit_tree(tr, y, depth=12, min_leaf=5)
R["deep_tree"] = dict(depth=12, train_mae=mae(predict(deep, tr), tr), test_mae=mae(predict(deep, te), te))
print("deep tree", R["deep_tree"])
t0 = time.time()
fpr = np.zeros(len(te))
for k in range(100):
    boot = rng.choice(tr, len(tr), replace=True)
    t = fit_tree(boot, y, depth=14, min_leaf=3, feats=lambda r: r.choice(X.shape[1], 3, replace=False))
    fpr += predict(t, te)
R["forest"] = dict(trees=100, depth=14, test_mae=mae(fpr / 100, te), secs=time.time() - t0)
print("forest", R["forest"])

# 6. too eager: learning rate 1.0
pred_tr, pred_te = np.full(len(tr), avg), np.full(len(te), avg)
c1 = []
for m in range(1, 1001):
    resid_all[tr] = y[tr] - pred_tr
    tree = fit_tree(tr, resid_all, depth=3)
    pred_tr += predict(tree, tr)
    pred_te += predict(tree, te)
    if m in KEEP:
        c1.append(dict(trees=m, train_mae=mae(pred_tr, tr), test_mae=mae(pred_te, te)))
R["boost_lr1"] = c1
print("lr 1.0:", [(c["trees"], round(c["train_mae"]), round(c["test_mae"])) for c in c1])
json.dump(R, open(os.path.join(HERE, "out", "results.json"), "w"), indent=1)
