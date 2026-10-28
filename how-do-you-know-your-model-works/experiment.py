"""
experiment.py - how do you know your model works? Cross-validation on Palmer Penguins.

Puts one published number on trial: a species model (5 nearest
neighbours on beak length + depth, Palmer Penguins) scored 97.1% (67 / 69) on ONE random test
split (seed 7, as published in our supervised-learning video). How much of that was the luck of the split?

  1. reproduce the published split
  2. the same model on 10,000 other random 80/20 splits: the spread of one test score
  3. 5-fold cross-validation: every penguin tested exactly once; the 5 fold scores; and how much
     the CV average itself moves over 1,000 reshuffles
  4. the trap: try k = 1..25 and report the best test score of one split, vs what that k really
     scores on penguins it was not chosen on

Data: Palmer Penguins (Gorman, Williams & Fraser 2014; palmerpenguins R package; CC0).

    python experiment.py --out out
"""
import os, csv, json, argparse, urllib.request
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="out")
ap.add_argument("--data", default="penguins.csv")
ap.add_argument("--splits", type=int, default=10000)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
if not os.path.exists(a.data):
    urllib.request.urlretrieve("https://raw.githubusercontent.com/allisonhorst/palmerpenguins/"
                               "main/inst/extdata/penguins.csv", a.data)

rows = list(csv.DictReader(open(a.data, encoding="utf-8")))
F = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
keep = [r for r in rows if all(r[f] != "NA" for f in F)]
SPECIES = ["Adelie", "Chinstrap", "Gentoo"]
X = np.array([[float(r["bill_length_mm"]), float(r["bill_depth_mm"])] for r in keep])
y = np.array([SPECIES.index(r["species"]) for r in keep])
N = len(y)
n_tr = int(0.8 * N)
R = {"penguins": N, "train": n_tr, "test": N - n_tr}

# all pairwise squared distances need re-standardising per split, so precompute nothing; it's small
def knn_acc(tr, te, k=5):
    mu, sd = X[tr].mean(0), X[tr].std(0)
    A, B = (X[tr] - mu) / sd, (X[te] - mu) / sd
    d = ((B[:, None, :] - A[None, :, :]) ** 2).sum(-1)
    nn = np.argsort(d, 1, kind="stable")[:, :k]
    votes = np.zeros((len(te), 3), int)
    for c in range(3):
        votes[:, c] = (y[tr][nn] == c).sum(1)
    pred = votes.argmax(1)
    return float((pred == y[te]).mean()), int((pred == y[te]).sum())


# 1. the published split (seed 7, as in the 10-24 experiment)
rng = np.random.default_rng(7)
idx = rng.permutation(N)
acc, right = knn_acc(idx[:n_tr], idx[n_tr:])
R["published_split"] = {"acc": round(acc, 4), "right": right, "of": N - n_tr}

# 2. 10,000 other random splits
rng = np.random.default_rng(2026)
accs, rights = [], []
for _ in range(a.splits):
    p = rng.permutation(N)
    ac, rt = knn_acc(p[:n_tr], p[n_tr:])
    accs.append(ac); rights.append(rt)
accs, rights = np.array(accs), np.array(rights)
counts = {int(r): int((rights == r).sum()) for r in np.unique(rights)}
R["many_splits"] = {"splits": a.splits, "mean": round(float(accs.mean()), 4),
                    "min": round(float(accs.min()), 4), "max": round(float(accs.max()), 4),
                    "range95": [round(float(np.percentile(accs, 2.5)), 4), round(float(np.percentile(accs, 97.5)), 4)],
                    "share_perfect": round(float((rights == N - n_tr).mean()), 4),
                    "share_at_or_above_published": round(float((accs >= acc - 1e-12).mean()), 4),
                    "right_counts": counts}

# 3. 5-fold cross-validation
def cv5(seed, k=5):
    p = np.random.default_rng(seed).permutation(N)
    folds = np.array_split(p, 5)
    scores, right = [], 0
    for i in range(5):
        te = folds[i]
        tr = np.concatenate([folds[j] for j in range(5) if j != i])
        ac, rt = knn_acc(tr, te, k)
        scores.append(ac); right += rt
    return scores, right

scores, right = cv5(0)
R["cv5_one_run"] = {"fold_scores": [round(s, 4) for s in scores], "fold_sizes": [len(f) for f in np.array_split(np.arange(N), 5)],
                    "all_penguins_right": right, "of": N, "acc": round(right / N, 4)}
cv_means = np.array([cv5(s)[1] / N for s in range(1000)])
R["cv5_reshuffled"] = {"runs": 1000, "mean": round(float(cv_means.mean()), 4),
                       "min": round(float(cv_means.min()), 4), "max": round(float(cv_means.max()), 4),
                       "range95": [round(float(np.percentile(cv_means, 2.5)), 4), round(float(np.percentile(cv_means, 97.5)), 4)]}

# 4. the trap: choose k by the test score of one split
rng = np.random.default_rng(99)
reported, honest = [], []
for _ in range(1000):
    p = rng.permutation(N)
    tr, te = p[:n_tr], p[n_tr:]
    scores_k = [knn_acc(tr, te, k)[0] for k in range(1, 26)]
    best_k = int(np.argmax(scores_k)) + 1
    reported.append(max(scores_k))
    # what that chosen k really scores: cross-validated on the same penguins with fresh folds
    honest.append(cv5(int(rng.integers(1 << 30)), best_k)[1] / N)
reported, honest = np.array(reported), np.array(honest)
R["choose_k_on_test"] = {"tries": 25, "runs": 1000, "reported_mean": round(float(reported.mean()), 4),
                         "honest_mean": round(float(honest.mean()), 4),
                         "share_reported_perfect": round(float((reported == 1.0).mean()), 4),
                         "inflation_points": round(100 * float((reported - honest).mean()), 2)}
json.dump(R, open(os.path.join(a.out, "results.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in R.items()}, indent=1))
