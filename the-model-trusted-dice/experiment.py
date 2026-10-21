"""
experiment.py - "The model trusted dice: L1 vs L2" (reel, 2026-11-04).

Predict a penguin's weight from bill length, bill depth and flipper length,
plus three columns of pure random noise ("the dice"). Compare a plain fit,
L2 (ridge) and L1 (lasso) as the penalty is turned up. Real Palmer Penguins
data (CC0), plain numpy: ridge in closed form, lasso by coordinate descent.

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
F = np.array([[float(r[k]) for k in ("bill_length_mm", "bill_depth_mm", "flipper_length_mm")] for r in R])
M = np.array([float(r["body_mass_g"]) for r in R])
NAMES = ["bill length", "bill depth", "flipper", "noise 1", "noise 2", "noise 3"]
X = np.c_[F, np.random.default_rng(0).normal(size=(len(M), 3))]      # the three dice columns


def setup(seed):
    """Standardise on the training split, so each weight is grams per standard deviation."""
    idx = np.random.default_rng(seed).permutation(len(R))
    tr, te = idx[:240], idx[240:]
    Z = (X - X[tr].mean(0)) / X[tr].std(0)
    mu = M[tr].mean()
    return Z[tr], M[tr] - mu, Z[te], M[te], mu


def lasso(A, b, lam, iters=4000):
    """Minimise (1/2n)|b - Aw|^2 + lam*|w|_1 by coordinate descent."""
    n = len(b); w = np.zeros(A.shape[1]); ss = (A * A).sum(0) / n
    for _ in range(iters):
        for j in range(len(w)):
            r = b - A @ w + A[:, j] * w[j]
            rho = A[:, j] @ r / n
            w[j] = np.sign(rho) * max(abs(rho) - lam, 0) / ss[j]
    return w


def ridge(A, b, lam):
    """Minimise |b - Aw|^2 + lam*|w|^2 (a different scale of lam from lasso's)."""
    return np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ b)


A, b, Zt, yt, mu = setup(0)
mae = lambda w: float(np.abs(Zt @ w + mu - yt).mean())
out = {"names": NAMES, "plain": {"w": ridge(A, b, 0).tolist(), "test_error_g": mae(ridge(A, b, 0))}, "l1": [], "l2": []}
print("weights (g per standard deviation):", ", ".join(NAMES))
for lam in (0, 15, 30, 40, 60):
    w = lasso(A, b, lam)
    out["l1"].append({"lam": lam, "w": w.tolist(), "zeros": int((w == 0).sum()), "test_error_g": mae(w)})
    print("L1 %5d  %s  zeros %d  test error %.0f g" % (lam, np.round(w, 1), (w == 0).sum(), mae(w)))
for lam in (0, 100, 800, 3200):
    w = ridge(A, b, lam)
    out["l2"].append({"lam": lam, "w": w.tolist(), "zeros": int((w == 0).sum()), "test_error_g": mae(w)})
    print("L2 %5d  %s  zeros %d  test error %.0f g" % (lam, np.round(w, 1), (w == 0).sum(), mae(w)))

print("\nfive different splits, L1 at 60:")
out["five_splits_l1_60"] = []
for s in range(5):
    A, b, Zt, yt, mu = setup(s)
    w = lasso(A, b, 60)
    kept = [NAMES[j] for j in range(6) if w[j] != 0]
    out["five_splits_l1_60"].append({"seed": s, "kept": kept})
    print("  split %d keeps %s" % (s, kept))
json.dump(out, open(os.path.join(OUT, "results.json"), "w"), indent=1)
