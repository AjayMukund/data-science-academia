"""
experiment.py - "One wrong row" (reel, 2026-11-03).

Fit a straight line for penguin weight against flipper length on 240 real
penguins. Then type ONE penguin's weight with an extra zero (4,650 g as
46,500 g), refit, and score both lines on the 102 untouched penguins. The typo
is ours; the Palmer Penguins data (CC0) contains no such error. Plain numpy.

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
x = np.array([float(r["flipper_length_mm"]) for r in R])
M = np.array([float(r["body_mass_g"]) for r in R])
idx = np.random.default_rng(0).permutation(len(R))
tr, te = idx[:240], idx[240:]
A = np.c_[x, np.ones(len(R))]
fit = lambda y: np.linalg.lstsq(A[tr], y[tr], rcond=None)[0]
mae = lambda w: float(np.abs(A[te] @ w - M[te]).mean())

w0 = fit(M)
i = int(tr[np.argmin(np.abs(M[tr] - 4650))])               # a typical penguin, 4,650 g
M2 = M.copy()
M2[i] = M[i] * 10                                          # the extra zero
w1 = fit(M2)
sq = (M2[tr] - A[tr] @ w1) ** 2                           # squared misses of the bent line
share = float(sq[list(tr).index(i)] / sq.sum())

out = {"typo": {"species": R[i]["species"], "flipper_mm": float(x[i]), "true_g": float(M[i]), "typed_g": float(M2[i])},
       "clean_line": {"g_per_mm": float(w0[0]), "test_error_g": mae(w0)},
       "typo_line": {"g_per_mm": float(w1[0]), "test_error_g": mae(w1)},
       "typo_share_of_squared_error": share,
       "mean_g": [float(M[tr].mean()), float(M2[tr].mean())], "median_g": [float(np.median(M[tr])), float(np.median(M2[tr]))]}
json.dump(out, open(os.path.join(OUT, "results.json"), "w"), indent=1)
print("the typo: %s, %.0f mm flipper, %.0f g typed as %.0f g" % (R[i]["species"], x[i], M[i], M2[i]))
print("slope   %.0f -> %.0f g per mm of flipper" % (w0[0], w1[0]))
print("error on the 102 untouched penguins: %.0f g -> %.0f g" % (mae(w0), mae(w1)))
print("the typo's share of all squared error: %.0f%%" % (100 * share))
print("mean %.0f -> %.0f g, median %.0f -> %.0f g" % (*out["mean_g"], *out["median_g"]))
