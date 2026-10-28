"""
experiment.py - classification with probabilities: logistic regression on the Challenger O-ring record.

Data: UCI "Challenger USA Space Shuttle O-Ring" (Draper 1993, CC BY 4.0;
https://archive.ics.uci.edu/dataset/92/challenger+usa+space+shuttle+o+ring), file
o-ring-erosion-or-blowby.data: the 23 shuttle flights before Challenger, each with the number of
its 6 field-joint O-rings that showed thermal distress (erosion or blow-by) and the launch
temperature (F). Challenger launched with a forecast of 31 F; the coldest earlier flight was 53 F.

Plain numpy, no fitting library:
  1. the record: which flights were damaged, at what temperature
  2. the damaged flights alone (the Rogers Commission's Figure 6) vs all flights (Figure 7)
  3. a straight line for "damaged or not": where it goes below 0 and above 1
  4. logistic regression, fitted by gradient descent: the S-curve and its probabilities
  5. how sure: refit on 10,000 bootstrap resamples of the 23 flights

    python experiment.py --out out
"""
import os, json, argparse, urllib.request
import numpy as np

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/space-shuttle/o-ring-erosion-or-blowby.data"
ap = argparse.ArgumentParser()
ap.add_argument("--data", default="download/o-ring-erosion-or-blowby.data")
ap.add_argument("--out", default="out")
ap.add_argument("--boot", type=int, default=10000)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
if not os.path.exists(a.data):
    os.makedirs(os.path.dirname(a.data) or ".", exist_ok=True)
    urllib.request.urlretrieve(URL, a.data)

rows = [l.split() for l in open(a.data) if l.strip()]
n_dist = np.array([int(r[1]) for r in rows])
T = np.array([float(r[2]) for r in rows])
order = np.array([int(r[4]) for r in rows])
y = (n_dist > 0).astype(float)
N = len(y)
R = {"flights": N, "damaged": int(y.sum()), "temp_min_max": [float(T.min()), float(T.max())],
     "challenger_forecast_F": 31.0, "coldest_before": float(T.min()),
     "degrees_colder_than_any": float(T.min() - 31.0),
     "flights": [{"order": int(o), "temp": float(t), "o_rings_damaged": int(d)} for o, t, d in zip(order, T, n_dist)]}

# --- 1/2. the record, and the damaged flights alone ------------------------------------------
dmg = y == 1
R["damaged_temps"] = sorted(T[dmg].tolist())
R["below_65"] = {"flights": int((T < 65).sum()), "damaged": int(y[T < 65].sum())}
R["at_or_above_65"] = {"flights": int((T >= 65).sum()), "damaged": int(y[T >= 65].sum())}
# Figure 6 view: among damaged flights only, does the count of damaged rings depend on temperature?
c = np.corrcoef(T[dmg], n_dist[dmg])[0, 1]
R["damaged_only_view"] = {"flights": int(dmg.sum()), "corr_temp_vs_rings": round(float(c), 3),
                          "temp_range": [float(T[dmg].min()), float(T[dmg].max())],
                          "rings_by_flight": sorted(zip(T[dmg].tolist(), n_dist[dmg].tolist()))}
# all flights view: share damaged in temperature bands
bands = [(50, 60), (60, 70), (70, 80), (80, 90)]
R["share_damaged_by_band"] = {"%d-%d" % b: [int(y[(T >= b[0]) & (T < b[1])].sum()), int(((T >= b[0]) & (T < b[1])).sum())]
                              for b in bands}

# --- 3. a straight line -------------------------------------------------------------------
m = float(((T - T.mean()) * (y - y.mean())).sum() / ((T - T.mean()) ** 2).sum())
c0 = float(y.mean() - m * T.mean())
R["straight_line"] = {"slope_per_F": round(m, 4), "at_31F": round(m * 31 + c0, 3), "at_53F": round(m * 53 + c0, 3),
                      "at_81F": round(m * 81 + c0, 3), "hits_zero_at_F": round(-c0 / m, 1), "hits_one_at_F": round((1 - c0) / m, 1)}

# --- 4. logistic regression by gradient descent (on centred temperature) -----------------------
TC = 70.0                                   # centre, so the two numbers learn at similar speed


def fit(t, yy, steps=200000, lr=0.05):
    w, b = 0.0, 0.0
    x = (t - TC) / 10.0                     # tens of degrees from 70 F
    for _ in range(steps):
        p = 1 / (1 + np.exp(-(w * x + b)))
        gw, gb = ((p - yy) * x).mean(), (p - yy).mean()
        w, b = w - lr * gw, b - lr * gb
        if abs(gw) < 1e-10 and abs(gb) < 1e-10:
            break
    return w, b


def newton(t, yy, iters=50):
    """Same model, fitted by Newton's method (a few steps instead of thousands) for the bootstrap.
    A perfectly separated resample has no finite best fit; its weights grow until the cap,
    which is what such a resample really says (a near-vertical step)."""
    x = np.c_[(t - TC) / 10.0, np.ones_like(t)]
    beta = np.zeros(2)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(x @ beta, -30, 30)))
        g = x.T @ (yy - p)
        H = x.T @ (x * (p * (1 - p))[:, None]) + 1e-9 * np.eye(2)
        step = np.linalg.solve(H, g)
        beta = beta + step
        if np.abs(step).max() < 1e-9 or np.abs(beta).max() > 50:
            break
    return beta[0], beta[1]


def prob(w, b, t):
    return 1 / (1 + np.exp(-(w * (t - TC) / 10.0 + b)))


w, b = fit(T, y)
ll = float((y * np.log(prob(w, b, T)) + (1 - y) * np.log(1 - prob(w, b, T))).sum())
R["logistic"] = {"w_per_10F": round(w, 4), "b_at_70F": round(b, 4), "log_likelihood": round(ll, 3),
                 "slope_per_F_logodds": round(w / 10, 4), "odds_multiplier_per_F_colder": round(float(np.exp(-w / 10)), 3),
                 "p": {str(t): round(float(prob(w, b, t)), 4) for t in (31, 40, 50, 53, 55, 60, 65, 70, 75, 81)},
                 "fifty_fifty_at_F": round(TC - 10 * b / w, 1),
                 # the conventional (unscaled) coefficients, to check against published fits (~15.04, -0.232)
                 "intercept_raw": round(float(b - w * TC / 10), 3), "slope_raw": round(float(w / 10), 4)}

# --- 5. how sure? bootstrap the 23 flights -------------------------------------------------
rng = np.random.default_rng(2026)
p31, p53, sep = [], [], 0
for _ in range(a.boot):
    i = rng.integers(0, N, N)
    tt, yy = T[i], y[i]
    if yy.sum() == 0 or yy.sum() == N:
        continue
    ww, bb = newton(tt, yy)
    p31.append(float(prob(ww, bb, 31))); p53.append(float(prob(ww, bb, 53)))
p31, p53 = np.array(p31), np.array(p53)
R["bootstrap"] = {"resamples_fitted": int(len(p31)),
                  "p31_median": round(float(np.median(p31)), 4), "p31_5th": round(float(np.percentile(p31, 5)), 4),
                  "share_p31_over_0.9": round(float((p31 > 0.9).mean()), 4),
                  "share_p31_over_0.5": round(float((p31 > 0.5).mean()), 4),
                  "p53_median": round(float(np.median(p53)), 4)}
json.dump(R, open(os.path.join(a.out, "results.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in R.items() if k != "flights"}, indent=1))
