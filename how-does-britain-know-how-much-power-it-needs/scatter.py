"""scatter.py - what the model's temperature rule is learned from (2021-2023): for each day, how much the temperature
changed since yesterday, against how far 5 pm demand landed from "yesterday + the usual weekday change". The slope
of this cloud is the rule used in the forecast (method E). No 18 Dec - 8 Jan.
  python scatter.py --data download"""
import argparse, json, os, runpy, sys
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True)
ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")); a = ap.parse_args()
sys.argv = ["experiment.py", "--data", a.data, "--out", a.out]
g = runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiment.py"))
D, X, T, idx, COEF_E = g["D"], g["X"], g["T"], g["idx"], g["COEF_E"]
P = 34                                                      # 17:00
pts = []
for d in D:
    k = idx[d]
    if d.year > 2023 or k < 30 or d not in T or D[k - 1] not in T or (d.month == 12 and d.day >= 18) or (d.month == 1 and d.day <= 8):
        continue
    wk = np.mean([X[k - 7 * w][P] - X[k - 7 * w - 1][P] for w in (1, 2, 3, 4)])
    pts.append((float(T[d] - T[D[k - 1]]), float(X[k][P] - (X[k - 1][P] + wk))))
t, y = np.array(pts).T
json.dump({"points": pts, "rule_gw_per_degree_5pm": float(COEF_E[P]), "fit_here": float((t @ y) / (t @ t)), "n": len(pts)},
          open(os.path.join(a.out, "scatter.json"), "w"), indent=1)
print(len(pts), "days; rule %.3f GW/degree at 5 pm; fit on this cloud %.3f" % (COEF_E[P], (t @ y) / (t @ t)))
