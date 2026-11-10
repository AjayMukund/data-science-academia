"""detail.py - the running example, Monday 18 November 2024: each forecast, and what each correction added.
  python detail.py --data download --day 2024-11-18"""
import argparse, json, os, runpy, sys, datetime as dt
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--day", required=True)
ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")); a = ap.parse_args()
sys.argv = ["experiment.py", "--data", a.data, "--out", a.out]
g = runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiment.py"))
D, X, T, idx, fc, COEF_E = g["D"], g["X"], g["T"], g["idx"], g["fc"], g["COEF_E"]
d = dt.date.fromisoformat(a.day); k = idx[d]
wk = np.mean([X[k - 7 * w] - X[k - 7 * w - 1] for w in (1, 2, 3, 4)], 0)
tc = COEF_E * (T[d] - T[D[k - 1]])
hh = lambda p: "%02d:%02d" % (p // 2, 30 * (p % 2))
pk = int(np.argmax(X[k]))
out = {"day": a.day, "weekday": d.strftime("%A"), "temp": T[d], "temp_yesterday": T[D[k - 1]],
       "actual": X[k].tolist(), "yesterday": X[k - 1].tolist(), "weekday_shift": wk.tolist(), "temp_shift": tc.tolist(),
       "E": fc(d, "E").tolist(), "C": fc(d, "C").tolist(),
       "peak": {"time": hh(pk), "actual": X[k][pk], "yesterday_same_time": X[k - 1][pk], "E_same_time": fc(d, "E")[pk],
                "weekday_shift": wk[pk], "temp_shift": tc[pk]},
       "mae": {m: float(np.abs(fc(d, m) - X[k]).mean()) for m in "ABCDE"},
       "max_miss": {m: float(np.abs(fc(d, m) - X[k]).max()) for m in "ABCDE"},
       "energy_gwh": {"actual": float(X[k].sum() / 2), "yesterday": float(X[k - 1].sum() / 2)}}
json.dump(out, open(os.path.join(a.out, "day_%s.json" % a.day), "w"), indent=1)
print(json.dumps({x: out[x] for x in ("day", "weekday", "temp", "temp_yesterday", "peak", "mae", "max_miss", "energy_gwh")}, indent=1))
print("shape (every 2 h): actual / yesterday / E")
for p in range(0, 48, 4): print(" ", hh(p), "%.1f %.1f %.1f" % (X[k][p], X[k - 1][p], fc(d, "E")[p]))
