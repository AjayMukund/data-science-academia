"""holidays.py - the honest turn: how the best method (E) does on Mondays, and on bank holidays and the day after.
  python holidays.py --data download"""
import argparse, json, os, runpy, sys, datetime as dt
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True)
ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")); a = ap.parse_args()
sys.argv = ["experiment.py", "--data", a.data, "--out", a.out]
g = runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiment.py"))
X, idx, fc, test = g["X"], g["idx"], g["fc"], g["test"]
err = lambda d, m: float(np.abs(fc(d, m) - X[idx[d]]).mean())
# England and Wales bank holidays in the test period (gov.uk)
HOL = ["2024-01-01", "2024-03-29", "2024-04-01", "2024-05-06", "2024-05-27", "2024-08-26", "2024-12-25", "2024-12-26",
       "2025-01-01", "2025-04-18", "2025-04-21", "2025-05-05", "2025-05-26", "2025-08-25"]
H = [dt.date.fromisoformat(s) for s in HOL]
out = {"E_by_weekday": {w: float(np.mean([err(d, "E") for d in test if d.strftime("%A") == w]))
                        for w in ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")},
       "A_mondays": float(np.mean([err(d, "A") for d in test if d.weekday() == 0])),
       "E_mondays": float(np.mean([err(d, "E") for d in test if d.weekday() == 0])),
       "E_all": float(np.mean([err(d, "E") for d in test])),
       "E_holidays": float(np.mean([err(d, "E") for d in H if d in idx and d in set(test)])),
       "E_day_after": float(np.mean([err(d + dt.timedelta(1), "E") for d in H if d + dt.timedelta(1) in set(test)])),
       "E_ordinary": float(np.mean([err(d, "E") for d in test if all(abs((d - h).days) > 1 for h in H)])),
       "holidays": {s: {"E": err(dt.date.fromisoformat(s), "E")} for s in HOL if dt.date.fromisoformat(s) in set(test)}}
d = dt.date(2025, 8, 25); k = idx[d]
out["aug_bank_holiday_2025"] = {"actual_peak": float(X[k].max()), "E_peak": float(fc(d, "E").max()),
                                "noon_actual": float(X[k][24]), "noon_E": float(fc(d, "E")[24]), "mae_E": err(d, "E")}
out["daily"] = [{"date": d.isoformat(), "A": err(d, "A"), "E": err(d, "E")} for d in test]   # every test day's miss
d2 = dt.date(2025, 8, 25)
out["aug_curves"] = {"actual": X[idx[d2]].tolist(), "E": fc(d2, "E").tolist()}
json.dump(out, open(os.path.join(a.out, "holidays.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "holidays"}, indent=1))
