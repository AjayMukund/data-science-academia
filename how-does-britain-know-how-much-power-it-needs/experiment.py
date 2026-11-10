"""
experiment.py - how does Britain know how much electricity it will need tomorrow? Forecasting, measured.

Data: NESO historic demand data, 2021-2025 (half-hourly national demand "ND", MW; NESO Open Data Licence:
"Supported by National Energy SO Open Data"), https://www.neso.energy/data-portal/historic-demand-data
Temperature: Open-Meteo historical archive (CC BY 4.0), daily mean at 2 m for London, Birmingham, Manchester, Glasgow,
averaged. Download with the commands in README.md into download/.

Forecast: tomorrow's 48 half-hours, made the day before. Score: mean absolute error in GW over the test period
(2024-01-01 to 2025-10-31); every model is fitted on 2021-2023 only.
  A  same as yesterday
  B  same as the same day last week
  C  average of the same weekday over the last 4 weeks
  E  yesterday, corrected for the weekday (how this weekday usually differs from the day before, over the last
     4 weeks) and for the temperature change since yesterday (slope learned per half-hour)
  D  C, plus a temperature correction learned per half-hour: demand gap ~ temperature gap (the temperature the
     forecast is for: here the actual day's mean, standing in for a weather forecast)

  python experiment.py --data download --out out
"""
import argparse, csv, datetime as dt, json, os
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"))
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)


def pdate(s):
    s = s.strip('"')                                   # the yearly files use three date styles
    for f in ("%d-%b-%Y", "%d-%b-%y", "%Y-%m-%d"):
        try:
            return dt.datetime.strptime(s.title() if s[3:6].isalpha() else s, f).date()
        except ValueError:
            pass
    raise ValueError(s)


days = {}
for y in range(2021, 2026):
    for r in csv.DictReader(open(os.path.join(a.data, "demand_%d.csv" % y), encoding="utf-8")):
        d, p = pdate(r["SETTLEMENT_DATE"]), int(r["SETTLEMENT_PERIOD"])
        days.setdefault(d, {})[p] = int(r["ND"])
# clock-change days have 46 or 50 half-hours: keep the first 48 periods, filling a short day's end from period 46
D = sorted(days)
X = np.array([[days[d].get(p, days[d].get(46)) for p in range(1, 49)] for d in D], float) / 1000.0     # GW
idx = {d: k for k, d in enumerate(D)}

temps = []
for c in ("london", "birmingham", "manchester", "glasgow"):
    j = json.load(open(os.path.join(a.data, "temp_%s.json" % c)))["daily"]
    temps.append(dict(zip(j["time"], j["temperature_2m_mean"])))
T = {d: np.mean([t[d.isoformat()] for t in temps]) for d in D if all(d.isoformat() in t for t in temps)}

res = {"days": len(D), "first": D[0].isoformat(), "last": D[-1].isoformat()}
TRAIN_END, TEST0, TEST1 = dt.date(2023, 12, 31), dt.date(2024, 1, 1), dt.date(2025, 10, 31)


def fc(d, how):
    k = idx[d]
    if how == "A":
        return X[k - 1]
    if how == "B":
        return X[k - 7]
    if how in ("C", "D"):
        base = X[[k - 7, k - 14, k - 21, k - 28]].mean(0)
        if how == "C":
            return base
        tgap = T[d] - np.mean([T[D[k - w]] for w in (7, 14, 21, 28)])
        return base + COEF * tgap
    if how == "E":
        wk = np.mean([X[k - 7 * w] - X[k - 7 * w - 1] for w in (1, 2, 3, 4)], 0)
        return X[k - 1] + wk + COEF_E * (T[d] - T[D[k - 1]])


# fit D's correction on 2021-2023: per half-hour slope of (actual - base) on the temperature gap
rows, gaps = [], []
for d in D:
    k = idx[d]
    if k < 28 or d > TRAIN_END or d not in T or any(D[k - w] not in T for w in (7, 14, 21, 28)):
        continue
    base = X[[k - 7, k - 14, k - 21, k - 28]].mean(0)
    rows.append(X[k] - base)
    gaps.append(T[d] - np.mean([T[D[k - w]] for w in (7, 14, 21, 28)]))
rows, gaps = np.array(rows), np.array(gaps)
COEF = (gaps @ rows) / (gaps @ gaps)                     # GW per degree, one per half-hour (no intercept)
rows, gaps = [], []
for d in D:
    k = idx[d]
    if k < 30 or d > TRAIN_END or d not in T or D[k - 1] not in T:
        continue
    wk = np.mean([X[k - 7 * w] - X[k - 7 * w - 1] for w in (1, 2, 3, 4)], 0)
    rows.append(X[k] - (X[k - 1] + wk))
    gaps.append(T[d] - T[D[k - 1]])
rows, gaps = np.array(rows), np.array(gaps)
COEF_E = (gaps @ rows) / (gaps @ gaps)
res["gw_per_degree"] = {"mean": float(COEF.mean()), "min": float(COEF.min()), "max": float(COEF.max()),
                        "at_17_30": float(COEF[35])}

test = [d for d in D if TEST0 <= d <= TEST1 and d in T and idx[d] >= 28]
err = {m: np.array([np.abs(fc(d, m) - X[idx[d]]).mean() for d in test]) for m in "ABCDE"}
res["test_days"] = len(test)
res["mae_gw"] = {m: float(e.mean()) for m, e in err.items()}
res["avg_demand_gw"] = float(np.mean([X[idx[d]].mean() for d in test]))
res["mae_pct"] = {m: float(e.mean() / res["avg_demand_gw"] * 100) for m, e in err.items()}
print("test days", len(test), "avg demand %.1f GW" % res["avg_demand_gw"])
print({m: "%.2f GW (%.1f%%)" % (res["mae_gw"][m], res["mae_pct"][m]) for m in "ABCDE"})

# worst days for C and D, with their weekday and temperature gap
def worst(m, n=12):
    o = np.argsort(-err[m])[:n]
    return [{"date": test[k].isoformat(), "weekday": test[k].strftime("%A"), "mae_gw": float(err[m][k]),
             "temp": float(T[test[k]]), "err_A": float(err["A"][k]), "err_B": float(err["B"][k]), "err_C": float(err["C"][k]), "err_D": float(err["D"][k]), "err_E": float(err["E"][k])} for k in o]


res["worst_C"], res["worst_D"], res["worst_E"] = worst("C"), worst("D"), worst("E")
# by weekday for A (yesterday breaks on Mondays and Saturdays)
res["A_by_weekday"] = {w: float(np.mean([e for d, e in zip(test, err["A"]) if d.strftime("%A") == w]))
                       for w in ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")}
# the daily shape: an average weekday vs Sunday in the test period, and the evening peak
res["shape"] = {w: [float(v) for v in np.mean([X[idx[d]] for d in test if d.strftime("%A") == w], 0)] for w in ("Wednesday", "Sunday")}
# full curves for the worst days and Christmas
keep = {x["date"] for x in res["worst_C"][:6] + res["worst_D"][:6] + res["worst_E"][:6]} | {"2024-12-25", "2024-12-18", "2025-01-08"}
res["curves"] = {s: {"actual": X[idx[dt.date.fromisoformat(s)]].tolist(),
                     **{m: fc(dt.date.fromisoformat(s), m).tolist() for m in "ABCDE"},
                     "temp": float(T[dt.date.fromisoformat(s)])} for s in sorted(keep) if dt.date.fromisoformat(s) in T}
json.dump(res, open(os.path.join(a.out, "results.json"), "w", encoding="utf-8"), indent=1)
res["E_gw_per_degree_17_30"] = float(COEF_E[35])
for m in ("E",):
    print("worst", m)
    for w in res["worst_" + m][:8]:
        print("  ", w)
print("A by weekday", {k: round(v, 2) for k, v in res["A_by_weekday"].items()})
print("GW per degree", res["gw_per_degree"])
