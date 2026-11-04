"""Why accuracy lies - California earthquakes, one forecast per day.

Question each day: will California have a magnitude 5+ quake TOMORROW?
Data: the USGS catalogue (public), every M4+ quake in a California box,
2000-01-01 to 2025-12-31. Re-downloaded if missing.

Forecasters, scored on every day:
  never     - always "no quake tomorrow"
  follow    - "yes" for the N days after any M4+ quake (aftershocks cluster)
Prints counts (tp, fp, fn, tn), accuracy, and how many quake days each caught.

    python experiment.py
"""
import csv
import datetime as dt
import json
import os
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(HERE, "download"), exist_ok=True)
os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
CSV = os.path.join(HERE, "download", "quakes_ca.csv")
URL = ("https://earthquake.usgs.gov/fdsnws/event/1/query?format=csv&starttime=2000-01-01"
       "&endtime=2026-01-01&minmagnitude=4.0&minlatitude=32.5&maxlatitude=42"
       "&minlongitude=-124.5&maxlongitude=-114&orderby=time-asc")
if not os.path.exists(CSV):
    urllib.request.urlretrieve(URL, CSV)

rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
rows = [r for r in rows if r["type"] == "earthquake"]
# the box also clips Nevada and Mexico: keep what USGS places in California
OUT = ("Nevada", ", NV", " NV ", "Mexico", "Baja", "B.C.", "Oregon", ", OR", "Arizona", ", AZ", "Utah")
rows = [r for r in rows if not any(o in r["place"] for o in OUT)]   # named quakes ("Lake Almanor") stay
day0, day1 = dt.date(2000, 1, 1), dt.date(2025, 12, 31)
days = [day0 + dt.timedelta(i) for i in range((day1 - day0).days + 1)]
big = {}      # day -> max magnitude
for r in rows:
    d = dt.date.fromisoformat(r["time"][:10])
    big[d] = max(big.get(d, 0), float(r["mag"]))

BIG = 5.0
truth = {d: big.get(d, 0) >= BIG for d in days}
n, pos = len(days), sum(truth.values())


def score(say):
    tp = sum(1 for d in days if say[d] and truth[d])
    fp = sum(1 for d in days if say[d] and not truth[d])
    fn = sum(1 for d in days if not say[d] and truth[d])
    tn = n - tp - fp - fn
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, acc=(tp + tn) / n, caught=tp, alarms=tp + fp)


out = {"days": n, "quake_days": pos, "M": BIG, "quakes_m4": len(rows),
       "quakes_m5": sum(1 for r in rows if float(r["mag"]) >= BIG)}
out["never"] = score({d: False for d in days})
for trig in (4.0, 4.5):
    for N in (1, 3, 7):
        say = {}
        for d in days:
            say[d] = any(big.get(d - dt.timedelta(k), 0) >= trig for k in range(1, N + 1))
        out["follow_m%.1f_%dd" % (trig, N)] = score(say)
out["quake_day_list"] = [(str(d), big[d]) for d in days if truth[d]]
say45 = {d: big.get(d - dt.timedelta(1), 0) >= 4.5 for d in days}
out["caught_by_rule"] = [str(d) for d in days if truth[d] and say45[d]]
# the biggest single quake days, for the story
top = sorted(((m, d) for d, m in big.items()), reverse=True)[:8]
out["top_days"] = [(str(d), m) for m, d in top]
for k, v in out.items():
    print(k, v)
json.dump(out, open(os.path.join(HERE, "out", "results.json"), "w"), indent=1, default=str)
