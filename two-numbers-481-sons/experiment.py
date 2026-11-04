"""The bell curve - Galton's families (1886 notebook, 934 grown children).

Measures: sons' and daughters' heights in cm, mean, SD, the share within one
and two SDs (vs the bell's 68% / 95%), a 2.5 cm histogram, and the "many
small causes" check: the miss after predicting from the parents is a bell too.
Data: Rdatasets HistData::GaltonFamilies (re-downloaded if missing).

    python experiment.py
"""
import csv
import json
import math
import os
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(HERE, "download"), exist_ok=True)
os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
CSV = os.path.join(HERE, "download", "GaltonFamilies.csv")
if not os.path.exists(CSV):
    urllib.request.urlretrieve("https://vincentarelbundock.github.io/Rdatasets/csv/HistData/GaltonFamilies.csv", CSV)
R = list(csv.DictReader(open(CSV)))
CM = 2.54


def describe(h):
    m, s = h.mean(), h.std()
    within = lambda k: float(np.mean(np.abs(h - m) <= k * s))
    lo = math.floor(h.min() / 2.5) * 2.5
    edges = np.arange(lo, h.max() + 2.5, 2.5)
    cnt, _ = np.histogram(h, edges)
    return dict(n=len(h), mean=round(m, 1), sd=round(s, 1), min=round(h.min(), 1), max=round(h.max(), 1),
                within1=round(within(1), 3), within2=round(within(2), 3),
                hist=[(round(float(a), 1), int(c)) for a, c in zip(edges, cnt)])


out = {}
for g in ("male", "female"):
    h = np.array([float(r["childHeight"]) for r in R if r["gender"] == g]) * CM
    out[g] = describe(h)
allh = np.array([float(r["childHeight"]) for r in R]) * CM
out["all"] = describe(allh)
# daughters scaled by Galton's own 1.08 sit on the sons' curve
# many small causes: predict each son from his parents, look at the misses
sons = [r for r in R if r["gender"] == "male"]
x = np.array([float(r["midparentHeight"]) for r in sons]) * CM
y = np.array([float(r["childHeight"]) for r in sons]) * CM
b, a = np.polyfit(x, y, 1)
miss = y - (a + b * x)
out["sons_from_parents"] = dict(slope=round(b, 2), miss=describe(miss))
for k, v in out.items():
    print(k, {kk: vv for kk, vv in (v.items() if isinstance(v, dict) else [])})
json.dump(out, open(os.path.join(HERE, "out", "results.json"), "w"), indent=1)
