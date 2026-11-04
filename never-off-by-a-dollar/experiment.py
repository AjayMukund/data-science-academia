"""Data leakage - Ames, Iowa house sales (De Cock 2011, 2,930 sales 2006-2010).

The job: guess a house's sale price before it sells. Three models, one 80/20 split
(seed 0) and 1,000 random splits for the spread:
  honest  - log price from living area, overall quality, year built, garage cars,
            basement area (all known before the sale)
  leaky   - the same plus "price per square foot" (sale price / living area),
            a column an analyst adds to "help": it contains the answer
  leaky, deployed - the leaky model on a house FOR SALE, where price per square foot
            does not exist yet: filled with its neighbourhood's training average
Error = median miss in dollars and in % of the price, on held-out houses.

    python experiment.py
"""
import csv
import json
import os
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(HERE, "download"), exist_ok=True)
os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
TXT = os.path.join(HERE, "download", "AmesHousing.txt")
if not os.path.exists(TXT):
    urllib.request.urlretrieve("https://jse.amstat.org/v19n3/decock/AmesHousing.txt", TXT)
R = list(csv.DictReader(open(TXT, encoding="latin-1"), delimiter="\t"))
# De Cock's own advice: drop the 5 houses over 4,000 sq ft (3 partial sales, 2 huge)
R = [r for r in R if float(r["Gr Liv Area"]) <= 4000]
num = lambda k: np.array([float(r[k]) if r[k] not in ("", "NA") else 0.0 for r in R])
price, area = num("SalePrice"), num("Gr Liv Area")
qual, built, cars, bsmt = num("Overall Qual"), num("Year Built"), num("Garage Cars"), num("Total Bsmt SF")
hood = np.array([r["Neighborhood"] for r in R])
ppsf = price / area
y = np.log(price)
base = np.column_stack([np.ones(len(R)), np.log(area), qual, built, cars, bsmt / 1000])


def fit(X, y):
    return np.linalg.lstsq(X, y, rcond=None)[0]


def run(seed):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(R))
    te, tr = idx[: len(R) // 5], idx[len(R) // 5:]
    res = {}
    w = fit(base[tr], y[tr])
    res["honest"] = np.exp(base[te] @ w)
    Xl = np.column_stack([base, np.log(ppsf)])
    wl = fit(Xl[tr], y[tr])
    res["leaky"] = np.exp(Xl[te] @ wl)
    # deployed: the price-per-foot column is filled with the neighbourhood's training average
    avg = {h: ppsf[tr][hood[tr] == h].mean() for h in set(hood)}
    fill = np.array([avg.get(h, ppsf[tr].mean()) if np.isfinite(avg.get(h, np.nan)) else ppsf[tr].mean() for h in hood[te]])
    Xd = Xl[te].copy()
    Xd[:, -1] = np.log(fill)
    res["deployed"] = np.exp(Xd @ wl)
    out = {}
    for k, p in res.items():
        miss = np.abs(p - price[te])
        out[k] = dict(median_miss=float(np.median(miss)), median_pct=float(np.median(miss / price[te])),
                      within10=float(np.mean(miss / price[te] <= 0.10)))
    out["weights_leaky"] = dict(zip(["const", "log_area", "qual", "built", "cars", "bsmt", "log_ppsf"], map(float, wl)))
    out["example"] = int(te[0])
    return out, te, res


one, te, res = run(0)
spread = {k: [] for k in ("honest", "leaky", "deployed")}
for s in range(1000):
    o, _, _ = run(s)
    for k in spread:
        spread[k].append(o[k]["median_pct"])
worse = sum(d > h for d, h in zip(spread["deployed"], spread["honest"]))
out = {"n": len(R), "median_price": float(np.median(price)), "seed0": one,
       "deployed_worse_splits": int(worse),
       "pct_1000": {k: dict(mean=float(np.mean(v)), lo=float(np.percentile(v, 2.5)), hi=float(np.percentile(v, 97.5)))
                    for k, v in spread.items()}}
# one held-out house to walk through
i = int(te[3])
out["walk"] = dict(row=i, hood=hood[i], area=float(area[i]), qual=float(qual[i]), built=float(built[i]),
                   price=float(price[i]), ppsf=float(ppsf[i]),
                   honest=float(res["honest"][3]), leaky=float(res["leaky"][3]), deployed=float(res["deployed"][3]))
print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(HERE, "out", "results.json"), "w"), indent=1)
