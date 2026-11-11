"""
experiment.py - could 1,000 trees have told New York what 683,788 did?
The 2015 census counted every street tree. We treat it as the truth and ask what samples would have said.
  1. The truth: share in good health (of every tree counted, alive, dead or stump).
  2. Random samples of 10 / 100 / 1,000 / 10,000 trees, 10,000 times each: the middle 95% of answers, the worst miss.
  3. Size of the place doesn't matter: 1,000 from Staten Island alone vs 1,000 from the whole city.
  4. Not random: 1,000 trees from one neighbourhood; the first 1,000 the census logged.
Seeded (numpy 0). Writes out/results.json and out/estimates.npz (every simulated answer).   python experiment.py
"""
import csv, json, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
rows = list(csv.DictReader(open(os.path.join(HERE, "data", "trees.csv"), encoding="utf-8")))
rows.sort(key=lambda r: int(r["tree_id"]))
good = np.array([r["health"] == "Good" for r in rows])
boro = np.array([r["boroname"] for r in rows]); nta = np.array([r["nta_name"] for r in rows])
N = len(rows); rng = np.random.default_rng(0)
R = {"trees": N, "good": int(good.sum()), "truth": float(good.mean()),
     "status": {s: int(sum(r["status"] == s for r in rows)) for s in ("Alive", "Dead", "Stump")},
     "health": {h: int(sum(r["health"] == h for r in rows)) for h in ("Good", "Fair", "Poor")}}
R["samples"] = {}
EST = {}
for n in (10, 100, 1000, 4000, 10000):
    est = np.array([good[rng.choice(N, n, replace=False)].mean() for _ in range(10000)])
    EST["n%d" % n] = est
    lo, hi = np.percentile(est, [2.5, 97.5])
    R["samples"][n] = dict(lo=float(lo), hi=float(hi), worst=float(np.abs(est - R["truth"]).max()),
                           within2=float(np.mean(np.abs(est - R["truth"]) <= 0.02)),
                           first5=[float(x) for x in est[:5]], hist=np.histogram(est, bins=40, range=(0.55, 1.0))[0].tolist())
    print(n, R["samples"][n]["lo"], R["samples"][n]["hi"], R["samples"][n]["worst"], R["samples"][n]["within2"])
# 3. place size
R["boroughs"] = {}
rng_b = np.random.default_rng(1)          # its own stream: adding a sample size above can't move these
for b in sorted(set(boro)):
    idx = np.where(boro == b)[0]
    est = np.array([good[rng_b.choice(idx, 1000, replace=False)].mean() for _ in range(10000)])
    R["boroughs"][b] = dict(trees=int(len(idx)), truth=float(good[idx].mean()),
                            lo=float(np.percentile(est, 2.5)), hi=float(np.percentile(est, 97.5)))
    print(b, R["boroughs"][b])
# 4. not random
R["neighbourhoods"] = {}
for n_ in sorted(set(nta)):
    idx = np.where(nta == n_)[0]
    if len(idx) >= 1000:
        R["neighbourhoods"][n_] = dict(trees=int(len(idx)), boro=str(boro[idx[0]]), truth=float(good[idx].mean()))
v = sorted(R["neighbourhoods"].items(), key=lambda kv: kv[1]["truth"])
print("neighbourhoods >=1000 trees:", len(v)); print("lowest", v[:3]); print("highest", v[-3:])
first = good[:1000].mean(); R["first1000"] = dict(share=float(first), boros=dict(zip(*np.unique(boro[:1000], return_counts=True))))
R["first1000"]["boros"] = {k: int(c) for k, c in R["first1000"]["boros"].items()}
print("first 1000 logged", first, R["first1000"]["boros"])
json.dump(R, open(os.path.join(HERE, "out", "results.json"), "w"), indent=1, default=str)
np.savez_compressed(os.path.join(HERE, "out", "estimates.npz"), **EST)
print("truth", R["truth"], R["status"], R["health"])
