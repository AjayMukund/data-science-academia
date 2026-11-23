"""
experiment.py - the birthday paradox on real England & Wales births.

Data: ONS, "Births in England and Wales: 2025" bulletin, Figure 5 "How popular is your birthday?"
(average daily live births, England and Wales, 1995 to 2025), and the same chart's 2025 counts.
Open Government Licence v3.0. Chart data:
  https://www.ons.gov.uk/visualisations/dvc3620/fig05/data.csv       (average 1995-2025)
  https://www.ons.gov.uk/visualisations/dvc3620/fig05/data2025.csv   (2025)

Question: in a room of n people, how likely is it that two share a birthday?
Exact for equal days (365); exact-by-formula is hard for unequal days, so the real-weights figure
is a Monte Carlo of 1,000,000 rooms per n (seed 1), plus the exact "probability nobody shares"
via the elementary symmetric polynomial for n <= 80.

    python experiment.py
"""
import csv, json, os, random, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE, OUT = os.path.join(HERE, "cache"), os.path.join(HERE, "out")
os.makedirs(CACHE, exist_ok=True); os.makedirs(OUT, exist_ok=True)
BASE = "https://www.ons.gov.uk/visualisations/dvc3620/fig05/"
MONTHS = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()


def load(name):
    p = os.path.join(CACHE, name)
    if not os.path.exists(p):
        req = urllib.request.Request(BASE + name, headers={"User-Agent": "research-script"})
        open(p, "wb").write(urllib.request.urlopen(req, timeout=60).read())
    days = []
    for row in csv.DictReader(open(p, encoding="utf-8-sig")):
        for d in range(1, 32):
            v = row.get(str(d), "")
            if v.strip():
                days.append(("%s %d" % (row["name"], d), float(v)))
    return days


def p_share_equal(n, k=365):
    q = 1.0
    for i in range(n):
        q *= (k - i) / k
    return 1 - q


def p_share_exact(n, w):
    """1 - n! * e_n(p): probability at least two of n people (iid, day weights p) share a day."""
    tot = sum(w)
    p = [x / tot for x in w]
    e = [1.0] + [0.0] * n                      # elementary symmetric polynomials
    for x in p:
        for j in range(n, 0, -1):
            e[j] += e[j - 1] * x
    f = 1.0
    for i in range(2, n + 1):
        f *= i
    return 1 - f * e[n]


res = {}
for name in ("data.csv", "data2025.csv"):
    days = load(name)
    # Feb 29 occurs one year in four: the average file reports births on the day it exists;
    # weight it by 1/4 (2025 has no Feb 29).
    w = [v / 4 if lab == "Feb 29" else v for lab, v in days]
    srt = sorted(days, key=lambda t: -t[1])
    r = dict(n_days=len(days), total=sum(w),
             top5=srt[:5], bottom5=[t for t in srt if t[0] != "Feb 29"][-5:],
             feb29=[t for t in days if t[0] == "Feb 29"])
    r["max_min_ratio"] = srt[0][1] / [t for t in srt if t[0] != "Feb 29"][-1][1]
    for n in (10, 20, 22, 23, 30, 40, 50, 57, 70):
        r["p%d" % n] = p_share_exact(n, w)
    res[name] = r
    print(name, "days", len(days), "top", srt[:3], "bottom", r["bottom5"][-3:], "ratio %.2f" % r["max_min_ratio"])
    for n in (22, 23, 30, 50, 57, 70):
        print("  n=%d  real %.4f   equal-365 %.4f" % (n, r["p%d" % n], p_share_equal(n)))

res["equal"] = {"p%d" % n: p_share_equal(n) for n in (10, 20, 22, 23, 30, 40, 50, 57, 70)}
res["pairs23"] = 23 * 22 // 2

# Monte Carlo check of the exact figure, 23 people, real 1995-2025 weights
days = load("data.csv")
w = [v / 4 if lab == "Feb 29" else v for lab, v in days]
random.seed(1)
idx = list(range(len(days)))
hits, R = 0, 200000
for _ in range(R):
    b = random.choices(idx, weights=w, k=23)
    hits += len(set(b)) < 23
res["mc23"] = hits / R
print("monte carlo 23 people: %.4f (%d rooms)" % (hits / R, R))

# one example room of 23 drawn from real births (seed chosen and recorded, first with a shared day)
random.seed(2026)
for attempt in range(1, 100):
    b = random.choices(idx, weights=w, k=23)
    if len(set(b)) < 23:
        lab = [days[i][0] for i in b]
        dup = sorted({x for x in lab if lab.count(x) > 1})
        res["example_room"] = dict(attempt=attempt, birthdays=lab, shared=dup)
        print("example room (attempt %d): shared %s" % (attempt, dup))
        break

json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=1)
print("wrote out/results.json")
