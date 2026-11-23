"""
experiment.py - why do a third of real numbers start with 1? (Benford's law)

Real numbers: every country's population, World Bank WDI indicator SP.POP.TOTL (CC BY 4.0), latest year with
data; plus land area (AG.LND.TOTL.K2) as a second real set. A made-up set: the same count of numbers drawn evenly
between the smallest and largest population (what "random-looking" invented figures do). Benford's prediction:
P(first digit d) = log10(1 + 1/d).

  python experiment.py            (downloads to cache/, writes out/results.json)
"""
import json, math, os, random, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE, OUT = os.path.join(HERE, "cache"), os.path.join(HERE, "out")
os.makedirs(CACHE, exist_ok=True); os.makedirs(OUT, exist_ok=True)


def wb(ind):
    path = os.path.join(CACHE, ind + ".json")
    if not os.path.exists(path):
        url = "https://api.worldbank.org/v2/country/all/indicator/%s?format=json&per_page=20000&mrnev=1" % ind
        req = urllib.request.Request(url, headers={"User-Agent": "research-script"})
        open(path, "wb").write(urllib.request.urlopen(req, timeout=60).read())
    meta = json.load(open(os.path.join(CACHE, "countries.json"))) if os.path.exists(os.path.join(CACHE, "countries.json")) else None
    if meta is None:
        req = urllib.request.Request("https://api.worldbank.org/v2/country?format=json&per_page=400",
                                     headers={"User-Agent": "research-script"})
        open(os.path.join(CACHE, "countries.json"), "wb").write(urllib.request.urlopen(req, timeout=60).read())
        meta = json.load(open(os.path.join(CACHE, "countries.json")))
    real = {c["id"] for c in meta[1] if c["region"]["value"] != "Aggregates"}      # countries, not regions
    rows = json.load(open(path))[1]
    latest = {}                       # the API's mrnev=1 can return several years: keep each country's latest
    for r in rows:
        c = r["countryiso3code"]
        if r["value"] and c in real and (c not in latest or r["date"] > latest[c]["date"]):
            latest[c] = r
    return [(r["country"]["value"], r["countryiso3code"], r["value"], r["date"]) for r in latest.values()]


def first(x):
    return int(str(int(x))[0]) if x >= 1 else int(("%e" % x)[0])


def dist(vals):
    c = [0] * 10
    for v in vals:
        c[first(v)] += 1
    return c[1:]


BEN = [math.log10(1 + 1 / d) for d in range(1, 10)]
pop = wb("SP.POP.TOTL")
area = wb("AG.LND.TOTL.K2")
res = {"benford": BEN}
for name, rows in (("population", pop), ("land_area", area)):
    vals = [r[2] for r in rows]
    c = dist(vals)
    res[name] = dict(n=len(vals), years=sorted({r[3] for r in rows}), counts=c,
                     share=[x / len(vals) for x in c], min=min(rows, key=lambda r: r[2])[:3],
                     max=max(rows, key=lambda r: r[2])[:3],
                     examples=sorted(rows, key=lambda r: -r[2])[:12])
    print(name, len(vals), "years", res[name]["years"])
    for d in range(9):
        print("  %d  %3d  %5.1f%%   Benford %5.1f%%" % (d + 1, c[d], 100 * c[d] / len(vals), 100 * BEN[d]))

random.seed(7)
lo, hi = res["population"]["min"][2], res["population"]["max"][2]
fake = [random.uniform(lo, hi) for _ in range(res["population"]["n"])]
cf = dist(fake)
res["fake_uniform"] = dict(n=len(fake), counts=cf, share=[x / len(fake) for x in cf], lo=lo, hi=hi)
print("made up, evenly between %d and %d" % (lo, hi))
for d in range(9):
    print("  %d  %3d  %5.1f%%" % (d + 1, cf[d], 100 * cf[d] / len(fake)))
# "invented by a person": a first digit chosen evenly, then ordinary random digits, 4-8 digits long
# (revised 2026-11-23: the first version padded with zeros, e.g. 30,000,926, which nobody would type)
def invent():
    k = random.randint(4, 8)
    return int(str(random.randint(1, 9)) + "".join(str(random.randint(0, 9)) for _ in range(k - 1)))
fake2 = [invent() for _ in range(len(fake))]
c2 = dist(fake2)
res["fake_even_digit"] = dict(n=len(fake2), counts=c2, share=[x / len(fake2) for x in c2])
# chi-square-ish distance from Benford, for each set
for k in ("population", "land_area", "fake_uniform", "fake_even_digit"):
    n = res[k]["n"]
    res[k]["chi2"] = sum((res[k]["counts"][d] - n * BEN[d]) ** 2 / (n * BEN[d]) for d in range(9))
    print(k, "chi2 vs Benford %.1f (df 8; 5%% cut-off 15.5)" % res[k]["chi2"])
# doubling: a population of 1x,xxx,xxx needs to double to reach 2; 5 -> 6 needs only +20%
json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=1)
print("wrote out/results.json")
