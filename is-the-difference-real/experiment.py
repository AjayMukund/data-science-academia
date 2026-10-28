"""
experiment.py - is the difference real? A/B testing on a real email experiment.

Data: Kevin Hillstrom's MineThatData E-Mail Analytics challenge (2008): 64,000 customers of a
retailer, randomly sent a men's email, a women's email or no email; visit / conversion / spend
recorded over two weeks. Publicly released for analysis:
http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv

Everything by re-running the experiment on the real customers (shuffling, re-drawing), no formulas:
  - the real result per group
  - SHUFFLE TEST: men's vs women's email - pool both groups, shuffle who got which email
    100,000 times, count how often luck alone makes a gap as big as the real one
  - A/A TEST: split the no-email customers into two random halves 100,000 times - the gap two
    IDENTICAL groups show, by luck alone
  - SMALL SAMPLES: re-run the experiment with only 1,000 (and 100) customers per email,
    100,000 times - how often the women's email "wins"

    python experiment.py --data hillstrom.csv --out out
"""
import os, csv, json, argparse, urllib.request
import numpy as np
URL = ("http://www.minethatdata.com/"
       "Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv")

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="hillstrom.csv", help="downloaded here on first run")
ap.add_argument("--out", default="out")
ap.add_argument("--runs", type=int, default=100000)
ap.add_argument("--seed", type=int, default=2026)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
if not os.path.exists(a.data):
    urllib.request.urlretrieve(URL, a.data)
rng = np.random.default_rng(a.seed)

rows = list(csv.DictReader(open(a.data, encoding="utf-8")))
seg = np.array([r["segment"] for r in rows])
buy = np.array([int(r["conversion"]) for r in rows])
visit = np.array([int(r["visit"]) for r in rows])
spend = np.array([float(r["spend"]) for r in rows])
R = {"customers": len(rows), "runs": a.runs, "seed": a.seed, "groups": {}}
for s in ("Mens E-Mail", "Womens E-Mail", "No E-Mail"):
    m = seg == s
    R["groups"][s] = {"customers": int(m.sum()), "visits": int(visit[m].sum()), "buyers": int(buy[m].sum()),
                      "visit_rate": round(float(visit[m].mean()), 4), "buy_rate": round(float(buy[m].mean()), 5),
                      "spend_total": round(float(spend[m].sum()), 2), "spend_per_customer": round(float(spend[m].mean()), 3)}
men, wom, none = (seg == "Mens E-Mail"), (seg == "Womens E-Mail"), (seg == "No E-Mail")
nm, nw = int(men.sum()), int(wom.sum())


def rate_gap_shuffled(x, n1, runs):
    """Shuffle x, call the first n1 group 1 and the rest group 2; return the rate gaps (1 - 2)."""
    out = np.empty(runs)
    k = x.sum()
    n = len(x)
    # the number of buyers landing in group 1 after a shuffle is hypergeometric - same as shuffling
    g1 = rng.hypergeometric(k, n - k, n1, size=runs)
    out = g1 / n1 - (k - g1) / (n - n1)
    return out, g1


# --- 1. the real gap, men's vs women's email ---------------------------------------------
real_gap = buy[men].mean() - buy[wom].mean()
pool = np.concatenate([buy[men], buy[wom]])
gaps, g1 = rate_gap_shuffled(pool, nm, a.runs)
as_big = int((np.abs(gaps) >= abs(real_gap) - 1e-12).sum())
# also do it the literal way on 2,000 shuffles, to show the shortcut matches
lit = []
for _ in range(2000):
    p = rng.permutation(pool)
    lit.append(p[:nm].mean() - p[nm:].mean())
lit = np.array(lit)
R["shuffle_test"] = {
    "real_gap_pct_points": round(100 * real_gap, 3),
    "real_buyer_gap": R["groups"]["Mens E-Mail"]["buyers"] - R["groups"]["Womens E-Mail"]["buyers"],
    "shuffled_runs": a.runs, "runs_as_big_as_real": as_big,
    "share_as_big": as_big / a.runs,
    "biggest_shuffled_gap_pct_points": round(100 * float(np.abs(gaps).max()), 3),
    "shuffled_gap_95pct_range_pct_points": [round(100 * float(np.percentile(gaps, 2.5)), 3),
                                            round(100 * float(np.percentile(gaps, 97.5)), 3)],
    "shuffled_buyers_in_men_95pct": [int(np.percentile(g1, 2.5)), int(np.percentile(g1, 97.5))],
    "literal_shuffle_check_95pct": [round(100 * float(np.percentile(lit, 2.5)), 3), round(100 * float(np.percentile(lit, 97.5)), 3)],
    "histogram_pct_points": np.histogram(100 * gaps, bins=41, range=(-0.41, 0.41))[0].tolist(),
}

# --- 2. A/A: two halves of the SAME no-email group ------------------------------------------
x = buy[none]
half = len(x) // 2
aa, ga = rate_gap_shuffled(x, half, a.runs)
R["aa_test"] = {"customers": int(len(x)), "half": half, "buyers": int(x.sum()),
                "gap_95pct_range_pct_points": [round(100 * float(np.percentile(aa, 2.5)), 3),
                                               round(100 * float(np.percentile(aa, 97.5)), 3)],
                "buyers_in_half_95pct": [int(np.percentile(ga, 2.5)), int(np.percentile(ga, 97.5))],
                "example_split_buyers": [int(ga[0]), int(x.sum() - ga[0])]}

# --- 3. the same experiment, smaller --------------------------------------------------------
small = {}
bm, bw = buy[men], buy[wom]
for n in (100, 1000, 5000):
    mi = rng.integers(0, len(bm), size=(a.runs, n))
    wi = rng.integers(0, len(bw), size=(a.runs, n))
    sm, sw = bm[mi].sum(1), bw[wi].sum(1)
    small[str(n)] = {"per_email": n, "women_wins_or_ties": round(float((sw >= sm).mean()), 4),
                     "women_wins": round(float((sw > sm).mean()), 4),
                     "men_buyers_median": int(np.median(sm)), "women_buyers_median": int(np.median(sw)),
                     "example_draws": [[int(sm[i]), int(sw[i])] for i in range(5)]}
R["smaller_experiments"] = small

# --- 4. how big is "real"? ------------------------------------------------------------------
R["per_1000_emails"] = {"extra_buyers_men_vs_women": round(1000 * real_gap, 2),
                        "extra_buyers_men_vs_none": round(1000 * (buy[men].mean() - buy[none].mean()), 2),
                        "extra_spend_men_vs_women": round(1000 * (spend[men].mean() - spend[wom].mean()), 1)}
json.dump(R, open(os.path.join(a.out, "results.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in R.items() if k != "shuffle_test"}, indent=1))
print(json.dumps({k: v for k, v in R["shuffle_test"].items() if k != "histogram_pct_points"}, indent=1))
