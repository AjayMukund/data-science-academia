"""
experiment.py - "Did money buy a lifeboat?" (reel, 2026-11-05).

Correlation is not causation, on the Titanic passenger list (Kaggle training
set, 891 passengers). Survival by fare overall, then within each class and
sex, then by fare per person (the fare divided by everyone sharing the
ticket). Stdlib only.

    python experiment.py          -> out/results.json
"""
import os, sys, csv, json, collections, statistics as st, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)
SRC = os.path.join(HERE, "download", "titanic.csv")
if not os.path.exists(SRC):
    os.makedirs(os.path.dirname(SRC), exist_ok=True)
    urllib.request.urlretrieve("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv", SRC)

R = list(csv.DictReader(open(SRC, encoding="utf-8")))
fare = lambda r: float(r["Fare"])
on_ticket = collections.Counter(r["Ticket"] for r in R)
per_person = lambda r: fare(r) / on_ticket[r["Ticket"]]
family = lambda r: int(r["SibSp"]) + int(r["Parch"]) + 1


def survival(rs):
    s = sum(int(r["Survived"]) for r in rs)
    return {"survived": s, "n": len(rs), "pct": round(100 * s / len(rs), 1)}


def by_median(rs, key):
    m = st.median(key(r) for r in rs)
    return [r for r in rs if key(r) > m], [r for r in rs if key(r) <= m]


more, less = by_median(R, fare)
out = {"paid_more": survival(more), "paid_less": survival(less),
       "first_class": [sum(r["Pclass"] == "1" for r in g) for g in (more, less)],
       "women_pct": [round(100 * sum(r["Sex"] == "female" for r in g) / len(g), 1) for g in (more, less)],
       "by_sex": {sx: survival([r for r in R if r["Sex"] == sx]) for sx in ("female", "male")}, "groups": {}}
print("paid more than the median fare: %s%% survived; paid less: %s%%" % (out["paid_more"]["pct"], out["paid_less"]["pct"]))
print("first class among them: %d vs %d; women: %s%% vs %s%%" % (*out["first_class"], *out["women_pct"]))
print("\nwithin each class and sex:           by fare            by fare per person   family size")
for c in "123":
    for sx in ("female", "male"):
        g = [r for r in R if r["Pclass"] == c and r["Sex"] == sx]
        m, l = by_median(g, fare)
        pm, pl = by_median(g, per_person)
        row = {"by_fare": [survival(m), survival(l)], "by_fare_per_person": [survival(pm), survival(pl)],
               "family_size": [round(st.mean(family(r) for r in m), 2), round(st.mean(family(r) for r in l), 2)]}
        out["groups"]["%s_%s" % (c, sx)] = row
        print("  class %s %-6s  more %5.1f%% less %5.1f%%  |  more %5.1f%% less %5.1f%%  |  %.1f vs %.1f"
              % (c, sx, row["by_fare"][0]["pct"], row["by_fare"][1]["pct"], row["by_fare_per_person"][0]["pct"],
                 row["by_fare_per_person"][1]["pct"], *row["family_size"]))
json.dump(out, open(os.path.join(OUT, "results.json"), "w"), indent=1)
