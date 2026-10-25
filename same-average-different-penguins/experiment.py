"""
experiment.py - "Same average, different penguins" (reel, 2026-11-08).

Standard deviation on the real Palmer Penguins data (CC0): the 58 Gentoo
females and all 167 penguins on Biscoe island have the same average weight,
4.7 kg, but very different spread. How often does guessing with the average
land within half a kilo, and why is Biscoe so wide? Stdlib only.

    python experiment.py          -> out/results.json
"""
import os, sys, csv, json, statistics as st, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)
sys.stdout.reconfigure(encoding="utf-8")
SRC = os.path.join(HERE, "download", "penguins.csv")
if not os.path.exists(SRC):
    os.makedirs(os.path.dirname(SRC), exist_ok=True)
    urllib.request.urlretrieve("https://raw.githubusercontent.com/allisonhorst/palmerpenguins/"
                               "main/inst/extdata/penguins.csv", SRC)

R = [r for r in csv.DictReader(open(SRC, encoding="utf-8")) if r["body_mass_g"] != "NA"]
mass = lambda rs: [float(r["body_mass_g"]) for r in rs]


def describe(v):
    """Standard deviation: square each distance from the average, average them, take the root."""
    m, sd = st.mean(v), st.pstdev(v)
    return {"n": len(v), "average_g": m, "standard_deviation_g": sd, "lightest_g": min(v), "heaviest_g": max(v),
            "within_500g_of_average": sum(abs(x - m) <= 500 for x in v) / len(v)}


groups = {"Gentoo females": mass([r for r in R if r["species"] == "Gentoo" and r["sex"] == "female"]),
          "every penguin on Biscoe": mass([r for r in R if r["island"] == "Biscoe"])}
out = {k: describe(v) for k, v in groups.items()}
out["Biscoe by species"] = {sp: describe(mass([r for r in R if r["island"] == "Biscoe" and r["species"] == sp]))
                            for sp in ("Adelie", "Gentoo")}
json.dump(out, open(os.path.join(OUT, "results.json"), "w"), indent=1)
for k in groups:
    d = out[k]
    print("%-24s n %3d  average %4.0f g ± %3.0f g  range %4.0f-%4.0f g  guess within 500 g: %2.0f%%"
          % (k, d["n"], d["average_g"], d["standard_deviation_g"], d["lightest_g"], d["heaviest_g"],
             100 * d["within_500g_of_average"]))
for sp, d in out["Biscoe by species"].items():
    print("  Biscoe %-7s n %3d  average %4.0f g" % (sp, d["n"], d["average_g"]))
