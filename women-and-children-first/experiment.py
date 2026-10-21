"""
experiment.py - "Was it really women and children first?"

Exploratory data analysis on the Titanic passenger list (Kaggle training set,
891 passengers; mirror
https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv):
one number, then group by group, then one group split by another. Stdlib +
numpy only.

    python experiment.py          -> out/results.json
"""
import os, sys, csv, re, json, urllib.request
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)
sys.stdout.reconfigure(encoding="utf-8")
SRC = os.path.join(HERE, "download", "titanic.csv")
if not os.path.exists(SRC):
    os.makedirs(os.path.dirname(SRC), exist_ok=True)
    urllib.request.urlretrieve("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv", SRC)

rows = list(csv.DictReader(open(SRC, encoding="utf-8")))


def rate(rs):
    """(survival %, survivors, group size)"""
    s = sum(int(r["Survived"]) for r in rs)
    return {"pct": round(100 * s / len(rs), 1), "survived": s, "n": len(rs)}


def child(r):
    return r["Age"].strip() != "" and float(r["Age"]) < 13


R = {"all": rate(rows)}
R["sex"] = {sx: rate([r for r in rows if r["Sex"] == sx]) for sx in ("female", "male")}
R["children"] = rate([r for r in rows if child(r)])
R["children_note"] = {"age_known": sum(1 for r in rows if r["Age"].strip()), "age_blank": sum(1 for r in rows if not r["Age"].strip())}
R["adult_men"] = rate([r for r in rows if r["Sex"] == "male" and r["Age"].strip() and float(r["Age"]) >= 13])
R["class"] = {c: rate([r for r in rows if r["Pclass"] == c]) for c in "123"}
R["sex_by_class"] = {c: {sx: rate([r for r in rows if r["Pclass"] == c and r["Sex"] == sx]) for sx in ("female", "male")}
                     for c in "123"}
R["children_by_class"] = {c: rate([r for r in rows if r["Pclass"] == c and child(r)]) for c in "123"}
# survival by age decade (known ages): the picture, not the average
aged = [r for r in rows if r["Age"].strip()]
R["age_decades"] = [[lo, rate([r for r in aged if lo <= float(r["Age"]) < lo + 10])] for lo in range(0, 80, 10)]
# group sizes, so a tiny group is never read as a big one
R["smallest_group"] = min(((c, v["n"]) for c, v in R["children_by_class"].items()), key=lambda t: t[1])

json.dump(R, open(os.path.join(OUT, "results.json"), "w"), indent=1)
print("everyone:            %(pct)s%% (%(survived)s of %(n)s)" % R["all"])
for sx in ("female", "male"):
    print("%-20s %s%% (%s of %s)" % (sx + ":", R["sex"][sx]["pct"], R["sex"][sx]["survived"], R["sex"][sx]["n"]))
print("children under 13:   %(pct)s%% (%(survived)s of %(n)s, age known)" % R["children"])
for c in "123":
    f, m = R["sex_by_class"][c]["female"], R["sex_by_class"][c]["male"]
    k = R["children_by_class"][c]
    print("class %s: women %s%% (n %s)  men %s%% (n %s)  children %s%% (n %s)" % (c, f["pct"], f["n"], m["pct"], m["n"],
                                                                                 k["pct"], k["n"]))
