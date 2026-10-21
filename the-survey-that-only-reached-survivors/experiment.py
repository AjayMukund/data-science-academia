"""
experiment.py - "The survey that only reached survivors" (reel, 2026-11-07).

Sampling bias on the Titanic passenger list (Kaggle training set, 891
passengers): a survey that can only reach the 342 survivors, against 1,000
random samples of 100 passengers from everyone. Stdlib only.

    python experiment.py          -> out/results.json
"""
import os, sys, csv, json, random, statistics as st, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)
SRC = os.path.join(HERE, "download", "titanic.csv")
if not os.path.exists(SRC):
    os.makedirs(os.path.dirname(SRC), exist_ok=True)
    urllib.request.urlretrieve("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv", SRC)

R = list(csv.DictReader(open(SRC, encoding="utf-8")))
S = [r for r in R if r["Survived"] == "1"]
FEATURES = {"women": lambda r: r["Sex"] == "female", "first class": lambda r: r["Pclass"] == "1",
            "third class": lambda r: r["Pclass"] == "3"}
share = lambda rs, f: sum(f(r) for r in rs) / len(rs)
rng = random.Random(0)
samples = [[R[i] for i in rng.sample(range(len(R)), 100)] for _ in range(1000)]

out = {"n_ship": len(R), "n_survivors": len(S), "features": {}}
print("%d passengers, %d survivors; 1,000 random samples of 100\n" % (len(R), len(S)))
print("              the ship  survivors  random 100s: middle 95%   highest")
for k, f in FEATURES.items():
    est = sorted(share(s, f) for s in samples)
    v = {"ship": share(R, f), "survivors": share(S, f), "random_mean": st.mean(est),
         "random_95": [est[25], est[974]], "random_highest": est[-1]}
    out["features"][k] = v
    print("  %-11s  %5.1f%%    %5.1f%%       %4.0f%% - %3.0f%%            %3.0f%%"
          % (k, 100 * v["ship"], 100 * v["survivors"], 100 * est[25], 100 * est[974], 100 * est[-1]))
json.dump(out, open(os.path.join(OUT, "results.json"), "w"), indent=1)
