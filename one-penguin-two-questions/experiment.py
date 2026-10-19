"""
experiment.py - "Supervised learning: regression vs classification" (master, 2026-10-24).

One penguin, two questions, on the real Palmer Penguins data (Gorman,
Williams & Fraser 2014; palmerpenguins R package, CC0; 344 penguins from
three islands in Antarctica): how heavy is it (a number - regression) and
which species is it (a name - classification)? Plain numpy. Results land in
out/results.json.

    python experiment.py
"""
import os, sys, csv, json, urllib.request
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)
sys.stdout.reconfigure(encoding="utf-8")
SRC = os.path.join(HERE, "download", "penguins.csv")
if not os.path.exists(SRC):
    os.makedirs(os.path.dirname(SRC), exist_ok=True)
    urllib.request.urlretrieve("https://raw.githubusercontent.com/allisonhorst/palmerpenguins/"
                               "main/inst/extdata/penguins.csv", SRC)

rows = list(csv.DictReader(open(SRC, encoding="utf-8")))
F = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
keep = [r for r in rows if all(r[f] != "NA" for f in F)]
SPECIES = ["Adelie", "Chinstrap", "Gentoo"]
X = np.array([[float(r[f]) for f in F] for r in keep])
y_sp = np.array([SPECIES.index(r["species"]) for r in keep])
bill_l, bill_d, flip, mass = X.T
R = {"n_rows": len(rows), "n_used": len(keep), "dropped_missing": len(rows) - len(keep),
     "species_counts": {s: int((y_sp == i).sum()) for i, s in enumerate(SPECIES)}}

rng = np.random.default_rng(7)
idx = rng.permutation(len(keep))
n_tr = int(0.8 * len(keep))
tr, te = idx[:n_tr], idx[n_tr:]
R["n_train"], R["n_test"] = len(tr), len(te)

# ---- regression: how heavy? -------------------------------------------------
base = float(mass[tr].mean())
R["reg_baseline"] = {"guess_g": round(base), "mae_g": round(float(np.abs(mass[te] - base).mean()))}
A = np.c_[flip[tr], np.ones(len(tr))]
slope, icpt = np.linalg.lstsq(A, mass[tr], rcond=None)[0]
pred = slope * flip[te] + icpt
mae = float(np.abs(mass[te] - pred).mean())
ss = 1 - ((mass[te] - pred) ** 2).sum() / ((mass[te] - mass[te].mean()) ** 2).sum()
R["reg_flipper"] = {"slope_g_per_mm": round(float(slope), 1), "intercept_g": round(float(icpt)),
                    "mae_g": round(mae), "r2": round(float(ss), 3)}
# all three measurements
A3 = np.c_[bill_l[tr], bill_d[tr], flip[tr], np.ones(len(tr))]
w3 = np.linalg.lstsq(A3, mass[tr], rcond=None)[0]
p3 = np.c_[bill_l[te], bill_d[te], flip[te], np.ones(len(te))] @ w3
R["reg_three"] = {"mae_g": round(float(np.abs(mass[te] - p3).mean()))}

# ---- classification: which species? ----------------------------------------
maj = int(np.bincount(y_sp[tr]).argmax())
R["cls_baseline"] = {"guess": SPECIES[maj], "acc": round(float((y_sp[te] == maj).mean()), 3)}


def knn(feat_tr, lab_tr, feat_te, k=5):
    mu, sd = feat_tr.mean(0), feat_tr.std(0)
    a, b = (feat_tr - mu) / sd, (feat_te - mu) / sd
    d = ((b[:, None, :] - a[None, :, :]) ** 2).sum(-1)
    nn = np.argsort(d, 1)[:, :k]
    votes = np.array([np.bincount(lab_tr[r], minlength=3) for r in nn])
    return votes.argmax(1), votes


for name, cols in (("flipper", [2]), ("bill", [0, 1]), ("bill_flipper", [0, 1, 2])):
    p, votes = knn(X[tr][:, cols], y_sp[tr], X[te][:, cols])
    acc = float((p == y_sp[te]).mean())
    cm = np.zeros((3, 3), int)
    for t, q in zip(y_sp[te], p):
        cm[t, q] += 1
    R["cls_" + name] = {"acc": round(acc, 3), "right": int((p == y_sp[te]).sum()),
                        "confusion": cm.tolist()}
    if name == "bill":
        BILL_P, BILL_V = p, votes

# ---- one penguin, both questions --------------------------------------------
cands = []
for j, i in enumerate(te):
    pm = slope * flip[i] + icpt
    cands.append((abs(pm - mass[i]), j, i))
cands.sort()
examples = []
for err, j, i in cands:
    if BILL_P[j] == y_sp[i] and BILL_V[j].max() == 5:
        examples.append(i)
    if len(examples) == 3:
        break
R["examples"] = [{"species": SPECIES[y_sp[i]], "island": keep[i]["island"], "sex": keep[i]["sex"],
                  "year": keep[i]["year"],
                  "bill_length_mm": float(bill_l[i]), "bill_depth_mm": float(bill_d[i]),
                  "flipper_length_mm": float(flip[i]), "mass_g": float(mass[i]),
                  "pred_mass_g": round(float(slope * flip[i] + icpt)),
                  "pred_species": SPECIES[BILL_P[list(te).index(i)]],
                  "votes": BILL_V[list(te).index(i)].tolist()} for i in examples]
# one the classifier gets wrong
wrong = [j for j in range(len(te)) if BILL_P[j] != y_sp[te[j]]]
R["wrong"] = [{"true": SPECIES[y_sp[te[j]]], "pred": SPECIES[BILL_P[j]], "votes": BILL_V[j].tolist(),
               "bill_length_mm": float(bill_l[te[j]]), "bill_depth_mm": float(bill_d[te[j]])} for j in wrong]

# per-species averages (for pictures)
R["species_means"] = {s: {f: round(float(X[y_sp == k][:, n].mean()), 1) for n, f in enumerate(F)}
                      for k, s in enumerate(SPECIES)}
R["points"] = [[round(float(bill_l[i]), 1), round(float(bill_d[i]), 1), int(y_sp[i]),
                float(flip[i]), float(mass[i]), int(i in set(te))] for i in range(len(keep))]

json.dump(R, open(os.path.join(OUT, "results.json"), "w", encoding="utf-8"), indent=1)
print(json.dumps({k: v for k, v in R.items() if k != "points"}, indent=1))
