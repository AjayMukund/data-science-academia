"""
experiment.py - "Can a computer sort penguins it was never told about?"

The same 342 Palmer Penguins as the supervised master, with the species
labels thrown away. Does k-means, told nothing but "find 3 groups", find the
three species? Plain numpy. Results land in out/results.json.

    python experiment.py
"""
import os, sys, csv, json, itertools, urllib.request
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
y = np.array([SPECIES.index(r["species"]) for r in keep])
R = {"n_used": len(keep), "species_counts": {s: int((y == i).sum()) for i, s in enumerate(SPECIES)}}


def kmeans(Z, k, seed, iters=100, trace=False):
    rng = np.random.default_rng(seed)
    C = Z[rng.choice(len(Z), k, replace=False)].copy()
    hist = [C.copy()]
    for _ in range(iters):
        lab = ((Z[:, None, :] - C[None]) ** 2).sum(2).argmin(1)
        newC = np.array([Z[lab == j].mean(0) if (lab == j).any() else C[j] for j in range(k)])
        if trace:
            hist.append(newC.copy())
        if np.allclose(newC, C):
            break
        C = newC
    lab = ((Z[:, None, :] - C[None]) ** 2).sum(2).argmin(1)
    inertia = float(((Z - C[lab]) ** 2).sum())
    return C, lab, inertia, hist


def best_of(Z, k, seeds=range(50)):
    return min((kmeans(Z, k, s) for s in seeds), key=lambda r: r[2])


def match(lab, k=3):
    """best one-to-one cluster->species map; returns (accuracy, map, table)"""
    T = np.zeros((k, 3), int)
    for c, s in zip(lab, y):
        T[c, s] += 1
    best = max(itertools.permutations(range(3), k), key=lambda p: sum(T[c, p[c]] for c in range(k)))
    return float(sum(T[c, best[c]] for c in range(k)) / len(y)), list(best), T.tolist()


def std(Z):
    return (Z - Z.mean(0)) / Z.std(0)


# ---- 1. beak only, scaled: the picture the video draws -----------------------
Zb = std(X[:, :2])
C, lab, inertia, _ = best_of(Zb, 3)
acc, mp, T = match(lab)
R["beak_scaled"] = {"acc": round(acc, 3), "correct": int(round(acc * len(y))), "table": T,
                    "map": [SPECIES[m] for m in mp],
                    "centres_mm": (C * X[:, :2].std(0) + X[:, :2].mean(0)).round(1).tolist()}
# an animated run (the one the video shows): the first seed whose run ends on the
# best answer exactly. Two answers tie to 0.006 inertia and differ by one
# borderline penguin (317 vs 318 right); the video uses the best one.
seed_t = next(s for s in range(1000) if abs(kmeans(Zb, 3, s)[2] - inertia) < 1e-6)
_, lab_t, _, hist = kmeans(Zb, 3, seed=seed_t, trace=True)
R["beak_trace"] = {"seed": int(seed_t), "correct": int(round(match(lab_t)[0] * len(y))), "steps": len(hist) - 1,
                   "centres_mm": [(h * X[:, :2].std(0) + X[:, :2].mean(0)).round(2).tolist() for h in hist],
                   "acc": round(match(lab_t)[0], 3)}

# ---- 2. all four measurements, scaled ---------------------------------------
Z4 = std(X)
C4, lab4, _, _ = best_of(Z4, 3)
acc4, mp4, T4 = match(lab4)
R["all4_scaled"] = {"acc": round(acc4, 3), "correct": int(round(acc4 * len(y))), "table": T4,
                    "map": [SPECIES[m] for m in mp4]}

# ---- 3. all four, NOT scaled: grams drown millimetres -----------------------
_, labr, _, _ = best_of(X, 3)
accr, mpr, Tr = match(labr)
R["all4_raw"] = {"acc": round(accr, 3), "correct": int(round(accr * len(y))), "table": Tr,
                 "map": [SPECIES[m] for m in mpr],
                 "spread": {f: round(float(X[:, i].std()), 1) for i, f in enumerate(F)}}

# ---- 4. how many groups? the elbow -----------------------------------------
R["elbow_beak"] = {k: round(best_of(Zb, k)[2], 1) for k in range(1, 8)}
R["elbow_all4"] = {k: round(best_of(Z4, k)[2], 1) for k in range(1, 8)}

# ---- 5. bad starts: how often does a single run land in a worse answer? -----
# one bad start in a hundred: the 1,000 single runs land in two families
# (inertia ~186.9, the species; ~303.7, a wrong split) - count by family
runs = [kmeans(Zb, 3, s) for s in range(1000)]
good = [r for r in runs if r[2] < 200]
bad = [r for r in runs if r[2] >= 200]
R["restarts_beak"] = {"runs": 1000, "good": len(good), "stuck": len(bad),
                      "good_inertia": round(min(r[2] for r in good), 1),
                      "stuck_inertia": round(min(r[2] for r in bad), 1) if bad else None,
                      "stuck_correct": int(round(match(bad[0][1])[0] * len(y))) if bad else None,
                      "stuck_table": match(bad[0][1])[2] if bad else None}

# ---- 6. sex: a second "true" grouping nobody asked for ------------------------
sx = np.array([r["sex"] for r in keep])
R["sex_counts"] = {s: int((sx == s).sum()) for s in set(sx)}

json.dump(R, open(os.path.join(OUT, "results.json"), "w"), indent=1)
print("beak, scaled:      %d of %d right" % (R["beak_scaled"]["correct"], R["n_used"]))
print("all four, scaled:  %d" % R["all4_scaled"]["correct"])
print("all four, raw:     %d   (grams drown millimetres)" % R["all4_raw"]["correct"])
print("1,000 random starts: %d good, %d stuck at %d right" % (R["restarts_beak"]["good"], R["restarts_beak"]["stuck"],
                                                       R["restarts_beak"]["stuck_correct"]))
print("elbow (beak):", R["elbow_beak"])
