"""detail.py - the running example: one reader, their 30 nearest readers, and why their hidden favourite came first.
  python detail.py --data download --reader 37188"""
import argparse, csv, json, os
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--reader", type=int, required=True)
ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")); a = ap.parse_args()
raw = np.loadtxt(os.path.join(a.data, "ratings.csv"), delimiter=",", skiprows=1, dtype=np.int32)
books = {int(b["book_id"]) - 1: b for b in csv.DictReader(open(os.path.join(a.data, "books.csv"), encoding="utf-8"))}
t = lambda k: books[k]["title"]
NU_ALL, NI = raw[:, 0].max(), raw[:, 1].max()
rng = np.random.default_rng(0)                                        # the same 10,000 readers as experiment.py
pick = np.sort(rng.choice(NU_ALL, 10000, replace=False)) + 1
remap = np.full(NU_ALL + 1, -1); remap[pick] = np.arange(10000)
sel = remap[raw[:, 0]] >= 0
u, i, r = remap[raw[sel, 0]], raw[sel, 1] - 1, raw[sel, 2].astype(np.float32)
hidden = {}
for row in np.where(r == 5)[0]:
    hidden[int(u[row])] = int(row)
keep = np.ones(len(u), bool); keep[list(hidden.values())] = False
M = np.zeros((10000, NI), np.float32); M[u[keep], i[keep]] = r[keep]; seen = M > 0
p = int(remap[a.reader]); target = int(i[hidden[p]])
mean = M.sum(1) / np.maximum(seen.sum(1), 1)
C = np.where(seen, M - mean[:, None], 0).astype(np.float32); nrm = np.linalg.norm(C, axis=1) + 1e-6
s = (C @ C[p]) / (nrm * nrm[p]); s[p] = -np.inf
nb = np.argsort(-s)[:30]; w = np.maximum(s[nb], 0)
score = (w[:, None] * C[nb]).sum(0) + 0.05 * (w[:, None] * seen[nb]).sum(0); score[seen[p]] = -np.inf
mine = np.where(seen[p])[0]
out = {"reader": a.reader, "rated": int(seen[p].sum()), "mean": float(mean[p]), "hidden": t(target),
       "stars": {int(k): int((M[p] == k).sum()) for k in range(1, 6)},
       "fives": [t(k) for k in mine if M[p, k] == 5],
       "ones_twos": [(t(k), int(M[p, k])) for k in mine if M[p, k] <= 2],
       "top10": [t(k) for k in np.argsort(-score)[:10]],
       "nearest": [], "nb_on_target": []}
for q, sim in zip(nb[:5], s[nb[:5]]):
    both = np.where(seen[p] & seen[q])[0]
    out["nearest"].append({"sim": float(sim), "rated": int(seen[q].sum()), "in_common": len(both),
                           "same_stars": int((M[p, both] == M[q, both]).sum()),
                           "examples": [(t(k), int(M[p, k]), int(M[q, k])) for k in both[:12]]})
for q, sim in zip(nb, s[nb]):
    if seen[q, target]:
        out["nb_on_target"].append({"sim": float(sim), "stars": int(M[q, target]), "their_mean": float(mean[q])})
out["readers_of_target_in_sample"] = int(seen[:, target].sum())
out["five_star_target_in_sample"] = int((M[:, target] == 5).sum())
json.dump(out, open(os.path.join(a.out, "reader_%d.json" % a.reader), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(json.dumps(out, indent=1, ensure_ascii=False))
