"""stats.py - figures for the "harsh raters" beat: how generous readers are, one generous reader's stars, and
reader 9885 vs their reading twin on all the books they share.
  python stats.py --data download"""
import argparse, csv, json, os
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True)
ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")); a = ap.parse_args()
raw = np.loadtxt(os.path.join(a.data, "ratings.csv"), delimiter=",", skiprows=1, dtype=np.int32)
books = {int(b["book_id"]) - 1: b["title"] for b in csv.DictReader(open(os.path.join(a.data, "books.csv"), encoding="utf-8"))}
NU_ALL, NI = raw[:, 0].max(), raw[:, 1].max()
rng = np.random.default_rng(0)
pick = np.sort(rng.choice(NU_ALL, 10000, replace=False)) + 1
remap = np.full(NU_ALL + 1, -1); remap[pick] = np.arange(10000)
sel = remap[raw[:, 0]] >= 0
u, i, r = remap[raw[sel, 0]], raw[sel, 1] - 1, raw[sel, 2].astype(np.float32)
hidden = {}
for row in np.where(r == 5)[0]:
    hidden[int(u[row])] = int(row)
keep = np.ones(len(u), bool); keep[list(hidden.values())] = False
M = np.zeros((10000, NI), np.float32); M[u[keep], i[keep]] = r[keep]; seen = M > 0
n = seen.sum(1); mean = M.sum(1) / np.maximum(n, 1)
five = (M == 5).sum(1) / np.maximum(n, 1)
out = {"readers": 10000, "mean_of_means": float(mean[n > 0].mean()),
       "share_mostly_five": float((five[n >= 20] >= 0.5).mean()), "readers_20plus": int((n >= 20).sum())}
# a generous reader with a similar number of books to 9885 (111): most stars are fives
cand = [p for p in range(10000) if 100 <= n[p] <= 120 and five[p] >= 0.75]
g = max(cand, key=lambda p: five[p])
out["generous"] = {"reader": int(pick[g]), "rated": int(n[g]), "mean": float(mean[g]),
                   "stars": {k: int((M[g] == k).sum()) for k in range(1, 6)}}
p = int(remap[9885])
C = np.where(seen, M - mean[:, None], 0).astype(np.float32); nrm = np.linalg.norm(C, axis=1) + 1e-6
s = (C @ C[p]) / (nrm * nrm[p]); s[p] = -np.inf
nb = np.argsort(-s)[:30]
q = int(nb[1])                                  # the 2nd nearest: the one with 21 books in common
both = np.where(seen[p] & seen[q])[0]
out["twin"] = {"rank": 2, "sim": float(s[q]), "rated": int(n[q]), "mean": float(mean[q]), "in_common": len(both),
               "me_mean": float(mean[p]),
               "books": [{"title": books[k], "me": int(M[p, k]), "them": int(M[q, k]),
                          "me_dev": float(M[p, k] - mean[p]), "them_dev": float(M[q, k] - mean[q])} for k in both]}
dev = np.array([[b["me_dev"], b["them_dev"]] for b in out["twin"]["books"]])
out["twin"]["same_side"] = int((np.sign(dev[:, 0]) == np.sign(dev[:, 1])).sum())
json.dump(out, open(os.path.join(a.out, "stats.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(json.dumps({k: v for k, v in out.items() if k != "twin"}, indent=1))
print("twin", {k: v for k, v in out["twin"].items() if k != "books"})
for b in out["twin"]["books"]: print(" %d %d %+.2f %+.2f %s" % (b["me"], b["them"], b["me_dev"], b["them_dev"], b["title"][:50]))
