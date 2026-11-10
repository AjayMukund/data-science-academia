"""
experiment.py - how does a recommender know what you'll like? Measured on Goodreads ratings (goodbooks-10k).

Plain numpy. Data: goodbooks-10k (Zygmunt Zajac, CC BY-SA 4.0), https://github.com/zygmuntz/goodbooks-10k -
6 million 1-5 star ratings of the 10,000 most-rated books on Goodreads, by 53,424 readers, in the order they were made.
Download ratings.zip and books.zip from its v1.0 release and unzip them into one folder.

To fit in memory we take 10,000 readers at random (seeded) and all 10,000 books.
The test: for every reader, hide the LAST book they gave 5 stars. Each method ranks every book that reader hasn't
rated; a "hit" is the hidden book landing in the top 10 of 10,000.

  1. Most popular: the same top 10 for everyone (most 5-star ratings).
  2. Readers like you: the 30 readers whose ratings agree most with yours (cosine on mean-centred ratings);
     score each book by how much more than usual they liked it.
  3. Taste numbers (matrix factorisation): every reader and every book gets k numbers, learned so that
     reader . book ~= their rating minus the average; score unseen books by that dot product.

  python experiment.py --data download --out out
"""
import argparse, csv, json, os, time
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"))
ap.add_argument("--readers", type=int, default=10000)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

raw = np.loadtxt(os.path.join(a.data, "ratings.csv"), delimiter=",", skiprows=1, dtype=np.int32)
books = {int(b["book_id"]) - 1: b for b in csv.DictReader(open(os.path.join(a.data, "books.csv"), encoding="utf-8"))}
title = lambda k: books[k]["title"]
NB_ALL, NU_ALL = raw[:, 1].max(), raw[:, 0].max()
res = {"all": {"ratings": len(raw), "readers": int(NU_ALL), "books": int(NB_ALL)}}

rng = np.random.default_rng(0)
pick = np.sort(rng.choice(NU_ALL, a.readers, replace=False)) + 1          # reader ids (1-based)
remap = np.full(NU_ALL + 1, -1)
remap[pick] = np.arange(a.readers)
sel = remap[raw[:, 0]] >= 0
order = np.arange(len(raw))[sel]                                         # file order = time order
u, i, r = remap[raw[sel, 0]], raw[sel, 1] - 1, raw[sel, 2].astype(np.float32)
NU, NI = a.readers, NB_ALL
res["sample"] = {"readers": NU, "ratings": int(sel.sum()), "median_per_reader": float(np.median(np.bincount(u, minlength=NU)))}

hidden = {}
for row in np.where(r == 5)[0]:
    hidden[int(u[row])] = int(row)                                       # the last 5-star wins (rows are in time order)
keep = np.ones(len(u), bool)
keep[list(hidden.values())] = False
M = np.zeros((NU, NI), np.float32)
M[u[keep], i[keep]] = r[keep]
seen = M > 0
res["tested_readers"] = len(hidden)


def rank_of(s, p):
    s = s.astype(np.float64).copy()
    s[seen[p]] = -np.inf
    return int((s > s[i[hidden[p]]]).sum()) + 1


def evaluate(score_fn):
    ranks = {p: rank_of(score_fn(p), p) for p in hidden}
    return sum(v <= 10 for v in ranks.values()), ranks


t0 = time.time()
pop = (M == 5).sum(0).astype(np.float64)
h_pop, rk_pop = evaluate(lambda p: pop)
res["popular"] = {"hits": h_pop, "top10": [title(k) for k in np.argsort(-pop)[:10]]}
print("popular", h_pop, "of", len(hidden), flush=True)

n_seen = seen.sum(1)
mean = np.where(n_seen > 0, M.sum(1) / np.maximum(n_seen, 1), 0).astype(np.float32)
C = np.where(seen, M - mean[:, None], 0).astype(np.float32)
nrm = np.linalg.norm(C, axis=1) + 1e-6
K_NB = 30


def neighbours(p):
    s = (C @ C[p]) / (nrm * nrm[p])
    s[p] = -np.inf
    nb = np.argpartition(-s, K_NB)[:K_NB]
    nb = nb[np.argsort(-s[nb])]
    return nb, np.maximum(s[nb], 0)


def score_nb(p):
    nb, w = neighbours(p)
    return (w[:, None] * C[nb]).sum(0) + 0.05 * (w[:, None] * seen[nb]).sum(0)


h_nb, rk_nb = evaluate(score_nb)
res["neighbours"] = {"k": K_NB, "hits": h_nb}
print("neighbours", h_nb, "(%.0fs)" % (time.time() - t0), flush=True)


def als(k, lam=10.0, iters=12, seed=0):
    g = np.random.default_rng(seed)
    gm = float(r[keep].mean())
    P, Q = g.normal(0, 0.1, (NU, k)), g.normal(0, 0.1, (NI, k))
    rows_u = [np.where(seen[p])[0] for p in range(NU)]
    rows_i = [np.where(seen[:, j])[0] for j in range(NI)]
    for _ in range(iters):
        for p in range(NU):
            js = rows_u[p]
            P[p] = np.linalg.solve(Q[js].T @ Q[js] + lam * np.eye(k), Q[js].T @ (M[p, js] - gm))
        for j in range(NI):
            ps = rows_i[j]
            Q[j] = 0 if len(ps) == 0 else np.linalg.solve(P[ps].T @ P[ps] + lam * np.eye(k), P[ps].T @ (M[ps, j] - gm))
    return P, Q, gm


res["taste"] = {}
TASTE = {}
for k in (2, 20):
    P, Q, gm = als(k)
    TASTE[k] = (P, Q)
    h, rk = evaluate(lambda p: P[p] @ Q.T)
    res["taste"][k] = {"hits": h}
    TASTE[k] += (rk,)
    print("taste k=%d" % k, h, "(%.0fs)" % (time.time() - t0), flush=True)

cnt = seen.sum(0)
big = np.where(cnt >= 200)[0]
P2, Q2, _ = TASTE[2]
res["taste2_axes"] = {d: {"low": [title(k) for k in big[np.argsort(Q2[big, d])][:8]],
                          "high": [title(k) for k in big[np.argsort(-Q2[big, d])][:8]]} for d in range(2)}

rk20 = TASTE[20][2]
ex = []
for p in hidden:
    nb, w = neighbours(p)
    ex.append({"reader": int(pick[p]), "rated": int(n_seen[p]), "hidden": title(i[hidden[p]]),
               "rank_popular": rk_pop[p], "rank_neighbours": rk_nb[p], "rank_taste20": rk20[p],
               "top_similarity": float(w[0])})
res["readers"] = ex
json.dump(res, open(os.path.join(a.out, "results.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("DONE %.0fs" % (time.time() - t0))
