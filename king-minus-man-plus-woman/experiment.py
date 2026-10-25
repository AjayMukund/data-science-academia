"""
experiment.py - "King minus man plus woman equals... king?" (reel, 2026-11-10).

GloVe
(glove-wiki-gigaword-300: 400,000 words x 300 numbers, from the gensim-data
release) and plain numpy, cosine similarity throughout. Every analogy is
ranked twice: over all 400,000 words, and with the three input words excluded
(what the famous demos do).

    python experiment.py          -> out/results.json  (downloads 376 MB of GloVe on first run)
    python experiment.py path/to/glove-wiki-gigaword-300.gz
"""
import gzip, json, os, sys, urllib.request
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)
URL = ("https://github.com/RaRe-Technologies/gensim-data/releases/download/"
       "glove-wiki-gigaword-300/glove-wiki-gigaword-300.gz")


def load(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        n, d = map(int, f.readline().split())
        words, M = [], np.empty((n, d), dtype=np.float32)
        for i, line in enumerate(f):
            p = line.rstrip().split(" ")
            words.append(p[0])
            M[i] = np.asarray(p[1:], dtype=np.float32)
    return words, M


def main(path):
    words, M = load(path)
    idx = {w: i for i, w in enumerate(words)}
    U = M / np.linalg.norm(M, axis=1, keepdims=True)
    v = lambda w: U[idx[w]]
    cos = lambda a, b: float(v(a) @ v(b))

    def analogy(a, b, c, k=6):
        q = v(a) - v(b) + v(c)
        q = q / np.linalg.norm(q)
        s = U @ q
        order = np.argsort(-s)
        inc = [(words[i], float(s[i])) for i in order[:k]]
        exc = [(words[i], float(s[i])) for i in order[:k + 3] if words[i] not in (a, b, c)][:k]
        return {"included": inc, "excluded": exc,
                "rank_included": {w: int(np.where(order == idx[w])[0][0]) + 1 for w, _ in exc[:2] + inc[:1]}}

    out = {"n_words": len(words), "dims": int(M.shape[1]),
           "similarity": {"king queen": cos("king", "queen"), "king man": cos("king", "man"),
                          "king strawberry": cos("king", "strawberry"), "man woman": cos("man", "woman")},
           "analogies": {}}
    for a, b, c in (("king", "man", "woman"), ("uncle", "man", "woman"), ("bigger", "big", "small"),
                    ("paris", "france", "japan")):
        out["analogies"]["%s - %s + %s" % (a, b, c)] = analogy(a, b, c)
    out["first_numbers"] = {w: [float(x) for x in M[idx[w]][:6]] for w in ("king", "queen", "strawberry")}   # raw, as stored
    # a picture: x = the man->woman direction, y = the main remaining variation of these words
    pic = ["man", "woman", "king", "queen", "uncle", "aunt", "boy", "girl", "prince", "princess"]
    g = v("woman") - v("man"); g /= np.linalg.norm(g)
    X = np.stack([v(w) for w in pic])
    R = X - np.outer(X @ g, g)
    R0 = R - R.mean(0)
    y = np.linalg.svd(R0, full_matrices=False)[2][0]
    if (v("king") - v("man")) @ y < 0:
        y = -y
    out["picture"] = {"axes": "x = the man->woman direction; y = the main remaining variation of these words",
                      "words": {w: [float(v(w) @ g), float(v(w) @ y)] for w in pic}}
    q = v("king") - v("man") + v("woman"); q /= np.linalg.norm(q)
    out["picture"]["king-man+woman"] = [float(q @ g), float(q @ y)]
    json.dump(out, open(os.path.join(OUT, "results.json"), "w", encoding="utf-8"), indent=1)
    print("%d words x %d numbers" % (len(words), M.shape[1]))
    print("similarity", {k: round(s, 3) for k, s in out["similarity"].items()})
    for k, r in out["analogies"].items():
        print("\n%s\n  all words:      %s\n  inputs removed: %s\n  ranks: %s" % (
            k, ", ".join("%s %.3f" % t for t in r["included"]), ", ".join("%s %.3f" % t for t in r["excluded"]),
            r["rank_included"]))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        src = sys.argv[1]
    else:
        src = os.path.join(HERE, "download", "glove-wiki-gigaword-300.gz")
        if not os.path.exists(src):
            os.makedirs(os.path.dirname(src), exist_ok=True)
            print("downloading GloVe (376 MB) ...")
            urllib.request.urlretrieve(URL, src)
    main(src)
