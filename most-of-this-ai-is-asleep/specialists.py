"""specialists.py - how many of the 768 experts have a clear specialty on our four texts (English, French, Python,
sums)? An expert "specialises" when its share of one kind of text beats its share of every other kind by a margin.
  python specialists.py --models <folder>"""
import argparse, json, os, re
import numpy as np
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # granite_np sits next to this file
import granite_np as G
HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser(); ap.add_argument("--models", required=True)
ap.add_argument("--out", default=os.path.join(HERE, "out")); a = ap.parse_args()
from huggingface_hub import snapshot_download
G.load(snapshot_download("ibm-granite/granite-3.0-1b-a400m-base", cache_dir=a.models, allow_patterns=["*.json", "*.safetensors"]))
def body(fn, start):
    t = open(os.path.join(HERE, "texts", fn), encoding="utf-8").read()
    return re.sub(r"\s+", " ", t[t.index(start):])
rng = np.random.default_rng(0)
TEXTS = {"english": body("alice_en.txt", "Alice was beginning"), "french": body("lerouge_fr.txt", "La petite ville de Verrières"),
         "python": open(os.path.join(HERE, "granite_np.py"), encoding="utf-8").read(),
         "sums": " ".join("%d + %d = %d." % (x, y, x + y) for x, y in rng.integers(10, 999, (200, 2)))}
L, E = G.CFG["num_hidden_layers"], G.CFG["num_local_experts"]
share = {}
for kind, txt in TEXTS.items():
    ids = G.tok.encode(txt).ids[:300]
    G.TRACE.clear(); G.forward(ids, [], every=True)
    share[kind] = np.stack([np.bincount(t.ravel(), minlength=E) / len(ids) for t in G.TRACE[0]])
S = np.stack([share[k] for k in TEXTS])                      # (4, L, E)
top = S.max(0); second = np.sort(S, 0)[-2]
res = {"experts": L * E, "margin_counts": {}}
for m in (0.3, 0.5, 0.7):
    res["margin_counts"][str(m)] = int((top - second > m).sum())
res["by_kind_0.5"] = {k: int(((S[i] - np.delete(S, i, 0).max(0)) > 0.5).sum()) for i, k in enumerate(TEXTS)}
res["never_used_any"] = int((S.max(0) == 0).sum())
# the running example for the animation: every layer's 8 experts for Alice's opening, both languages,
# and the router's real scores and blend weights in layer 21 for the first word piece
routes = {}
for lang, fn, start in (("en", "alice_en.txt", "Alice was beginning"), ("fr", "alice_fr.txt", "ALICE, assise")):
    ids = G.tok.encode(body(fn, start)).ids[:24]
    G.TRACE.clear(); G.GATES.clear(); G.forward(ids, [], every=True)
    routes[lang] = {"pieces": [G.tok.decode([x]) for x in ids],
                    "experts": [[sorted(int(e) for e in G.TRACE[0][l][n]) for l in range(L)] for n in range(len(ids))]}
    lg, top, g = G.GATES[21]
    routes[lang]["l21_first"] = {"scores": [float(v) for v in lg[0]], "chosen": [int(v) for v in top[0]], "weights": [float(v) for v in g[0]]}
json.dump(routes, open(os.path.join(a.out, "routes.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
json.dump(res, open(os.path.join(a.out, "specialists.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
