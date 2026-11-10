"""
experiment.py - "most of the model is asleep for every word": a mixture-of-experts model, measured.

Model: IBM Granite 3.0 1B-A400M base (Apache-2.0), run in plain numpy by granite_np.py.
Texts (public domain, Project Gutenberg): Alice's Adventures in Wonderland (#11, English) and Le Rouge et le Noir
(#798, French); Python: this project's own granite_np.py; sums: generated "a + b = c" lines.

  1. Size: every parameter vs what one token actually uses (8 of 32 experts per layer).
  2. Who handles what: per layer, which experts each kind of text is sent to; the most specialised experts.
  3. Fewer or more experts per token (k = 1, 2, 4, 8, 16, 32): next-word accuracy on each text, and time.
  4. The running example: one sentence, which experts each of its words wakes.

  python experiment.py --models <folder for the 5.5 GB download> --out out
"""
import argparse, json, os, re, time
import numpy as np
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # granite_np sits next to this file
import granite_np as G

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--models", required=True)
ap.add_argument("--out", default=os.path.join(HERE, "out"))
ap.add_argument("--tokens", type=int, default=300)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

from huggingface_hub import snapshot_download
G.load(snapshot_download("ibm-granite/granite-3.0-1b-a400m-base", cache_dir=a.models,
                         allow_patterns=["*.json", "*.safetensors"]))
res = {}

# 1. size
total, per_expert, L, E = G.params()
k8 = G.CFG["num_experts_per_tok"]
experts_all = per_expert * L * E
res["size"] = {"total": int(total), "per_expert": int(per_expert), "layers": L, "experts": E, "k": k8,
               "expert_params": int(experts_all), "active": int(total - experts_all + per_expert * L * k8)}
print("size", res["size"], flush=True)


# 2/3. texts
def body(fn, start_pat):
    t = open(os.path.join(HERE, "texts", fn), encoding="utf-8").read()
    return re.sub(r"\s+", " ", t[t.index(start_pat):])


rng = np.random.default_rng(0)
sums = " ".join("%d + %d = %d." % (x, y, x + y) for x, y in rng.integers(10, 999, (200, 2)))
TEXTS = {"english": body("alice_en.txt", "Alice was beginning"),
         "french": body("lerouge_fr.txt", "La petite ville de Verrières"),
         "python": open(os.path.join(HERE, "granite_np.py"), encoding="utf-8").read(),
         "sums": sums}
ids = {k: G.tok.encode(v).ids[:a.tokens] for k, v in TEXTS.items()}

res["by_k"] = {}
usage = {}
for k in (1, 2, 4, 8, 16, 32):
    res["by_k"][k] = {}
    for kind, tid in ids.items():
        G.TRACE.clear()
        t0 = time.time()
        lg = G.forward(tid, [], route_k=k, every=True)       # one pass; every position predicts the next token
        secs = time.time() - t0
        guess = lg[:-1].argmax(-1)
        truth = np.array(tid[1:])
        z = lg[:-1] - lg[:-1].max(-1, keepdims=True)
        nll = -(z[np.arange(len(truth)), truth] - np.log(np.exp(z).sum(-1)))
        if k == 8:
            usage[kind] = np.stack([np.bincount(t.ravel(), minlength=E) for t in G.TRACE[0]])   # (layers, experts)
        res["by_k"][k][kind] = {"secs": round(secs, 2), "right": int((guess == truth).sum()), "of": len(truth),
                                "surprise": float(nll.mean())}
    print("k=%d" % k, {kind: (v["right"], v["secs"]) for kind, v in res["by_k"][k].items()}, flush=True)

# who handles what: an expert's share of each kind's tokens, per layer (k=8, so 8 of 32 is the even share: 25%)
share = {kind: u / len(ids[kind]) for kind, u in usage.items()}
spec = []
for kind in share:
    others = np.max([share[o] for o in share if o != kind], 0)
    for Lr in range(L):
        for e in range(E):
            spec.append((float(share[kind][Lr, e] - others[Lr, e]), kind, Lr, e,
                         {o: round(float(share[o][Lr, e]), 3) for o in share}))
spec.sort(key=lambda s: -s[0])
best = {}
for gap, kind, Lr, e, sh in spec:
    best.setdefault(kind, [])
    if len(best[kind]) < 3:
        best[kind].append({"layer": Lr, "expert": e, "gap": round(gap, 3), "share": sh})
res["specialists"] = best
res["never_used"] = {kind: int((u == 0).sum()) for kind, u in usage.items()}
for kind, b in best.items():
    print("specialist", kind, b[0], flush=True)
json.dump(res, open(os.path.join(a.out, "results.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
