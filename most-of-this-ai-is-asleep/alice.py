"""
alice.py - the running example: the opening of Alice's Adventures in Wonderland, in English and in the 1869 French
translation (Henri Bue, Project Gutenberg #55456), through Granite 1B-A400M. Which experts wake for each?

  python alice.py --models <folder> --out out
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
a = ap.parse_args()
from huggingface_hub import snapshot_download
G.load(snapshot_download("ibm-granite/granite-3.0-1b-a400m-base", cache_dir=a.models,
                         allow_patterns=["*.json", "*.safetensors"]))


def body(fn, start):
    t = open(os.path.join(HERE, "texts", fn), encoding="utf-8").read()
    return re.sub(r"\s+", " ", t[t.index(start):])


EN = body("alice_en.txt", "Alice was beginning")
FR = body("alice_fr.txt", "ALICE, assise")
res = {"en_first": EN[:200], "fr_first": FR[:200]}
L, E = G.CFG["num_hidden_layers"], G.CFG["num_local_experts"]

# 1. the first sentence in each language: per layer, which experts did its words use?
used = {}
for lang, txt in (("en", EN), ("fr", FR)):
    ids = G.tok.encode(txt).ids[:300]
    G.TRACE.clear()
    lg = G.forward(ids, [], every=True)
    tr = G.TRACE[0]
    used[lang] = np.stack([np.bincount(t.ravel(), minlength=E) / len(ids) for t in tr])      # share of words per expert
    res[lang] = {"tokens": len(ids), "next_word_right": int((lg[:-1].argmax(-1) == np.array(ids[1:])).sum()),
                 "words": [G.tok.decode([x]) for x in ids[:16]],
                 "layer21_experts_first16": [sorted(int(e) for e in tr[21][n]) for n in range(16)]}
    print(lang, res[lang]["next_word_right"], "/", len(ids) - 1, flush=True)

# 2. the French specialist (layer 21 expert 18) and the biggest English/French gaps, same story
res["l21_e18"] = {"en": float(used["en"][21, 18]), "fr": float(used["fr"][21, 18])}
gap = used["fr"] - used["en"]
top = np.dstack(np.unravel_index(np.argsort(-np.abs(gap).ravel())[:10], gap.shape))[0]
res["biggest_gaps"] = [{"layer": int(l), "expert": int(e), "en": round(float(used["en"][l, e]), 3),
                        "fr": round(float(used["fr"][l, e]), 3)} for l, e in top]
print("L21 E18", res["l21_e18"], flush=True)

# 3. the same prompt with 1, 2, 8 and 32 experts awake
prompt = "Alice was beginning to get very tired of"
res["prompt"] = prompt
res["by_k"] = {}
for k in (1, 2, 4, 8, 32):
    t0 = time.time()
    res["by_k"][k] = {"text": G.generate(prompt, 12, route_k=k, stop=None), "secs": round(time.time() - t0, 1)}
    print(k, repr(res["by_k"][k]["text"]), flush=True)
json.dump(res, open(os.path.join(a.out, "alice.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("DONE")
