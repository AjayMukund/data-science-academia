"""
experiment.py - why does a chatbot answer you instead of continuing your sentence? Base vs tuned, measured.

Two releases of the same network, run by us in plain numpy (qwen_np.py):
  Qwen/Qwen2.5-0.5B           the base model: trained only to continue text
  Qwen/Qwen2.5-0.5B-Instruct  the same model after instruction tuning and preference training (Apache-2.0)
We ask both the same questions (greedy, so a re-run gives the same words):
  1. "What is the capital of X?" as plain text to the base model, and in the chat format to the tuned one:
     does it answer, or keep writing more of the page?
  2. Does the base model know the answer anyway? Its next-word scoreboard after "The capital of X is".
  3. A wrong claim ("The capital of Australia is Sydney, right?"): who agrees?

  python experiment.py --models C:/models/hf --out out
"""
import argparse, json, os, sys
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--models", default=os.path.expanduser("~/models/hf"), help="Hugging Face cache folder")
ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"))
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qwen_np as Q
from huggingface_hub import snapshot_download

PAT = ["config.json", "tokenizer.json", "model.safetensors"]
REAL = {"France": "Paris", "Germany": "Berlin", "Spain": "Madrid", "China": "Beijing", "Russia": "Moscow",
        "Egypt": "Cairo", "Canada": "Ottawa", "Australia": "Canberra", "Brazil": "Brasilia", "India": "New Delhi",
        "Mexico": "Mexico City", "Kenya": "Nairobi", "Greece": "Athens", "Turkey": "Ankara", "Poland": "Warsaw",
        "Sweden": "Stockholm", "Norway": "Oslo", "Peru": "Lima", "Ireland": "Dublin", "Portugal": "Lisbon"}
# a well-known wrong answer for each (the biggest or best-known other city)
WRONG = {"France": "Lyon", "Germany": "Munich", "Spain": "Barcelona", "China": "Shanghai", "Russia": "St Petersburg",
         "Egypt": "Alexandria", "Canada": "Toronto", "Australia": "Sydney", "Brazil": "Rio de Janeiro",
         "India": "Mumbai", "Mexico": "Guadalajara", "Kenya": "Mombasa", "Greece": "Thessaloniki",
         "Turkey": "Istanbul", "Poland": "Krakow", "Sweden": "Gothenburg", "Norway": "Bergen", "Peru": "Cusco",
         "Ireland": "Cork", "Portugal": "Porto"}
END_IDS = None


def greedy(text, n=40):
    ids = Q.tok.encode(text).ids
    cache = []
    z = Q.forward(ids, cache)
    out = []
    for _ in range(n):
        i = int(np.argmax(z))
        if i in END_IDS:
            break
        out.append(i)
        z = Q.forward([i], cache)
    return Q.tok.decode(out)


def scoreboard(text, top=5):
    p = Q.probs(Q.forward(Q.tok.encode(text).ids, []))
    o = np.argsort(-p)[:top]
    return [(Q.tok.decode([int(i)]), round(float(p[i]), 4)) for i in o]


def chat_prompt(user):
    return ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n%s<|im_end|>\n"
            "<|im_start|>assistant\n" % user)


res = {}
for name, rid in (("base", "Qwen/Qwen2.5-0.5B"), ("tuned", "Qwen/Qwen2.5-0.5B-Instruct")):
    Q.T.clear(); Q.CFG.clear()
    Q.load(snapshot_download(rid, cache_dir=a.models, allow_patterns=PAT))
    END_IDS = {Q.tok.token_to_id("<|im_end|>"), Q.tok.token_to_id("<|endoftext|>")}
    r = {"model": rid, "ask": {}, "knows": {}, "wrong_claim": {}}
    wrap = (lambda u: u + "\n") if name == "base" else chat_prompt     # the base model has no chat format
    for c, city in REAL.items():
        out = greedy(wrap("What is the capital of %s?" % c), 40)
        r["ask"][c] = out
        r["knows"][c] = scoreboard("The capital of %s is" % c)
        claim = "The capital of %s is %s, right?" % (c, WRONG[c])
        r["wrong_claim"][c] = greedy(wrap(claim), 30)
        print(name, c, "|", out[:70].replace("\n", " / "), "|", r["knows"][c][0], "|",
              r["wrong_claim"][c][:60].replace("\n", " / "), flush=True)
    # 4. which city does it find more likely after "The capital of X is"? (whole name: every token's log-probability)
    r["prefers_right"] = {}
    for c, city in REAL.items():
        lp = {}
        for cand in (city, WRONG[c]):
            pre = Q.tok.encode("The capital of %s is" % c).ids
            ids = pre + Q.tok.encode(" " + cand).ids
            cache, tot = [], 0.0
            z = Q.forward(pre, cache)
            for i in ids[len(pre):]:
                tot += float(np.log(Q.probs(z)[i]))
                z = Q.forward([i], cache)
            lp[cand] = round(tot, 3)
        r["prefers_right"][c] = {"logp": lp, "right_more_likely": lp[city] > lp[WRONG[c]]}
    print(name, "prefers the real capital:", sum(v["right_more_likely"] for v in r["prefers_right"].values()), "/ 20",
          flush=True)
    res[name] = r
    json.dump(res, open(os.path.join(a.out, "results.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
# the same numbers, nudged: one weight row in both files, and how many numbers each file holds
import struct, glob
def _read(rid, key):
    path = os.path.join(snapshot_download(rid, cache_dir=a.models, allow_patterns=PAT), "model.safetensors")
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]; h = json.loads(f.read(n)); m = h[key]
        s, e = m["data_offsets"]; f.seek(8 + n + s); raw = f.read(e - s)
    total = sum(int(np.prod(v["shape"])) for k, v in h.items() if k != "__metadata__")
    return (np.frombuffer(raw, np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(m["shape"]), total
key = "model.layers.0.self_attn.q_proj.weight"
wb, nb = _read("Qwen/Qwen2.5-0.5B", key)
wt, nt = _read("Qwen/Qwen2.5-0.5B-Instruct", key)
json.dump({"tensor": key, "row0_base": [round(float(x), 4) for x in wb[0, :6]], "row0_tuned": [round(float(x), 4) for x in wt[0, :6]],
           "mean_abs_change": float(np.abs(wb - wt).mean()), "mean_abs_weight": float(np.abs(wb).mean()), "params": nt,
           "params_base": nb}, open(os.path.join(a.out, "weights.json"), "w"), indent=1)
print("weights: same count", nb == nt, nt)
print("DONE")
