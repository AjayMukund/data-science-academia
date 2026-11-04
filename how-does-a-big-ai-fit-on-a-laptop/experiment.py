"""
experiment.py - how small can a model's numbers get before it stops knowing things? Quantization, measured.

Qwen2.5-0.5B-Instruct (Apache-2.0) in plain numpy (qwen_np.py, the same port as the RAG and KV-cache videos).
Every weight matrix is rounded to fewer bits, in groups of 32 numbers that share one scale (16-bit), the
way common laptop formats do it: 8, 4, 3 and 2 bits. Biases and the small norm weights stay as they are.
For each level we measure:
  - the file size the weights would take,
  - the 50 capitals quiz from the hallucination video (asked in chat, greedy), and
  - one free answer, to read what the damage looks like.

  python experiment.py --models C:/models/hf --out out
"""
import argparse, json, os, sys, time
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

Q.load(snapshot_download("Qwen/Qwen2.5-0.5B-Instruct", cache_dir=a.models,
                         allow_patterns=["config.json", "tokenizer.json", "model.safetensors"]))
ORIG = {k: v.copy() for k, v in Q.T.items()}

REAL = {"France": "Paris", "Germany": "Berlin", "Spain": "Madrid", "China": "Beijing", "Russia": "Moscow",
        "Egypt": "Cairo", "Canada": "Ottawa", "Australia": "Canberra", "Brazil": "Brasilia", "India": "New Delhi",
        "Mexico": "Mexico City", "Kenya": "Nairobi", "Greece": "Athens", "Turkey": "Ankara", "Poland": "Warsaw",
        "Sweden": "Stockholm", "Norway": "Oslo", "Peru": "Lima", "Ireland": "Dublin", "Portugal": "Lisbon",
        "Austria": "Vienna", "Belgium": "Brussels", "Denmark": "Copenhagen", "Finland": "Helsinki",
        "Hungary": "Budapest", "Argentina": "Buenos Aires", "Chile": "Santiago", "Colombia": "Bogota",
        "Cuba": "Havana", "Thailand": "Bangkok", "Vietnam": "Hanoi", "Indonesia": "Jakarta",
        "Pakistan": "Islamabad", "Iran": "Tehran", "Iraq": "Baghdad", "Syria": "Damascus",
        "Saudi Arabia": "Riyadh", "Nigeria": "Abuja", "Ghana": "Accra", "Ethiopia": "Addis Ababa",
        "Morocco": "Rabat", "South Korea": "Seoul", "Philippines": "Manila", "New Zealand": "Wellington",
        "Switzerland": "Bern", "Netherlands": "Amsterdam", "Ukraine": "Kyiv", "Romania": "Bucharest",
        "Bangladesh": "Dhaka", "Nepal": "Kathmandu"}
ALIAS = {"Kyiv": ["Kiev"], "Brasilia": ["Brasília"], "Bogota": ["Bogotá"]}
FREE = "Describe the Eiffel Tower in one sentence."
GROUP = 32


def quantize(w, bits):
    """Round w to a signed integer grid per group of GROUP numbers along each row; return the rounded-back floats."""
    qmax = 2 ** (bits - 1) - 1
    r, c = w.shape
    g = w.reshape(r, c // GROUP, GROUP)
    s = np.abs(g).max(-1, keepdims=True) / qmax
    s = s.astype(np.float16).astype(np.float32)             # the shared scale is stored in 16 bits
    s[s == 0] = 1
    q = np.clip(np.round(g / s), -qmax, qmax)
    return (q * s).reshape(r, c).astype(np.float32)


def set_level(bits):
    n_q = n_keep = 0
    for k, v in ORIG.items():
        if v.ndim == 2 and bits:
            Q.T[k] = quantize(v, bits)
            n_q += v.size
        else:
            Q.T[k] = v
            n_keep += v.size
    if not bits:                                              # the original file: bfloat16, 2 bytes a number
        return n_q + n_keep, (n_q + n_keep) * 2
    return n_q + n_keep, n_q * bits / 8 + n_q / GROUP * 2 + n_keep * 2


def chat(user, n=40):
    ans, ps, _ = Q.chat(user, n=n)
    return ans


def right(ans, city):
    return any(x.lower() in ans.lower() for x in [city] + ALIAS.get(city, []))


res = {"model": "Qwen/Qwen2.5-0.5B-Instruct", "group": GROUP, "levels": {}}
w = ORIG["layers.0.self_attn.q_proj.weight"][0, :8]
res["example_weights"] = {"original": [round(float(x), 4) for x in w]}
for bits in (0, 8, 4, 3, 2):
    lab = "16-bit (original)" if not bits else "%d-bit" % bits
    t0 = time.perf_counter()
    n, size = set_level(bits)
    if bits:
        res["example_weights"][lab] = [round(float(x), 4) for x in Q.T["layers.0.self_attn.q_proj.weight"][0, :8]]
    rows = {}
    for c, city in REAL.items():
        ans = chat("What is the capital of %s? Answer with the city name only." % c, n=8)
        rows[c] = {"answer": ans, "right": right(ans, city)}
    score = sum(r["right"] for r in rows.values())
    free = chat(FREE, n=40)
    res["levels"][lab] = {"bits": bits or 16, "params": n, "mb": round(size / 1e6, 1), "capitals_right": score,
                          "answers": rows, "eiffel": free, "secs": round(time.perf_counter() - t0)}
    print("%-18s %7.1f MB  capitals %2d/50  | %s" % (lab, size / 1e6, score, free.replace("\n", " ")), flush=True)
    json.dump(res, open(os.path.join(a.out, "results.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
# the share of weights that round to exactly zero (every matrix in layers 0, 12 and 23)
zeros = {}
for bits in (8, 4, 3, 2):
    qmax = 2 ** (bits - 1) - 1
    z = tot = 0
    for k, w in ORIG.items():
        if w.ndim != 2 or not any(k.startswith("layers.%d." % n) for n in (0, 12, 23)):
            continue
        g = w.reshape(w.shape[0], -1, GROUP)
        s = np.abs(g).max(-1, keepdims=True) / qmax
        q = np.round(g / np.where(s == 0, 1, s))
        z += int((q == 0).sum()); tot += q.size
    zeros[bits] = round(z / tot, 3)
json.dump({"layers_sampled": [0, 12, 23], "share_rounded_to_zero": zeros}, open(os.path.join(a.out, "zeros.json"), "w"), indent=1)
print("rounded to zero:", zeros)
print("DONE")
