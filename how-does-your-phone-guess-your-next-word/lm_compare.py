"""The same test for a small language model: GPT-2 (124M parameters, OpenAI 2019, MIT licence) reads The Hound of the
Baskervilles and suggests its top 3 next words, scored on the SAME positions as the word-pair counter.
Positions used: the next word is a single GPT-2 token (so either can name it whole). CPU only.
Needs torch + transformers (a scratch venv, never the render venv):  python lm_compare.py"""
import os, re, json, collections, torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
ROOT = os.path.dirname(os.path.abspath(__file__))
import importlib.util
spec = importlib.util.spec_from_file_location("exp", os.path.join(ROOT, "experiment.py"))
# rebuild the counter exactly as experiment.py does (it downloads and counts; re-running is cheap)
exp = importlib.util.module_from_spec(spec); spec.loader.exec_module(exp)
raw = exp.text(exp.HELD).replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
raw = re.sub(r"[ \t]*\r?\n[ \t]*", " ", raw); raw = re.sub(r" +", " ", raw).strip()
wm = [(m.start(), m.group().lower()) for m in re.finditer(r"[A-Za-z]+(?:'[a-z]+)?", raw)]
start2idx = {s: i for i, (s, _) in enumerate(wm)}
tok = GPT2TokenizerFast.from_pretrained("gpt2"); model = GPT2LMHeadModel.from_pretrained("gpt2").eval()
enc = tok(raw, return_offsets_mapping=True); ids = enc["input_ids"]; off = enc["offset_mapping"]
W, S = 1024, 512
top3 = {}
with torch.no_grad():
    for s in range(0, len(ids) - 1, S):
        chunk = ids[max(0, s - (W - S)): s + S]; base = max(0, s - (W - S))
        logits = model(torch.tensor([chunk])).logits[0]
        for j in range(len(chunk)):
            g = base + j
            if g >= s - (0 if s == 0 else 0) and g not in top3: top3[g] = torch.topk(logits[j], 3).indices.tolist()
        print("tokens", min(s + S, len(ids)), "/", len(ids), flush=True)
n = hit_lm = hit_ng = hit_lm1 = hit_ng1 = 0
for j in range(2, len(ids) - 2):
    t1 = tok.convert_ids_to_tokens(ids[j + 1]); t2 = tok.convert_ids_to_tokens(ids[j + 2])
    if not re.fullmatch(r"Ġ[A-Za-z]+", t1) or re.match(r"[A-Za-z']", t2): continue      # next word = one whole token
    st = off[j + 1][0] + 1
    if st not in start2idx: continue
    k = start2idx[st]
    if k < 2: continue
    word = wm[k][1]; a, b = wm[k - 2][1], wm[k - 1][1]
    sug, _ = exp.suggest(a, b)
    lm = [tok.decode([t]).strip().lower() for t in top3[j]]
    n += 1; hit_ng += word in sug; hit_lm += word in lm; hit_ng1 += word == sug[0]; hit_lm1 += word == lm[0]
R = {"positions": n, "of_all_words": len(wm), "counter_top3_pct": round(100 * hit_ng / n, 1), "gpt2_top3_pct": round(100 * hit_lm / n, 1),
     "counter_top1_pct": round(100 * hit_ng1 / n, 1), "gpt2_top1_pct": round(100 * hit_lm1 / n, 1)}
# prompts the counter has never seen
demo = {}
for p in ["I left my laptop on the", "Holmes picked up the phone and", "My dear"]:
    x = tok(p, return_tensors="pt").input_ids
    with torch.no_grad(): lg = model(x).logits[0, -1]
    pr = torch.softmax(lg, -1); tp = torch.topk(pr, 5)
    demo[p] = [(tok.decode([i]).strip(), round(100 * v.item(), 1)) for v, i in zip(tp.values, tp.indices)]
    ws = [w.lower() for w in re.findall(r"[A-Za-z]+", p)]
    demo[p + " | counter"] = exp.suggest(ws[-2], ws[-1]) if len(ws) > 1 else None
R["demo"] = demo
json.dump(R, open(os.path.join(ROOT, "lm_results.json"), "w"), indent=1)
print(json.dumps(R, indent=1))
