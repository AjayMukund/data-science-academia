"""
generate_ai.py - the AI-written side: Qwen2.5-0.5B-Instruct (Apache 2.0) writes one ~100-word encyclopedia paragraph
on each of the same 50 topics. Two settings:
  shipped: the model's own default sampling (temperature 0.7, top-p 0.8, repetition penalty 1.05)
  warm:    temperature 1.0, no cut-offs (the twist: the same AI, plain sampling, dice rolled wider)
Seeded per topic. Writes out/ai.json.   python generate_ai.py --models <hf cache folder>
"""
import argparse, json, os, re, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from topics import TOPICS
ap = argparse.ArgumentParser(); ap.add_argument("--models", required=True); a = ap.parse_args()
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)
torch.set_grad_enabled(False); torch.set_num_threads(8)
from transformers import AutoTokenizer, AutoModelForCausalLM
M = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(M, cache_dir=a.models)
model = AutoModelForCausalLM.from_pretrained(M, cache_dir=a.models, torch_dtype=torch.float32).eval()
human = {h["topic"]: h for h in json.load(open(next(p for p in (os.path.join(OUT, "human.json"), os.path.join(os.path.dirname(OUT), "data", "human.json")) if os.path.exists(p)), encoding="utf-8"))}

def trim(t):
    t = re.sub(r"\s+", " ", t.replace("*", "").replace("#", "")).strip()
    sents = re.split(r"(?<=[.!?])\s+", t); keep, n = [], 0
    for s in sents:
        w = len(s.split())
        if n and n + w > 110: break
        keep.append(s); n += w
    return " ".join(keep)

res = []
for i, t in enumerate(TOPICS):
    title = human[t]["title"]
    msgs = [{"role": "user", "content": f"Write one paragraph of about 100 words for an encyclopedia about: {title}."}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
    row = dict(topic=t, title=title)
    for name, kw in (("shipped", dict(temperature=0.7, top_p=0.8, top_k=20, repetition_penalty=1.05)),
                     ("warm", dict(temperature=1.0, top_p=1.0, top_k=0, repetition_penalty=1.0))):
        torch.manual_seed(1000 + i)
        out = model.generate(ids, do_sample=True, max_new_tokens=190, **kw)
        row[name] = trim(tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True))
    res.append(row); print(i, t, len(row["shipped"].split()), len(row["warm"].split()), flush=True)
json.dump(res, open(os.path.join(OUT, "ai.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
