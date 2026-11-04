"""
experiment.py - does an AI get better at maths when it "thinks" first? Measured.

Qwen2.5-0.5B-Instruct (Apache-2.0) in plain numpy (qwen_np.py), greedy, on the first 50 problems of
GSM8K's test set (grade-school word problems; OpenAI, MIT licence). Each problem is asked three ways:
  direct  "... Answer with the final number only." (it often starts explaining anyway, and is cut off at 16 tokens)
  forced  no room to think: the reply is pre-started with "The final answer is" and the next number is read
  steps   "... Think step by step, then give the final answer on the last line as 'Answer: <number>'."
We record the answer, whether it matches GSM8K's, how many tokens it wrote and how long it took.

  python experiment.py --models C:/models/hf --out out
"""
import argparse, json, os, re, sys, time, urllib.request
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--models", default=os.path.expanduser("~/models/hf"), help="Hugging Face cache folder")
ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"))
ap.add_argument("--n", type=int, default=50)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qwen_np as Q
from huggingface_hub import snapshot_download

URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
cache = os.path.join(a.out, "gsm8k_test.jsonl")
if not os.path.exists(cache):
    urllib.request.urlretrieve(URL, cache)
PROBS = [json.loads(l) for l in open(cache, encoding="utf-8")][:a.n]

Q.load(snapshot_download("Qwen/Qwen2.5-0.5B-Instruct", cache_dir=a.models,
                         allow_patterns=["config.json", "tokenizer.json", "model.safetensors"]))
DIRECT = "%s\nAnswer with the final number only."
STEPS = "%s\nThink step by step, then give the final answer on the last line as 'Answer: <number>'."


def forced(question):
    """The reply is pre-started with "The final answer is", so the very next tokens must be the answer."""
    prompt = ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n%s<|im_end|>\n"
              "<|im_start|>assistant\nThe final answer is" % question)
    cache = []
    z = Q.forward(Q.tok.encode(prompt).ids, cache)
    end = Q.tok.token_to_id("<|im_end|>")
    out = []
    for _ in range(10):
        i = int(np.argmax(z))
        if i == end:
            break
        out.append(i)
        z = Q.forward([i], cache)
    return "The final answer is" + Q.tok.decode(out), len(out)


def gold(p):
    return float(p["answer"].split("####")[-1].strip().replace(",", ""))


def number(text, last=True):
    m = re.search(r"Answer:\s*\$?(-?[\d,]*\.?\d+)", text)
    if m:
        return float(m.group(1).replace(",", ""))
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text)
    if not nums:
        return None
    s = (nums[-1] if last else nums[0]).replace(",", "").rstrip(".")
    try:
        return float(s)
    except ValueError:
        return None


rows = []
for i, p in enumerate(PROBS):
    g = gold(p)
    r = {"i": i, "question": p["question"], "gold": g}
    for mode, tmpl, n in (("direct", DIRECT, 16), ("steps", STEPS, 400)):
        t0 = time.perf_counter()
        ans, ps, toks = Q.chat(tmpl % p["question"], n=n)
        secs = time.perf_counter() - t0
        got = number(ans, last=(mode == "steps"))
        r[mode] = {"answer": ans, "number": got, "right": got is not None and abs(got - g) < 1e-6,
                   "tokens": len(toks), "secs": round(secs, 1), "hit_cap": len(toks) >= n}
    t0 = time.perf_counter()
    ans, n_tok = forced(p["question"])
    got = number(ans, last=False)
    r["forced"] = {"answer": ans, "number": got, "right": got is not None and abs(got - g) < 1e-6,
                   "tokens": n_tok, "secs": round(time.perf_counter() - t0, 1), "hit_cap": False}
    rows.append(r)
    print("%2d gold %-8g forced %-8s %s | direct %-8s %s | steps %-8s %s (%d tok, %.0fs)" % (
        i, g, r["forced"]["number"], "OK" if r["forced"]["right"] else "--", r["direct"]["number"], "OK" if r["direct"]["right"] else "--", r["steps"]["number"],
        "OK" if r["steps"]["right"] else "--", r["steps"]["tokens"], r["steps"]["secs"]), flush=True)
    summ = {m: {"right": sum(x[m]["right"] for x in rows), "of": len(rows),
                "mean_tokens": round(float(np.mean([x[m]["tokens"] for x in rows])), 1),
                "mean_secs": round(float(np.mean([x[m]["secs"] for x in rows])), 1),
                "hit_cap": sum(x[m]["hit_cap"] for x in rows)} for m in ("direct", "forced", "steps")}
    summ["only_steps_right"] = sum(x["steps"]["right"] and not x["direct"]["right"] for x in rows)
    summ["only_direct_right"] = sum(x["direct"]["right"] and not x["steps"]["right"] for x in rows)
    summ["only_steps_right_vs_forced"] = sum(x["steps"]["right"] and not x["forced"]["right"] for x in rows)
    summ["only_forced_right"] = sum(x["forced"]["right"] and not x["steps"]["right"] for x in rows)
    json.dump({"model": "Qwen/Qwen2.5-0.5B-Instruct", "dataset": "GSM8K test, first %d" % a.n,
               "prompts": {"direct": DIRECT, "forced": "assistant reply pre-started with: The final answer is", "steps": STEPS}, "summary": summ, "rows": rows},
              open(os.path.join(a.out, "results.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("DONE", summ)
