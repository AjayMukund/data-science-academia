"""
experiment.py - give an AI a calculator: does running code fix its maths? Measured.

Qwen2.5-0.5B-Instruct (Apache-2.0) in plain numpy (qwen_np.py), greedy, on the same first 50 GSM8K test problems as
the reasoning video. New mode: the model writes a short Python program; we run it (a separate, isolated Python
process, 5-second limit, no network use expected) and read the number it prints. Compared with the reasoning video's
"steps" answers (its results.json), problem by problem.

  python experiment.py --models C:/models/hf --steps ../why-does-an-ai-get-maths-right-when-it-shows-its-working/out/results.json --out out
"""
import argparse, json, os, re, subprocess, sys, tempfile, time
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--models", default=os.path.expanduser("~/models/hf"))
ap.add_argument("--steps", required=True, help="the reasoning experiment's results.json (same 50 problems)")
ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"))
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qwen_np as Q
from huggingface_hub import snapshot_download

PREV = json.load(open(a.steps, encoding="utf-8"))
ROWS = PREV["rows"]
Q.load(snapshot_download("Qwen/Qwen2.5-0.5B-Instruct", cache_dir=a.models,
                         allow_patterns=["config.json", "tokenizer.json", "model.safetensors"]))
ASK = ("%s\nWrite a short Python program that calculates the answer and prints only the final number. "
       "Put the code in one ```python block.")


def code_of(text):
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.S)
    return m.group(1) if m else None


def run(code):
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(code)
        path = f.name
    try:
        p = subprocess.run([sys.executable, "-I", path], capture_output=True, text=True, timeout=5,
                           cwd=tempfile.gettempdir(), env={"PATH": "", "SYSTEMROOT": os.environ.get("SYSTEMROOT", "")})
        return p.stdout.strip(), p.stderr.strip()[-300:], p.returncode
    except subprocess.TimeoutExpired:
        return "", "timeout", -1
    finally:
        os.remove(path)


def last_number(s):
    nums = re.findall(r"-?\d[\d,]*\.?\d*", s or "")
    if not nums:
        return None
    try:
        return float(nums[-1].replace(",", "").rstrip("."))
    except ValueError:
        return None


rows = []
for r in ROWS:
    t0 = time.perf_counter()
    ans, ps, toks = Q.chat(ASK % r["question"], n=300)
    code = code_of(ans)
    out, err, rc = run(code) if code else ("", "no code block", None)
    got = last_number(out)
    ok = got is not None and abs(got - r["gold"]) < 1e-6
    rows.append({"i": r["i"], "gold": r["gold"], "reply": ans, "code": code, "stdout": out, "stderr": err, "returncode": rc,
                 "number": got, "right": ok, "tokens": len(toks), "secs": round(time.perf_counter() - t0, 1),
                 "steps_right": r["steps"]["right"], "forced_right": r["forced"]["right"]})
    print("%2d gold %-8g code %-8s %s | steps %s | rc %s %s" % (r["i"], r["gold"], got, "OK" if ok else "--",
          "OK" if r["steps"]["right"] else "--", rc, (err or "")[:40].replace("\n", " ")), flush=True)
    summ = {"code_right": sum(x["right"] for x in rows), "of": len(rows),
            "ran_ok": sum(x["returncode"] == 0 for x in rows), "no_code": sum(x["code"] is None for x in rows),
            "steps_right": sum(x["steps_right"] for x in rows),
            "both": sum(x["right"] and x["steps_right"] for x in rows),
            "only_code": sum(x["right"] and not x["steps_right"] for x in rows),
            "only_steps": sum(x["steps_right"] and not x["right"] for x in rows),
            "mean_tokens": round(float(np.mean([x["tokens"] for x in rows])), 1),
            "mean_secs": round(float(np.mean([x["secs"] for x in rows])), 1)}
    json.dump({"model": "Qwen/Qwen2.5-0.5B-Instruct", "prompt": ASK, "summary": summ, "rows": rows},
              open(os.path.join(a.out, "results.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("DONE", summ)
