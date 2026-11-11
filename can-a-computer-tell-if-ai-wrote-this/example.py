"""
example.py - the running example, word by word: the three Moon paragraphs (Wikipedia, AI shipped, AI at temperature
1.0), read by GPT-2 small. For every word-piece: GPT-2's top 5 guesses, the rank of the piece that came, its probability.
Writes out/moon.json.   python example.py --models <hf cache folder>
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser(); ap.add_argument("--models", required=True); a = ap.parse_args()
sys.argv = [sys.argv[0], "--models", a.models]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "detect.py"), encoding="utf-8").read()
src = src.split("human = json.load")[0]            # reuse the model + score() only
ns = {"__file__": os.path.join(os.path.dirname(os.path.abspath(__file__)), "detect.py")}; exec(compile(src, "detect.py", "exec"), ns)
S = json.load(open(os.path.join(OUT, "scores.json"), encoding="utf-8"))
res = {}
for k in ("human", "shipped", "warm"):
    t = [r for r in S[k] if r["topic"] == "Moon"][0]["text"]
    res[k] = dict(text=t, **ns["score"](t, keep=True))
    r = res[k]; print(k, round(r["ppl"], 1), "top1", round(r["top1"], 2), "top10", round(r["top10"], 2), r["n"])
res["declaration"] = S["old"]["declaration"]
r = res["declaration"]; print("declaration", round(r["ppl"], 2), round(r["top1"], 2), round(r["top10"], 2), r["n"])
json.dump(res, open(os.path.join(OUT, "moon.json"), "w", encoding="utf-8"), ensure_ascii=False)
for k in ("human", "shipped"):
    r = res[k]
    print(k, " ".join(f"{p.strip()}[{rk}]" for p, rk in list(zip(r["pieces"], r["rank"]))[:30]))
r = res["declaration"]; print(" ".join(f"{p.strip()}[{rk}]" for p, rk in list(zip(r["pieces"], r["rank"]))[:30]))
