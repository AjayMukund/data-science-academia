"""
detect.py - the detector: GPT-2 small (124M, MIT) reads each paragraph and, at every word-piece, we ask how surprised it
was (its probability for the piece that actually came, and that piece's rank among its 50,257 guesses).
  perplexity = exp(mean surprise): low = predictable = "looks AI-written".
Scores: 50 Wikipedia openings (human), 50 AI paragraphs (shipped settings), 50 AI paragraphs (warm, temperature 1.3),
and four famous old texts (public domain). The rule: perplexity under a cut-off -> "AI". The cut-off is chosen on
topics 1-25 and tested on topics 26-50. Writes out/scores.json.   python detect.py --models <hf cache folder>
"""
import argparse, json, math, os
import numpy as np, torch
ap = argparse.ArgumentParser(); ap.add_argument("--models", required=True); a = ap.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "out")
torch.set_grad_enabled(False)
from transformers import GPT2TokenizerFast, GPT2LMHeadModel
tok = GPT2TokenizerFast.from_pretrained("openai-community/gpt2", cache_dir=a.models)
lm = GPT2LMHeadModel.from_pretrained("openai-community/gpt2", cache_dir=a.models).eval()

def score(text, keep=False):
    ids = tok(text, return_tensors="pt").input_ids[:, :1024]
    ids = torch.cat([torch.tensor([[tok.bos_token_id]]), ids], 1)     # start from <|endoftext|>
    lp = torch.log_softmax(lm(ids).logits[0, :-1].double(), -1)
    tgt = ids[0, 1:]
    got = lp[torch.arange(len(tgt)), tgt]
    rank = (lp > got[:, None]).sum(-1) + 1
    nll = -got.numpy()
    r = dict(ppl=float(math.exp(nll.mean())), n=int(len(tgt)), top1=float((rank == 1).float().mean()),
             top10=float((rank <= 10).float().mean()), words=len(text.split()))
    if keep:
        r["pieces"] = [tok.decode([int(t)]) for t in tgt]; r["rank"] = rank.tolist(); r["prob"] = np.exp(-nll).tolist()
        top = lp.topk(5, -1)
        r["guess"] = [[[tok.decode([int(i)]), float(math.exp(v))] for v, i in zip(top.values[k], top.indices[k])]
                      for k in range(len(tgt))]
    return r

human = json.load(open(next(p for p in (os.path.join(OUT, "human.json"), os.path.join(os.path.dirname(OUT), "data", "human.json")) if os.path.exists(p)), encoding="utf-8"))
ai = json.load(open(os.path.join(OUT, "ai.json"), encoding="utf-8"))
OLD = {
 "declaration": "We hold these truths to be self-evident, that all men are created equal, that they are endowed by their "
   "Creator with certain unalienable Rights, that among these are Life, Liberty and the pursuit of Happiness. That to "
   "secure these rights, Governments are instituted among Men, deriving their just powers from the consent of the "
   "governed.",
 "genesis": "In the beginning God created the heaven and the earth. And the earth was without form, and void; and "
   "darkness was upon the face of the deep. And the Spirit of God moved upon the face of the waters. And God said, Let "
   "there be light: and there was light.",
 "gettysburg": "Four score and seven years ago our fathers brought forth on this continent, a new nation, conceived in "
   "Liberty, and dedicated to the proposition that all men are created equal. Now we are engaged in a great civil war, "
   "testing whether that nation, or any nation so conceived, can long endure.",
 "sonnet18": "Shall I compare thee to a summer's day? Thou art more lovely and more temperate: Rough winds do shake the "
   "darling buds of May, And summer's lease hath all too short a date.",
}
S = {"human": [], "shipped": [], "warm": [], "old": {}}
for h, x in zip(human, ai):
    assert h["topic"] == x["topic"]
    S["human"].append(dict(topic=h["topic"], text=h["text"], **score(h["text"], keep=h["topic"] == "Honey bee")))
    S["shipped"].append(dict(topic=h["topic"], text=x["shipped"], **score(x["shipped"], keep=h["topic"] == "Honey bee")))
    S["warm"].append(dict(topic=h["topic"], text=x["warm"], **score(x["warm"], keep=h["topic"] == "Honey bee")))
    print(h["topic"], round(S["human"][-1]["ppl"], 1), round(S["shipped"][-1]["ppl"], 1), round(S["warm"][-1]["ppl"], 1), flush=True)
for k, t in OLD.items():
    S["old"][k] = dict(text=t, **score(t, keep=True)); print(k, round(S["old"][k]["ppl"], 1))

ph = np.array([r["ppl"] for r in S["human"]]); ps = np.array([r["ppl"] for r in S["shipped"]]); pw = np.array([r["ppl"] for r in S["warm"]])
def acc(cut, hh, aa): return (np.sum(hh >= cut) + np.sum(aa < cut)) / (len(hh) + len(aa))
cands = np.sort(np.concatenate([ph[:25], ps[:25]]))
best = max(cands, key=lambda c: (acc(c, ph[:25], ps[:25]), -abs(c - np.median(cands))))
cut = float(best)
S["rule"] = dict(cut=cut, train_acc=float(acc(cut, ph[:25], ps[:25])), test_acc=float(acc(cut, ph[25:], ps[25:])),
                 all_acc=float(acc(cut, ph, ps)), human_flagged=int(np.sum(ph < cut)), ai_caught=int(np.sum(ps < cut)),
                 warm_caught=int(np.sum(pw < cut)),
                 test_human_flagged=int(np.sum(ph[25:] < cut)), test_ai_caught=int(np.sum(ps[25:] < cut)),
                 test_warm_caught=int(np.sum(pw[25:] < cut)),
                 old_flagged={k: bool(v["ppl"] < cut) for k, v in S["old"].items()},
                 median=dict(human=float(np.median(ph)), shipped=float(np.median(ps)), warm=float(np.median(pw))),
                 top10=dict(human=float(np.mean([r["top10"] for r in S["human"]])),
                            shipped=float(np.mean([r["top10"] for r in S["shipped"]])),
                            warm=float(np.mean([r["top10"] for r in S["warm"]]))))
print(json.dumps(S["rule"], indent=1))
json.dump(S, open(os.path.join(OUT, "scores.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
