"""
experiment.py - "What machine learning actually is" (master, 2026-10-23).

Rules written by hand vs rules learned from examples, on the real SMS Spam
Collection (UCI #228, Almeida & Gomez Hidalgo 2011, CC BY 4.0; 5,574 texts,
download/SMSSpamCollection). Plain numpy: a multinomial Naive Bayes filter.
The dataset downloads itself on first run. Every number lands in out/results.json.

    python experiment.py
"""
import os, re, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)
sys.stdout.reconfigure(encoding="utf-8")

SRC = os.path.join(HERE, "download", "SMSSpamCollection")
if not os.path.exists(SRC):                       # first run: fetch from UCI
    import io, urllib.request, zipfile
    os.makedirs(os.path.dirname(SRC), exist_ok=True)
    url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
    zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(url).read())).extract(
        "SMSSpamCollection", os.path.dirname(SRC))
rows = [l.rstrip("\n").split("\t", 1) for l in
        open(SRC, encoding="utf-8")]
labels = np.array([1 if r[0] == "spam" else 0 for r in rows])
texts = [r[1] for r in rows]
N = len(texts)
rng = np.random.default_rng(42)
idx = rng.permutation(N)
n_tr = int(0.8 * N)
tr, te = idx[:n_tr], idx[n_tr:]


def toks(t):
    return re.findall(r"[a-z0-9£$']+", t.lower())


def score(pred, y):
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    return {"acc": (tp + tn) / len(y), "caught": tp, "spam": tp + fn, "missed": fn,
            "false_alarms": fp, "ham": fp + tn}


R = {"n": N, "n_spam": int(labels.sum()), "n_train": len(tr), "n_test": len(te),
     "test_spam": int(labels[te].sum())}

# ---- 1. rules a person writes ------------------------------------------------
RULES1 = ["free", "win", "winner", "prize", "cash", "urgent", "claim"]
RULES2 = RULES1 + ["txt", "call now", "£", "mobile", "reply", "stop", "won", "offer", "award"]


def rule_pred(words, items):
    out = []
    for i in items:
        t = texts[i].lower()
        out.append(1 if any(w in t for w in words) else 0)
    return np.array(out)


for name, words in (("rules7", RULES1), ("rules16", RULES2)):
    R[name] = {"words": words, **score(rule_pred(words, te), labels[te])}

# the minimal pair: real messages with "free" in them, one of each kind
free_ham = [texts[i] for i in te if labels[i] == 0 and re.search(r"\bfree\b", texts[i].lower())]
free_spam = [texts[i] for i in te if labels[i] == 1 and re.search(r"\bfree\b", texts[i].lower())]
R["free_ham"] = free_ham[:8]
R["free_spam"] = free_spam[:8]
R["free_counts"] = {"ham_with_free_test": len(free_ham), "spam_with_free_test": len(free_spam)}

# ---- 2. a filter that learns -------------------------------------------------
def train_nb(items):
    vocab = {}
    cnt = [dict(), dict()]
    docs = np.zeros(2)
    for i in items:
        c = labels[i]
        docs[c] += 1
        for w in toks(texts[i]):
            vocab.setdefault(w, len(vocab))
            cnt[c][w] = cnt[c].get(w, 0) + 1
    V = len(vocab)
    tot = [sum(cnt[c].values()) for c in (0, 1)]
    def logp(w, c):
        return np.log((cnt[c].get(w, 0) + 1) / (tot[c] + V))
    prior = np.log(docs / docs.sum())
    return {"vocab": vocab, "cnt": cnt, "tot": tot, "V": V, "prior": prior, "logp": logp}


def nb_score(m, t):
    ws = toks(t)
    s = [m["prior"][c] + sum(m["logp"](w, c) for w in ws) for c in (0, 1)]
    return s[1] - s[0]                       # >0 means spam


def nb_pred(m, items):
    return np.array([1 if nb_score(m, texts[i]) > 0 else 0 for i in items])


M = train_nb(tr)
R["learned"] = score(nb_pred(M, te), labels[te])
R["learned"]["vocab"] = M["V"]

# what it learned: words most tied to spam / ham (seen >= 10 times)
lr = []
for w in M["vocab"]:
    n = M["cnt"][0].get(w, 0) + M["cnt"][1].get(w, 0)
    if n >= 10:
        lr.append((float(M["logp"](w, 1) - M["logp"](w, 0)), w, M["cnt"][1].get(w, 0), M["cnt"][0].get(w, 0)))
lr.sort()
R["spam_words"] = [(w, s, h, round(x, 2)) for x, w, s, h in lr[::-1][:15]]
R["ham_words"] = [(w, s, h, round(x, 2)) for x, w, s, h in lr[:15]]
R["free_weight"] = round(float(M["logp"]("free", 1) - M["logp"]("free", 0)), 2)

# how the two read the "free" pair
def verdicts(msgs):
    return [(m, int(any(w in m.lower() for w in RULES1)), round(float(nb_score(M, m)), 2)) for m in msgs]
R["pair_verdicts"] = {"ham": verdicts(free_ham[:8]), "spam": verdicts(free_spam[:8])}

# ---- 3. more examples, better -------------------------------------------------
curve = []
for n in (10, 30, 100, 300, 1000, 3000, len(tr)):
    m = train_nb(tr[:n])
    curve.append((n, score(nb_pred(m, te), labels[te])))
R["curve"] = [(n, round(s["acc"], 4), s["caught"], s["false_alarms"]) for n, s in curve]

# a message the rules miss and the model catches, and the reverse
missed = [i for i in te if labels[i] == 1 and rule_pred(RULES1, [i])[0] == 0 and nb_pred(M, [i])[0] == 1]
R["rules_miss_model_catches"] = [texts[i] for i in missed[:6]]
flagged = [i for i in te if labels[i] == 0 and rule_pred(RULES1, [i])[0] == 1 and nb_pred(M, [i])[0] == 0]
R["rules_flag_model_clears"] = [texts[i] for i in flagged[:6]]

json.dump(R, open(os.path.join(OUT, "results.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(json.dumps({k: v for k, v in R.items() if k not in ("free_ham", "free_spam", "pair_verdicts",
                                                              "rules_miss_model_catches", "rules_flag_model_clears")},
                 indent=1, ensure_ascii=False))
print("\nham with 'free':"); [print("  ", m[:90]) for m in free_ham[:6]]
print("spam with 'free':"); [print("  ", m[:90]) for m in free_spam[:4]]
print("rules miss, model catches:"); [print("  ", m[:90]) for m in R["rules_miss_model_catches"][:4]]
print("rules flag, model clears:"); [print("  ", m[:90]) for m in R["rules_flag_model_clears"][:4]]

# ---- 4. the two cold-open messages, word by word ---------------------------
def contributions(t):
    ws = toks(t)
    return {"prior": round(float(M["prior"][1] - M["prior"][0]), 2),
            "words": [(w, round(float(M["logp"](w, 1) - M["logp"](w, 0)), 2)) for w in ws],
            "total": round(float(nb_score(M, t)), 2)}

FRIEND = next(m for m in free_ham if m.startswith("Can... I'm free"))
SCAM = next(m for m in free_spam if m.startswith("Free entry in 2 a wkly comp"))
R["friend"] = {"text": FRIEND, **contributions(FRIEND)}
R["scam"] = {"text": SCAM, **contributions(SCAM)}
R["ashwini"] = next(texts[i] for i in te if "ashwini" in texts[i].lower())
json.dump(R, open(os.path.join(OUT, "results.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("\nfriend", R["friend"]); print("scam", R["scam"]["total"], R["scam"]["words"][:12])
