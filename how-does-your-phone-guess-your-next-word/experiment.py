"""How does your phone guess your next word? A word-pair (n-gram) predictor trained on the Sherlock Holmes books.
Data: Project Gutenberg (public domain in the US). Plain Python; downloads the texts on first run."""
import os, re, urllib.request, collections, json, random
ROOT = os.path.dirname(os.path.abspath(__file__)); DL = os.path.join(ROOT, "download"); os.makedirs(DL, exist_ok=True)
BOOKS = {244: "A Study in Scarlet", 2097: "The Sign of the Four", 1661: "The Adventures of Sherlock Holmes",
         834: "The Memoirs of Sherlock Holmes", 108: "The Return of Sherlock Holmes", 3289: "The Valley of Fear",
         2350: "His Last Bow", 2852: "The Hound of the Baskervilles"}
HELD = 2852                                    # the Hound is held out: the test the predictor never read
def text(n):
    p = os.path.join(DL, "%d.txt" % n)
    if not os.path.exists(p) or os.path.getsize(p) == 0:
        import time
        for i, url in enumerate(["https://gutenberg.pglaf.org/cache/epub/%d/pg%d.txt" % (n, n),   # the official mirror (www answers 503 to scripts)
                                 "https://aleph.pglaf.org/cache/epub/%d/pg%d.txt" % (n, n)] * 3):
            try:
                rq = urllib.request.Request(url, headers={"User-Agent": "research-script"})
                data = urllib.request.urlopen(rq, timeout=120).read(); open(p, "wb").write(data); break
            except Exception as e:
                print("retry", n, e); time.sleep(10 * (i + 1))
    t = open(p, encoding="utf-8-sig", errors="replace").read()
    a = re.search(r"\*\*\* ?START OF.*?\*\*\*", t); b = re.search(r"\*\*\* ?END OF", t)
    return t[a.end():b.start()]
def words(t):
    t = t.replace("’", "'").replace("‘", "'")
    return re.findall(r"[A-Za-z]+(?:'[a-z]+)?", t)
W = {n: [w.lower() for w in words(text(n))] for n in BOOKS}
train = [w for n in BOOKS if n != HELD for w in W[n]]; test = W[HELD]
R = {"train_words": len(train), "test_words": len(test), "vocab": len(set(train))}
uni = collections.Counter(train)
bi = collections.defaultdict(collections.Counter); tri = collections.defaultdict(collections.Counter)
for a, b in zip(train, train[1:]): bi[a][b] += 1
for a, b, c in zip(train, train[1:], train[2:]): tri[(a, b)][c] += 1
def top(c, k=3):
    s = sum(c.values()); return [(w, n, round(100 * n / s, 1)) for w, n in c.most_common(k)], s
R["top_words"] = [(w, n, round(100 * n / len(train), 2)) for w, n in uni.most_common(5)]
for ctx in [("my", "dear"), ("i",), ("i", "have"), ("sherlock",), ("baker",), ("dear",), ("my",), ("the",), ("no", "doubt")]:
    c = tri[ctx] if len(ctx) == 2 else bi[ctx[0]]
    R["after " + " ".join(ctx)] = top(c, 5)
# a phone keyboard: three suggestions from the last two words, falling back to the last word, then to the commonest words
def suggest(a, b, k=3):
    if tri[(a, b)]: return [w for w, _ in tri[(a, b)].most_common(k)], "pair"
    if bi[b]: return [w for w, _ in bi[b].most_common(k)], "word"
    return [w for w, _ in uni.most_common(k)], "none"
hits = hit1 = 0; src = collections.Counter(); hit_src = collections.Counter()
for a, b, c in zip(test, test[1:], test[2:]):
    s, how = suggest(a, b); src[how] += 1
    if c in s: hits += 1; hit_src[how] += 1
    if c == s[0]: hit1 += 1
n = len(test) - 2
R["test_positions"] = n; R["top3_hit_pct"] = round(100 * hits / n, 1); R["top1_hit_pct"] = round(100 * hit1 / n, 1)
R["context_source_pct"] = {k: round(100 * v / n, 1) for k, v in src.items()}
R["hit_pct_by_source"] = {k: round(100 * hit_src[k] / src[k], 1) for k in src}
R["unseen_pairs_in_test_pct"] = round(100 * src["word"] / n + 100 * src["none"] / n, 1)
# always take the top suggestion: what does it write?
def ramble(a, b, n=24):
    out = [a, b]
    for _ in range(n): out.append(suggest(out[-2], out[-1])[0][0])
    return " ".join(out)
R["greedy_from 'i have'"] = ramble("i", "have"); R["greedy_from 'my dear'"] = ramble("my", "dear")
R["greedy_from 'the hound'"] = ramble("the", "hound")
# unseen contexts that a person would complete easily
for ctx in [("dear", "watson"), ("my", "phone"), ("the", "internet"), ("my", "laptop")]:
    R["seen " + " ".join(ctx)] = sum(tri[ctx].values())
json.dump(R, open(os.path.join(ROOT, "results.json"), "w"), indent=1)
for k, v in R.items(): print(k, ":", v)
