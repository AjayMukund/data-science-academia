"""Why does "the" make up about 1 in 20 of everything you read? Zipf's law on Jane Austen's six novels
(Project Gutenberg, public domain). Plain Python; downloads from the official mirror on first run."""
import os, re, json, math, time, urllib.request, collections
ROOT = os.path.dirname(os.path.abspath(__file__)); DL = os.path.join(ROOT, "download"); os.makedirs(DL, exist_ok=True)
BOOKS = {1342: "Pride and Prejudice", 158: "Emma", 161: "Sense and Sensibility", 105: "Persuasion",
         121: "Northanger Abbey", 141: "Mansfield Park"}
def text(n):
    p = os.path.join(DL, "%d.txt" % n)
    if not os.path.exists(p) or os.path.getsize(p) == 0:
        for i, url in enumerate(["https://gutenberg.pglaf.org/cache/epub/%d/pg%d.txt" % (n, n),
                                 "https://aleph.pglaf.org/cache/epub/%d/pg%d.txt" % (n, n)] * 3):
            try:
                data = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "research-script"}), timeout=120).read()
                open(p, "wb").write(data); break
            except Exception as e:
                print("retry", n, e); time.sleep(10 * (i + 1))
    t = open(p, encoding="utf-8-sig", errors="replace").read()
    a = re.search(r"\*\*\* ?START OF.*?\*\*\*", t); b = re.search(r"\*\*\* ?END OF", t)
    return t[a.end():b.start()].replace("’", "'").replace("‘", "'")
W = []
per = {}
for n, name in BOOKS.items():
    w = [x.lower() for x in re.findall(r"[A-Za-z]+(?:'[a-z]+)?", text(n))]
    per[name] = w; W += w
C = collections.Counter(W); N = len(W); V = len(C)
R = {"words": N, "different_words": V}
top = C.most_common()
R["top10"] = [(w, c, round(100 * c / N, 2)) for w, c in top[:10]]
R["the_one_in"] = round(N / C["the"], 1)
# Zipf: rank x frequency, and the share of text covered by the top k words
R["rank_x_share"] = [(r, top[r - 1][0], round(100 * top[r - 1][1] / N, 3), round(r * top[r - 1][1] / N, 3)) for r in (1, 2, 3, 10, 100, 1000)]
cum = 0; cover = {}
for i, (w, c) in enumerate(top, 1):
    cum += c
    for k in (10, 100, 1000):
        if i == k: cover[k] = round(100 * cum / N, 1)
R["share_covered_by_top"] = cover
half = next(i for i, s in enumerate(__import__("itertools").accumulate(c for _, c in top), 1) if s >= N / 2)
R["words_that_make_half_the_text"] = half
R["used_once"] = sum(1 for c in C.values() if c == 1); R["used_once_pct_of_vocab"] = round(100 * R["used_once"] / V, 1)
# log-log slope (least squares, ranks 10..5000)
xs = [math.log10(r) for r in range(10, 5001)]; ys = [math.log10(top[r - 1][1]) for r in range(10, 5001)]
mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
R["loglog_slope_10_5000"] = round(sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs), 2)
# does it hold book by book? share of "the" and the 2nd word in each
R["per_book"] = {name: (len(w), round(100 * w.count("the") / len(w), 2), collections.Counter(w).most_common(3)) for name, w in per.items()}
R["examples_used_once"] = sorted([w for w, c in C.items() if c == 1 and len(w) > 6])[:5000:500]
json.dump(R, open(os.path.join(ROOT, "results.json"), "w"), indent=1)
for k, v in R.items(): print(k, ":", v)
# book by book: how many different words make half of each novel?
import itertools
R["half_by_book"] = {}
for name, w in per.items():
    cs = sorted(collections.Counter(w).values(), reverse=True)
    R["half_by_book"][name] = next(i for i, s in enumerate(itertools.accumulate(cs), 1) if s >= len(w) / 2)
R["one_in"] = {w: round(N / C[w]) for w in ("the", "it", "great", "dr")}
json.dump(R, open(os.path.join(ROOT, "results.json"), "w"), indent=1)
print("half_by_book", R["half_by_book"], "one_in", R["one_in"])
R["top100"] = [(w, c) for w, c in top[:100]]
R["rank_counts_1_to_5000"] = [c for _, c in top[:5000]]
json.dump(R, open(os.path.join(ROOT, "results.json"), "w"), indent=1)
