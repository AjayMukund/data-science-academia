"""
fetch_human.py - the human-written side: the opening paragraph of 50 English Wikipedia articles (CC BY-SA 4.0),
trimmed to whole sentences up to ~110 words. Writes out/human.json.
Wikipedia changes: the paragraphs we measured (with their revision IDs) ship as data/human.json; this script refreshes them.   python fetch_human.py
"""
import json, os, re, sys, time, requests
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from topics import TOPICS
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)
S = requests.Session(); S.headers["User-Agent"] = "dsa-research/1.0 (educational explainer)"
res = []
for t in TOPICS:
    for k in range(8):
        q = S.get("https://en.wikipedia.org/w/api.php", params=dict(action="query", prop="extracts|revisions", exintro=1,
                  explaintext=1, titles=t, format="json", redirects=1, rvprop="ids|timestamp"), timeout=30)
        try: r = q.json(); break
        except ValueError: time.sleep(5 * (k + 1))
    time.sleep(1.5)
    p = next(iter(r["query"]["pages"].values()))
    txt = re.sub(r"\s+", " ", p["extract"]).strip()
    sents = re.split(r"(?<=[.!?])\s+", txt)
    keep, n = [], 0
    for s in sents:
        w = len(s.split())
        if n and n + w > 110: break
        keep.append(s); n += w
    res.append(dict(topic=t, title=p["title"], revid=p["revisions"][0]["revid"], text=" ".join(keep)))
    print(t, n)
json.dump(res, open(os.path.join(OUT, "human.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
