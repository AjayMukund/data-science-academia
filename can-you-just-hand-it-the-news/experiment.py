"""
experiment.py - giving a model a library (retrieval-augmented generation), run by us end to end.

Qwen2.5-0.5B-Instruct was released in September 2024. We ask it 12 questions about what happened
after that (the 2026 World Cup, the 2025 conclave ...):
  1. closed book: just the question. What does it say, and how sure is it of its first word?
  2. a library: the opening section of 24 Wikipedia pages - the 12 that hold the answers and
     12 look-alikes (the edition before: the 2022 World Cup, the 2013 conclave ...), each cut into
     chunks of about three sentences.
  3. search: every chunk and every question turned into 384 numbers by all-MiniLM-L6-v2; the
     3 closest chunks by cosine similarity are retrieved.
  4. open book: those chunks are pasted above the question, with the instruction to answer only
     from them and to say so when they don't hold the answer. How many right now?
  5. the wrong drawer: the same question with only the look-alike page's chunks. Does it say
     "I don't know", or answer from last time?

Pages are read at the exact revision we used (REVS), so a fresh run reads the same text.
Models are downloaded from Hugging Face into --models (about 1.1 GB).

    python experiment.py --models path/to/a/model/folder --out out
"""
import os, sys, re, json, argparse
from html.parser import HTMLParser
import numpy as np
import requests
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # qwen_np.py, minilm_np.py sit beside this file

ap = argparse.ArgumentParser()
ap.add_argument("--models", required=True, help="folder to download the models into")
ap.add_argument("--out", default="out")
ap.add_argument("--fetch-latest", action="store_true", help="ignore REVS and read today's pages")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
sys.stdout.reconfigure(encoding="utf-8")

# question, the page that answers it, its look-alike, the words that count as right
QS = [("Who won the 2026 FIFA World Cup?", "2026 FIFA World Cup", "2022 FIFA World Cup", ["Spain"]),
      ("Who won the 2025 UEFA Champions League final?", "2025 UEFA Champions League final",
       "2024 UEFA Champions League final", ["Paris Saint-Germain", "PSG"]),
      ("Who was elected pope in 2025?", "2025 papal conclave", "2013 papal conclave", ["Leo XIV", "Prevost"]),
      ("Who won the 2025 Indian Premier League?", "2025 Indian Premier League", "2024 Indian Premier League",
       ["Royal Challengers"]),
      ("Who won the 2025 ICC Champions Trophy?", "2025 ICC Champions Trophy", "2017 ICC Champions Trophy", ["India"]),
      ("Who won the 2025 Nobel Peace Prize?", "2025 Nobel Peace Prize", "2024 Nobel Peace Prize", ["Machado"]),
      ("Who won the men's singles at Wimbledon in 2025?", "2025 Wimbledon Championships – Men's singles",
       "2024 Wimbledon Championships – Men's singles", ["Sinner"]),
      ("Who won Super Bowl LIX?", "Super Bowl LIX", "Super Bowl LVIII", ["Eagles", "Philadelphia"]),
      ("Which film won Best Picture at the 2025 Oscars?", "97th Academy Awards", "96th Academy Awards", ["Anora"]),
      ("Who won the 2024 United States presidential election?", "2024 United States presidential election",
       "2020 United States presidential election", ["Trump"]),
      ("Who won the 2025 Formula One World Championship?", "2025 Formula One World Championship",
       "2024 Formula One World Championship", ["Norris"]),
      ("Who won the 2025 FIFA Club World Cup?", "2025 FIFA Club World Cup", "2023 FIFA Club World Cup", ["Chelsea"])]
REVS = {   # the revisions we read (2026-10-29); --fetch-latest reads today's instead
    '2026 FIFA World Cup': 1381009618,
    '2022 FIFA World Cup': 1382432110,
    '2025 UEFA Champions League final': 1379805629,
    '2024 UEFA Champions League final': 1370459237,
    '2025 papal conclave': 1381557334,
    '2013 papal conclave': 1370556610,
    '2025 Indian Premier League': 1375648040,
    '2024 Indian Premier League': 1381156513,
    '2025 ICC Champions Trophy': 1378230609,
    '2017 ICC Champions Trophy': 1376412528,
    '2025 Nobel Peace Prize': 1379415461,
    '2024 Nobel Peace Prize': 1379415583,
    "2025 Wimbledon Championships – Men's singles": 1372142791,
    "2024 Wimbledon Championships – Men's singles": 1380368735,
    'Super Bowl LIX': 1381201528,
    'Super Bowl LVIII': 1374061874,
    '97th Academy Awards': 1380826865,
    '96th Academy Awards': 1370729225,
    '2024 United States presidential election': 1382580275,
    '2020 United States presidential election': 1381608142,
    '2025 Formula One World Championship': 1370436518,
    '2024 Formula One World Championship': 1382454930,
    '2025 FIFA Club World Cup': 1381821239,
    '2023 FIFA Club World Cup': 1375910110,
}

SYSTEM = "You are a helpful assistant."
RAG_SYSTEM = ("You are a helpful assistant. Answer the question using only the passages provided. "
              "If the passages do not contain the answer, say that you don't know.")
NO = ("don't know", "do not know", "not contain", "does not mention", "doesn't mention", "not mentioned",
      "no information", "not provided", "cannot answer", "can't answer", "not specified", "unable to",
      "cannot provide", "cannot predict", "i'm sorry")
K = 3                                   # chunks pasted into the prompt
H = {"User-Agent": "DataScienceAcademia-research/1.0 (https://github.com/AjayMukund/data-science-academia)"}
API = "https://en.wikipedia.org/w/api.php"


class Paras(HTMLParser):
    """The text of the <p> paragraphs, without infoboxes, footnote markers or pronunciations."""
    VOID = ("br", "img", "wbr", "meta", "link", "hr", "input", "source")

    def __init__(self):
        super().__init__(); self.out, self.cur, self.skip = [], None, 0

    def handle_starttag(self, tag, attrs):
        if tag in self.VOID: return
        cls = dict(attrs).get("class") or ""
        if self.skip:
            self.skip += 1
        elif tag in ("sup", "style", "table", "figure") or "IPA" in cls or "noprint" in cls or "rt-commentedText" in cls:
            self.skip = 1
        elif tag == "p" and self.cur is None:
            self.cur = []

    def handle_endtag(self, tag):
        if tag in self.VOID: return
        if self.skip:
            self.skip -= 1
        elif tag == "p" and self.cur is not None:
            t = re.sub(r"\s+", " ", "".join(self.cur)).strip()
            if t: self.out.append(t)
            self.cur = None

    def handle_data(self, data):
        if self.cur is not None and not self.skip: self.cur.append(data)


def intro(title):
    """The page's opening section at the pinned revision (or today's, with --fetch-latest)."""
    p = {"action": "parse", "prop": "text|revid", "section": 0, "format": "json", "formatversion": 2, "redirects": 1}
    if title in REVS and not a.fetch_latest:
        p["oldid"] = REVS[title]
    else:
        p["page"] = title
    r = requests.get(API, headers=H, params=p, timeout=60).json()["parse"]
    x = Paras(); x.feed(r["text"])
    t = " ".join(x.out)
    t = re.sub(r"\s*\(\s*[;,]?\s*\)", "", t)          # brackets emptied by removed pronunciations
    t = re.sub(r"\s+([,.;:])", r"\1", t)
    return t, r["revid"]


def chunks(text, n=3):
    """About three sentences per chunk."""
    s = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])", text)
    return [" ".join(s[i:i + n]) for i in range(0, len(s), n)]


def right(ans, ok): return any(w.lower() in ans.lower() for w in ok)


def named(q, toks, ps):
    """How sure it was of the name it gave: the first capitalised word it did not copy from the question
    ("South" in "The 2026 FIFA World Cup was won by the South Korean team")."""
    qw = set(re.findall(r"[A-Za-z]+", q.lower())) | {"the", "i", "as", "in", "it"}
    for t, p in zip(toks, ps):
        w = t.strip()
        if w[:1].isupper() and w.lower().strip(".,'\"") not in qw:
            return [w, round(p, 4)]
    return ["", None]

def declined(ans): return any(w in ans.lower() for w in NO)
def passages(cs): return "\n\n".join("[%d] %s" % (i + 1, c) for i, c in enumerate(cs))


# 1. the library
pages, revs = {}, {}
for q in QS:
    for t in (q[1], q[2]):
        pages[t], revs[t] = intro(t)
json.dump(revs, open(os.path.join(a.out, "revs.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
lib = [(t, i, c) for t in pages for i, c in enumerate(chunks(pages[t]))]
print("library:", len(pages), "pages,", len(lib), "chunks", flush=True)

# 2. models
from huggingface_hub import snapshot_download
os.environ.setdefault("MINILM_CACHE", os.path.join(a.models, "minilm"))
import minilm_np as M
import qwen_np as Q
Q.load(snapshot_download("Qwen/Qwen2.5-0.5B-Instruct", cache_dir=a.models,
                         allow_patterns=["config.json", "tokenizer.json", "model.safetensors"]))
E = np.array([M.embed(c) for _, _, c in lib])

rows = []
for q, page, look, ok in QS:
    sims = E @ M.embed(q)
    order = [int(i) for i in np.argsort(-sims)]
    top = order[:K]
    holds = lambda i: lib[i][0] == page and right(lib[i][2], ok)
    gold = next(r for r, i in enumerate(order) if holds(i)) + 1
    closed, pc, tc = Q.chat(q, SYSTEM)
    opened, po, to = Q.chat("Passages:\n%s\n\nQuestion: %s" % (passages([lib[i][2] for i in top]), q), RAG_SYSTEM)
    wrong = [c for t, _, c in lib if t == look]
    wd, pw, tw = Q.chat("Passages:\n%s\n\nQuestion: %s" % (passages(wrong[:K]), q), RAG_SYSTEM)
    rows.append({"q": q, "page": page, "lookalike": look,
                 "closed": closed, "p_closed": round(pc[0], 4), "closed_right": right(closed, ok),
                 "closed_declined": declined(closed), "closed_name": named(q, tc, pc),
                 "scoreboard_closed": [[t, round(p, 4)] for t, p in Q.first_word_scores(q, SYSTEM)],
                 "retrieved": [{"page": lib[i][0], "sim": round(float(sims[i]), 4), "holds_answer": holds(i),
                                "text": lib[i][2]} for i in top],
                 "answer_chunk_rank": gold, "best_lookalike_sim": round(float(max(sims[i] for i in order if lib[i][0] == look)), 4),
                 "open": opened, "p_open": round(po[0], 4), "open_right": right(opened, ok), "open_declined": declined(opened),
                 "open_name": named(q, to, po),
                 "wrong_drawer": wd, "p_wrong": round(pw[0], 4), "wrong_right": right(wd, ok),
                 "wrong_declined": declined(wd), "wrong_name": named(q, tw, pw)})
    r = rows[-1]
    print("%-50s | closed: %-40s | top-%d holds: %-5s rank %d | open: %-40s | wrong drawer: %s" % (
        q[:50], closed[:40].replace("\n", " "), K, any(x["holds_answer"] for x in r["retrieved"]), gold,
        opened[:40].replace("\n", " "), wd[:60].replace("\n", " ")), flush=True)

# the numbers shown on screen for the running example: the first 8 of 384 for the question and its best
# chunk from each World Cup page
q0, page0, look0, _ = QS[0]
qv = M.embed(q0)
best = lambda t: max((i for i in range(len(lib)) if lib[i][0] == t), key=lambda i: float(E[i] @ qv))
VEC = {"dims": int(len(qv)), "question": [round(float(x), 3) for x in qv[:8]],
       page0: [round(float(x), 3) for x in E[best(page0)][:8]], look0: [round(float(x), 3) for x in E[best(look0)][:8]],
       "sim": {page0: round(float(E[best(page0)] @ qv), 4), look0: round(float(E[best(look0)] @ qv), 4)}}

S = {"pages": len(pages), "chunks": len(lib), "questions": len(QS), "k": K,
     "closed_right": sum(r["closed_right"] for r in rows), "closed_declined": sum(r["closed_declined"] for r in rows),
     "mean_p_closed": round(float(np.mean([r["p_closed"] for r in rows])), 4),
     "top1_hits": sum(r["answer_chunk_rank"] == 1 for r in rows),
     "topk_hits": sum(r["answer_chunk_rank"] <= K for r in rows),
     "open_right": sum(r["open_right"] for r in rows), "open_declined": sum(r["open_declined"] for r in rows),
     "wrong_drawer_right": sum(r["wrong_right"] for r in rows),
     "wrong_drawer_declined": sum(r["wrong_declined"] for r in rows),
     "wrong_drawer_answered": sum(not r["wrong_declined"] for r in rows)}
json.dump({"summary": S, "rows": rows, "vectors": VEC, "library": [{"page": t, "i": i, "text": c} for t, i, c in lib]},
          open(os.path.join(a.out, "results.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(json.dumps(S, indent=1))
