# King minus man plus woman equals... king?

"King − man + woman = queen" is the famous word-embedding demo. Run it over
all 400,000 words of GloVe and the nearest word is **king**; queen is second.
The standard demo removes the words you typed in first. Also: Paris − France
+ Japan → Tokyo with nothing removed, and uncle − man + woman → uncle, a hair
ahead of aunt.

```bash
python experiment.py      # downloads GloVe (376 MB) on first run
```

Data: GloVe `glove-wiki-gigaword-300` (Pennington, Socher & Manning, Stanford
NLP; Public Domain Dedication and License), via the gensim-data release. Plain
numpy, cosine similarity. See `FINDINGS.md`. Try: your own analogies, or
`bigger - big + small` (a narrow miss).
