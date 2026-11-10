# Mixture-of-experts master (11-16): most of the model is asleep for every word — measured (2026-11-10)

**Model:** IBM Granite 3.0 1B-A400M base (Apache-2.0), run by us in plain numpy (`granite_np.py`, float32, greedy).
24 layers; each has attention plus **32 experts** (small feed-forward networks, 1,572,864 numbers each). For every
word piece, a router scores all 32 and the top **8** run; their outputs are mixed by softmax weights over those 8.
**Texts:** Alice's Adventures in Wonderland, English (Gutenberg #11) and the 1869 French translation by Henri Bué
(Gutenberg #55456), both public domain; Le Rouge et le Noir (#798, French), this project's `granite_np.py` (Python),
generated "a + b = c" sums. 300 word pieces of each.
**Code:** `experiment.py` → `out/results.json`, `alice.py` → `out/alice.json`. Deterministic.

## Size
- **1,334,625,280** numbers in all (1.33 billion); 1,207,959,552 of them in the 768 experts (24 × 32).
- Per word piece: attention + embeddings + 8 experts per layer = **428,655,616** (429 million, **32%**). The other
  ~906 million are idle for that word.
- Every number must still be in memory: 5.3 GB in float32 (the download is 5.5 GB). It saves work, not space.

## The running example — the opening of Alice, in English and in French
- "Alice was beginning to get very tired of" → (8 experts) **" the constant bickering between her and her sister."**
- Next word piece guessed right, English Alice: **255 of 299**; French Alice: 138 of 299.
- **Layer 21, expert 18:** woken for **96%** of the French Alice's word pieces, **7%** of the English.
  (On the mixed texts: 94% of the French novel, 7% English, 4% Python, 1% sums.)
- Other specialists (mixed texts, k=8): layer 20 expert 3 takes 94% of the sums' word pieces (11% of English);
  layer 8 expert 19 takes 66% of Python (1% of English).

## How many experts to wake (trained with 8)
| experts woken per word | 1 | 2 | 4 | **8** | 16 | 32 (all) |
|---|---|---|---|---|---|---|
| English Alice, next word right of 299 | 0 | 11 | 123 | **255** | 167 | 126 |
- Same prompt: 1 → "'s atomr],rozenlikecoreis included forb"; 2 → " daddy and (basic associated research where
  Crox, Y"; 4 → " the constant bickering and nattering of the other"; **8 → " the constant bickering between her
  and her sister."**; 32 → " the whole thing. "I'm going to be".
- Waking all 32 is 4x the expert work and WORSE: the router and the mixing weights were trained for 8.

## How many experts specialise (`specialists.py` → `out/specialists.json`)
- On our four texts (English, French, Python, sums; 300 word pieces each, k=8), an expert counts as a clear specialist
  when one kind of text uses it at least 50 points more often than any other kind: **34 of 768** (margin 0.3: 123;
  0.7: 3). By kind: sums 23, French 6, Python 3, English 2. Every expert was used by at least one text.
  → "Most of the 768, we couldn't name" holds (734 of 768 show no clear specialty on these texts).
- Layer 21, the word piece "Alice": the 8 chosen are 2, 4, 5, 7, 11, 13, 27, 28; the blend gives **94%** of the
  weight to one of them (the rest 0.6-1.2% each). `out/routes.json` holds every layer's 8 for the first 24 word pieces
  of both openings.

## Cautions
- No timing claims: our timings were taken while other jobs ran. Talk about work (numbers used), not seconds.
- "Specialist" is measured on our texts only; most experts aren't this clean, and we didn't name them all.
- 255/299 on Alice is high partly because Alice is famous (likely in its training data); say "the opening of Alice".
- One small model. Large MoE models (Mixtral, DeepSeek) use the same idea at bigger scale; we measured this one.
