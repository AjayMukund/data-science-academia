# Recommender master (11-15): how does it know what you'll like? — measured findings (2026-11-10)

**Data:** goodbooks-10k (Zygmunt Zajac, **CC BY-SA 4.0**, github.com/zygmuntz/goodbooks-10k): 5,976,479 Goodreads
ratings (1-5 stars) of the 10,000 most-rated books by 53,424 readers. (MovieLens was rejected: its licence forbids
revenue-bearing use without permission.)
**Code:** plain numpy, `python experiment.py --data download` → `out/results.json`; `python detail.py --data download
--reader 9885` → `out/reader_9885.json`. A seeded sample of **10,000 readers** (1,115,666 ratings, median 110 per
reader) against all 10,000 books, to fit in memory. Seeded: re-runs match.

## The test
For every reader, hide the last book they gave 5 stars (9,930 readers had one). Each method ranks the ~9,900 books
that reader hasn't rated; a **hit** = the hidden book in the top 10.

| method | hits of 9,930 | rate |
|---|---|---|
| Bestseller list (most 5-star ratings; the same 10 for everyone) | **297** | 3.0%, ~1 in 33 |
| Readers like you (30 most similar readers) | **627** | 6.3%, ~1 in 16 — **2.1x** |
| Taste numbers (matrix factorisation, 20 numbers) | 246 | 2.5% (worse than the bestsellers here; not in the script) |

- The bestseller top 10: Harry Potter 1, The Hunger Games, To Kill a Mockingbird, Harry Potter 7, 3, 4, 6, 5, 2,
  The Hobbit.
- "Similar" = cosine between two readers' ratings, each measured from that reader's own average (a 4 from a harsh
  rater counts for more than a 4 from someone who gives everything 5). Score of a book = the similar readers'
  above-average liking of it, weighted by how similar they are.

## The running example — reader 9885 (anonymous Goodreads user id)
- Rated **111** books; a harsh rater: **8** five-stars (2: 6, 3: 40, 4: 57, 5: 8). Average 3.60.
- Five stars: The Fault in Our Stars, The Giver, The Book Thief, Big Little Lies, Still Alice, The Storyteller,
  The Good Girl, The Girl in the Spider's Web. Two stars: The Maze Runner, Miss Peregrine's Home for Peculiar Children,
  Little Bee, The Dressmaker, A Mother's Reckoning, The Cellar.
- Hidden: **A Man Called Ove** (Fredrik Backman; no other Backman book rated).
- Bestseller list: Ove ranks **319th**. Readers like you: **4th** of the books they hadn't rated.
- Their top 10 from readers like them: The Help, Harry Potter 1, The Hunger Games, **A Man Called Ove**, Catching Fire,
  Wonder, The Glass Castle, The Nightingale, Harry Potter 3, The Cuckoo's Calling.
- Of the 30 most similar readers, **7 had read Ove: 4 gave it 5 stars, 3 gave it 4.** None lower.
- One of the nearest (2nd): **21 books in common**, same stars on 8; both 5 stars for The Fault in Our Stars;
  Divergent 4 vs 5, Gone Girl 4 vs 5, Miss Peregrine's 2 vs 3. Against each reader's own average (9885: 3.60, twin: 4.18)
  they land on the same side on **16 of the 21** (`stats.py` → `out/stats.json`). NOT The Giver: 5 vs 4 is above 9885's
  average but below the twin's (draft 1 of the script said it; corrected 2026-11-10).
- Generous raters: reader **20597** rated 105 books, **all 105 five stars**; 20% of the 10,000 give 5 stars to at least
  half their books. Average rating across readers: 3.91.
- Ove in the 10,000-reader sample: 417 readers, 163 gave it 5 stars.

## Cautions for the script
- 1 in 16 means it still misses 15 times out of 16 (top 10 of ~9,900; the hidden book is ONE right answer, the other
  nine may be good too — we can't tell).
- A brand-new reader with no ratings has no similar readers: they get the bestseller list (by construction).
- Popular books creep back in: 4 of reader 9885's top 10 are Harry Potter / Hunger Games.
- Real services (Netflix, Amazon, Goodreads) use far more signals (clicks, time, what you skip); we can't see inside
  them. This is the oldest idea at their core ("collaborative filtering"), not their system.
- Reader ids are anonymous numbers in a public dataset; don't invent a name or a life for 9885.
