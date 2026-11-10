# How does it know what you'll love next? A recommender, measured on Goodreads

"Readers like you" (user-based collaborative filtering) against a bestseller list, in plain NumPy, on goodbooks-10k:
6 million real Goodreads ratings of the 10,000 most-rated books. For each of 10,000 readers we hide the last book they
gave five stars; a hit is that book landing in their top 10. The bestseller list finds it for 297 of 9,930 readers;
the 30 most similar readers (cosine on ratings measured from each reader's own average) for 627, more than double.
The running example: reader 9885 rated 111 books and gave five stars to only 8; the hidden one, A Man Called Ove,
is 319th on the bestseller list and 4th for readers like them.

```bash
pip install numpy
mkdir download && cd download
curl -L -O https://github.com/zygmuntz/goodbooks-10k/releases/download/v1.0/ratings.zip
curl -L -O https://github.com/zygmuntz/goodbooks-10k/releases/download/v1.0/books.zip
unzip ratings.zip && unzip books.zip && cd ..
python experiment.py --data download              # ~15 minutes; out/results.json
python detail.py --data download --reader 9885    # the running example; out/reader_9885.json
python stats.py --data download                   # the generous rater and the reading twin; out/stats.json
```

Data: goodbooks-10k by Zygmunt Zajac, CC BY-SA 4.0 (https://github.com/zygmuntz/goodbooks-10k). Reader numbers are the
dataset's anonymous ids. Seeded, so a re-run gives the same numbers. See `FINDINGS.md`.

Things to try: more or fewer neighbours (`K_NB`), plain cosine without the mean-centring, more readers, or the
"taste numbers" (matrix factorisation) at other sizes: here, with 20 numbers, it scored 246 hits, below the bestsellers.
