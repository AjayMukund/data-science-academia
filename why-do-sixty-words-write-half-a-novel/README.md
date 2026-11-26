# Why do sixty words write half of a Jane Austen novel? Zipf's law

Jane Austen's six novels (Project Gutenberg, public domain): 731,189 words, 14,297 different words. **About sixty words make up half of each novel (57 to 62, book by book). "The" is one word in 27; the 10th word one in 71; the 100th one in 740: ten times further down the list, about ten times rarer.** A third of her vocabulary is used exactly once.

```bash
python experiment.py      # plain Python; downloads the six novels from the Gutenberg mirror on first run
```

See `FINDINGS.md`. Try: run it on any other author's books and see whether the sixty holds.
