# Why do sixty words write half of a Jane Austen novel? (Zipf's law)

Reel 12-07 (replaces the withdrawn Shazam reel, 2026-11-26: it duplicated the masters session's 12-04 fingerprinting
master). Run: `python experiment.py` (plain Python, downloads from the Gutenberg mirror) -> results.json.

## Data
Jane Austen's six novels, Project Gutenberg (public domain): Pride and Prejudice, Emma, Sense and Sensibility,
Persuasion, Northanger Abbey, Mansfield Park. **731,189 words, 14,297 different words.** (Not the 12-03 Holmes books.)

## Findings
- **"The"** is the commonest word: 26,694 times, **1 word in 27** (3.65%). Then to, and, of, a, her, I, in, was, it.
- **About sixty words write half of each novel**: 60 overall; book by book 57 (Emma, Sense and Sensibility, Mansfield
  Park), 58 (Pride and Prejudice), 59 (Persuasion), 62 (Northanger Abbey).
- Ranked biggest first: the 10th word ("it") is 1 in 71; the 100th ("great") 1 in 740; the 1,000th 1 in 10,016.
  Ten times further down the list, about ten times rarer: rank x share stays 0.10-0.14 from rank 10 to 1,000 (Zipf's law).
  Log-log slope, ranks 10-5,000: -1.31 (Zipf's ideal is -1; the top few words are flatter than the law).
- Top 10 words cover 22.8% of all text; top 100: 57.7%; top 1,000: 85.2%.
- **A third of her vocabulary is used exactly once**: 4,564 of 14,297 words (31.9%) - e.g. "officiousness",
  "tranquilize", "abandoned", "inaction".
- GPT-2's tokenizer keeps " the" and " great" as one token each, but splits " officiousness" into 3 (offic|ious|ness)
  and " tranquilize" into 2: common words get their own piece, rare ones are built from parts.

## Script cautions
- Say "about sixty" (57-62), "one word in twenty-seven". Don't claim Zipf is exact: it holds from about the 10th word on.
- The tokenizer point is a light link to AI, not a claim about how well models handle rare words (not measured).
