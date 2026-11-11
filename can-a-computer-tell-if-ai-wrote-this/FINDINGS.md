# AI-detector master (11-20): can a computer tell if AI wrote this? — measured (2026-11-11)

**Models:** GPT-2 small (124M, MIT) as the detector; Qwen2.5-0.5B-Instruct (Apache 2.0) as the AI writer. Both run on
our laptop's CPU. **Human text:** the opening ~100 words of 50 English Wikipedia articles (CC BY-SA 4.0, revision IDs in
`out/human.json`). **Old texts:** public domain.
**Code:** `topics.py`, `fetch_human.py` → `out/human.json`; `generate_ai.py` → `out/ai.json`; `detect.py` →
`out/scores.json`; `example.py` → `out/moon.json`. Seeded: re-runs match.

## The detector
- GPT-2 reads a paragraph one word-piece at a time and, before each one, guesses what comes next (a probability for
  each of its 50,257 pieces). **Surprise** = how unlikely the real next piece was. Average surprise → **perplexity**:
  low = every word was a word GPT-2 expected = "looks AI-written".
- The rule: perplexity under **13.6** → "AI". The cut-off was chosen on topics 1–25 (94% right) and tested on 26–50
  (**96%** right). On all 100: **95%** right.

## 50 human vs 50 AI paragraphs on the same topics
- Human (Wikipedia): perplexity median **23.9** (12.5 – 49.4). **2 of 50 wrongly flagged as AI** (Volcano 12.5,
  Tornado 12.9).
- AI (the chatbot's own default settings, temperature 0.7): median **10.2** (6.2 – 16.9). **47 of 50 caught**.
- Share of word-pieces that were in GPT-2's top 10 guesses: human 71%, AI 84%.

## The running example: the Moon
- Wikipedia's Moon paragraph: perplexity **20.9** → human ✓ (76% of pieces in the top 10; 36% the #1 guess).
- The AI's Moon paragraph: **11.9** → AI ✓ (84% top 10). It also says the Moon has "vast oceans of water" and an
  atmosphere "of nitrogen and oxygen" — wrong, fluently. The detector doesn't check facts.
- Word by word (rank of the real piece among GPT-2's guesses): human "The[2] Moon[780] is[1] the[2] only[2]
  natural[56] satellite[2] of[2] Earth[3]"; AI "the[1] name[37] given[2] to[1] the[1] … captured[22] the[1]
  imag[3]inations[1] of[1] humans[49] for[1] centuries[1]".

## The famous failure: old texts it has read
- The Declaration of Independence ("We hold these truths…", 69 pieces): perplexity **1.95** → **"AI"**. **90%** of its
  pieces were GPT-2's #1 guess. Genesis 1 (King James): **3.6** → "AI". The Gettysburg Address (30.9) and Shakespeare's
  Sonnet 18 (114.7) pass as human.
- Why: GPT-2 learned from web pages that quote them over and over; predictable to it ≠ written by a machine.

## The twist: same AI, dice rolled more freely
- Same model, same prompts, but plain sampling (temperature 1.0, no cut-offs): median perplexity **25.9**, above the
  human median. **Only 3 of 50 caught: 47 slip through as "human".**
- The Moon, temperature 1.0: **22.9** → "human" (it reads cleanly).
- Honesty: 17 of these 50 contain stray foreign characters or nonsense a reader would notice; of the 33 that read
  cleanly, 30 slip through. Don't claim all 47 are convincing.
- Asking it to "write in a casual, personal voice" (5 topics tried) did NOT fool the detector (11–14) — not in the video.

## Cautions
- A small detector and a small AI on 100 short paragraphs; commercial detectors use bigger models, same idea.
- Wikipedia is edited, formal text; students' essays differ.
- "AI-written" means "predictable to GPT-2", nothing more.
