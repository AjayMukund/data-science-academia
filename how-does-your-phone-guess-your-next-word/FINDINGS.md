# How does your phone guess your next word? (n-grams -> language models)

Reel 12-03. Run: `python experiment.py` (plain Python, downloads the books) -> results.json;
`python lm_compare.py` in a scratch venv with torch + transformers -> lm_results.json.

## Data
Eight Sherlock Holmes books by Arthur Conan Doyle, Project Gutenberg (public domain in the US; downloaded from the
official mirror gutenberg.pglaf.org, www answers 503 to scripts): A Study in Scarlet, The Sign of the Four, Adventures,
Memoirs, Return, The Valley of Fear, His Last Bow - the counter reads these (**519,976 words**, 17,520 different);
**The Hound of the Baskervilles is held back** as the test it never read (59,636 words).

## The counter (a word-pair / three-word table, the classic phone-keyboard method)
- "the" is 5.5% of every word it reads.
- After **"Sherlock"**: "Holmes" 320 of 347 times (**92%**). After **"Baker"**: "Street" 93 of 107 (87%).
- After **"my dear"**: **"Watson" 74 of 177 (42%)**, "fellow" 35 (20%), "sir" 15 (8.5%).
- After "I": have 9.0%, had 5.9%, am 5.8%.
- Three suggestions, like a phone keyboard (last two words; if never seen, the last word; else the commonest words).
  On the unread Hound: the right word is among the three **24.3%** of the time; first suggestion right 14.6%.
- **26.6%** of the Hound's word pairs never appear in the seven books: the counter falls back to guessing from one word.
  "my phone", "my laptop", "the internet": seen **0** times.
- Always taking the first suggestion, it loops: "my dear watson said he i have no doubt that the man who had been a
  very serious matter small said the inspector and **i have no doubt that the man**..."

## The same test for a language model (GPT-2, 124M parameters, OpenAI 2019, MIT licence; CPU)
Scored on the same 54,844 positions (92% of the Hound's words: those where the next word is one whole GPT-2 token):
| | right word among 3 suggestions | first suggestion right |
|---|---|---|
| the counter (read 7 Holmes books) | **25.9%** | 15.6% |
| GPT-2 (read the web, not tuned on Holmes) | **48.3%** | 29.8% |
- GPT-2 handles what the counter never saw: "I left my laptop on the" -> floor 13%, couch 12%, bed, desk.
  "Holmes picked up the phone and" -> called 14.5%, said 8.3%, asked 6.1% (counter, never having seen "phone":
  the / i / a).
- But "My dear" -> GPT-2 says "friend" (9.9%), not Watson: it learned how words behave in general, not Holmes.
- Caveat: GPT-2's web training text may have included the Hound itself. Say "a small AI model", give its number, and do
  not claim it never saw the book.

## Background (for the script, verified)
- Google's Gboard paper (Hard et al. 2018, "Federated Learning for Mobile Keyboard Prediction") compares a neural
  network keyboard model against an n-gram baseline: phones moved from counting to small neural networks. Say "early
  phone keyboards counted; newer ones use small neural networks" - nothing more specific.
