# How does your phone guess your next word? From counting to language models

Seven Sherlock Holmes books (Project Gutenberg, public domain) teach a word-pair counter, the classic phone-keyboard method. **After "my dear" it suggests "Watson": 74 times out of 177.** On The Hound of the Baskervilles, which it never read, the right word is among its three suggestions 24.3% of the time, and 26.6% of the Hound's word pairs were never seen. On the same 54,844 positions, GPT-2 (124M parameters) gets 48.3% against the counter's 25.9%.

```bash
python experiment.py      # plain Python; downloads the eight books from the Gutenberg mirror on first run
python lm_compare.py      # needs torch + transformers (downloads GPT-2, ~550 MB); CPU, a few minutes
```

See `FINDINGS.md`. Try: hold out a different book, or type your own two words into `suggest()`.
