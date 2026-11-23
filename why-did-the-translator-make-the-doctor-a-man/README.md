# Why did the translator make the doctor a man?

Turkish "o" means he or she, so "O bir doktor." carries no gender. An open Turkish-to-English model (Helsinki-NLP/opus-mt-tr-en, Apache-2.0) translated "O bir <job>." for 50 jobs: **He 42, She 8**, never "they". Jobs that are mostly men: he, 26 of 26. Jobs that are mostly women: she, only 8 of 24. For "teacher" its first word is a near coin toss (He 0.43, She 0.37), but beam search takes the bigger number every time. Real shares of women: US BLS Current Population Survey 2025, Table 11 (public domain).

```bash
pip install torch transformers sentencepiece
python experiment.py --model <folder with opus-mt-tr-en>   # the 50 jobs and the he/she probabilities
python probe.py --model <folder with opus-mt-tr-en>        # adjectives and context sentences
```

Download the model from the Hugging Face Hub (`Helsinki-NLP/opus-mt-tr-en`); it runs on a laptop CPU. See `FINDINGS.md`. Try: add a context word ("O bir doktor ve o bir anne.") and watch the pronoun follow it.
