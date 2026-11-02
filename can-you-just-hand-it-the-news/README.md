# Can you just hand an AI the news? Retrieval-augmented generation, end to end

A small chat model, Qwen2.5-0.5B-Instruct (released September 2024), is asked 12 questions about
events from 2025 and 2026. With no library it gets 0 right (4 refusals, 8 confident inventions).
Then it gets a library: the opening sections of 24 Wikipedia pages (12 hold the answers, 12 are
look-alikes from the edition before), cut into 106 chunks. all-MiniLM-L6-v2 turns every chunk and
the question into 384 numbers, and the 3 closest by cosine similarity are pasted above the question.
It now gets 5 right. Handed only the look-alike page, it answers 10 of 12 from last time's page:
"Argentina" won the 2026 World Cup, 88% sure.

Both models run in plain NumPy (`qwen_np.py`, `minilm_np.py`), no PyTorch needed.

```bash
pip install numpy scipy requests tokenizers huggingface_hub
python experiment.py --models path/to/a/model/folder   # downloads ~1.1 GB of weights; ~10 minutes on a laptop CPU
```

Pages are read from Wikipedia at the exact revisions we used (`REVS` in `experiment.py`), so a fresh
run reads the same text; `--fetch-latest` reads today's pages instead. Results go to `out/results.json`.

See `FINDINGS.md`. Things to try: retrieve 5 chunks instead of 3; use smaller or larger chunks; swap in
Qwen2.5-1.5B-Instruct as the reader; add a re-ranker. This is one small model and one small library:
real systems add better search, re-ranking and citations.
