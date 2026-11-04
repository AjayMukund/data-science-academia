# Why doesn't a long chat take forever? The KV cache, measured

A small chat model, Qwen2.5-0.5B-Instruct, holds one real 48-message chat about a hike with a dog called
Biscuit; it writes every reply itself. For each of its 24 answers we time two ways of producing it:
keeping every token's key and value from all 24 layers (the KV cache), and re-reading the whole chat for
every new token. At question 24 (1,916 tokens) re-reading costs 20.9 s per token on our laptop; with the
cache, 0.33 s (63x), and the scores match to within 0.00003. Then the cost: 12 KB of notes per token for
this model, 320 KB for Qwen2.5-72B (10.7 GB for one 32,768-token chat). Finally, with the first message
dropped, it is asked the dog's name: "a poodle", then "Buddy".

The model runs in plain NumPy (`qwen_np.py`), no PyTorch needed.

```bash
pip install numpy tokenizers huggingface_hub
# optional: a long plain-text book for the "pasted book" recall test (we used Alice in Wonderland,
# Project Gutenberg #11: https://www.gutenberg.org/ebooks/11)
python experiment.py --models path/to/a/model/folder --book alice.txt   # ~1 GB of weights; ~30 minutes on a laptop CPU
```

Replies are greedy, so a re-run writes the same chat; the timings are your machine's. Results go to
`out/results.json`. See `FINDINGS.md`.

Things to try: a longer chat (where does the re-read cost become impossible?); drop the last message
instead of the first; ask the dog's name in your own words; compute the notes per token for another
model from its `config.json` (layers x key/value heads x head size x 2 x 2 bytes). Real chat servers run
on GPUs and are far faster than this CPU run; the ratio is what carries over.
