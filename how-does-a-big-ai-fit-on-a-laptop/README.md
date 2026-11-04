# How does a big AI fit on a laptop? Quantization, measured

Qwen2.5-0.5B-Instruct (494 million numbers, 988 MB at 16 bits) has every weight matrix rounded to 8, 4, 3 and
2 bits, in groups of 32 numbers sharing one 16-bit scale, then answers a 50-capitals quiz and describes the
Eiffel Tower. 8-bit: 525 MB, 44 of 50. 4-bit: 278 MB, 38 of 50, still fluent. 3-bit: 216 MB, 0 of 50, and
"8.8.8.8.8.8.8.9.111". 2-bit: gibberish.

The model runs in plain NumPy (`qwen_np.py`), no PyTorch needed. The rounded numbers are turned back into
floats for the maths, so the answers are exactly what the stored integers would give.

```bash
pip install numpy tokenizers huggingface_hub
python experiment.py --models path/to/a/model/folder   # ~1 GB of weights, ~4 GB of RAM; ~10 minutes on a laptop CPU
```

Greedy decoding, so a re-run gives the same words. Results go to `out/results.json`. See `FINDINGS.md`.

Things to try: groups of 64 or 128; keep the embedding table at 8 bits; protect the largest 1% of weights in
each row (the idea behind smarter formats) and see whether 3 bits recovers. This is the simplest rounding:
real laptop formats (k-quants, GPTQ, AWQ) do better at low bits.
