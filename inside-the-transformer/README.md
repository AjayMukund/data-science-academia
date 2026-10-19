# How ChatGPT works inside, step by step

GPT-2 small, run layer by layer in plain NumPy on "The Eiffel Tower is in the
city of". See what it would answer after each of its 12 blocks, the real
attention arithmetic, and what breaks when parts are switched off.

```bash
python ../common/download_gpt2.py                      # once, 548 MB
python experiment.py ../models/gpt2/model.safetensors   # the layer-by-layer readout, knock-outs
python pipeline.py ../models/gpt2/model.safetensors     # tokens, attention scores, generation
```

See `FINDINGS.md`. Try: your own sentence.
