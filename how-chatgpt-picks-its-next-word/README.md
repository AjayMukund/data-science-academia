# The cat sat on the… how ChatGPT picks its next word

GPT-2's real probability list for "The cat sat on the", and what greedy
decoding, sampling, temperature, top-k and top-p do to it.

```bash
python ../common/download_gpt2.py                           # once, 548 MB
python gpt2_core.py    ../models/gpt2/model.safetensors   # the next-word list
python greedy_exp.py   ../models/gpt2/model.safetensors   # greedy decoding loops
python sampling_exp.py ../models/gpt2/model.safetensors   # sampling and temperature
python conf_exp.py     ../models/gpt2/model.safetensors   # top-k / top-p on confident prompts
```

See `FINDINGS.md`. Try: your own prompt, or a temperature of 2.
