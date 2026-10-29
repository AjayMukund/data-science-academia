# Why does AI make things up? 100 questions to GPT-2

GPT-2 small (124M, OpenAI 2019), run in plain NumPy, asked "What is the capital of X?" for 50 real
countries and 50 invented ones (Brastovia, Quelloria, ...). It gets 39 of the real ones right and
answers all 50 invented ones, never once saying the country doesn't exist; a worked example of
declining doesn't change that. Its confidence is lower when inventing (34% vs 63%) but the two
overlap: a 40% cut-off catches 35 inventions, loses 6 right answers and lets 15 through.

```bash
pip install numpy tiktoken
# download model.safetensors from https://huggingface.co/openai-community/gpt2 (about 550 MB)
python experiment.py --weights path/to/model.safetensors   # about 10 minutes on a laptop CPU
```

See `FINDINGS.md`. Things to try: sample instead of greedy and ask each question ten times; add
more examples of declining; try a larger model and compare. This is a small 2019 model: it says
nothing about the rates of today's assistants.
