# Can a computer tell if AI wrote this? A perplexity detector, measured

GPT-2 small reads a paragraph and, at every word-piece, scores how surprised it was; the average (perplexity) under
13.6 means "AI". On 50 Wikipedia openings against 50 paragraphs written by Qwen2.5-0.5B-Instruct on the same topics,
it catches 47 of 50 AI paragraphs and wrongly flags 2 of 50 human ones (95%). It also calls the Declaration of
Independence AI-written (perplexity 1.95), and the same chatbot sampled at temperature 1.0 slips past 47 times in 50.

```bash
pip install torch transformers numpy requests
python generate_ai.py --models path/to/hf-cache   # Qwen writes 2 x 50 paragraphs -> out/ai.json (~35 min on CPU)
python detect.py --models path/to/hf-cache        # GPT-2 scores everything -> out/scores.json
python example.py --models path/to/hf-cache       # the Moon, word by word -> out/moon.json
```

The 50 Wikipedia paragraphs we measured ship in `data/human.json` (CC BY-SA 4.0, with each article's revision ID),
because Wikipedia keeps changing; `fetch_human.py` fetches today's versions instead. See `FINDINGS.md`. Things to try:
a bigger detector (gpt2-medium); your own writing; a stricter or looser cut-off and how the two error types trade.
