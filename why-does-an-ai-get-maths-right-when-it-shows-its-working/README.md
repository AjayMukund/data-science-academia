# Why does an AI get maths right when it shows its working? Measured

A small chat model, Qwen2.5-0.5B-Instruct, answers the first 50 problems of GSM8K's test set three ways:
forced to answer at once (its reply is pre-started with "The final answer is"), asked for "the number only",
and asked to think step by step. Forced: 2 of 50 right, 5 tokens, 1.7 s each on our laptop. Step by step:
20 of 50, 284 tokens, 52.7 s. The sheep problem (#6): "120" at once, 20 → 80 → 160 → 260 step by step.
And tidy steps can still be wrong: Eliza's overtime (#9) is paid on all 45 hours, $940 instead of $460.

The model runs in plain NumPy (`qwen_np.py`), no PyTorch needed. GSM8K's test file is downloaded from
github.com/openai/grade-school-math (MIT licence) on the first run.

```bash
pip install numpy tokenizers huggingface_hub
python experiment.py --models path/to/a/model/folder   # ~1 GB of weights; ~45 minutes on a laptop CPU
```

Greedy decoding, so a re-run gives the same answers; the timings are your machine's. Results go to
`out/results.json`. See `FINDINGS.md`.

Things to try: more problems (`--n 200`); a bigger model (Qwen2.5-1.5B-Instruct); ask it to check its own
answer; sample five working-outs and take the most common answer. This prompts a small model to show its
working; reasoning models are trained to do it, and we did not run one.
