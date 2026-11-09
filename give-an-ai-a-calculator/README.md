# Give an AI a calculator: does running code fix its maths? Measured

A small chat model, Qwen2.5-0.5B-Instruct, answers the first 50 GSM8K maths word problems by writing a short Python
program, which we run in a separate, isolated Python process (5-second limit) to read the number it prints. Compared
problem by problem with the same model's step-by-step answers from
[`why-does-an-ai-get-maths-right-when-it-shows-its-working`](../why-does-an-ai-get-maths-right-when-it-shows-its-working):
20 of 50 right either way, but not the same 20 (14 both, 6 only code, 6 only steps). Code fixes arithmetic slips
(Mishka's shopping: "3 × $16.50 = $54.50" becomes 49.5, total 243) but not wrong plans (Eliza's overtime: an exact,
wrong $510). When the two methods agree (17 problems), 14 are right.

The model runs in plain NumPy (`qwen_np.py`), no PyTorch needed.

```bash
pip install numpy tokenizers huggingface_hub
# first run the reasoning experiment, which this one compares against:
(cd ../why-does-an-ai-get-maths-right-when-it-shows-its-working && python experiment.py --models path/to/models)
python experiment.py --models path/to/models \
    --steps ../why-does-an-ai-get-maths-right-when-it-shows-its-working/out/results.json   # ~15 minutes on a laptop CPU
```

Greedy decoding, so a re-run writes the same programs; timings are your machine's. Results go to `out/results.json`.
See `FINDINGS.md`. The generated programs run with no network access expected and a time limit, but they are code
written by a model: read `out/results.json` before reusing any of it.

Things to try: ask for code and steps together and keep the answer only when they agree; let the model see the error
message and try again; a bigger model (Qwen2.5-1.5B-Instruct).
