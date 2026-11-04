# Reasoning master (11-10): does an AI get better at maths when it "thinks" first? — measured findings (2026-11-04)

**Model:** Qwen2.5-0.5B-Instruct (Apache-2.0), run by us in plain numpy (`qwen_np.py`), greedy, on a
laptop CPU. **Problems:** the first 50 of GSM8K's test set (grade-school maths word problems; OpenAI, MIT licence).
`python experiment.py --models <hf cache>` writes `out/results.json` (log `out_run.log`); the first run kept as `out/results_run1.json` in our working copy
is the first run (direct + steps only), kept for reference.

## Three ways to ask
| mode | how | right | tokens written | time per problem |
|---|---|---|---|---|
| **forced** | reply pre-started with "The final answer is": no room to think | **2 / 50** | 5.2 | 1.7 s |
| direct | "...Answer with the final number only." | 1 / 50 | 9.6 | 2.5 s |
| **steps** | "...Think step by step, then give the final answer on the last line as 'Answer: <number>'." | **20 / 50** | 284 | 52.7 s |

- "direct" is NOT a fair no-thinking score: 24 of 50 times the model ignored "number only", started explaining,
  and hit the 16-token cap. Use **forced** for "answer at once" (it always writes a number).
- Steps right where forced was wrong: **19**. Forced right where steps was wrong: **1** (problem 38, gold 10).
- So: **10x** as many right answers (2 → 20) for about **31x** the time (1.7 s → 52.7 s) and ~55x the tokens.
- Steps hit the 400-token cap 4 times (counted as written; those answers are mostly wrong).

## The running example — problem 6 (the sheep)
"Toulouse has twice as many sheep as Charleston. Charleston has 4 times as many sheep as Seattle. How many sheep do
Toulouse, Charleston, and Seattle have together if Seattle has 20 sheep?" Answer **260**.
- forced: "The final answer is **120**."
- steps: Seattle 20 → Charleston 4 × 20 = 80 → Toulouse 2 × 80 = 160 → 20 + 80 + 160 = **260**. Right.

## The opposite case — problem 9 (Eliza's overtime)
"$10 an hour for the first 40 hours, overtime 1.2 times her rate, she worked 45 hours." Answer **$460**.
- steps: "$10 × 1.2 = $12 per hour" (right), "Regular $10 × 40 = $400" (right), then "**Overtime $12 × 45 hours = $540**"
  (wrong: overtime is the 5 extra hours, $60), total "**$940**". Confident, tidy, wrong.
- Other step failures look the same: James's sprints, 3 × 3 × 60 = 540, it wrote 60 × 3 = 180 (dropped a step);
  Henry's bike trip, it wrote 20 − 15 = 5 (gold 25). Steps that look like reasoning do not guarantee it.

## Reasoning models — DeepSeek-R1 paper (arXiv 2501.12948, v2 of 4 Jan 2026), read verbatim §1–§2.3
- R1-Zero: RL (GRPO) straight on the base model, "bypass the conventional supervised fine-tuning (SFT) phase".
- Rewards are rule-based: "accuracy rewards evaluate whether the response is correct" (maths answer in a set
  format) and "format rewards" for reasoning inside <think> tags; no neural reward model for reasoning.
- "a steady increase in thinking time throughout training ... generating hundreds to thousands of tokens".
- AIME 2024 pass@1 "jumping from an initial 15.6% to 77.9%" (v2; v1 reported 71.0%: quote v2).
- An "aha moment": "a sudden increase in the use of the word 'wait' during reflections".
- Limits: "poor readability and language mixing".

## Cautions for the script
- We PROMPTED a small model to show steps; reasoning models are TRAINED to think (we did not run one).
- One small model, 50 problems, greedy: not a benchmark score for any product.
- Timings are ours (CPU, numpy); the ratio is the point.
