# Tool-use master (11-12): give an AI a calculator — measured findings (2026-11-04)

**Model:** Qwen2.5-0.5B-Instruct (Apache-2.0), plain numpy (`qwen_np.py`), greedy. **Problems:** the same
first 50 GSM8K test problems as the reasoning master. **New mode:** "Write a short Python program that calculates the
answer and prints only the final number"; we run the program (separate isolated Python process, 5 s limit) and read
what it prints. Compared problem by problem with the reasoning master's step-by-step answers
(the reasoning experiment's `out/results.json`). `python experiment.py --models <hf cache> --steps ../why-does-an-ai-get-maths-right-when-it-shows-its-working/out/results.json`.

| mode | right | tokens written | time per problem |
|---|---|---|---|
| steps (reasoning master) | **20 / 50** | 284 | 52.7 s |
| **code, run by us** | **20 / 50** | 113 | 16.2 s |

- Same score, **different problems**: both right 14; only code 6; only steps 6.
- Code ran without an error 49 times of 50 (one reply had no code block).
- **Agreement is a check:** the two methods gave the same number on 17 problems, and **14 of those 17 were right
  (82%)**; on the 33 where they disagreed, code alone was right 6 times (18%).
- Code was about 3x faster here: shorter answers (113 vs 284 tokens).

## The running example — problem 26 (Mishka's shopping): the sum the calculator fixes
"3 pairs of shorts at $16.50, 3 pairs of pants at $22.50, 3 pairs of shoes at $42. How many dollars?" Answer **243**.
- steps: right plan, wrong sum: "3 × $16.50 = **$54.50**" (it is $49.50), then $54.50 + $67.50 + $126 = **$248**.
- code: the same plan, `3 * 16.50` computed by Python → **243.0**. Right.

## The opposite case — problem 9 (Eliza's overtime): the plan the calculator can't fix
- steps (reasoning master): overtime paid on all 45 hours → **$940**.
- code: `earnings = 10 * 45 + 12 * (45 - 40)` → **$510**: every sum exact, the plan wrong (regular rate on all 45 hours
  PLUS overtime on 5). Right answer **$460**.
- Other code-only wins: problem 3 (sprints) 540 vs steps' 180; problem 16 230 vs 80.

## Cautions for the script
- One small model, 50 problems, greedy, our own sandboxed runner; real assistants call tools differently.
- "The arithmetic is exact" is true of the code we ran; the errors that remain are in what the code says to compute.
- Agreement (82%) is measured on 17 problems: a small number; say "most of the time".
