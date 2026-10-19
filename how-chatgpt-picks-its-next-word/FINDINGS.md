# Sampling master — measured findings (2026-10-17)

GPT-2 small (openai-community/gpt2) run in numpy from
`../embeddings/gpt2.safetensors`; vocabulary 50,257. Scripts in the session
folder (`gpt2_core.py`, `greedy_exp.py`, `conf_exp.py`).

## "The cat sat on the" — the real next-token list

| rank | token | probability |
|---|---|---|
| 1 | floor | 7.64% |
| 2 | bed | 6.53% |
| 3 | couch | 5.41% |
| 4 | ground | 5.21% |
| 5 | edge | 4.78% |
| 6 | bench | 3.16% |
| 7 | table | 3.15% |
| 8 | sofa | 2.88% |
| 9 | other | 1.99% |
| 10 | back | 1.90% |
| … | | |
| 26 | roof | 0.51% |
| 31 | **mat** | **0.40%** |

- The famous "mat" is 31st. The top 10 hold only 42.6% of the probability.
- "windowsill" and "carousel" (the temperature/top-k reels' toy words) are
  2 tokens each in GPT-2 — the reels' five-word list was a simplified toy
  with illustrative scores. Say so if comparing.
- Raw logits sit near −80 (GPT-2's offset; softmax ignores it). Show
  probabilities, or score differences, never the raw −80s.

## Temperature on the same list

| T | "floor" | top-10 share |
|---|---|---|
| 0.5 | 24.9% | 92.3% |
| 1.0 | 7.6% | 42.6% |
| 1.5 | 1.8% | 11.9% |

## Top-k and top-p

"The cat sat on the": top-p 0.5 keeps 17 tokens; 0.9 keeps **728**;
0.95 keeps 1,729. Top-k 40 keeps 61.7% of the probability.

Confident prompts:

| prompt | top choice | top-p 0.9 keeps | top-k 40 holds |
|---|---|---|---|
| "Thank you very" | much 99.2% | **1** | 99.9% |
| "The United States of" | America 96.6% | **1** | 98.2% |
| "1, 2, 3, 4," | 5 87.1% | **3** | 98.2% |
| "Barack" | Obama 72.7% | 149 | 85.5% |
| "Once upon a" | time 17.7% | 1,956 | 51.6% |

GPT-2 small is NOT confident on "The capital of France is" (top: "the"
8.5%) — don't use it.

## Greedy decoding loops (60 tokens, always the top token)

- "The cat sat on the" → "…the floor, and the cat was still asleep.
  "I'm sorry, I'm sorry," she said. "I'm sorry. I'm sorry. …" — "I'm sorry"
  repeated 8+ times.
- "I think that" → "I think that's a good thing. I think that's a good
  thing. …" — the sentence repeated 8 times.

## Sampling at T = 1 (seed 7, 10 tokens each)

- "The cat sat on the attached rooftop tree, its saddle, surrounded by its"
- "The cat sat on the wall. "So do you??? Head over check"
- "The cat sat on the ground, sitting in a corner of her plastic bag"
