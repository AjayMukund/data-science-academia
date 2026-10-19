# Transformer-layer master — measured findings (2026-10-17)

GPT-2 small (openai-community/gpt2, 12 layers, d = 768) run in numpy by
`experiment.py`; full output in `output.txt`. Weights live OUTSIDE the
repository: run `python ../common/download_gpt2.py` (548 MB); re-download
from Hugging Face if missing.

"After layer N" readouts are the **logit lens** (nostalgebraist, 2020): the
model's own final layer norm + unembedding applied to the middle of the
stack. It is an approximation, not what the model "thinks" - say "if it
stopped here, it would say...", never "layer 9 believes Rome".

## The example: "The Eiffel Tower is in the city of"

Tokens: `The | E | iff | el | Tower | is | in | the | city | of` (Eiffel is 3 tokens).

| after | top guess | Paris |
|---|---|---|
| embedding | ` destro` (noise) | rank 22,680 |
| layers 1-5 | **the** (69-81%, 41% at 5) | rank 5,042 → 658 |
| layer 6 | the 5.8%, **England** 5.3% | rank 261 |
| layer 7 | **East**, England, southwest | rank 103 |
| layer 8 | Ing, **Rome** 8.1% | rank 64 |
| layer 9 | **Rome 19.2%**, London, Chicago | 2.55%, rank 10 |
| layer 10 | **London 28.5%**, Paris 24.8% | rank 2 |
| layer 11 | **Paris 18.3%**, London 17.5% | rank 1 |
| layer 12 (final) | **Paris 7.0%**, London 5.8%, New 3.1% | rank 1 |

- The final answer is right but barely: Paris 7.0% vs London 5.8%. Small model.
- Confidence DROPS on the last layer (18.3% → 7.0%). The same happens on
  other prompts (cold 64% → 16%, basketball 62% → 14%). Known GPT-2 behaviour;
  don't explain it, admit it.
- Same shape elsewhere: "Michael Jordan plays the sport of" says **baseball**
  at layer 8 (29%), basketball from layer 9. "The opposite of hot is":
  "now"/"not" until layer 8, cold at 9. "The United States of" → America
  appears at 7, 87% at 9.
- Easy completions settle immediately: "...is in New" → York 87% after layer 1.

## What "of" looks at (attention, Eiffel prompt)

- From layer 3 to 10 the strongest head on "of" points at **city**
  (85%, 93%, 100%, 79% in layers 3-6; head-average on city 27-33% at 3-5).
- Direct attention from "of" to E/iff/el/Tower is tiny (1-6%). Don't claim
  "of reads Eiffel" - the information reaches "of" indirectly.
- Most remaining attention lands on the first token "The" (63-83% in layers
  6-11): GPT-2's "attention sink", a known parking spot. Don't draw it as meaning.

## One layer = attention + feed-forward

| part | parameters | share |
|---|---|---|
| attention | 2,362,368 | 33.3% |
| feed-forward | 4,722,432 | **66.6%** |
| layer norms | 3,072 | 0.04% |
| one layer | 7,087,872 | |

- Feed-forward widens 768 → **3,072** → 768 (4x).
- 12 layers = 85.1M; embeddings 39.4M; total ≈ 124.4M (the "124M" model).
- GPT-3 (Brown et al. 2020, Table 2.1): **96 layers**, d = 12,288,
  **175B** parameters. Same two-part floor, 96 times.

## Knock-outs (Eiffel prompt, Paris 7.0% baseline)

| removed | Paris | top guess |
|---|---|---|
| any ONE layer's attention or feed-forward (layers 2-12) | 3.3-12.8% | Paris or London |
| layer 1's attention / feed-forward | 0.02% / 0.01% | the |
| **attention, layers 7-12** | 0.44% (rank 20) | New, London, Los - still cities |
| **feed-forward, layers 7-12** | **0.06% (rank 158)** | the, ",", a - no longer a place |
| attention everywhere | 0.01% | the |
| feed-forward everywhere | 0.00% | " Suddenly" |
| **the highway (x = f(x), no skip connection)** | **0.0000%** | "-" 39.5%, "2" 17.8% |

- No single upper floor is essential: the stack is redundant.
- Without the residual "highway" the model outputs junk on every prompt
  tested (York 86.7% → 0.0003%).

## Each floor edits, it doesn't rewrite (the last word "of")

- Layer 1 replaces the raw embedding almost completely (cos 0.17 with it).
- **Layers 2-10 each change the vector by only 20-43%** of its size; cosine
  with the previous floor stays 0.93-0.98.
- The last two layers change it most (0.90x, 0.96x) - where the readout
  swings from London to Paris and confidence moves.
- The vector's length grows from 55 to 417 through the stack (why layer
  norm rescales before every block).

## Rejected candidates

- "The capital of Japan is": final top is "the" (Tokyo #2). "Water freezes at
  zero degrees": final top " C". "The Colosseum...": Rome 1.4%, rank 4.
  "Big Ben...": London rank 5. Use none of them.
- "The Brandenburg Gate is in the city of" → Berlin 16.8%, rank 1: a usable
  backup example.

## v2 additions — the architecture walk-through (`pipeline.py`, `pipeline_output.txt`)

- Token IDs: The 464 · E 412 · iff 733 · el 417 · Tower 8765 · is 318 · in 287 · the 262 · city 1748 · of 286.
- Embedding rows are 768 numbers; " city" starts `+0.00 +0.09 +0.13 -0.09 -0.04 -0.17`.
  The table row depends only on the token, never on where it sits ("The" 464 and
  " the" 262 are different tokens - the sentence has no repeated token).
- **Attention arithmetic, layer 3 head 4, query " of"** (q·k, ÷8, softmax):
  city −10.6 → −1.32 → **84.5%**; the −35.5 → 3.8%; of −36.6 → 3.3%; The −37.7 → 2.8%;
  Tower −37.8 → 2.8%; in −43.3 → 1.4%; is −44.8 → 1.2%; el −63.0, iff −65.9, E −78.6 → ~0%.
- **Heads differ (layer 3):** head 2 mostly looks one token back (iff→E 99%, the→in 86%,
  in→is 77%); head 4 sends "of" to city 85% and "el" to iff 84%.
- **Output:** top of 50,257 - Paris 7.00%, London 5.84%, New 3.10%, Amsterdam 2.92%, Berlin 2.35%.
- **Generation (always the top token):** Paris (7.0%) , (35.6%) and (12.0%) the (14.0%) E (3.0%) iff (100%)
  → "The Eiffel Tower is in the city of Paris, and the Eiff".
