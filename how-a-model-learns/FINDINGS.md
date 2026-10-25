# "How a model learns" master — measured findings (2026-10-18)

A real training run, done by us: `experiment.py` trains a character-level
language model (8 characters of context -> 16-number letter embeddings ->
256 tanh units -> 72 next-character probabilities; **52,680 numbers**) from
random on *Alice's Adventures in Wonderland* (Project Gutenberg #11, public
domain, `corpus/alice_raw.txt`). Plain numpy, hand-written backprop, seeded.
Logs in `out/` (`sweep.json`, `train.json`, `final_params.npz`).

## The data
- 143,018 characters, 72 distinct characters. 90% for training (128,708
  "guess the next letter" questions), 10% held out (14,294).
- Chance loss = ln 72 = **4.28** (guessing evenly among 72 characters).

## What it writes (prompt "Alice was ", temperature 0.6, seed 1)
| step | sample |
|---|---|
| 0 | `Alice was Z‘?‘JTtS_!o_NpKW?PDHnJW“‘l[JC’[;fqez(…` |
| 100 | `Alice was mod d the s it t  a the wor eus toar in the sar bos dod…` (seed 2: `and the git th it t ln s tor…`) |
| 300 | `Alice was hem” pthe seesa dad heryd that Anthe weenoupther…` |
| 1,000 | `Alice was ly ving land the Qouse wour could no she reat in the roll…` |
| 3,000 | `Alice was nt the changed the Gryphon iton the Mouse thing it was a stou…` |
| 10,000 | `Alice was now it a sught to see what what when she was in a sow the pid the Duch` |
| 60,000 | `Alice was pidy interbled had to flow all creature with some play at the next…` |

## Loss = surprise
- Training-batch loss: 4.26 (step 0) → 2.58 (100) → 2.29 (500) → 2.18 (1,000)
  → 1.70 (5,000) → 1.43 (10,000) → 1.36 (30,000) → 0.89 (end).
- Held-out loss: 4.28 → 2.70 (100) → 2.48 (300) → 2.23 (1,000) → 1.99 (3,000)
  → **1.86 (10,000) → 1.85 (60,000)**: no gain on unseen text after 10k steps
  while the training loss keeps falling. It is memorising Alice (overfitting).
- p(correct next letter) on the book's first sentence ("Alice was beginning to get very tired…"):
  step 0 ≈ 1–2% every letter (mean surprise 4.25); step 1,000: " " after
  "beginning" 94%, n 54%, e 34% (mean 1.95); step 60,000: the letters of
  "beginning" 84–100%, but the "v" of "very" 0% (mean 0.80). This sentence is
  in the training text.

## One number, swept (the valley)
- The output bias for "e", trained value 1.66. Swept ±6 on 4,000 held-out
  windows: loss 2.15 at the left end, 2.63 at the right, **minimum 1.807
  exactly at the trained value**.

## Learning rate (3,000 steps each, held-out loss)
| lr | result |
|---|---|
| 0.01 | 2.54 (slow) |
| 0.03 | 2.34 |
| 0.1 | 2.07 |
| **0.3** | **1.87** (used for the main run) |
| 1 | 1.84 |
| 3 | 3.45, batch loss spiking to **9-12** (worse than random, 4.28) |
| 10 | 10.87, batch loss 22-25 |

## What it taught itself (letter embeddings, cosine)
- "." nearest "!" (0.65), then "”" and "?" (0.46); "?" nearest "!" (0.67).
- Upper/lower case pairs: a↔A 0.53, o↔O 0.49, t↔T 0.42.
- Vowels do NOT cluster - don't claim it.

## Scale
- Our run: 60,000 steps × 64 guesses = **3,840,000 guesses**, 142 s on a CPU.
- GPT-3: **175B parameters** (Brown et al. 2020, abstract), trained on 300B
  tokens (§2.2).
- Llama 3: "pretrained on **over 15 trillion tokens**", 8B and 70B parameters
  (Meta Llama 3 model card, Hugging Face).

## One knob, three models (`learning_rate.py`)
Same model, book, seed and 3,000 steps; held-out error (random guessing = 4.28):

| learning rate | error every 500 steps | final | writes ("Alice was ", seed 1) |
|---|---|---|---|
| 0.01 | 4.28 2.95 2.75 2.65 2.58 2.54 2.54 | **2.54** | `mode ouse the ea d mimetthe dood the sa loog se, the therer d the gher` |
| 0.3 | 4.28 2.37 2.21 2.09 2.04 1.90 1.84 | **1.87** | `not anked not in a conems hamrelly ulle not of the thing to gay, then` |
| 3 | 4.28 **10.06** 9.83 10.56 11.44 5.78 3.40 | **3.45** | `lunt so lat uuplad inEevlen sut deil hlatle s ou tor ha gut o tuoren t` |

At 3 the error is more than twice random guessing from step 500 to 2,000; it
recovers only after step 2,400, when the schedule cuts every run's step to a
tenth, and still finishes worst.
