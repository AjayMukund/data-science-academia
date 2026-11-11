# Handwriting master (11-19): how does a computer read your handwriting? — measured (2026-11-10)

**Data:** MNIST (LeCun, Cortes, Burges; **CC BY-SA 3.0**): 60,000 training and 10,000 test handwritten digits, 28 × 28
grey pixels (784 numbers each), written by US Census Bureau staff and high-school students. Mirror:
storage.googleapis.com/cvdf-datasets/mnist.
**Code:** plain numpy. `experiment.py` → `out/results.json`, `out/net.npz`, `out/averages.npy`; `pair.py` →
`out/pair.json`. Seeded: re-runs match.

## The simple rule: compare with the average digit
- The average picture of each digit, from the 60,000. Read a test digit as the closest average (sum of squared pixel
  differences): **82.0%** of the 10,000 right (1,800 wrong).

## The network: 784 → 64 → 10
- 784 pixels in; 64 hidden neurons (each a weighted sum of all 784 pixels, negatives cut to zero); 10 outputs turned
  into probabilities. **50,890** numbers to learn.
- Untrained (random weights): **9.2%** — no better than guessing.
- Training: show it the 60,000, 64 at a time; after each batch, nudge every number to make the right digit a little
  more likely. After pass 1: **92.7%**; pass 5: 96.5%; pass 10: **97.3%** (268 of 10,000 wrong).
  **About 3 seconds** for all ten passes on our laptop.
- What the hidden neurons learned (their 784 weights as pictures): mostly speckle with faint strokes and blobs;
  nobody chose them.
- Of the 1,800 digits the rule gets wrong, the network gets **1,593** right.
- Most common confusions: **4 read as 9 (12)**, 2 as 8 (11), 7 as 2 (9).

## The running example — test digit 65: an open-topped 4
- The rule: distance to the average 9 **31.5**, to the average 4 **34.2** → reads **9**. Wrong.
- The network: **4, 98.6% sure** (9: 0.9%). 38 of its 64 neurons fire for it.
- **The minimal pair:** close the top with a short stroke (**10 pixels** changed): the network drops to **46% "4"**,
  31% "9", 20% "3". Still says 4, but no longer sure. (The rule now reads it as 5.) Bigger strokes push it towards 3,
  not cleanly to 9 — don't claim a flip.

## The honest turn — test digit 9587: a 9 it reads as a 4
- A 9 with a small closed loop and a stray dot top-left. The network: **4, 99.8% sure** (99.85%).
- Rub out the stray dot (9 pixels): **still 4, 99.5%**. The dot isn't why. It can't tell us why.

## Cautions
- 97.3% on this test set; real-world handwriting (other people, pens, scans) differs.
- Postal and bank readers use far bigger systems; this is a small teaching network.
- "Sure" = the probability the network outputs, not a guarantee (9587 shows it).
