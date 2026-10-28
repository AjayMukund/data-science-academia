# How do you know your model works? Findings

**Data:** Palmer Penguins (Gorman, Williams & Fraser 2014; palmerpenguins R package; CC0),
344 penguins, 2 with missing measurements dropped -> **342**.
**Model on trial:** the species model from our supervised-learning video - 5 nearest neighbours on beak length +
beak depth (standardised on the training penguins), 80/20 split: 273 train / 69 test.
Run: `python experiment.py --out out` (numpy only).

## 1. The published number
- Seed-7 split (as published): **97.1% - 67 of 69** test penguins right. Reproduced.

## 2. One test score is one roll of the dice (10,000 other random 80/20 splits)
- Mean **96.6%**. Worst **88.4% (61/69)**, best **100% (69/69)**.
- 95% of splits score **92.8% to 100%**. **6.8%** of splits are perfect.
- Distribution of right answers (of 69): 61: 7, 62: 31, 63: 143, 64: 464, 65: 1,192, 66: 2,376,
  67: 2,971, 68: 2,139, 69: 677.

## 3. 5-fold cross-validation
- Every penguin is a test penguin exactly once (folds of 69, 69, 68, 68, 68).
- One run: fold scores **97.1, 98.6, 95.6, 95.6, 97.1%**; overall **331 / 342 = 96.8%**.
- Re-shuffled 1,000 times: the CV score stays within **95.0-97.7%** (95%: 95.9-97.4%) - a
  1.5-point band, against 7 points for a single split.

## 4. The trap: choosing the setting by the test score
- Try k = 1..25 on one split, report the best: average reported **97.6%**; **16.3%** of the time
  it reports a perfect 100%.
- What the chosen k really scores (cross-validated on fresh folds): **96.1%**.
- The habit inflates the score by **1.5 points** on average.
