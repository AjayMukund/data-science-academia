# Findings: never grade on the homework

342 penguins with a bill length and weight; 240 to learn from, 102 held back
(numpy `default_rng(seed).permutation`, seeds 0-4). Inputs: bill length, bill
depth, flipper length (scaled for the memoriser). Error = mean absolute error.

| split | memoriser, homework | memoriser, test | line, homework | line, test |
|---|---|---|---|---|
| 0 | **0 g** | **379 g** | 314 g | **306 g** |
| 1 | 0 g | 356 g | 308 g | 314 g |
| 2 | 0 g | 319 g | 317 g | 293 g |
| 3 | 0 g | 333 g | 307 g | 319 g |
| 4 | 0 g | 376 g | 301 g | 335 g |

- On its own training data the most similar penguin to each one is itself,
  so the memoriser's homework error is exactly zero, every split.
- The line beats the memoriser on new penguins on all five splits.
