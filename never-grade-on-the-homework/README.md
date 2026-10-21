# Never grade on the homework: train vs test

A memoriser (1-nearest-neighbour: copy the weight of the most similar penguin
it has already seen) scores a perfect 0 g error on the penguins it learned
from, and misses new ones by 379 g, worse than a plain straight line. On the
real [Palmer Penguins](https://allisonhorst.github.io/palmerpenguins/) (Horst, Hill & Gorman; collected by Dr. Kristen Gorman and the Palmer Station LTER; CC0).

```bash
python experiment.py      # downloads the data on first run
```

See `FINDINGS.md`. Try: use 5 nearest neighbours instead of 1, and watch the
homework score stop being perfect while the test score improves.
