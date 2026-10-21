# One wrong row: how one typo drags a regression line

Fit penguin weight against flipper length, then type one penguin's weight
with an extra zero (4,650 g as 46,500 g) and refit. One row out of 240 bends
the line from 49 to 35 grams per millimetre and adds 100 g of error on
penguins it never touched. On the real [Palmer Penguins](https://allisonhorst.github.io/palmerpenguins/) (Horst, Hill & Gorman; collected by Dr. Kristen Gorman and the Palmer Station LTER; CC0); the typo is ours.

```bash
python experiment.py      # downloads the data on first run
```

See `FINDINGS.md`. Try: fit the median of the residuals instead (a robust
line), or drop rows outside a plausible range before fitting.
