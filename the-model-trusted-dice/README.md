# The model trusted dice: L1 vs L2

Hide three columns of pure random noise ("dice") among real penguin
measurements. A plain fit gives the first dice column four times the weight
of bill length. L2 (ridge) shrinks every weight but never to zero; L1 (lasso)
sets all three dice to exactly zero. On the real [Palmer Penguins](https://allisonhorst.github.io/palmerpenguins/) (Horst, Hill & Gorman; collected by Dr. Kristen Gorman and the Palmer Station LTER; CC0).

```bash
python experiment.py      # downloads the data on first run
```

See `FINDINGS.md`. Try: add ten dice columns instead of three, or make two
real columns nearly identical and compare which one L1 keeps.
