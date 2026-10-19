# Can a computer sort penguins it was never told about?

Unsupervised learning on the real [Palmer Penguins](https://allisonhorst.github.io/palmerpenguins/)
data (344 penguins, CC0). We throw the species away, ask k-means for three
groups using only beak length and depth, and only afterwards bring the names
back to grade it. Plain numpy, no scikit-learn.

```bash
python experiment.py      # downloads the data on first run
```

It prints the headline numbers and writes everything to `out/results.json`.
See `FINDINGS.md`.

Things to try:
- Change the random seed range in `best_of` and count how often a single
  run gets stuck.
- Delete the `std()` call in the "all four" run and watch weight take over.
- Ask for 2, 4 or 5 groups and look at what the extra group splits.
