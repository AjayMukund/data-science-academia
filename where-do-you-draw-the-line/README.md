# Where do you draw the line? Classification thresholds

569 breast biopsies, 212 cancers, each scored out-of-fold by a small logistic model. At 0.5: 202 caught, 10 missed, 7 healthy flagged. Catching all 212 flags 231 of 357 healthy, because the last two cancers score almost zero. On the [Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) data, UCI, CC BY 4.0. A 1990s research dataset, not a clinical test.

```bash
python experiment.py      # downloads the data on first run
```

See `FINDINGS.md`. Try: use a different set of three measurements and compare where the hard cases land.
