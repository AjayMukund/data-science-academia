# Two numbers, 481 sons: the bell curve

Francis Galton's 481 sons average 176 cm with a spread of 6.7 cm; from those two numbers the bell curve expects 70 over six foot, and Galton counted 83. Mix in the daughters and the single bell expects 28 people under five foot; there were 6. Galton's 1886 notebook via the [HistData](https://cran.r-project.org/package=HistData) package (`GaltonFamilies`).

```bash
python experiment.py      # downloads the data on first run
```

See `FINDINGS.md`. Try: fit one bell to the daughters alone and check its tails the same way.
