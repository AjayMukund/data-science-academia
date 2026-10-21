# Was it really "women and children first"?

Exploratory data analysis on the Titanic passenger list (891 passengers): one
number, then group by group, then one group split by another - the move that
finds the answer. Stdlib + numpy only.

```bash
python experiment.py      # downloads the data on first run
```

It prints survival by sex, age, class and class x group, and writes
everything to `out/results.json`. See `FINDINGS.md`.

Things to try: split by port of embarkation, or by family size, and check the
group sizes before you believe a rate.
