# How do you know your model works? Cross-validation

A simple species model (5 nearest neighbours on beak length and depth, Palmer Penguins, 342
birds) scored 97.1% on one random test set of 69. On 10,000 other random test sets the same
model scores anywhere from 88% to 100%. 5-fold cross-validation gives 96.8% (331 / 342) and
stays within 95.0-97.7% when reshuffled. Choosing the number of neighbours by the test score
reports 97.6%; that choice really scores 96.1%.

```bash
pip install numpy
python experiment.py            # downloads penguins.csv on first run, writes out/results.json
```

See `FINDINGS.md`. Things to try: 10 folds instead of 5; add flipper length; do the choice of
k inside the training folds (nested cross-validation) and compare.
