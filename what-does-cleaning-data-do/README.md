# What does cleaning data actually do?

The Titanic passenger list (891 passengers, 866 blank cells). A model has to
learn who survived from it. This script measures what each cleaning choice
does: to WHO is left in the data, and to how well a model does on passengers
it never saw. Plain numpy: logistic regression, scored with 10 x 5-fold
cross-validation.

```bash
python experiment.py      # downloads the data on first run
```

It prints the headline numbers and writes everything to `out/results.json`.
See `FINDINGS.md`.

Things to try:
- Delete every row with a blank and look at the class mix of what is left.
- Turn the cabin blank into a yes/no column, then add the title from the name.
- Remove the three £512 fares and see what you lose.
