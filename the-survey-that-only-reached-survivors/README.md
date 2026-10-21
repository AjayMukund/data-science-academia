# The survey that only reached survivors: sampling bias

Survey the Titanic after the sinking and you can only reach the 342
survivors: the ship looks 68% women (it was 35%). A random 100 passengers
land between 26% and 44%, and not one of 1,000 such samples reaches 68%.
On the Titanic passenger list (Kaggle training set, 891 passengers; mirror: https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv; a public historical record).

```bash
python experiment.py      # downloads the data on first run
```

See `FINDINGS.md`. Try: survey only first-class passengers, or only people
who boarded at one port, and see which numbers tilt.
