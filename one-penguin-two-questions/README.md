# One penguin, two questions: regression vs classification

On the real [Palmer Penguins](https://allisonhorst.github.io/palmerpenguins/)
data (344 penguins, CC0): predict a penguin's **weight** (a number:
regression, a fitted line) and its **species** (a name: classification,
5 nearest neighbours), each scored on penguins the model never saw.

```bash
python experiment.py      # downloads the data on first run
```

See `FINDINGS.md`. Try: predict species from flipper length alone, then from
the beak - and see why the measurement has to fit the question.
