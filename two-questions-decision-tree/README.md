# Two questions to name a penguin: decision trees

A decision tree built from scratch in numpy (Gini impurity, every feature and
every cut-off tried) names a penguin's species with yes/no questions. One
question: 74.5% of 102 new penguins. Two: 98 of 102. Grown until every
training penguin is right (12 end points): 99 of 102. On the real
[Palmer Penguins](https://allisonhorst.github.io/palmerpenguins/) data
(Horst, Hill & Gorman; collected by Dr. Kristen Gorman and the Palmer Station
LTER; CC0).

```bash
python experiment.py      # downloads the data on first run, prints the 3-question tree
```

See `FINDINGS.md`. Try: other seeds for the split, or `min samples per leaf`
as a stopping rule instead of depth.
