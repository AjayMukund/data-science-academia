# Why do many trees beat one? A random forest, measured on the Titanic

A decision tree and a random forest, written in plain NumPy (no scikit-learn), on the Titanic passenger list (891
passengers, 7 facts each). One tree grown until every leaf is pure has 162 leaves: 98.5% right on its 712 training
passengers, 77.2% on new ones (averaged over 20 splits). Ten trees grown on resamples of the same passengers disagree
on 43 of 179 new ones. A forest of 100 (each tree on its own resample, choosing from 3 random facts per question) scores
81.8% and beats the single tree in all 20 splits. The running example: Anna Salkjelsvik (passenger 107), whom the single
tree calls "died", ten trees split 5-5, and the forest says survived, 63 votes to 37. She survived.

```bash
pip install numpy
# the Titanic passenger list (Kaggle training set, 891 rows); the copy we used:
curl -L -o titanic.csv https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv
python experiment.py --data titanic.csv   # about a minute; results in out/results.json
```

Seeded, so a re-run gives the same numbers. See `FINDINGS.md`.

Things to try: more facts per question (`mtry`), shallower trees in the forest, 500 trees, or a different split seed.
One dataset of 891 people: forests usually beat single trees, but it isn't guaranteed.
