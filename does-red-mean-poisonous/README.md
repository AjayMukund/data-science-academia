# Does red mean poisonous? How a decision tree chooses its questions

A plain-NumPy decision tree (CART, Gini impurity) on the UCI Mushroom data: 8,124 mushrooms from
a 1981 field guide, 22 details each, edible or poisonous. The tree tries all 116 yes/no
questions and keeps the one that leaves its two piles least mixed. "Red cap" barely helps
(49.5% mixed); "smells of nothing" wins (19%), beating "smells foul", which makes a perfectly
pure pile but leaves the other a mess. Grown until pure, the tree gets all 1,625 held-out
mushrooms right, but every one of them matches a mushroom it learned from in 21 of 22 details.

```bash
pip install numpy
python experiment.py            # downloads the data on first run, writes out/results.json
```

See `FINDINGS.md`. Things to try: split so that whole groups of near-twins stay together and
watch the score fall; replace Gini with "share the majority gets right" and see which question
wins; cap the depth and count how many poisonous mushrooms are called edible.

Never eat a wild mushroom because a model said so. The source field guide itself says there
is no simple rule for telling edible from poisonous.
