# Findings: two questions to name a penguin

342 penguins, 240 to learn from and 102 held back (seed 0).

| questions deep | end points | training | new penguins (of 102) |
|---|---|---|---|
| 1 | 2 | 80.8% | 76 (74.5%) |
| **2** | 4 | 95.4% | **98** |
| 3 | 7 | 96.2% | 100 |
| until perfect | 12 | **100%** | 99 |

The two-question tree: flipper ≤ 207.5 mm? No → bill depth ≤ 17.6 mm? yes →
Gentoo (88 of the 89 Gentoos). Yes → bill length ≤ 44.7 mm? yes → Adelie,
no → Chinstrap.

Five splits (seeds 0-4), new penguins right: two questions 98 / 96 / 96 / 97
/ 98; grown until perfect 99 / 98 / 96 / 98 / 98. More questions never did
worse here, and never more than 2 penguins better.
