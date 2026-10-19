# "One penguin, two questions" master — measured findings (2026-10-19)

Data: **Palmer Penguins** (Gorman, Williams & Fraser 2014; palmerpenguins R
package; CC0) - 344 penguins, three species, three islands in Antarctica.
2 penguins with missing measurements dropped -> **342**. `experiment.py`
(plain numpy) writes `out/results.json`. Seeded split: **273 train / 69 test**.

## Question one: how heavy? (a number - regression)
- Body mass ranges 2,700-6,300 g. Guessing the average (**4,222 g**) is off by
  **650 g** on average (test set).
- Line from flipper length: **+49.3 g per mm** (intercept -5,683 g); off by
  **286 g** on average on unseen penguins (R² 0.79).
- Adding the two beak measurements: off by 292 g - no better.

## Question two: which species? (a name - classification)
Species counts: Adelie 151, Chinstrap 68, Gentoo 123.
- Always "Adelie": **44.9%** right.
- 5 nearest neighbours on flipper length alone: **62.3%** (43 / 69) - Adelie
  and Chinstrap flippers average 190.0 and 195.8 mm.
- 5 nearest neighbours on beak length + depth: **97.1% (67 / 69)**.
- Beak + flipper: 98.6% (68 / 69).

## Our penguin (a real test-set bird)
Adelie, female, Dream Island, 2008: flipper 190 mm, beak 38.1 x 18.6 mm,
**3,700 g**. The line predicts **3,681 g (19 g off)**; neighbours vote
**5 / 5 Adelie**.

## The catch
Two test penguins misnamed by the beak model; one is an **Adelie with a
45.8 mm beak** (depth 18.9) - all five neighbours Chinstrap.
