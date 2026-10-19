# Findings

Data: Palmer Penguins (Gorman, Williams & Fraser 2014; palmerpenguins R
package; CC0). 342 penguins with all four measurements: 151 Adelie,
68 Chinstrap, 123 Gentoo. The species column is removed before clustering
and used only to grade the result. k-means (Lloyd's algorithm), best of 50
random starts unless stated.

## The headline
k-means on beak length and depth (both scaled to the same spread), asked for
3 groups, recovers the species for **317 of 342 penguins (92.7%)**.

| group | Adelie | Chinstrap | Gentoo | centre (length, depth mm) |
|---|---|---|---|---|
| 1 | **147** | 5 | 1 | 38.8, 18.3 |
| 2 | 4 | **54** | 6 | 49.9, 18.7 |
| 3 | 0 | 9 | **116** | 47.1, 15.0 |

k-means returns group numbers, not names; the species names are matched
afterwards.

A near-tie: two answers differ by one borderline penguin (317 vs 318 right)
and their total squared distances differ by only 0.006. We report the
tighter one, 317.

## The run shown step by step
From three random penguins, the pins move four times. Penguins switching
groups after each move: 25, 4, 2, 0. After the fourth move nobody switches,
so the pins have settled.

## Bad starts
1,000 single runs from random starts: **990** find the species and **10**
get stuck in a worse answer (223 of 342 right), which cuts the Adelies in two
and lumps Chinstraps with Gentoos. Running several times and keeping the
tightest answer fixes it.

## Scale matters
All four measurements **unscaled**: body mass spreads by 801 g while beak
depth spreads by 2.0 mm, so distance is almost entirely weight. The three
groups become weight bands (2,700-3,975 g, 4,000-4,925 g, 4,950-6,300 g):
**198 of 342 (57.9%)**. The same four, **scaled**: **313 of 342 (91.5%)**,
Gentoo perfect. Four scaled measurements do slightly worse than the beak
alone (313 vs 317): more columns is not automatically better.

## How many groups? (beak, scaled; total squared distance)
k=1 684.0 · 2 340.2 · **3 186.9** · 4 147.8 · 5 114.0 · 6 94.4 · 7 82.6.
The drop from 2 to 3 groups is 153; from 3 to 4 only 39, so the curve bends
at 3. The curve always falls, so the bend is a judgement, not a proof.
