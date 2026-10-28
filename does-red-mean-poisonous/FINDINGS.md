# Does red mean poisonous? - measured findings

**Data:** UCI Mushroom (Schlimmer 1987), CC BY 4.0,
https://archive.ics.uci.edu/dataset/73/mushroom (file `agaricus-lepiota.data`).
- 8,124 mushrooms, each with 22 described features. 4,208 are edible and 3,916 poisonous.
- The rows are "hypothetical samples corresponding to 23 species of gilled mushrooms" from *The Audubon Society Field Guide to North American Mushrooms* (1981).
- The dataset merged the guide's "unknown edibility, not recommended" class into poisonous.
- The dataset notes quote the guide: "there is no simple rule for determining the edibility of a mushroom".

**Method:** `python experiment.py` writes `out/results.json`. The tree is plain-numpy CART:
- Every question has the form "is feature X value V?". There are 116 such questions.
- Each question is scored by the weighted Gini impurity of the two piles it makes.
- Gini impurity is described as **"mixed"**: pick two mushrooms from a pile at random (with replacement); how often is one edible and the other poisonous? For two classes that is 2p(1-p).
- The tree learns from 80% (seed 0, `numpy.random.default_rng(0).permutation`): 6,499 to learn from (3,340 edible, 3,159 poisonous) and 1,625 held back (868 edible, 757 poisonous).

## The cold open (whole book, 8,124)
| Question | Edible | Poisonous |
|---|---|---|
| Red cap | 624 | **876** |
| Not red | 3,584 | 3,040 |

## Choosing the first question (learning pile, 6,499)
Before any question, the pile is **49.96% mixed** ("half the time, a coin flip").

| Question | Rank of 116 | Mixed after | Yes pile (e / p) | No pile (e / p) |
|---|---|---|---|---|
| **Smells of nothing** (odor = none) | **1** | **18.75%** | **2,712 / 91** | 628 / 3,068 |
| Smells foul | 2 | 30.3% | **0 / 1,763** (pure) | 3,340 / 1,396 (a mess) |
| Bruises | 11 | 37.4% | 2,170 / 496 | 1,170 / 2,663 |
| Grows in woods | 68 | 49.2% | 1,488 / 1,023 | 1,852 / 2,136 |
| **Red cap** | **82** | **49.5%** | 495 / 703 | 2,845 / 2,456 |
| Flat cap | 113 | 49.9% | 1,252 / 1,261 | 2,088 / 1,898 |

- **"Smells foul" makes a perfectly pure pile (all 1,763 poisonous) and still loses**, because its other pile stays mixed. "Smells of nothing" leaves both piles tidier.
- Within the smells-of-nothing pile, mixed = 2(91/2803)(2712/2803) = 6.3%.

## The tree, deeper (learned on 6,499; tested on the 1,625 held back)
| Questions deep | End points | Learning right (of 6,499) | **Held back right (of 1,625)** | Poisonous called edible |
|---|---|---|---|---|
| 0 | 1 | 3,340 | 868 | 757 |
| 1 | 2 | 5,780 | 1,424 | 29 |
| 2 | 4 | 6,212 | 1,540 | 18 |
| **3** | **7** | 6,405 | **1,599** | **7** |
| 4 | 9 | 6,450 | 1,610 | 5 |
| 5 | 11 | 6,492 | 1,620 | 5 |
| **until pure** | **16** (deepest path 9 questions) | 6,499 | **1,625 (100%)** | **0** |

**The 3-deep tree** (learning counts, edible / poisonous):
- Smells of nothing, spore print green: **0 / 56**, poisonous.
- Smells of nothing, not green, yellow cap: 0 / 18, poisonous.
- Smells of nothing, not green, not yellow: **2,712 / 17**, edible. This pile hides 17 poisonous mushrooms.
- Smells of something, club stalk root, no bruises: 0 / 31.
- Smells of something, club stalk root, bruises: 407 / 0.
- Smells of something, not club, rooted: 144 / 0.
- Smells of something, not club, not rooted: 77 / 3,037, poisonous.

## The honest turn
- **None of the 1,625 held-back mushrooms is an exact copy** of one it learned from.
- **Every one of them** has a learning mushroom that matches it in **21 of the 22 details** (nearest differs by exactly 1, for all 1,625).
- All 8,124 rows are distinct, but they are variations on 23 species from one book.
- The perfect score therefore tests near-twins, not new mushrooms.

## What the script may say
- 624 red edible vs 876 red not (whole book).
- 116 questions.
- About half the time, a coin flip.
- Red cap: 49%. Woods: 49%. Foul: all 1,763 poisonous, the rest a mess, 30%.
- Smells of nothing: 2,700 edible / 91 poisonous; mixed 50 → 19.
- Spore print green: 56, all poisonous.
- 3 questions: 1,599 of 1,625, but 7 poisonous called edible.
- Until pure: 16 end points, all 1,625 right.
- Near-twin: same in 21 of 22 details. 23 species, one 1981 book. "No simple rule."
- **Never present the tree as a guide to eating mushrooms.**
