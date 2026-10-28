# When will Old Faithful erupt? - measured findings

**Data:** R's built-in `faithful` dataset (Härdle 1991; see Azzalini & Bowman 1990, *Applied Statistics* 39:357-365).
- 272 eruptions of Old Faithful, Yellowstone.
- Two columns: eruption length (minutes) and the wait until the next eruption (minutes).
- R's own documentation says the eruption times are heavily rounded.
- Fetched from the Rdatasets mirror: https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/datasets/faithful.csv

**Method:** `python experiment.py` writes `out/results.json`.
- Plain numpy, no fitting library.
- Learn on 200 eruptions, test on 72 (seed 0, `numpy.random.default_rng(0).permutation`).

**What the Park Service says** (checked 2026-10-28):
- https://www.nps.gov/teachers/classrooms/predict-old-faithful.htm gives the prediction table: eruption "< 3 minutes", next interval "68 minutes"; "> 3 minutes", "94 minutes".
- The page warns "any prediction may not prove to be exactly right".
- https://home.nps.gov/features/yell/ofvec/exhibits/eruption/prediction/predict8.htm says: "The time of the next eruption is predicted by using the duration of the current eruption."
- **Neither page gives an accuracy figure. Do not claim one.**
- Today's intervals are longer than in our record, so the NPS numbers (68 / 94) are not our numbers (55 / 80).

## The problem
| | Value |
|---|---|
| Waits | **43 to 96 min**, average **70.9** |
| Eruption lengths | 1.6 to 5.1 min |
| Guessing the average (70.8, learned) on the 72 test waits | misses by **11.55 min** on average; within ±10 min **37.5%** |

## A line is two numbers, scored by its squared misses (200 learning eruptions)
| Line | Slope (min of wait per min of eruption) | Sum of squared misses |
|---|---|---|
| Flat (the average) | 0 | **37,475** |
| Too gentle | 5 | **15,649** |
| Too steep | 20 | **30,370** |
| **Best** | **10.66**, starting at 33.7 | **7,072** |

The "bowl": the total squared miss against slope (intercept through the means) bottoms out between 10 and 11. See `bowl` in results.json.

## The line finds itself (gradient descent from flat, learning rate 0.02)
| Step | Slope | Sum of squared misses |
|---|---|---|
| 1 | 0.57 | 35,097 |
| 10 | 1.29 | 32,823 |
| 100 | 3.97 | 20,215 |
| 500 | 9.16 | 7,733 |
| **1,226** | within 0.1 of best | — |
| 2,000 | 10.66 | 7,072 |

"About twelve hundred steps."

## The test (72 eruptions never seen)
- **Line:** misses by **4.57 min** on average (said "four and a half").
- **Line:** within ±10 min on **90.3%** ("nine times in ten"); worst miss 13.9 min.
- **Line predictions:** 2 min → 55.0; 3 min → 65.7; 4 min → 76.4; 5 min → 87.0.
- **R²** on the learning eruptions: 0.81.

## The honest turn: two clumps
- 97 eruptions are under 3 minutes and 175 are 3 or over.
- Only **12 of 272** fall between 2.5 and 3.5 minutes.
- The mean wait is **54.5** after a short eruption and **80.0** after a long one.
- Inside each clump the slope is only 5.2 (short) and 6.3 (long), about half the overall 10.7. The line mostly sorts short from long.
- **Two guesses** (learned: **54.6** after a short eruption, **80.1** after a long one) on the same 72:
  - average miss **4.34 min**, which beats the line's 4.57;
  - within ±10 min **97.2%** (70 / 72), which beats the line's 90.3%.
- Two guesses with a 3-minute cut is the shape of the Park Service's own rule.

## What the script may say
- "about every seventy minutes"; "forty-three"; "ninety-six"
- "off by eleven and a half minutes"
- 272 eruptions
- "thirty-seven thousand / thirty thousand / fifteen thousand / seven thousand"
- "about twelve hundred steps"; "ten point seven minutes of wait for every minute of eruption"
- "seventy-two eruptions"; "four and a half"; "nine times in ten"
- "twelve between two and a half and three and a half"
- "fifty-five" / "eighty"; "four point three"; "ninety-seven times in a hundred"
- NPS: "under three minutes, about sixty-eight; over three, about ninety-four"
- "the geyser has slowed since our record"
