# Could the numbers have warned them? - measured findings

**Data:** UCI "Challenger USA Space Shuttle O-Ring" (Draper 1993), CC BY 4.0, https://archive.ics.uci.edu/dataset/92/challenger+usa+space+shuttle+o+ring
- File `o-ring-erosion-or-blowby.data`: the 23 shuttle flights before Challenger, each with launch temperature (°F) and how many of its 6 field-joint O-rings showed thermal distress (erosion or blow-by).
- The UCI page says the Challenger launch "was forecast for 31 degrees F", that no previous launch was below 53 °F, and that 31 °F needs "heavy extrapolation".
- A flight counts as **damaged** if any O-ring showed distress.

**Method:** `python experiment.py` writes `out/results.json`.
- Plain numpy.
- Logistic regression by gradient descent, with the bootstrap refits by Newton's method.

**History, checked against the Rogers Commission report** (https://www.nasa.gov/history/rogersrep/v1ch6.htm, v1p146.htm; read 2026-10-28):
- Challenger launched 28 January 1986.
- The report's Figure 2 lists the joint temperature as 31 °F; its coldest earlier flight, STS 51-C, was 53 °F.
- Figure 6 plots "flights with incidents of O-ring thermal distress"; Figure 7 plots "flights with and without incidents".
- Finding: "A careful analysis of the flight history of O-ring performance would have revealed the correlation of O-ring damage and low temperature."
- The report also says: "the probability of O-ring distress is increased to almost a certainty if the temperature of the joint is less than 65."
- The vehicle broke apart 73 seconds after launch.
- **Do not claim** who drew which chart on the eve of launch, or why the zero-damage flights were left off: not verified from the primary source.

## The record
| | Value |
|---|---|
| Flights | 23, of which **7 damaged** |
| Temperatures flown | 53-81 °F |
| Damaged flights' temperatures | 53, 57, 58, 63, 70, 70, 75 |
| **Below 65 °F** | **4 of 4 damaged** |
| **65 °F and above** | **3 of 19 damaged** |
| By band | 50s 3/3, 60s 1/7, 70s 3/12, 80s 0/1 |
| Challenger forecast | **31 °F**, **22 degrees** colder than any flight |

## Damaged flights alone (Figure 6's view)
- 7 flights between 53 and 75 °F, with 2, 1, 1, 1, 1, 1 and 2 rings damaged.
- Correlation of temperature with rings damaged: **0.02**, i.e. no pattern.

## A straight line through damaged (1) / clean (0)
- **175%** at 31 °F (1.746).
- 92% at 53 °F.
- **−12%** at 81 °F.
- The line leaves 0-1 above 77.7 °F and below 51.0 °F.

## Logistic regression (all 23 flights)
- Raw coefficients: intercept **15.043**, slope **−0.2322 per °F**. These match the published fit of this data.
- Each degree colder multiplies the odds of damage by 1.26.

| Temperature | P(damage) |
|---|---|
| 81 °F | 2% |
| 75 °F | 9% |
| **70 °F** | **23%** |
| **65 °F** | **49%** (50/50 at **64.8 °F**) |
| 60 °F | 75% |
| **53 °F** | **94%** |
| 40 °F | 99.7% |
| **31 °F** | **99.96%** |

## How sure: 10,000 bootstrap resamples of the 23 flights
- 9,998 resamples fitted; 2 had no damaged flight at all and were skipped.
- P(31 °F) median 99.98%; 5th percentile 96.8%.
- **97.6%** of resamples give over 90%; 99.1% give over 50%.
- Every resample still extrapolates from 53 °F down to 31 °F.

## What the script may say
- "January 1986"; "thirty-one degrees"; "fifty-three"; "twenty-two degrees colder"
- "six rubber rings"; "seven of the twenty-three"
- "no pattern"
- "sixteen flights with no damage"; "four out of four"; "three out of nineteen"
- "a hundred and seventy-five percent"; "minus twelve"
- "fifty percent at sixty-five"; "seventy degrees: twenty-three percent"; "fifty-three: ninety-four"
- "ninety-nine point nine six"
- "ten thousand"; "ninety-seven times in a hundred, over ninety percent"
- "seventy-three seconds"
- The Commission's finding, quoted.
