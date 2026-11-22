# Full-moon master (11-28): are more babies born on a full moon? — measured (2026-11-22)

**Data:** every US birth by date, 2000-2014: **5,479 days, 62.2 million births** (62,187,024) (Social Security Administration via
FiveThirtyEight, CC BY 4.0, `data/births.csv`). **Full moons:** computed (Meeus, *Astronomical Algorithms* ch. 49);
185 in the period; checked against two published times (2000-01-21 04:41 UTC, 2014-12-06 12:27 UTC: both match to the
minute). A birth's date is its US date, so a full moon's day is its US Eastern date.
**Code:** `experiment.py` → `out/results.json`. Plain numpy, seeded.

## Method (what "expected" means)
- Each day against the **median of the same weekday 1, 2 and 3 weeks either side** (removes weekends, seasons, trend).
- US federal holidays and the days around Christmas/New Year are left out of every test.
- **What checking changed:** a first version compared each day with its month's same-weekday average and found a
  "+1.4% full-moon effect, p < 0.001". It was Thanksgiving: a holiday drags its week's average down, so an ordinary
  Thursday nearby looks 17% busy. With the holiday-proof baseline the effect is gone. (A good aside for the honest turn.)
- The test: move every full moon to a random day 3-26 days away (each date on its own), 10,000 times, and ask how
  often the random days look as busy as the real full moons. That fraction is the p-value.

## The answer
- Full-moon days (175 after holidays are left out): **+0.16%** above expected, about **23 extra babies a day on
  ~11,460**. Random days do as well or better **51% of the time (p = 0.51)**. **No effect.**
- 51% of full-moon days are above expected, 49% below.
- Random sets of days swing between −0.29% and +0.54% (middle 95%).
- The "supermoon" of 19 March 2011 (a Saturday; the closest full moon since 1993, per NASA): **7,786 born, 7,868
  expected (−1.0%)**.

## What does move births (scheduled inductions and C-sections, i.e. people's calendars)
- Weekday averages: **Tuesday 13,122**, Wednesday 12,911, Thursday 12,846, Friday 12,596, Monday 11,898,
  Saturday 8,563, **Sunday 7,518** (−43% vs Tuesday).
- **Friday the 13th: −6.2%** (25 of them; **all 25 below expected**; p < 0.0001). e.g. 13 April 2001: 10,881 vs 12,316
  (−11.6%).
- Christmas Day −40%, 4 July −25%, Halloween −12%, Valentine's Day +4.7%.

## Fishing (the twist)
- Test each of the 30 days of the moon's cycle on its own: **1 of 30** comes out "significant" (p < 0.05): 10 days
  before the full moon, +0.45%, p = 0.038. By chance alone we'd expect about 1.5 of 30.
- So: a "lunar birth peak 10 days before the full moon" headline is exactly what fishing produces.

## Cautions
- Births by date only (no hour), so a full moon at 02:00 counts for its whole day; a window of ±1 day changes nothing
  material (check before quoting).
- US only; the effect being tested is natural labour, and many births are scheduled.
