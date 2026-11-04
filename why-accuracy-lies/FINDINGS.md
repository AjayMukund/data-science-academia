# Why accuracy lies - California earthquakes

Run: `python experiment.py` (2026-11-04).
Data: USGS earthquake catalogue (public domain), every M4+ quake in the box
32.5-42 N, 124.5-114 W, 2000-01-01 to 2025-12-31, minus events USGS places in
Nevada, Mexico, Oregon, Arizona or Utah (named quakes like "Lake Almanor" and
"2004 Parkfield, California Earthquake" are kept - an include-filter on "CA"
lost them). 776 quakes M4+, 54 of M5+.

Question per day: will California have a magnitude 5+ quake tomorrow?

| | |
|---|---|
| days | **9,497** (26 years) |
| days with an M5+ | **48** - one day in **198** |

| Forecaster | Alarms | Caught (of 48) | Right on (of 9,497) | Accuracy |
|---|---|---|---|---|
| always "no" | 0 | **0** | 9,449 | **99.5%** |
| "yes" the day after any M4.5+ | **153** | **5** | 9,306 | **98.0%** |
| "yes" the day after any M4+ | 544 | 7 | 8,919 | 93.9% |
| "yes" for 7 days after any M4+ | 2,972 | 20 | 6,517 | 68.6% |
| always "yes" | 9,497 | 48 | 48 | **0.5%** |

- The aftershock rule's alarms are right **1 in 31** (5 / 153) vs **1 in 198**
  by chance: about 6x better, and it scores LOWER than "never".
- Caught (M4.5+ the day before): 2004-09-29 M5.0 (day after Parkfield M5.97);
  **2019-07-05 M5.4 (day after Ridgecrest M6.4) and 2019-07-06 M7.1 Ridgecrest
  (day after the 5.4)**; 2020-06-24 M5.8 Lone Pine; 2023-05-12 M5.2 (day after
  Lake Almanor M5.5).
- 2019-07-04 M6.4 itself: no M4.5+ on 07-03, so the rule missed it.
- Biggest days: 2019-07-06 7.1, 2003-12-22 6.5 (San Simeon), 2022-12-20 6.4,
  2019-07-04 6.4.
- Honest line: USGS says no scientist has ever predicted a major earthquake;
  the rule only knows shaking follows shaking.
