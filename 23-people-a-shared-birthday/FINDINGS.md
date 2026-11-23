# Birthday reel (11-29): 23 people, a shared birthday? — measured 2026-11-23

**Data:** ONS, "Births in England and Wales: 2025" bulletin, Figure 5 "How popular is your birthday?"
(Open Government Licence v3.0). The chart's own data files:
- `data.csv`: average daily live births per calendar date, England and Wales, **1995 to 2025** (366 dates)
- `data2025.csv`: live births per date in **2025** (365 dates)

`experiment.py` -> `out/results.json`. Feb 29 is weighted 1/4 (it exists one year in four).

## The answer (23 people)
| | P(two share a birthday) |
|---|---|
| every day equally likely (365) | 50.73% |
| real births, 1995-2025 average | **50.72%** |
| real births, 2025 | 51.21% |
| Monte Carlo, 200,000 rooms of 23, 1995-2025 weights | 50.73% |
- Spoken: "fifty-one percent" (50.7). 22 people: 47.6%, so **23 is the first size over a half**.
- 23 people = **253 pairs** (23 x 22 / 2). 49.3% chance that no pair matches.

## Births are not even, and it barely matters
- Most common date (1995-2025 average): **26 September, 1,927 births a day**; 25 Sep 1,926; 23 Sep 1,920.
- Least common: **26 December, 1,294**; 25 Dec 1,360; 1 Jan 1,526. Busiest/quietest = 1.49x (49% more).
- 2025 alone: top 28 May 1,844; bottom 26 Dec 1,052.
- Yet 23 people: 50.72% (real) vs 50.73% (even). The unevenness moves the answer by under 0.5 points.
  (Uneven days raise the match chance a little; the 1995-2025 figure also includes Feb 29 as a 366th day, which
  lowers it a hair; net about zero.)
- Why September peaks: not measured here. Don't claim a cause in the voice.

## Bigger rooms (1995-2025 weights)
- 30 people (a school class): **70.6%** -> "seventy-one percent"
- 50 people: 97.0%; **57 people: 99.0%** -> "ninety-nine percent"; 70: 99.9%

## Example room (seed 2026, first draw)
- 23 birthdays drawn from the real weights; the shared date is **24 October**. The scene can show this room.

## Use (stakes, general practice)
- The same calculation tells engineers how soon two different files or messages get the same short fingerprint
  (a hash collision; "birthday attack" in security). General practice; no specific numbers claimed.
