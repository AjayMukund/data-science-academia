# Margin-of-error master (11-21): could 1,000 trees have told New York what 683,788 did? — measured (2026-11-11)

**Data:** NYC 2015 Street Tree Census (TreesCount! 2015-16), NYC Parks via NYC Open Data, dataset `uvpi-gqnh`
(free to use under the NYC Open Data terms). 683,788 records: every street tree, dead tree and stump the census
logged, with health, species, neighbourhood and position. Counted by NYC Parks staff and **more than 2,200
volunteers**, from May 2015 into 2016 (NYC Parks). (Parks' own headline counts 666,134 street trees mapped; we use the
published data file and say "683,788 in the city's data, stumps included" — never "666,134".)
**Code:** `fetch.py` → `data/trees.csv` (not committed); `experiment.py` → `out/results.json`; `detail.py` →
`out/map_all.png`, `out/sample.json`. Plain numpy, seeded.

## The truth (the whole census)
- **528,850 of 683,788 in good health: 77.3%** (77.34%). Fair 96,504; poor 26,818; dead 13,961; stumps 17,654.

## Random samples, 10,000 of each size (the middle 95% of answers)
| Trees asked | 95% of samples land in | ± | Worst of 10,000 |
|---|---|---|---|
| 10 | 50% – 100% | — | 57 points off |
| 100 | 69% – 85% | ±8 | 18 points |
| **1,000** | **74.7% – 79.9%** | **±2.6** | 5.6 points |
| 4,000 | 76.1% – 78.6% | ±1.3 | 2.7 |
| 10,000 | 76.5% – 78.2% | ±0.8 | 1.7 |
- Four times the trees halves the error (1,000 → 4,000: ±2.6 → ±1.3); 100 → 10,000 (100×): ±8 → ±0.8 (10×).
- 1,000 random trees land within 2 points of the truth **87%** of the time.
- Our worked random 1,000 (seed 1): **78.2%**.

## The size of the place doesn't matter
1,000 random trees, 95% range, per borough (10,000 draws each, on their own seeded stream so other runs can't move them):
| Borough | Trees | Truth | 95% range | ± |
|---|---|---|---|---|
| Manhattan | 65,423 | 72.4% | 69.6 – 75.2 | 2.8 |
| Bronx | 85,203 | 78.2% | 75.6 – 80.7 | 2.55 |
| Staten Island | 105,318 | 78.5% | 75.9 – 81.0 | 2.55 |
| Brooklyn | 177,293 | 78.0% | 75.4 – 80.5 | 2.55 |
| Queens | 250,551 | 77.4% | 74.8 – 80.0 | 2.6 |
| **All NYC** | 683,788 | 77.3% | 74.7 – 79.9 | 2.6 |
- Theory agrees: 1.96 × sqrt(p(1−p)/1000) is 2.75 for Manhattan (p = 0.724), 2.60 for the city (p = 0.773).
- ±2.8 for 65,000 trees, ±2.6 for 683,788 (the gap comes from Manhattan's lower health share, not its size): a spoonful tastes the pot, whatever its size.

## Not random: 1,000 trees from one place
- 183 neighbourhoods have 1,000+ trees. Lowest: **Breezy Point-Belle Harbor-Rockaway Park-Broad Channel** (Queens, on
  the sea), 4,351 trees, **52.2%** good (our 1,000: 52.5%). Highest: **Westerleigh** (Staten Island), 4,954 trees,
  **88.1%** (our 1,000: 88.6%).
- Same 1,000 trees each: 52% / 88% / random 78% vs truth 77%. **Who you ask beats how many.**
- Why the Rockaways are low: the census doesn't say. (Hurricane Sandy's salt water in 2012 is a plausible guess; say
  it as a guess or not at all.)
- Curiosity, not for the video: the first 1,000 records by tree ID read 60.4%.

## Cautions
- "Good" is the volunteer's judgement on the day (Good / Fair / Poor); we treat the census as the truth.
- The ± is the middle 95% of 10,000 simulated samples, not a formula.
