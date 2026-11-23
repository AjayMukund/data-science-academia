# Regression-to-the-mean reel (11-30): why does last season's top team usually slip? — measured 2026-11-23

**Data:** openfootball/football.json (CC0 1.0, public domain), English Premier League match results, every
complete season in the repository: **2010-11 to 2025-26 (16 seasons)**. Tables rebuilt from results (3 a win,
1 a draw); every champion's points match the published tables. Team names normalised across files.
`experiment.py` -> `out/results.json`.

## The example: Liverpool
- 2024-25 champions, **84 points**. 2025-26: **60 points, 5th**.

## Every champion, next season (15 champions, 2010-11 to 2024-25)
- **12 of 15 scored fewer points** the next season; 3 more (Man Utd 2011-12, City 2021-22, City 2023-24 seasons).
- Mean **89.7 -> 75.1**: an average drop of **14.6 points** ("fifteen").
- 4 of 15 retained the title (all Manchester City: 2018-19, and 2021-22 to 2023-24).
- Biggest falls: Leicester 81 -> 44 (2016-17), Chelsea 87 -> 50 (2015-16).

## By finishing position (teams in both seasons, n = 15 each)
- 1st: 89.7 -> 75.1 (-14.6); 2nd: 82.0 -> 76.2 (-5.8)
- 13th-17th (survived, below average): 41.1 -> 43.2 (+2.1): **they climb**
- All 255 team-seasons: r = 0.71; slope 0.78 (a team keeps about 78% of its distance from average).

## The luck test: a league of dice
- 20 identical teams, every match drawn at the real result rates (home 44.7%, draw 24.2%, away 31.0%),
  2,000 simulated seasons (seed 11).
- The dice champion averages **67.9 points** ("sixty-eight"); the same team next season **52.6** ("fifty-three"),
  i.e. the league average. Luck alone makes a champion, and it does not come back.
- Spread top-to-bottom: dice league 30.9 points; real league 66.2: **about twice as spread out**, so skill is real.

## Big jumps
- The 10 biggest one-season gains averaged **+27.0 points**; a year later they had kept **+11.2** ("eleven").
  (Chelsea 50 -> 93 -> 70; Leicester 41 -> 81 -> 44; Forest 36 -> 65 -> 44.)

## Not claimed
- Why Liverpool in particular dropped (injuries, transfers, form): not measured. The voice says some fall was
  expected, not that it was all luck.

## For the pictures (added 2026-11-23)
- `results.json` also holds every rebuilt table (`tables`) and one dice-league season pair (`dice_example`,
  `random.Random(2026)`, first pair whose clear champion scores exactly 68 and then 53: try 69). The reel draws that
  one season; the voice's 68 and 53 are the 2,000-season averages, which it matches.
