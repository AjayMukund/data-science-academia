# "Who's the best batter in IPL history?" reel (12-02): findings (2026-11-23)

**Data:** Cricsheet IPL ball-by-ball JSON (cricsheet.org), Open Data Commons Attribution License
(credit "Cricsheet"), every match 18 April 2008 to 31 May 2026; the files are shared with
`research_data/ipl_toss/download/ipl`. `python experiment.py` -> `results.json` (plain Python).
Balls faced exclude wides; super overs excluded; strike rate = 100 x runs / balls.

| Claim | Number |
|---|---|
| Batters who faced a ball | **738**; league strike rate **133.4** |
| No minimum, #1 | **Kwena Maphaka: 8 off 2, strike rate 400** (two fours, Rajasthan Royals, 18 May 2025) |
| Next | Auqib Nabi 14 off 4 (350), L Wood 9 off 3 (300), B Stanlake 5 off 2 (250) |
| Strike rate zero | **25 batters** (e.g. 0 off 1) |
| Under 10 balls (125 batters) | strike rates **0 to 400** |
| 1,000+ balls (88 batters) | strike rates **107 to 174** |
| Andre Russell | 2,651 off 1,522, **174.2**; best with 1,000+ balls |
| Fix: add 100 balls at 133.4 to everyone | Maphaka 400 -> **138.7, 1st -> 119th**; Russell 174.2 -> 171.7 (5th) |
| Vaibhav Suryavanshi | 1,028 off 449, **229.0** -> **211.6, still 1st** |

- Russell "faced fifteen hundred balls" (1,522). Median balls of the top 20 by raw strike rate: 46; bottom 20: 2.
- The 100-ball padding is a shrinkage (Bayesian) average; rating sites pull few-review items toward the average the same way.
- **Luck test** (random.Random(2026), balls drawn from all 285,686 real non-wide deliveries): two boundaries off two balls **2.96%, about 1 in 34**; 1,028+ runs off 449 balls **0 of 100,000** (best strike rate 169.7).
- Do not state anyone's age; the data has no ages.
