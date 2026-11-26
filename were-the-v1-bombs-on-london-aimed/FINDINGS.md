# Were the V-1 flying bombs aimed at particular streets? (clustering illusion, Poisson)

Reel 12-04. Run: `python experiment.py` (plain Python, seed 1946) -> results.json.

## Source
R. D. Clarke (1946), "An application of the Poisson distribution", Journal of the Institute of Actuaries 72, 481.
Only his summary table survives (no map). Figures are facts from a published paper; the counts are reproduced in
Feller's textbook and in the VGAM R package (dataset `V1`).

## The data (Clarke's table)
- 144 km² of south London, cut into **576 squares** of 1/4 km² each; **537 hits**. Mean 0.9323 a square.

| hits in a square | 0 | 1 | 2 | 3 | 4 | 5+ |
|---|---|---|---|---|---|---|
| squares observed | **229** | 211 | 93 | 35 | 7 | 1 |
| pure chance predicts | 226.74 | 211.39 | 98.54 | 30.62 | 7.14 | 1.57 |

- Chi-square **1.17**, 4 degrees of freedom, **p = 0.88**: what landed is what chance predicts (reproduced here exactly).
- 229 of 576 squares (40%) never took a hit; 43 squares took 3 or more.

## Random scatters (ours, 10,000 runs of 537 random hits on 576 squares)
- **Every one of 10,000** random scatters has at least one square with 4+ hits; the busiest square is typically 5.
- Empty squares range 200-260 per scatter.
- Example scatter (run 1): counts 233 / 200 / 106 / 26 / 8 / 3; its busiest 2x2 block holds **10 hits** against an
  even share of 3.7. Clusters and empty patches with no aiming at all.

## Honest caveats (Shaw & Shaw, Significance 16(5), 2019, "The flying bomb and the actuary"; supplementary analysis online)
- Clarke chose an area "where the mean number seemed roughly constant". Across the whole County of London the V-1s
  were NOT uniform: they were skewed to the south (p = 4.8e-6 county-wide). Clarke's result holds within that southern region.
- British intelligence fed misinformation in late 1944 to increase the shortfall onto the suburbs (per the article).
- So: aimed at London, yes - but within a few kilometres they could not pick a street. Say "inside this patch of south
  London, the hits fell exactly like chance"; never "the bombs fell at random over London".
- Totals: more than 2,300 V-1s fell on London (~5,500 killed); 10,486 launched, 2,420 (23%) fell inside the London
  Civil Defence Region (Evans & Delaney 2018, cited in the supplement). Not needed for the script; no casualty imagery.

## The everyday turn
Spotify engineering blog, 28 Feb 2014, "How to shuffle songs?": users complained that true random shuffle
(Fisher-Yates) played songs by the same artist back to back, so Spotify changed it to spread each artist out
- making shuffle LESS random so it would FEEL random. https://engineering.atspotify.com/2014/02/how-to-shuffle-songs
