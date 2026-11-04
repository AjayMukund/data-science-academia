# PCA - wheat grains

Run: `python experiment.py` (2026-11-04).
Data: UCI 236 "Seeds" (**CC BY 4.0**; Charytanowicz et al. 2010, Institute of
Agrophysics, Polish Academy of Sciences, Lublin): **210** wheat kernels, 70
each of **Kama, Rosa and Canadian**, 7 measurements from soft X-ray images:
area, perimeter, compactness, length, width, asymmetry, groove length.

Five measurements are one idea - size:
- area vs perimeter correlation **0.994**; vs width 0.971; vs length 0.950.
- Rosa is the big grain (area 18.3), Canadian the small one (11.9), Kama in
  between (14.3).

PCA (standardised columns, numpy eigh):

| Direction | Share of the spread | Made of |
|---|---|---|
| 1 | **71.9%** | area .44, perimeter .44, width .43, length .42, groove .39, compactness .28: **size** |
| 2 | **17.1%** | asymmetry .72, compactness -.53, groove .38: **shape** |
| 3 | 9.7% | |
| 4-7 | **1.3% together** | |

Directions 1+2 hold **89%**.

Naming the variety (leave-one-out, nearest variety centre):

| Using | Correct (of 210) |
|---|---|
| all 7 measurements | **194** (92.4%) |
| **directions 1 + 2** | **195** (92.9%) |
| best raw pair (area + groove length) | 192 (91.4%) |
| direction 1 alone | 184 (87.6%) |
| area alone | 183 (87.1%) |
| worst raw pair (compactness + asymmetry) | 148 (70.5%) |

- Honest turn: PCA never sees the variety names. It finds where the grains
  differ most, which here happens to be what separates the varieties; a good
  hand-picked pair gets 192. Direction 1 alone is barely better than area
  alone (184 vs 183): its first find is "size", which area already measures.
- Variety centres on directions 1, 2: Kama (-0.36, -1.02), Rosa (2.68, 0.41),
  Canadian (-2.32, 0.61). Points in `pc12.npy`.
