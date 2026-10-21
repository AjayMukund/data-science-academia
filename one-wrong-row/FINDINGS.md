# Findings: one wrong row

240 training penguins, 102 untouched test penguins (seed 0). The typo: an
Adelie with a 184 mm flipper, 4,650 g typed as 46,500 g. The real dataset
contains no such error.

- Slope: **49 -> 35 g per mm** of flipper.
- Error on the 102 untouched penguins: **310 g -> 410 g**.
- Least squares squares each miss, so the typo carries **97%** of the bent
  line's total squared error; the other 239 penguins share 3%.
- Training mean 4,248 -> 4,423 g (+175 g); median 4,150 -> 4,150 g.
