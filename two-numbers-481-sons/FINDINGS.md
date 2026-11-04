# The bell curve - Galton's families

Run: `python experiment.py` (2026-11-04).
Data: Francis Galton's 1886 family notebook (public domain), as
HistData::GaltonFamilies via Rdatasets: 205 families, 934 grown children
(481 sons, 453 daughters), heights in inches, converted to cm (x 2.54).
70% of sons are recorded to the whole inch - bin by the inch (2.54 cm), or
the histogram combs.

| | Sons | Daughters | Everyone |
|---|---|---|---|
| n | **481** | 453 | 934 |
| average | **175.9 cm** | 162.8 cm | 169.5 cm |
| spread (SD) | **6.7 cm** | 6.0 cm | 9.1 cm |
| within 1 spread | **66%** (bell 68%) | 70% | 62% |
| within 2 spreads | **97%** (bell 95%) | 96% | 98% |

The two-number prediction (sons, from average + spread only):
- over six foot (72 in = 182.9 cm): bell says **14.6% = 70 sons**; Galton
  counted **83** (17.3%)
- under five foot: bell 0.1 people; actual 0

Many small causes: predict each son from his parents' average height (slope
0.71); what is left - everything the parents don't explain - is itself a
bell: spread **5.8 cm**, 68% within one spread, 96% within two.

Where it breaks (everyone together, sons + daughters as one group):
- under five foot: bell expects **28 people**; there were **6**
- over six foot: bell 7.1% (66), actual 83
Two bells, one inside the other, are not one bell.
