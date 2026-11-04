# Where do you draw the line? - Wisconsin biopsies

Run: `python experiment.py` (2026-11-04).
Data: UCI Breast Cancer Wisconsin (Diagnostic), Wolberg, Street & Mangasarian
(1995), **CC BY 4.0**: **569** digitised fine-needle biopsies, 30 measurements
of the cell nuclei each, labelled **212 malignant / 357 benign**.
(Replaced the banknote set: its documentation never says which class is
genuine, so the story could not say which side was fake.)

Score: logistic regression on three measurements (largest nucleus area,
roughest outline = worst concave points, worst texture), **out-of-fold**
(10-fold, seed 0): every biopsy scored by a model that never saw it.
AUC **0.992**. Scores pile at the ends: 172 of 212 cancers score 0.9+, 317 of
357 benign score under 0.1.

| Threshold | Cancers caught (of 212) | Missed | Healthy flagged (of 357) |
|---|---|---|---|
| 0.9 | 172 | 40 | **0** |
| 0.7 | 193 | 19 | 1 |
| **0.5** (default) | **202** | **10** | **7** |
| 0.3 | 207 | 5 | 18 |
| 0.2 | 209 | 3 | 25 |
| 0.1 | 210 | 2 | 40 |
| 0.01 | 210 | 2 | 125 |
| **0.0009** (lowest cancer score) | **212** | **0** | **231** (65%) |

- The last two cancers score almost zero: the model is sure they are benign.
  Catching them costs 191 more healthy people flagged than at 0.1.
- One measurement alone (largest nucleus area): AUC 0.968; at 0.5, 177 caught,
  14 flagged; catching all 212 flags 234.
- Honest line: a 1990s research dataset, not a hospital's test; in real
  clinics a flag means a second look, not a diagnosis.
