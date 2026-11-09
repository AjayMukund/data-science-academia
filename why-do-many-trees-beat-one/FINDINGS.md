# Random forest master (11-11): why do many trees beat one? — measured findings (2026-11-04)

**Data:** Titanic passengers, 891 (`titanic.csv`, the public Kaggle/Stanford file), 342
survived. Features: class, sex, age (blank → −1), siblings/spouses, parents/children, fare, port. Target: survived.
**Code:** plain numpy CART (Gini, binary splits) and a forest (each tree on a bootstrap resample, each question chosen
from 3 random features of the 7), `python experiment.py --data titanic.csv` → `out/results.json`. Seeded: re-runs match.

## 1. One tree, grown until every leaf is pure (split: 712 learn, 179 kept back)
- **162 leaves**. On its own 712: **98.5%** right. On the 179 it never saw: **81.0%** on this split.
- Across **20 different splits**: one tree averages **77.2%** (worst 71.5%, best 82.7%). The 81.0% was a lucky split.
- Shallower trees on this split: depth 1 77.1%, 3 78.2%, 6 80.4%, 8 79.9% test.

## 2. Shake the data: ten trees from ten resamples of the same 712
- A resample draws 712 passengers from the 712 with replacement (some twice, some not at all).
- Ten trees score **73.7% to 79.9%** on the same 179, and **disagree (3 to 7 votes of 10) on 43 of the 179**.

## 3. The forest
| trees | 1 | 3 | 5 | 10 | 25 | 50 | 100 | 200 |
|---|---|---|---|---|---|---|---|---|
| test accuracy | 74.9% | 79.3% | 76.5% | 81.0% | 81.0% | 81.6% | **82.1%** | 81.0% |
- 100 trees, across the 20 splits: **81.8%** average (worst 76.5%, best 84.9%); **better than the single tree on all 20**.
- 200 trees is no better than 100 (81.0% vs 82.1% on the main split): the gain levels off.
- First question of the 100 trees: **sex 43**, fare 22, class 21, siblings/spouses 7, parents/children 4, port 3, age 0.

## The running example — Anna Kristine Salkjelsvik (passenger 107, in the 179 kept back)
- 21, third class, ticket £7.65, travelling alone (no siblings/spouse, no parents/children aboard), boarded at Southampton.
- **She survived.** The one deep tree: **"didn't survive"**. The ten resampled trees: **5 yes, 5 no**.
  The forest of 100: **63 say survived**. Right.
- Why the deep tree said no: its 5 questions for her are female? → third class? → fare ≤ £23.35? → age > 19.5?
  → age ≤ 21.5? Her leaf holds **4 training passengers, aged 20, 20, 21, 21, all of whom died**, while of the **86**
  training passengers who share her first three answers (third-class women on cheaper tickets), **49 survived**.
  The tree decided her fate on four people: memorising, not learning.
- The tree's first questions (training passengers): sex (712: 268 survived) → women: class ≤ 2 (249, 184 survived) /
  men: fare ≤ £7.91 (463, 84 survived).

## Cautions for the script
- 1 in 5 is still wrong: who reached a lifeboat depended on things not in the file.
- One tree is a flowchart you can read; 100 trees are not (say so).
- This is one dataset of 891; "forests beat trees" is typical but not guaranteed.
- Anna is a real passenger who survived; the people the trees got wrong are real too — no drama, just the record.
