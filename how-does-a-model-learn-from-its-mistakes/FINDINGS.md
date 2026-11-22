# Boosting master (11-30): how does a model learn from its own mistakes? — measured (2026-11-22)

**Data:** California Housing (Pace & Barry 1997, from the **1990 US census**; StatLib, public; scikit-learn's mirror):
**20,640 districts**, each with its median house value and income, house age, rooms, bedrooms, people, homes and
location. **1,000+ districts are capped at $500,001** in the data. Split: 16,512 to learn from, **4,128 it never sees**.
**Code:** `experiment.py` → `out/results.json`. Plain numpy (trees on 64-bin features), seeded. Errors are mean absolute
error on the 4,128 unseen districts ("off by, on average").

## The ladder
| Model | Off by, on average |
|---|---|
| Guess the average ($206,663) for every district | **$92,511** |
| One small tree (3 questions), full size | $58,600 |
| Boosting, 1 tree (added at a tenth) | $87,732 |
| Boosting, 10 trees | $62,572 |
| Boosting, 100 trees | **$36,130** |
| Boosting, 300 trees | $32,447 |
| Boosting, 1,000 trees | **$29,767** (on the districts it learned from: $27,601) |
| One deep tree (12 questions deep) | $40,298 (learned-from: $29,430) |
| Random forest, 100 deep trees averaged | $32,707 |
- 1,000 boosted trees: **6 seconds** on our laptop.
- How boosting works here: tree 1 learns the prices; every later tree learns **the mistakes so far** (true minus
  current guess), and is added at a tenth of its size (learning rate 0.1).
- Tree 1's first question: **income ≤ $50,900?** (income is the 1990 median household income); then income ≤ $31,100,
  rooms per home, people per home.

## The running example: a district by the beach in Santa Monica
- Test district nearest Santa Monica pier (34.00, −118.48): **true value $450,000**; income $39,722; houses 25 years old;
  1,000 homes. The model never saw it.
- Its guess, tree by tree: start (the average) **$206,663** → 1 tree $213,635 → 2 $220,319 → 3 $226,529 → 10
  **$252,042** → 30 $296,904 → 100 **$362,324** → 300 $402,439 → 1,000 **$432,142** ($17,858 short).
- Over all unseen districts: boosting moves **83%** closer than the average guess; **17% end further away**. Median
  final miss: $19,822.
- One of the 17%: the district nearest Pasadena city hall, true $187,500: average $206,663 → 1,000 trees $246,204
  (worse).

## The honest turn: too eager
- Same trees added at full size (learning rate 1.0): unseen error $58,600 (1) → $32,901 (100) → **$32,210 (300) →
  $33,256 (1,000)**, while the learned-from error keeps falling to **$19,299**. It starts memorising its own districts.
- Full-size steps bottom out at **340 trees ($32,043)**, then rise steadily (every 50 trees) to $33,256 at 1,000.
- At a tenth (0.1) the unseen error is still falling at 1,000 trees ($29,767).

## Cautions
- 1990 prices, capped at $500,001 (the model can never say more than the cap's neighbourhood).
- One random split; the ranking of methods can move by a few hundred dollars on another split.
