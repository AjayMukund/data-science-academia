# Data leakage - Ames, Iowa house sales

Run: `python experiment.py` (2026-11-04).
Data: De Cock (2011), "Ames, Iowa: Alternative to the Boston Housing Data
Set", Journal of Statistics Education - 2,930 home sales, Ames Assessor's
Office, 2006-2010. Free for education. De Cock's advice followed: the 5 homes
over 4,000 sq ft dropped -> **2,925 sales**, median price **$160,000**.

Job: guess a house's sale price before it sells. Log-linear model (numpy
lstsq), 80/20 split, seed 0; spread from 1,000 random splits.

| Model | Inputs | Median miss | % of price | Within 10% |
|---|---|---|---|---|
| honest | living area, overall quality, year built, garage cars, basement area | **$15,800** | 9.4% | 54% |
| leaky (in testing) | the same + **price per square foot** | **$0** | 0.000% | 100% |
| leaky, on a house for sale | price per foot unknown -> its neighbourhood's average | **$21,200** | 12.8% | 41% |

- Over 1,000 splits: honest **8.8%** (8.1-9.5), deployed leaky **13.0%**
  (11.9-14.2): the leaky model is worse on every split.
- The leaky model's weights: living area exactly 1, price per foot exactly 1,
  **quality, year built, garage and basement exactly 0**. It stopped looking
  at the house: price = area x price per foot, an identity, not a pattern.
- Walk-through house (row 610): North Ames, 1,445 sq ft, quality 6, built
  1964, sold **$158,000** ($109.34 per sq ft). Honest guess $167,900; leaky
  **$158,000 to the dollar**; leaky on sale day (neighbourhood average price
  per foot) $168,000.
- The leak is a column that only exists after the sale. Real ones: "price
  per square foot", "days on market", "final appraisal".
