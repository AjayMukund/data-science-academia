# Benford reel (11-28): why do a third of real numbers start with 1? — measured 2026-11-22

**Data:** World Bank World Development Indicators (CC BY 4.0), via the public API, countries only (aggregates
dropped). Population SP.POP.TOTL, **2025, 217 countries**; land area AG.LND.TOTL.K2 (2021-24, 216 countries) as a
second check. `experiment.py` -> `out/results.json`.

## Real numbers: country populations, 2025 (217)
| first digit | countries | share | Benford log10(1+1/d) |
|---|---|---|---|
| 1 | **67** | **30.9%** | 30.1% |
| 2 | 35 | 16.1% | 17.6% |
| 3 | 28 | 12.9% | 12.5% |
| 4 | 19 | 8.8% | 9.7% |
| 5 | 24 | 11.1% | 7.9% |
| 6 | 15 | 6.9% | 6.7% |
| 7 | 11 | 5.1% | 5.8% |
| 8 | 9 | 4.1% | 5.1% |
| 9 | **9** | **4.1%** | 4.6% |
- Chi-square vs Benford 3.9 (8 df; a misfit would exceed 15.5): fits.
- The top of the list: India 1,463,865,525; China 1,406,585,000; US 341,784,857; ... Bangladesh 175,686,899;
  Russia 143,513,328; Ethiopia 135,472,051; Mexico 131,946,900. Smallest: Tuvalu 9,492.
- Land area (216): 1s 27.3%, 9s 5.6%, chi-square 2.2: fits too (not in the script: no fact dumps).

## Why (the growth mechanism)
- A town of 10,000 growing 5% a year reaches 100,000 in 47.2 years. It starts with a 1 (10,000-19,999) for
  **14.2 years**: it must double. It starts with a 9 (90,000-99,999) for **2.2 years**: +11% does it.
- 14.2 / 47.2 = **30%** of its time on a 1 (= log10 2); 2.2 / 47.2 = 4.6% on a 9 (= log10 10/9). Exactly Benford.

## Made-up numbers (217, seed 7)
- Invented the way people do: first digit chosen evenly, then ordinary random digits, 4-8 digits long.
  1s **25 (11.5%)**, 9s **27 (12.4%)**; every digit between 7.8% and 14.7%. Chi-square 113.5: fails Benford badly.
- (Revised 2026-11-23. The first version padded with zeros, e.g. 30,000,926, and gave 9s 33 (15.2%); replaced
  because nobody invents figures like that.)
- (An even spread between the smallest and largest population gives 36.9% 1s because 1.0-1.46 billion dominates
  that range; not used: it confuses the point.)

## Use
- Auditors and fraud examiners run this check on accounts and expense claims to flag figures that look invented
  (general practice, e.g. Nigrini's forensic accounting work). Say it as practice, no specific case or rate.

## Not claimed
- Benford needs numbers spread over several orders of magnitude (heights, prices with a cap, phone numbers do not
  follow it). Not in the voice; it may go in the caption.
