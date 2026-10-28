# Is the difference real? Findings

**Data:** Kevin Hillstrom, MineThatData E-Mail Analytics and Data Mining Challenge (2008):
64,000 customers of an online retailer, each RANDOMLY assigned one of three treatments - a men's
merchandise email, a women's merchandise email, or no email - with visits, purchases ("conversion")
and spend recorded over the following two weeks. Publicly released for analysis:
http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv
(3.96 MB).

Run: `python experiment.py --data hillstrom.csv --out out` (numpy only; seed 2026; 100,000 re-runs).
The "shuffle" is done as a hypergeometric draw (exactly what shuffling labels does to a 0/1 column);
2,000 literal shuffles give the same 95% range (-0.193 to +0.191 points), checked in the script.

## The real result
| group | customers | visited | bought | buyers | spend / customer |
|---|---|---|---|---|---|
| Men's email | 21,307 | 18.3% | 1.25% | **267** | $1.42 |
| Women's email | 21,387 | 15.1% | 0.88% | **189** | $1.08 |
| No email | 21,306 | 10.6% | 0.57% | 122 | $0.65 |

Men's vs women's: a gap of **78 buyers**, 0.369 percentage points.

## A/A: how much do two IDENTICAL groups differ by luck?
- Split the 21,306 no-email customers (122 buyers) into two random halves of 10,653.
- First split: **60 vs 62** buyers. Over 100,000 splits, 95% of halves get **50 to 72** buyers
  (rate gap within +-0.207 points).

## The shuffle test: men's vs women's
- Pool both groups (42,694 customers, 456 buyers), shuffle who got which email, 100,000 times.
- 95% of shuffles give the "men's" group **207 to 248** buyers (gap -0.193 to +0.191 points).
  Largest shuffled gap: 0.435 points.
- Real: **267** buyers in the men's group (+0.369 points).
- Shuffles with a gap at least as big as the real one: **27 of 100,000** (0.027%, ~1 in 3,700).
  That share is what statisticians call the p-value (two-sided).

## The same experiment, smaller (100,000 re-draws each, customers drawn from the real groups)
| emails per version | women's email wins or ties | women's wins outright | median buyers men / women |
|---|---|---|---|
| 100 | 55.0% | 25.9% | 1 / 1 |
| 1,000 | **24.4%** | 18.0% | 12 / 9 |
| 5,000 | 4.1% | 3.3% | 62 / 44 |

## Real is not the same as big
- Per 1,000 emails: men's email gives **3.7** more buyers than the women's, **6.8** more than no
  email; **$345** more spend than the women's.
- Visits are not the question here; both emails beat no email by a wide margin (18.3% / 15.1% vs 10.6%).
