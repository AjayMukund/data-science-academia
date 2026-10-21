# The screen that lied: findings

Run: `python pipeline.py` (plain Python 3 standard library: urllib, csv, sqlite3).
Writes `out/results.json` and `out/titanic.db`.

**Data:** the Titanic passenger list (Kaggle training set, 891 passengers), a public
historical record compiled from Encyclopedia Titanica, fetched from
`https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv`.
The same file as `what-does-cleaning-data-do` and `women-and-children-first`.

## Extract
- 60,302 bytes, plain ASCII text, 892 lines (header + 891 passengers), 12 columns.
- Download time on our laptop: median **0.25 s** over 9 runs (0.22-0.28 s; a cold first run took 0.5-0.6 s). The raw bytes are saved to `out/titanic.csv`.

## Transform (rules, applied to every row)
- Age blank for **177** passengers -> filled with the median age for the passenger's
  title: Mr 30, Mrs 35, Miss 21, Master 3.5, other titles 44.5 (4 of the 177 are Masters).
- Port (Embarked) blank for **2** -> filled with "S" (Southampton, the most common).
- Cabin known for **204**, blank for **687** -> becomes a yes/no `has_cabin` column.
- Title pulled from every name (891).
- 891 rows in, 891 rows out, nothing dropped. Time: **~3 ms** (median 3.3 ms over 30 runs, 3.1-5.4;
  one cold first run measured 8 ms).

## Load
- SQLite table `passengers`, 891 rows written, 891 counted back. File 36,864 bytes. 0.1-0.4 s (disk).

## The question (one SQL query, ~0.3 ms: median 0.32 ms over 30 runs)
`SELECT pclass, sex, ROUND(100.0*AVG(survived),1), COUNT(*) FROM passengers GROUP BY pclass, sex`

| class | sex | survived | n |
|---|---|---|---|
| 1 | female | 96.8% | 94 |
| 1 | male | 36.9% | 122 |
| 2 | female | 92.1% | 76 |
| 2 | male | 15.7% | 108 |
| 3 | female | 50.0% | 144 |
| 3 | male | 13.5% | 347 |

(Matches the 10-27 EDA master: 97 / 92 / 50 for women by class.)

## The turn: the source quietly changes (SIMULATED)
We renamed one header in the downloaded file, `Sex` -> `Gender`, and ran it again.

- **Without checks:** every step succeeds. 891 rows loaded, **0 women found**, women's
  survival NULL. No exception, no warning.
- **With checks** (required columns present before transforming; every sex is male or
  female after): the run stops in Transform with
  `ValueError: missing column(s) ['Sex']`. Load never runs, so the database still
  holds the previous good run.

- Women in the source: 314, men: 577 (before the change).
