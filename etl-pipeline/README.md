# The screen that lied: an ETL pipeline, and how it fails quietly

A small, real Extract -> Transform -> Load pipeline on the Titanic passenger
list (891 passengers). Extract downloads the plain-text file; Transform fills
177 blank ages from the median for each title, fills 2 missing ports and turns
687 empty cabins into a yes/no column; Load writes everything to SQLite and
answers one SQL query (survival by class and sex).

Then the turn: rename one column in the source (`Sex` -> `Gender`) and run it
again. Without checks, every step succeeds, all 891 rows load, and the query
finds 0 women. With two checks (the columns we need before transforming, valid
values after), the run stops at step two:

    ValueError: missing column(s) ['Sex']

Plain Python 3, standard library only (urllib, csv, sqlite3).

```bash
python pipeline.py      # downloads the data, writes out/results.json
```

Timings are the fastest of several runs on our laptop; yours will differ.
See `FINDINGS.md`.

Things to try:
- Rename a different column (`Age`, `Cabin`) and see which check catches it.
- Change the values instead of the header (`male` -> `M`) and watch check 2.
- Add a third check: the row count against the previous run.
