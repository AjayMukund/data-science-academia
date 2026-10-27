# Do New Yorkers tip 13% or 22%?

One month of New York yellow-taxi trips (NYC TLC, August 2026: 3,336,716 trips, 3.2x a
spreadsheet's row limit), analysed with pandas in 17 lines. Sum of tips over sum of fares
says 12.9%; grouped by payment type, card riders tip 21.9%, cash rides record a tip of zero
($121 across 325,427 rides), and app-booked rides show 1.2%. The 13% averaged real tips with
tips that were never written down. Card tips by hour: 23.3% at 6 p.m., 17.5% at 4 a.m.

```bash
pip install pandas pyarrow
python download.py --month 2026-08
python analysis.py --data yellow_tripdata_2026-08.parquet --out out
```

Every snippet in `analysis.py` is run exactly as written and its printed output saved to
`out/results.json`. Timings are the fastest of 5 runs on our laptop. See `FINDINGS.md`.

Things to try:
- Group card tips by weekday, or by pickup zone (`PULocationID`).
- Compare tips as a share of the fare with tips as a share of the total.
- Run the same 17 lines on another month and see whether the cash and app gaps change.
