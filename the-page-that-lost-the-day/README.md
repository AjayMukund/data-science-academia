# The page that lost the day: big data on one laptop

Which English Wikipedia page was read most on 20 October 2026? The answer is in
Wikipedia's public hourly pageview files: 24 files, 151.6 million lines, 5 GB of
text. This folder measures three ways to answer it on one ordinary laptop:

1. **Hold it all in memory**: about 82 bytes per line, so the day needs ~11.5 GB
   (measured to a 3 GB safety limit, then projected).
2. **Stream it**: read a line, add its views to that page's running total, forget
   the line. 9 million page totals fit in 0.84 GB; 2 min 22 s on one worker.
3. **Map-reduce on 8 workers**: each worker counts whole hourly files (map), splits
   its counts into 8 bins by page name, and each worker adds up one bin (reduce).
   53 s, the identical top 10, but only 2.7x faster than one worker.

The winner, Kim Jong Un (351,098 reads), led only 3 of the 24 hours; Jailer 2 led 15
and finished second. See `FINDINGS.md`.

Plain Python 3, standard library only.

```bash
python download.py --day 2026-10-20 --dest pageviews     # 24 files, ~1.4 GB
python experiment.py --data pageviews --out out           # ~15 min, writes out/results.json
python race.py --data pageviews --out out                 # the hour-by-hour race
```

Timings are the fastest of 3 runs on our laptop (4 cores / 8 threads, 16 GB); yours
will differ. The naive test stops at `--naive-limit-gb` (default 3) so it cannot run
your machine out of memory.

Things to try:
- Change `BUCKETS` and the pool size, and watch where more workers stop helping.
- Rank a different wiki (`de`, `fr`, `ja`) by changing `EN`.
- Download a second day and see whether the same page wins.
