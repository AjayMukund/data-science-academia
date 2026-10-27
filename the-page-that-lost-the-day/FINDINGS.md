# The page that lost the day: findings

Question: which English Wikipedia page was read most on one day?

**Data:** Wikimedia's public hourly pageview files for **2026-10-20** (UTC), 24 files from
`https://dumps.wikimedia.org/other/pageviews/2026/2026-10/` (CC0, Wikimedia Foundation analytics
data). One line per (wiki, page, hour): `en Some_Page 123 0`. English Wikipedia = project `en`
(desktop) + `en.m` (mobile), summed per title. Excluded from the ranking: Main_Page, `-`, and
non-article namespaces (Special:, Wikipedia:, Portal:, Help:, File:, Category:, Template:, User:,
Talk:).

Run: `python download.py --dest pageviews`, then `python experiment.py --data pageviews --out out` (results.json), `python race.py --data pageviews` (race.json), `python sample.py --data pageviews`.
Plain Python 3 standard library (gzip, multiprocessing). Machine: our laptop, Intel i3-10105
(4 cores / 8 threads), 16 GB RAM, about 4 GB free while measuring.

## A. How big is one day
- 24 files, **1.38 GB** compressed, **5.03 GB** unpacked text, **151,648,607 lines** (every wiki).
- A spreadsheet stops at 1,048,576 rows: this day is **144.6** spreadsheets' worth.
- One plain read of everything, no counting: 93 s.

## B. The ordinary way: hold it all in memory (stopped at a safety limit)
- Each line held as a Python bytes object costs **81.6 bytes** of memory.
- Hour by hour: 1 h 0.36 GB, 2 h 0.75, 4 h 1.50, 8 h **3.12 GB** (41.0 M lines) -> stopped at the
  3 GB limit we set.
- Projected for all 151.6 M lines (scaled by rows, since later hours are busier): **11.5 GB**,
  against ~4 GB free (16 GB total). Said as a projection.

## C. One worker, streaming (read a line, add to that page's tally, forget the line)
- **9,028,757** distinct English pages; the tallies take **0.84 GB**.
- **142.0 s** (2 min 22 s), fastest of 3 runs (144.6 / 142.0 / 142.4).
- A year at this speed: 142.0 x 365 = **14.4 hours** (projection).

## D. Map-reduce, 8 workers
- MAP: each worker counts whole hourly files and splits its counts into 8 buckets by page name
  (crc32 % 8), written to disk. REDUCE: worker k adds up bucket k from all 24 files; the 8 top-10
  lists are merged.
- **53.3 s** total, fastest of 3 (53.3 / 53.9 / 54.3): map 45.1 s, reduce 8.2 s.
- **2.66x** faster than one worker, not 8x: 4 physical cores (8 threads), and the shuffle wrote and
  read back **1.01 GB** of partial counts.
- Each worker peaked at 0.29 GB. Buckets hold 1.13 M pages each (even split).
- Top-10 identical to C, page for page and view for view (asserted).
- A year on 8 workers: 53.3 x 365 = **5.4 hours** (projection).

## The answer
| # | page | views |
|---|---|---|
| 1 | Kim_Jong_Un | 351,098 |
| 2 | Jailer_2 | 303,515 |
| 3 | Elizabeth_Holmes | 257,751 |
| 4 | You_Can_See_Everything | 208,954 |
| 5 | Helen_Haines | 164,342 |
| 6 | Luke_McCaffrey | 141,933 |
| 7 | ChatGPT | 139,736 |
| 8 | Deaths_in_2026 | 134,792 |
| 9 | Hunter_Schafer | 122,232 |
| 10 | Richard_Riehle | 112,361 |

## The race (race.json)
- Hourly leader of all English pages: Jailer_2 in **15 of 24 hours** (05-19 UTC), Kim_Jong_Un 3,
  Elizabeth_Holmes 3, Luke_McCaffrey 2, Xavi_Espart 1. At 12:00 Jailer_2 led (13,989 vs 12,748).
- Kim_Jong_Un's reads came early: 35,004 / 30,668 / 28,103 in hours 00-02; **136,208 by the end of
  hour 04**, a **90,246** lead over Jailer_2.
- Jailer_2 out-read Kim_Jong_Un every hour from 05 to 19 and cut the lead to **39,567** (end of 19).
  Kim_Jong_Un led the running total in every hour of the day; final 351,098 vs 303,515.
- So any single hour, or any sample of hours, would name the wrong winner most of the time.

## Not ours
- MapReduce: J. Dean and S. Ghemawat, "MapReduce: Simplified Data Processing on Large Clusters",
  OSDI 2004 (Google).
- Spreadsheet row limit 1,048,576: Microsoft Excel specifications.
