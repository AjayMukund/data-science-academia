# Extrapolation - the men's mile world record

Run: `python experiment.py` (2026-11-04).
Data: Wikipedia, "Mile run world record progression", IAAF / World Athletics
era table: 32 ratified records 1913-1999, plus one awaiting ratification
(Josh Kerr, 3:42.66, London, 18 July 2026 - confirmed on World Athletics'
own race report, worldathletics.org, 18 Jul 2026). Records are facts; no
licence issue.

| | |
|---|---|
| first | **4:14.4**, John Paul Jones, 31 May 1913 |
| Bannister | 3:59.4, 6 May 1954 (first under four) |
| last ratified | **3:43.13**, Hicham El Guerrouj, Rome, 7 July 1999 |
| next | **3:42.66**, Josh Kerr, 18 July 2026 (pending): **0.47 s in 27 years** |

A straight line through all 32 ratified records:
- falls **3.9 s per decade** (about four seconds a decade)
- expects **3:30.9 in 2025** and **3:30.3 on the day Kerr ran**
- reaches **3:30 in 2027**, **3:00 in 2104**, **0:00 in 2562**, then negative

The line that worked: fitted on the **14 records up to Bannister** (1913 - 6 May
1954), it predicts **3:44.06 for 7 July 1999**, the day El Guerrouj ran
**3:43.13**: **0.93 s off, 45 years ahead**.

Corrected 2026-11-04 (caught by a consistency check): the first run fitted
15 records (it took in Landy, June 1954) and read the line at 1 January 1999,
giving "3:43.15, two hundredths off". Read on the record's real date that fit
says 3:42.96 (0.17 s); fitted only to Bannister, as the script says, 3:44.06.

Other fits for reference: up to 1975 (24 records), 4.2 s/decade, 3:00 in 2093.
