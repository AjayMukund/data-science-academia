# Do New Yorkers tip 13% or 22%? Findings

**Data:** NYC Taxi & Limousine Commission, Yellow Taxi Trip Records, **August 2026**
(free public data, https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page),
`yellow_tripdata_2026-08.parquet` (59 MB).

Run (pandas 2.3.3 + pyarrow 25.0.1):
`python analysis.py --data yellow_tripdata_2026-08.parquet --out out`. Every code snippet is in `SNIPPETS`, executed exactly as written; its real printed output and its time
(fastest of 5 runs on our laptop) are in `out/results.json`.

## The file
- **3,336,716 trips, 21 columns** - 3.18x a spreadsheet's 1,048,576-row limit.
- `pd.read_parquet` loads it in **0.29 s**.
- Cleaning: **15** trips dated outside August (the earliest 2009-01-01); trips with fare <= 0
  (refunds, voids) dropped too -> **3,319,452** trips.

## The two answers
- All trips: tips = **12.9%** of the fare (sum of tips / sum of fares).
- By payment type (one groupby): card **21.9%**, cash **0.0%**, type 0 **1.2%**, no-charge 0,
  dispute 0.
- **Cash:** 325,427 trips, **$121.08** of tips in total; 99.994% record a tip of zero. Cash tips
  are not written into the file.
- **Type 0:** 920,219 trips; request sources HV0003 (628,761) and HV0005 (181,225) - the TLC
  base licences of the ride-hailing apps - i.e. 88% booked through ride-hailing apps; their tips
  are 1.2% of the fare. Where those tips go is not in this file (we do not know).
- **Card only:** 2,051,305 trips, **21.9%** of the fare.

## By hour (card trips only)
- Lowest **04:00, 17.5%**; highest **18:00, 23.3%**.
- All trips, the same hours: 04:00 **6.6%** vs 16:00 15.0% - "less than half", an artefact.
- Why: at 04:00 **56.9%** of trips are type 0 (app-booked), vs 20% at midday; cash share stays
  7-13% all day. The missing app tips, not cash, drag the night down.

## Speed
- All seven snippets together: **1.06 s** (load 0.29, look 0.04, clean 0.38, total 0.01,
  groupby 0.06, card filter 0.18, by hour 0.10).
- 17 lines of code in total.

## What we cannot know
- What cash riders tip: the file records zero. 21.9% is what CARD riders tip.
