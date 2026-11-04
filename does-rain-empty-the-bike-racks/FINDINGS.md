# Group-by in pandas - Seoul bike rentals

Run (scratch venv with pandas 2.3.3, NOT the render venv):
`python analysis.py` (2026-11-04). Every snippet shown on
screen is in `SNIPPETS`, executed exactly as written; its real printed output
and time (fastest of 5 on our laptop) are in `out/results.json`.

Data: UCI 560, "Seoul Bike Sharing Demand" (**CC BY 4.0**; Sathishkumar V E &
Yongyun Cho, 2020): Seoul's public bikes, rentals per hour with the weather,
1 Dec 2017 - 30 Nov 2018. 8,760 hours; **295** hours when the system was
closed dropped -> **8,465 hours**. 516 rainy hours (6.1%).

| Snippet | Output | Time |
|---|---|---|
| load + drop closed hours | `8465 hours` | 0.013 s |
| average per hour | **729** | 0.0001 s |
| `groupby("raining")` | dry **766**, rain **167** | 0.0009 s |
| `groupby("Hour")` | 4 am 137, **8 am 1,050**, 1 pm 758, **6 pm 1,554**, 11 pm 694 | 0.001 s |
| `groupby(["Hour", "raining"])` | 8 am: dry **1,096**, rain **289**; 6 pm: dry **1,657**, rain **265** | 0.004 s |
| `groupby("Seasons")` | winter 226, spring 746, autumn 924, summer 1,034 | 0.001 s |

- Rain cuts rentals by 78% overall (766 -> 167): about a fifth remain.
- **The twist:** dry, 6 pm is the busiest hour (1,657 vs 1,096 at 8 am); in the
  rain the morning wins (289 vs 265). The evening falls 84%, the morning 74%.
  Why is not in the data (likely: commuters must get to work, can choose how
  to get home) - a guess, not a finding.
- Rain is not bunched at any hour (3-11% of each hour), so the split is fair.
- Busiest single hour: 19 June 2018, 6 pm, **3,556** bikes.
