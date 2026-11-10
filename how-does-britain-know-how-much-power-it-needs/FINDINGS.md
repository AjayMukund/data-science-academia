# Forecasting master (11-17): how does Britain know how much power it will need tomorrow? — measured (2026-11-10)

**Data:** NESO historic demand data 2021-2025: national demand ("ND", the demand the transmission grid sees, net of
rooftop solar and small wind), every half-hour. NESO Open Data Licence — commercial use allowed with the attribution
**"Supported by National Energy SO Open Data"** (must be in every description). Temperature: Open-Meteo historical
archive (CC BY 4.0), daily mean at 2 m, average of London, Birmingham, Manchester and Glasgow.
**Code:** plain numpy. `experiment.py` (all methods, all test days) → `out/results.json`; `detail.py --day 2024-11-18`
→ `out/day_2024-11-18.json`; `holidays.py` → `out/holidays.json`. Fitted on 2021-2023; tested on **670 days**,
1 Jan 2024 – 31 Oct 2025. Average demand in the test period **25.9 GW**.
(Spam was the first pick for 11-17 and was dropped: the ml_basics master already built a spam filter.)

## The forecast: tomorrow's 48 half-hours, made the day before; score = average miss, GW
| method | average miss | of demand |
|---|---|---|
| A. same as yesterday | 1.86 GW | 7.2% |
| B. same day last week | 2.07 GW | 8.0% |
| C. average of the last 4 same weekdays | 2.03 GW | 7.8% |
| D. C + temperature | 1.77 GW | 6.8% |
| **E. yesterday + "what this weekday usually adds" + temperature change** | **1.51 GW** | **5.8%** |
- "Yesterday" by weekday: Monday 3.21 GW, Saturday 2.51 (the days after a change of routine); Wednesday 1.28.
  E on Mondays: **1.71** (halved).
- Temperature: about **0.45 GW per degree colder** across the day (0.53 at 5:30 pm), learned from 2021-2023.
- E's temperature input is the actual day's mean temperature, standing in for a weather forecast (a real forecaster
  would use the forecast; say so).

## The running example — Monday 18 November 2024 (first cold snap of that winter)
- Average **3.4 °C**, **2.2 degrees colder** than Sunday (5.6 °C).
- Peak **41.8 GW at 5 pm**. Sunday at 5 pm: **37.9 GW**. Energy that day 790 GWh vs Sunday 668 GWh.
- "Same as yesterday": average miss **5.1 GW**, worst **10.8 GW** (morning: 8 am actual 35.8 vs Sunday 25.3).
- E at 5 pm: Sunday 37.9 + **4.2 GW** (Monday vs Sunday, last 4 weeks) + **0.6 GW** (2.2 degrees colder) = **42.7**
  vs actual 41.8. E's average miss that day **1.3 GW**.

## The honest turn — holidays and sunshine
- E on bank holidays: **2.4 GW**; the day after: **3.0 GW**; ordinary days: **1.45 GW**.
- August bank holiday, Monday 25 Aug 2025: at noon E expected **23.1 GW**; the grid needed **15.2 GW**. (A sunny
  holiday: rooftop solar also hides demand from the grid. Our model knows neither holidays nor sunshine.)
- Christmas Day and Boxing Day are among the worst days for every weekday-average method (C: 8.5 and 7.6 GW).

## Cautions for the script
- 1 GW is roughly one large power station (Sizewell B: 1,198 MW net, EDF). Use "roughly".
- This is our model, not NESO's. The real forecasters use weather forecasts, holidays, TV schedules and much more;
  we can't see inside their system.
- Half-hour data averages out the famous half-time "TV pickup" spike; don't claim it from this data.
