# How does Britain know how much power it needs tomorrow? Day-ahead forecasting, measured

Forecast tomorrow's 48 half-hours of Great Britain's electricity demand, the day before, in plain NumPy. Fitted on
2021-2023, tested on 670 days (January 2024 to October 2025). "Same as yesterday" misses by 1.86 GW on average;
yesterday + what that weekday usually adds + a temperature term misses by 1.51 GW, and Mondays drop from 3.21 to 1.71.
The running example: Monday 18 November 2024, the first cold snap of that winter: 41.8 GW at 5 pm against Sunday's
37.9; the forecast said 42.7. Where it fails: bank holidays (the August 2025 one: 23.1 GW expected at noon, 15.2 needed).

```bash
pip install numpy
python download.py                                  # NESO demand 2021-2025 and Open-Meteo temperatures into download/
python experiment.py --data download                # all five methods; out/results.json
python detail.py --data download --day 2024-11-18   # the running example; out/day_2024-11-18.json
python holidays.py --data download                  # Mondays, bank holidays, every day's miss; out/holidays.json
python scatter.py --data download                   # what the temperature rule learns from; out/scatter.json
```

Data: NESO historic demand data, Supported by National Energy SO Open Data (NESO Open Data Licence); temperatures from
the Open-Meteo historical weather archive (CC BY 4.0), the daily mean of London, Birmingham, Manchester and Glasgow.
The forecast uses each day's actual temperature, standing in for a weather forecast. This is our simple model, not
NESO's. See `FINDINGS.md`.

Things to try: a holiday flag, rooftop solar (the files include embedded solar generation), a separate temperature
rule for summer, or forecasting a week ahead.
