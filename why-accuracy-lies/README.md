# Right 99.5% of the time, never warned once: why accuracy lies

A forecaster that always says "no big California earthquake tomorrow" is right on 99.5% of the 9,497 days from 2000 to 2025, and warned of none of the 48 days with a magnitude 5+ quake. A simple aftershock rule scores lower (98.0%) and catches 5 of them, Ridgecrest's 7.1 among them. On the [USGS earthquake catalogue](https://earthquake.usgs.gov/fdsnws/event/1/) (public domain).

```bash
python experiment.py      # downloads the data on first run
```

See `FINDINGS.md`. Try: change the magnitude that counts as "big" (`BIG`) or the rule's trigger, and watch accuracy and the catch move in opposite directions.
