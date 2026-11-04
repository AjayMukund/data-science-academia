# Does rain empty Seoul's bike racks? Group-by in pandas

A year of Seoul's public bikes, 8,465 hours. One group-by: dry hours rent 766 bikes, rainy hours 167. By hour: peaks at 8 am and 6 pm. By both: dry, the evening wins (1,657 vs 1,096); in the rain the morning does (289 vs 265). Every snippet in `SNIPPETS` is the code shown in the video, run exactly as written. On [Seoul Bike Sharing Demand](https://archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand), UCI, CC BY 4.0. Needs pandas.

```bash
python analysis.py      # downloads the data on first run
```

See `FINDINGS.md`. Try: group by `Seasons` and `raining` together.
