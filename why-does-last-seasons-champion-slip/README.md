# Why does last season's champion usually slip? Regression to the mean

Premier League results 2010-11 to 2025-26 (openfootball/football.json, CC0). **12 of the last 15 champions got fewer points the next season, by 14.6 on average.** A league of 20 identical teams whose every match is settled by dice still crowns a champion on 68 points, who averages 53 the next season. Real tables are about twice as spread out, so skill is real; but first place takes skill and luck, and the luck re-rolls.

```bash
python experiment.py      # downloads the season files on first run
```

See `FINDINGS.md`. Try: the same test for the team that finishes 17th.
