# Who's the best batter in IPL history? Why small samples lie

Cricsheet's IPL ball-by-ball data (every match 2008 to 31 May 2026, ODC-By). **The best strike rate is 400, from 8 runs off 2 balls; 25 batters sit on zero.** Give every batter 100 extra balls at the league's 133.4 and the 400 falls from 1st to 119th, while Vaibhav Suryavanshi (229 off 449) stays 1st on 212. In 100,000 simulated innings of 449 average balls, none reached 229.

```bash
python experiment.py      # downloads Cricsheet's IPL archive on first run (~40 s with the simulation)
```

See `FINDINGS.md`. Cricsheet adds each new season to the archive, so after IPL 2027 the numbers will move. Try: change the 100 extra balls to 50 or 300 and watch the leaderboard.
