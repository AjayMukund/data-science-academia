# Are more babies born on a full moon? 62 million US births, checked

Every US birth by day, 2000 to 2014 (62,187,024 births over 5,479 days), against the 175 full moons we could test.
Each day is compared with the median of the same weekday one, two and three weeks either side, with holidays left
out. Full-moon days come out +0.16% (about 23 extra births on an average day of ~11,460), and a random day does at
least that well 51% of the time (p = 0.514). Only 1 of the 30 days of the lunar cycle passes p < 0.05, which is what
luck alone gives. Meanwhile every Friday the 13th falls below its expected count (−6.24% on average, 25 of 25),
Halloween −12.2% and Valentine's Day +4.7%: the dates people choose for planned births move the numbers. The moon doesn't.

```bash
pip install numpy
python experiment.py          # a few seconds; writes out/results.json
```

Data: US births by date (Social Security Administration via FiveThirtyEight, CC BY 4.0), in `data/births.csv`.
Full moons are computed in the script (Meeus, *Astronomical Algorithms*, ch. 49).
A first version without the holiday filter found a false +1.4% "full-moon effect": Thanksgiving pulled its week's
baseline down. See `FINDINGS.md`. Things to try: new moons; the days after a full moon; leave the holidays in.
