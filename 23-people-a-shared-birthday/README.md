# 23 people, a shared birthday? The birthday paradox on real births

Using average births per calendar date in England and Wales, 1995 to 2025 (ONS, Open Government Licence), a room of 23 has a shared birthday **50.7%** of the time (equal days: 50.7%; 22 people: 47.6%; 30: 70.6%; 57: 99.0%). The trick: 23 people make 253 pairs. Births are uneven (26 September averages 1,927 a day, Boxing Day 1,294), and it barely changes the answer.

```bash
python experiment.py      # downloads the ONS data on first run; exact answer plus a simulation check
```

See `FINDINGS.md`. Try: the smallest room that passes 90%.
