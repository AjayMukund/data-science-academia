# When will Old Faithful erupt? Linear regression from scratch

R's classic `faithful` data: 272 eruptions of Old Faithful, each with how long it lasted and how
long until the next. A line is fitted with no library: scored by its sum of squared misses, then
found by gradient descent from a flat line (about 1,200 steps to 10.7 minutes of wait per minute
of eruption). On 72 held-out eruptions the line misses by 4.57 minutes against 11.55 for
guessing the average. But the data is two clumps, and two plain guesses (about 55 minutes after
a short eruption, 80 after a long one) do better still: 4.34 minutes, 97% within ten. The
National Park Service's own prediction rule has the same shape.

```bash
pip install numpy
python experiment.py            # downloads the data on first run, writes out/results.json
```

See `FINDINGS.md`. Things to try: change the learning rate and count the steps; fit a separate
line inside each clump; move the 3-minute cut and watch the two-guess score.
