# Could the numbers have warned them? Logistic regression on the Challenger O-rings

The 23 space shuttle flights before Challenger, each with its launch temperature and whether any
of its six field-joint O-rings showed heat damage (UCI dataset, CC BY 4.0). The seven damaged
flights alone show no temperature pattern; add the sixteen clean flights and every flight below
65°F was damaged (4 of 4, against 3 of 19 above). A straight line gives impossible chances (175% at
31°F); logistic regression, fitted here by gradient descent in plain NumPy, gives 23% at 70°F, 94%
at 53°F and 99.96% at 31°F, the forecast on launch morning, 22 degrees colder than any flight.
10,000 bootstrap refits show how much of that is the data and how much is extrapolation.

```bash
pip install numpy
python experiment.py            # downloads the data on first run, writes out/results.json
```

See `FINDINGS.md`. Things to try: fit on the erosion-only file instead; model the number of
damaged rings (binomial) rather than damaged-or-not; drop the clean flights and watch the fit fail.
