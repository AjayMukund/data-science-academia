# Is the difference real? A/B testing without formulas

A real randomised experiment (Kevin Hillstrom's MineThatData email challenge, 2008): 64,000
customers sent a men's email, a women's email or no email. The men's email sold to 267, the
women's to 189. Is that a real difference, or luck?

Everything is measured by re-running the experiment on the real customers:
- two random halves of the SAME no-email group: 60 vs 62 buyers; 50-72 over 100,000 splits
- shuffling who got which email 100,000 times: the men's group gets 207-248 buyers; only 27
  shuffles match the real 267 (about 1 in 3,700 - the p-value)
- the same test with 1,000 emails each: the weaker email wins or ties about 1 time in 4

```bash
pip install numpy
python experiment.py            # downloads the data (4 MB) on first run, writes out/results.json
```

See `FINDINGS.md`. Things to try: test visits instead of purchases; compare each email
against no email; find the smallest test size where the men's email wins 95% of the time.
