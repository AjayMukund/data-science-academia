# Were the V-1 bombs on London aimed? Why randomness clumps

R. D. Clarke (1946) cut 144 km² of south London into 576 squares and counted 537 flying-bomb hits. **229 squares had none and 43 had three or more, which looked aimed. Pure chance predicts 227 empty squares, and 7 squares with four hits; the real counts were 229 and 7 (chi-square 1.17, p = 0.88).** In 10,000 blind scatters of 537 hits, every one had a square with four or more.

```bash
python experiment.py      # plain Python, no download, seeded
```

See `FINDINGS.md`, including the 2019 re-analysis (Shaw & Shaw): across all of London the bombs leaned south; Clarke's result holds inside his patch. Try: drop the hits on a 12 x 12 grid instead and see how clustered chance looks.
