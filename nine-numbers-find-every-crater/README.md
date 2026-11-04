# Nine numbers find every crater: convolution

One 3x3 grid (-1 0 1 / -2 0 2 / -1 0 1) slid over NASA's Moon sums to 0 on flat ground and to -389 across a plain's edge; every total written down as brightness draws every crater rim. Image: NASA Goddard, "Full Moon" (GSFC_20171208_Archive_e000868), built from Lunar Reconnaissance Orbiter data, public domain.

```bash
python experiment.py      # downloads the data on first run
```

See `FINDINGS.md`. Try: swap `EDGE` for a 3x3 of your own and look at `out/moon_edges.png`.
