# Never off by a dollar: data leakage

An honest house-price model for Ames, Iowa misses a typical home by $15,800. Add a "price per square foot" column and it is never off by a dollar, because the column contains the answer; on a house still for sale it misses by $21,200, worse on 1,000 of 1,000 splits. On [De Cock's Ames housing data](https://jse.amstat.org/v19n3/decock.pdf) (2,930 sales, 2006-2010).

```bash
python experiment.py      # downloads the data on first run
```

See `FINDINGS.md`. Try: add `Yr Sold` or `Mo Sold` as a feature and see whether time leaks too.
