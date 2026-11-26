# How does a shop know what you'll buy next? Confidence and lift

UCI Online Retail (a UK online gift shop, Dec 2010 to Dec 2011, CC BY 4.0): 19,957 baskets. **The green Regency teacup is in 1,015 baskets; 480 of them also hold the roses teacup (confidence 47.3%, lift 8.9). Party bunting turns up in 79 teacup baskets, but that's 7.8% against 8.4% of all baskets: lift 0.9, no link.**

```bash
python experiment.py      # plain Python; reads the .xlsx directly (no pandas), downloads it on first run (~23 MB)
```

See `FINDINGS.md`. Try: put a different product in `GREEN REGENCY TEACUP AND SAUCER`'s place and rank its companions by count, then by lift.
