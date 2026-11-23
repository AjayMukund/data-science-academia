# Why do a third of real numbers start with 1? Benford's law

The populations of 217 countries (World Bank, 2025, CC BY 4.0): **67 start with a 1 (31%)** and only 9 with a 9 (4%), just as Benford's law predicts (chi-square 3.9). Numbers invented the way people invent them spread evenly instead: 1s 11.5%, 9s 12.4% (chi-square 113.5). The reason is growth: a town growing 5% a year spends 14.2 years with a 1 in front and 2.2 years with a 9.

```bash
python experiment.py      # downloads from the World Bank API on first run
```

See `FINDINGS.md`. Try: run it on the land-area indicator (AG.LND.TOTL.K2), or on any column of amounts you have.
