# Why do scientists squash their charts? Log scales and doubling

Our World in Data (CC BY 4.0): transistors per microprocessor (Rupp 2022) and the training compute of 536 notable AI models (Epoch AI). **Transistors went from 2,308 (1971) to 58.2 billion (2021), 25 million times; on a log scale it is one straight line, doubling every 1.98 years. AI training compute since 2010 doubles every 5.7 months: Grok 4 (2025, an estimate) used about a billion times AlexNet's (2012).**

```bash
python experiment.py      # plain Python; downloads both series from Our World in Data on first run
```

See `FINDINGS.md`. Try: fit only 2000-2021 and see whether the chip doubling time changed.
