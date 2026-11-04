# Seven measurements, two ideas: PCA

210 wheat grains of three varieties, seven X-ray measurements each. PCA's first direction (size) holds 72% of the spread and the second (shape) 17%; on those two numbers the nearest variety centre names 195 of 210, against 194 with all seven. On the [Seeds](https://archive.ics.uci.edu/dataset/236/seeds) data, UCI, CC BY 4.0.

```bash
python experiment.py      # downloads the data on first run
```

See `FINDINGS.md`. Try: drop the five size measurements and run PCA on the other two.
