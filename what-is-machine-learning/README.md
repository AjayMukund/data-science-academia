# What machine learning actually is

Can a few hand-written rules catch spam? We tried, on 5,574 real text messages
(the [SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection),
Almeida & Gómez Hidalgo, 2011, CC BY 4.0), and compared them with a filter
that learns from examples (multinomial Naive Bayes, written from scratch).

```bash
python experiment.py      # downloads the dataset on first run
```

It prints and saves (`out/results.json`): how many spam texts the 7-word and
16-word rules catch and how many real messages they wrongly flag, the same for
the learned filter, the words it found most "spammy", and how its accuracy
grows with the number of examples. See `FINDINGS.md` for our results.

Try: add your own rule words, or change the training-set size.
