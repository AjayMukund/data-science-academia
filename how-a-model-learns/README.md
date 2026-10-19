# How a model learns

A character-level language model (52,680 numbers) trained from random on
*Alice's Adventures in Wonderland*, with backpropagation written by hand in
NumPy so every step of learning is visible.

```bash
python experiment.py sweep          # three learning rates: too small, right, too big
python experiment.py train 0.3 60000   # the main run: prints what it writes as it learns
```

The book downloads from Project Gutenberg on first run. See `FINDINGS.md`.
Try: a different book, a bigger hidden layer, or a learning rate of 3.
