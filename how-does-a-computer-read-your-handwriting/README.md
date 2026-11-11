# How does a computer read your handwriting? A neural network from scratch

A 784-64-10 neural network written in plain NumPy reads MNIST's handwritten digits: 9.2% right before training,
92.7% after one pass, 97.3% after ten (about three seconds on a laptop CPU), against 82.0% for the obvious rule of
comparing each digit with the average digit. Test digit 65, an open-topped 4, is read as a 4 at 98.6%; ten pixels
that close its top drop that to 46% "4" / 31% "9". A real 9 (test digit 9587) is read as a 4 at 99.8%.

```bash
pip install numpy
# download the four MNIST .gz files from https://storage.googleapis.com/cvdf-datasets/mnist/ into download/
python experiment.py --data download   # training, accuracy curve, confusions -> out/results.json, out/net.npz
python pair.py --data download         # the open 4, the ten pixels, the 9 read as a 4 -> out/pair.json
```

See `FINDINGS.md`. Data: MNIST (LeCun, Cortes, Burges; CC BY-SA 3.0). Things to try: `--hidden 16` or `--hidden 256`;
fewer or more passes (`--epochs`); draw your own digit as a 28x28 array and see what the network says.
