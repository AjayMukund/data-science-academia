"""
experiment.py - how does a computer read your handwriting? A small neural network, trained from scratch in numpy.

Data: MNIST (Yann LeCun, Corinna Cortes, Christopher Burges; CC BY-SA 3.0): 60,000 training and 10,000 test
handwritten digits, 28x28 grey pixels, written by US Census Bureau employees and high-school students.
Mirror: https://storage.googleapis.com/cvdf-datasets/mnist/ (download the four .gz files into download/).

  1. The simple rule: the average picture of each digit (from the training set); read a digit as the closest average.
  2. A neural network, 784 -> 64 -> 10 (ReLU, softmax), trained with mini-batch gradient descent; test accuracy
     before training and after each pass through the 60,000.
  3. What it learned: each hidden neuron's 784 weights, as a 28x28 picture.
  4. Its mistakes on the 10,000: how many, which confusions, the most confident wrong answers.

  python experiment.py --data download --out out
"""
import argparse, gzip, json, os, time
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"))
ap.add_argument("--hidden", type=int, default=64)
ap.add_argument("--epochs", type=int, default=10)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)


def load(name, offset, shape):
    return np.frombuffer(gzip.open(os.path.join(a.data, name)).read(), np.uint8, offset=offset).reshape(shape)


Xtr = load("train-images-idx3-ubyte.gz", 16, (-1, 784)).astype(np.float32) / 255
ytr = load("train-labels-idx1-ubyte.gz", 8, (-1,)).astype(np.int64)
Xte = load("t10k-images-idx3-ubyte.gz", 16, (-1, 784)).astype(np.float32) / 255
yte = load("t10k-labels-idx1-ubyte.gz", 8, (-1,)).astype(np.int64)
res = {"train": len(ytr), "test": len(yte), "pixels": 784}

# 1. the simple rule: closest average digit
avg = np.stack([Xtr[ytr == d].mean(0) for d in range(10)])
dist = ((Xte[:, None, :] - avg[None]) ** 2).sum(-1)
rule_pred = dist.argmin(1)
res["rule"] = {"accuracy": float((rule_pred == yte).mean()), "wrong": int((rule_pred != yte).sum())}
np.save(os.path.join(a.out, "averages.npy"), avg)
print("closest-average rule: %.1f%%" % (100 * res["rule"]["accuracy"]), flush=True)

# 2. the network
rng = np.random.default_rng(0)
H = a.hidden
W1 = rng.normal(0, np.sqrt(2 / 784), (784, H)).astype(np.float32); b1 = np.zeros(H, np.float32)
W2 = rng.normal(0, np.sqrt(2 / H), (H, 10)).astype(np.float32); b2 = np.zeros(10, np.float32)
res["params"] = int(W1.size + b1.size + W2.size + b2.size)


def forward(X):
    h = np.maximum(X @ W1 + b1, 0)
    z = h @ W2 + b2
    z = z - z.max(1, keepdims=True)
    p = np.exp(z)
    return h, p / p.sum(1, keepdims=True)


def acc():
    return float((forward(Xte)[1].argmax(1) == yte).mean())


res["curve"] = [acc()]
print("untrained: %.1f%%" % (100 * res["curve"][0]), flush=True)
lr, B = 0.1, 64
t0 = time.time()
for ep in range(a.epochs):
    order = rng.permutation(len(ytr))
    for i in range(0, len(order), B):
        idx = order[i:i + B]
        X, y = Xtr[idx], ytr[idx]
        h, p = forward(X)
        g = p.copy(); g[np.arange(len(y)), y] -= 1; g /= len(y)            # d loss / d logits
        gW2 = h.T @ g; gb2 = g.sum(0)
        gh = (g @ W2.T) * (h > 0)
        gW1 = X.T @ gh; gb1 = gh.sum(0)
        W2 -= lr * gW2; b2 -= lr * gb2; W1 -= lr * gW1; b1 -= lr * gb1
    res["curve"].append(acc())
    print("after pass %d: %.2f%%  (%.0fs)" % (ep + 1, 100 * res["curve"][-1], time.time() - t0), flush=True)
res["train_secs"] = round(time.time() - t0, 1)
np.savez(os.path.join(a.out, "net.npz"), W1=W1, b1=b1, W2=W2, b2=b2)

# 3/4. what it learned, and its mistakes
h, p = forward(Xte)
pred = p.argmax(1)
wrong = np.where(pred != yte)[0]
conf = np.zeros((10, 10), int)
for t, q in zip(yte, pred):
    conf[t, q] += 1
pairs = sorted(((int(conf[i, j]), i, j) for i in range(10) for j in range(10) if i != j), reverse=True)[:6]
res["net"] = {"accuracy": float((pred == yte).mean()), "wrong": int(len(wrong)),
              "top_confusions": [{"true": i, "read_as": j, "count": c} for c, i, j in pairs]}
conf_wrong = sorted(wrong, key=lambda k: -p[k, pred[k]])[:12]
res["confident_mistakes"] = [{"i": int(k), "true": int(yte[k]), "read_as": int(pred[k]),
                              "p_read": float(p[k, pred[k]]), "p_true": float(p[k, yte[k]])} for k in conf_wrong]
# digits the simple rule gets wrong and the network gets right, confidently
fixed = [k for k in range(len(yte)) if rule_pred[k] != yte[k] and pred[k] == yte[k] and p[k, yte[k]] > 0.95]
res["rule_wrong_net_right"] = {"count": int(sum((rule_pred != yte) & (pred == yte))), "examples":
                               [{"i": int(k), "true": int(yte[k]), "rule_read": int(rule_pred[k]), "p": float(p[k, yte[k]])} for k in fixed[:20]]}
json.dump(res, open(os.path.join(a.out, "results.json"), "w"), indent=1)
print("network %.2f%%, %d wrong; top confusions %s" % (100 * res["net"]["accuracy"], len(wrong), pairs[:3]))
