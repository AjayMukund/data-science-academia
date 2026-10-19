"""
experiment.py - a real, tiny training run for the "how a model learns" master.

A character-level language model (Bengio-style MLP: 8 characters of context ->
embedding -> tanh hidden layer -> next-character probabilities) trained from
scratch on Alice's Adventures in Wonderland (Project Gutenberg #11, public
domain, corpus/alice_raw.txt). Plain numpy with hand-written backprop - no PyTorch, so every
step of learning is visible in the code.

    python experiment.py sweep     # learning-rate sweep
    python experiment.py train     # the main run + logs

Everything is seeded; results land in out/. The book downloads itself on first run.
"""
import os, sys, json, time
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)
sys.stdout.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------- data
SRC = os.path.join(HERE, "corpus", "alice_raw.txt")
if not os.path.exists(SRC):                       # first run: fetch from Project Gutenberg
    import urllib.request
    os.makedirs(os.path.dirname(SRC), exist_ok=True)
    urllib.request.urlretrieve("https://www.gutenberg.org/cache/epub/11/pg11.txt", SRC)
raw = open(SRC, encoding="utf-8").read()
a = raw.index("*** START OF"); a = raw.index("\n", a) + 1
b = raw.index("*** END OF")
text = raw[a:b]
text = text[text.index("CHAPTER I."):]                  # skip the contents list
text = " ".join(text.split())                            # one line, single spaces
chars = sorted(set(text))
V = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
data = np.array([stoi[c] for c in text], dtype=np.int64)
CTX = 8
n_split = int(0.9 * len(data))


def windows(arr):
    X = np.lib.stride_tricks.sliding_window_view(arr[:-1], CTX)
    Y = arr[CTX:]
    return X, Y


Xtr, Ytr = windows(data[:n_split])
Xva, Yva = windows(data[n_split:])

E, H = 16, 256


def init(seed=0):
    r = np.random.default_rng(seed)
    return {"C": r.normal(0, 1, (V, E)),
            "W1": r.normal(0, 1, (CTX * E, H)) * (5 / 3) / np.sqrt(CTX * E),
            "b1": np.zeros(H),
            "W2": r.normal(0, 1, (H, V)) * 0.01,
            "b2": np.zeros(V)}


def forward(P, X):
    emb = P["C"][X].reshape(len(X), -1)
    h = np.tanh(emb @ P["W1"] + P["b1"])
    logits = h @ P["W2"] + P["b2"]
    return emb, h, logits


def probs_of(logits):
    z = logits - logits.max(1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(1, keepdims=True)


def loss_of(P, X, Y):
    _, _, lg = forward(P, X)
    p = probs_of(lg)
    return float(-np.log(p[np.arange(len(Y)), Y] + 1e-12).mean())


def step(P, X, Y, lr):
    emb, h, logits = forward(P, X)
    p = probs_of(logits)
    B = len(Y)
    loss = float(-np.log(p[np.arange(B), Y] + 1e-12).mean())
    d = p
    d[np.arange(B), Y] -= 1
    d /= B
    g = {"W2": h.T @ d, "b2": d.sum(0)}
    dh = d @ P["W2"].T * (1 - h ** 2)
    g["W1"] = emb.T @ dh
    g["b1"] = dh.sum(0)
    demb = (dh @ P["W1"].T).reshape(B, CTX, E)
    gC = np.zeros_like(P["C"])
    np.add.at(gC, X, demb)
    g["C"] = gC
    for k in P:
        P[k] -= lr * g[k]
    return loss, g


def sample(P, prompt, n=70, temp=0.6, seed=1):
    r = np.random.default_rng(seed)
    s = prompt
    for _ in range(n):
        ctx = np.array([[stoi.get(c, 0) for c in s[-CTX:].rjust(CTX)]])
        _, _, lg = forward(P, ctx)
        p = probs_of(lg / temp)[0]
        s += chars[r.choice(V, p=p)]
    return s


def probe(P, sentence):
    """p(correct next char) at every position of `sentence` after CTX chars."""
    ids = [stoi[c] for c in sentence]
    out = []
    for i in range(CTX, len(ids)):
        _, _, lg = forward(P, np.array([ids[i - CTX:i]]))
        p = probs_of(lg)[0]
        out.append((sentence[i], float(p[ids[i]])))
    return out


def train(lr, steps, batch=64, seed=0, log_every=100, hooks=None):
    P = init(seed)
    r = np.random.default_rng(seed + 1)
    curve = []
    for t in range(steps + 1):
        if hooks and t in hooks:
            hooks[t](P, t)
        if t == steps:
            break
        ix = r.integers(0, len(Xtr), batch)
        lr_t = lr if t < 0.8 * steps else lr / 10
        loss, _ = step(P, Xtr[ix], Ytr[ix], lr_t)
        if not np.isfinite(loss):
            curve.append((t, float("nan")))
            break
        if t % log_every == 0:
            curve.append((t, loss))
    return P, curve


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "train"
    print("corpus: %d characters, vocab %d, train windows %d, val %d"
          % (len(text), V, len(Xtr), len(Xva)))
    print("chance loss ln(V) = %.3f" % np.log(V))
    print("first 120 chars:", text[:120])

    if mode == "sweep":
        res = {}
        for lr in (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0):
            t0 = time.time()
            P, curve = train(lr, 3000, log_every=250)
            vl = loss_of(P, Xva[:20000], Yva[:20000]) if np.all(np.isfinite(P["W2"])) else float("nan")
            res[lr] = {"curve": curve, "val": vl}
            print("lr %-5g  val loss %.3f  curve %s  (%.0fs)"
                  % (lr, vl, " ".join("%.2f" % c for _, c in curve), time.time() - t0))
        json.dump({str(k): v for k, v in res.items()}, open(os.path.join(OUT, "sweep.json"), "w"), indent=1)

    else:
        lr = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
        steps = int(sys.argv[3]) if len(sys.argv) > 3 else 30000
        ckpts = [0, 100, 300, 1000, 3000, 10000, steps]
        log = {"lr": lr, "steps": steps, "V": V, "chars": len(text), "samples": {},
               "probe": {}, "val": {}}
        PROBE = "Alice was beginning to get very tired of sitting by her sister on the bank"

        def hook(P, t):
            log["samples"][t] = [sample(P, "Alice was ", seed=s) for s in (1, 2)]
            log["probe"][t] = probe(P, PROBE)
            log["val"][t] = loss_of(P, Xva[:20000], Yva[:20000])
            print("\n--- step %d  val loss %.3f" % (t, log["val"][t]))
            for s in log["samples"][t]:
                print("   ", s)

        t0 = time.time()
        P, curve = train(lr, steps, hooks={c: hook for c in ckpts}, log_every=50)
        log["curve"] = curve
        log["secs"] = time.time() - t0
        log["guesses"] = steps * 64

        # one weight, swept: the loss landscape along a single parameter
        ix = np.random.default_rng(9).integers(0, len(Xva), 4000)
        w0 = float(P["b2"][stoi["e"]])
        sl = []
        for dw in np.linspace(-6, 6, 49):
            P["b2"][stoi["e"]] = w0 + dw
            sl.append((float(w0 + dw), loss_of(P, Xva[ix], Yva[ix])))
        P["b2"][stoi["e"]] = w0
        log["slice"] = {"param": "output bias for 'e'", "w0": w0, "points": sl}
        np.savez(os.path.join(OUT, "final_params.npz"), **P)
        json.dump(log, open(os.path.join(OUT, "train.json"), "w"), indent=1)
        print("\ntrained %d steps in %.0fs; %d guesses" % (steps, log["secs"], log["guesses"]))
