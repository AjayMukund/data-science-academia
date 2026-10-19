"""
experiment.py - measurements for the transformer-layer master.

GPT-2 small run in numpy (no PyTorch needed).
Weights are NOT kept in the project: download openai-community/gpt2
model.safetensors into a folder outside it and pass the path.

    python ../common/download_gpt2.py          # once: GPT-2 small, 548 MB, into ../models/gpt2/
    python experiment.py ../models/gpt2/model.safetensors

Measures:
  1. the "logit lens": after every layer, what would the model say next
     if it stopped there? (final layer norm + unembedding applied early)
  2. parameters inside one layer: attention vs feed-forward
  3. knock-outs: remove one layer's attention, or its feed-forward, or the
     residual "highway", and see what happens to the answer
  4. how far each layer moves the word's vector
"""

import sys, json, struct
import numpy as np
import tiktoken

sys.stdout.reconfigure(encoding="utf-8")
T = {}
with open(sys.argv[1], "rb") as f:
    hlen = struct.unpack("<Q", f.read(8))[0]
    head = json.loads(f.read(hlen))
    base = 8 + hlen
    for k, m in head.items():
        if k == "__metadata__":
            continue
        s, e = m["data_offsets"]
        f.seek(base + s)
        T[k] = np.frombuffer(f.read(e - s), dtype=np.float32).reshape(m["shape"])

enc = tiktoken.get_encoding("gpt2")


def ln(x, g, b):
    mu = x.mean(-1, keepdims=True)
    v = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(v + 1e-5) * g + b


def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


def attn(L, x):
    h = "h.%d." % L
    n = len(x)
    a = ln(x, T[h + "ln_1.weight"], T[h + "ln_1.bias"])
    q, k, v = np.split(a @ T[h + "attn.c_attn.weight"] + T[h + "attn.c_attn.bias"], 3, axis=-1)
    q, k, v = [t.reshape(n, 12, 64).transpose(1, 0, 2) for t in (q, k, v)]
    s = q @ k.transpose(0, 2, 1) / 8.0 + np.triu(np.full((n, n), -1e10), 1)
    s = np.exp(s - s.max(-1, keepdims=True))
    s /= s.sum(-1, keepdims=True)
    return (s @ v).transpose(1, 0, 2).reshape(n, 768) @ T[h + "attn.c_proj.weight"] + T[h + "attn.c_proj.bias"]


def mlp(L, x):
    h = "h.%d." % L
    m = ln(x, T[h + "ln_2.weight"], T[h + "ln_2.bias"])
    return gelu(m @ T[h + "mlp.c_fc.weight"] + T[h + "mlp.c_fc.bias"]) @ T[h + "mlp.c_proj.weight"] + T[h + "mlp.c_proj.bias"]


def run(ids, skip_attn=(), skip_mlp=(), no_highway=False):
    """Returns the residual stream after the embedding and after every layer."""
    x = T["wte.weight"][ids] + T["wpe.weight"][:len(ids)]
    states = [x]
    for L in range(12):
        a = 0 if L in skip_attn else attn(L, x)
        x = (a if no_highway else x + a)
        m = 0 if L in skip_mlp else mlp(L, x)
        x = (m if no_highway else x + m)
        states.append(x)
    return states


def probs(x_last):
    z = ln(x_last, T["ln_f.weight"], T["ln_f.bias"]) @ T["wte.weight"].T
    z = z.astype(np.float64) - z.max()
    p = np.exp(z)
    return p / p.sum()


def tok(w):
    ids = enc.encode(w)
    assert len(ids) == 1, (w, ids)
    return ids[0]


def show(p, k=3):
    o = np.argsort(-p)[:k]
    return ", ".join("%r %.1f%%" % (enc.decode([int(i)]), 100 * p[i]) for i in o)


PROMPTS = [
    ("The Eiffel Tower is in the city of", " Paris"),
    ("The Statue of Liberty is in New", " York"),
    ("Michael Jordan plays the sport of", " basketball"),
    ("The opposite of hot is", " cold"),
    ("The United States of", " America"),
    ("Barack", " Obama"),
    ("The capital of Japan is", " Tokyo"),
    ("Water freezes at zero degrees", " Celsius"),
]

print("=== 1. final answers ===")
for text, ans in PROMPTS:
    ids = enc.encode(text)
    p = probs(run(ids)[-1][-1])
    t = tok(ans)
    rank = int((p > p[t]).sum()) + 1
    print("%-42r -> %-16r p=%.1f%% rank %d | top: %s" % (text, ans, 100 * p[t], rank, show(p)))

print("\n=== 2. logit lens, layer by layer ===")
for text, ans in PROMPTS:
    ids = enc.encode(text)
    t = tok(ans)
    st = run(ids)
    print("\n%r  (target %r)" % (text, ans))
    for L, x in enumerate(st):
        p = probs(x[-1])
        rank = int((p > p[t]).sum()) + 1
        print("  after %-9s target p=%6.2f%% rank %-6d top: %s"
              % ("embed" if L == 0 else "layer %d" % L, 100 * p[t], rank, show(p, 3)))

print("\n=== 3. parameters ===")
attn_p = sum(T["h.0." + k].size for k in ("attn.c_attn.weight", "attn.c_attn.bias",
                                          "attn.c_proj.weight", "attn.c_proj.bias"))
mlp_p = sum(T["h.0." + k].size for k in ("mlp.c_fc.weight", "mlp.c_fc.bias",
                                         "mlp.c_proj.weight", "mlp.c_proj.bias"))
ln_p = sum(T["h.0." + k].size for k in ("ln_1.weight", "ln_1.bias", "ln_2.weight", "ln_2.bias"))
layer = attn_p + mlp_p + ln_p
total = sum(v.size for v in T.values() if v.dtype == np.float32 and v.ndim > 0
            and not str(v.shape) == "()")
emb = T["wte.weight"].size + T["wpe.weight"].size
print("one layer: attention %d (%.1f%%), feed-forward %d (%.1f%%), norms %d, total %d"
      % (attn_p, 100 * attn_p / layer, mlp_p, 100 * mlp_p / layer, ln_p, layer))
print("12 layers %d, embeddings %d, total (approx, excl. attn.bias buffers) %d" % (12 * layer, emb, 12 * layer + emb + 2 * 768))
print("feed-forward width: %d -> %d -> %d" % (768, T["h.0.mlp.c_fc.weight"].shape[1], 768))

print("\n=== 4. knock-outs ===")
for text, ans in PROMPTS[:4]:
    ids = enc.encode(text)
    t = tok(ans)
    base = probs(run(ids)[-1][-1])[t]
    print("\n%r  baseline %r %.1f%%" % (text, ans, 100 * base))
    for L in range(12):
        pa = probs(run(ids, skip_attn=(L,))[-1][-1])
        pm = probs(run(ids, skip_mlp=(L,))[-1][-1])
        print("  layer %-2d  no attention: %6.2f%% (%s) | no feed-forward: %6.2f%% (%s)"
              % (L + 1, 100 * pa[t], show(pa, 1), 100 * pm[t], show(pm, 1)))
    pn = probs(run(ids, no_highway=True)[-1][-1])
    print("  no highway (x = f(x)): %.4f%%  top: %s" % (100 * pn[t], show(pn, 3)))

print("\n=== 5. how far each layer moves the last word ===")
for text, ans in PROMPTS[:2]:
    st = run(enc.encode(text))
    print("\n%r" % text)
    x0 = st[0][-1]
    for L in range(1, 13):
        prev, cur = st[L - 1][-1], st[L][-1]
        d = np.linalg.norm(cur - prev) / np.linalg.norm(prev)
        c0 = cur @ x0 / np.linalg.norm(cur) / np.linalg.norm(x0)
        cp = cur @ prev / np.linalg.norm(cur) / np.linalg.norm(prev)
        print("  layer %-2d  change %.2f x stream  cos(prev) %.3f  cos(embed) %.3f  norm %.1f"
              % (L, d, cp, c0, np.linalg.norm(cur)))


print("\n=== 6. what the last word looks at (Eiffel prompt) ===")
text = "The Eiffel Tower is in the city of"
ids = enc.encode(text)
print("tokens:", [enc.decode([i]) for i in ids])


def attn_weights(L, x):
    h = "h.%d." % L
    n = len(x)
    a = ln(x, T[h + "ln_1.weight"], T[h + "ln_1.bias"])
    q, k, v = np.split(a @ T[h + "attn.c_attn.weight"] + T[h + "attn.c_attn.bias"], 3, axis=-1)
    q, k = [t.reshape(n, 12, 64).transpose(1, 0, 2) for t in (q, k)]
    s = q @ k.transpose(0, 2, 1) / 8.0 + np.triu(np.full((n, n), -1e10), 1)
    s = np.exp(s - s.max(-1, keepdims=True))
    return s / s.sum(-1, keepdims=True)          # (heads, n, n)


st = run(ids)
words = [enc.decode([i]) for i in ids]
for L in range(12):
    w = attn_weights(L, st[L])[:, -1, :]         # every head, last token's row
    best = np.unravel_index(np.argmax(w[:, 1:]), w[:, 1:].shape)   # ignore token 0 sink
    hd, j = best[0], best[1] + 1
    print("  layer %-2d  strongest head %2d -> %-8r %.0f%%   head-average on tokens: %s"
          % (L + 1, hd, words[j], 100 * w[hd, j],
             " ".join("%s:%.0f" % (t.strip() or "_", 100 * v) for t, v in zip(words, w.mean(0)))))

print("\n=== 7. group knock-outs (Eiffel prompt, Paris baseline) ===")
t = tok(" Paris")
for name, kw in [("attention off, layers 1-12", dict(skip_attn=range(12))),
                 ("feed-forward off, layers 1-12", dict(skip_mlp=range(12))),
                 ("attention off, layers 7-12", dict(skip_attn=range(6, 12))),
                 ("feed-forward off, layers 7-12", dict(skip_mlp=range(6, 12))),
                 ("attention off, layers 2-12", dict(skip_attn=range(1, 12))),
                 ("feed-forward off, layers 2-12", dict(skip_mlp=range(1, 12)))]:
    p = probs(run(ids, **kw)[-1][-1])
    rank = int((p > p[t]).sum()) + 1
    print("  %-32s Paris %.2f%% rank %d   top: %s" % (name, 100 * p[t], rank, show(p, 3)))

print("\n=== 8. same floors, different city ===")
for text2, ans in [("The Colosseum is in the city of", " Rome"),
                   ("Big Ben is in the city of", " London"),
                   ("The Brandenburg Gate is in the city of", " Berlin")]:
    ids2 = enc.encode(text2)
    t2 = tok(ans)
    p = probs(run(ids2)[-1][-1])
    print("  %-42r %r %.1f%% rank %d  top: %s" % (text2, ans, 100 * p[t2], int((p > p[t2]).sum()) + 1, show(p, 3)))
