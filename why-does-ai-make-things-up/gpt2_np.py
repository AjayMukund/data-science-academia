"""
gpt2_np.py - GPT-2 small (124M) in plain numpy: the forward pass and nothing else.

Weights: openai-community/gpt2 model.safetensors, downloaded once to a folder you choose
(https://huggingface.co/openai-community/gpt2). Tokenizer: tiktoken's "gpt2".
"""
import json, struct
import numpy as np
import tiktoken

enc = tiktoken.get_encoding("gpt2")
T = {}


def load(path):
    with open(path, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        head = json.loads(f.read(hlen))
        base = 8 + hlen
        for k, m in head.items():
            if k == "__metadata__":
                continue
            s, e = m["data_offsets"]
            f.seek(base + s)
            T[k] = np.frombuffer(f.read(e - s), dtype=np.float32).reshape(m["shape"])


def _ln(x, g, b):
    mu = x.mean(-1, keepdims=True)
    v = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(v + 1e-5) * g + b


def _gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


def _attn(L, x):
    h = "h.%d." % L
    n = len(x)
    a = _ln(x, T[h + "ln_1.weight"], T[h + "ln_1.bias"])
    q, k, v = np.split(a @ T[h + "attn.c_attn.weight"] + T[h + "attn.c_attn.bias"], 3, axis=-1)
    q, k, v = [t.reshape(n, 12, 64).transpose(1, 0, 2) for t in (q, k, v)]
    s = q @ k.transpose(0, 2, 1) / 8.0 + np.triu(np.full((n, n), -1e10), 1)
    s = np.exp(s - s.max(-1, keepdims=True))
    s /= s.sum(-1, keepdims=True)
    return (s @ v).transpose(1, 0, 2).reshape(n, 768) @ T[h + "attn.c_proj.weight"] + T[h + "attn.c_proj.bias"]


def _mlp(L, x):
    h = "h.%d." % L
    m = _ln(x, T[h + "ln_2.weight"], T[h + "ln_2.bias"])
    return _gelu(m @ T[h + "mlp.c_fc.weight"] + T[h + "mlp.c_fc.bias"]) @ T[h + "mlp.c_proj.weight"] + T[h + "mlp.c_proj.bias"]


def next_probs(ids):
    """Probability of every one of the 50,257 tokens coming next."""
    x = T["wte.weight"][ids] + T["wpe.weight"][:len(ids)]
    for L in range(12):
        x = x + _attn(L, x)
        x = x + _mlp(L, x)
    z = _ln(x[-1], T["ln_f.weight"], T["ln_f.bias"]) @ T["wte.weight"].T
    z = z.astype(np.float64) - z.max()
    p = np.exp(z)
    return p / p.sum()


def greedy(text, n=6):
    """Append the most likely token n times; return the continuation and each step's probability."""
    ids = enc.encode(text)
    out, ps = [], []
    for _ in range(n):
        p = next_probs(ids)
        i = int(np.argmax(p))
        out.append(i); ps.append(float(p[i])); ids.append(i)
    return enc.decode(out), ps
