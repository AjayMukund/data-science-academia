import sys, json, struct, numpy as np, tiktoken
sys.stdout.reconfigure(encoding="utf-8")
T = {}
with open(sys.argv[1], "rb") as f:
    hlen = struct.unpack("<Q", f.read(8))[0]; head = json.loads(f.read(hlen)); base = 8 + hlen
    for k, m in head.items():
        if k == "__metadata__": continue
        s, e = m["data_offsets"]; f.seek(base + s)
        T[k] = np.frombuffer(f.read(e - s), dtype=np.float32).reshape(m["shape"])
def ln(x, g, b): mu = x.mean(-1, keepdims=True); v = ((x-mu)**2).mean(-1, keepdims=True); return (x-mu)/np.sqrt(v+1e-5)*g + b
def gelu(x): return 0.5*x*(1+np.tanh(np.sqrt(2/np.pi)*(x+0.044715*x**3)))
def forward(ids, positions=True, causal=True):
    n = len(ids); x = T["wte.weight"][ids] + (T["wpe.weight"][:n] if positions else 0)
    for L in range(12):
        h = "h.%d." % L
        a = ln(x, T[h+"ln_1.weight"], T[h+"ln_1.bias"])
        q, k, v = np.split(a @ T[h+"attn.c_attn.weight"] + T[h+"attn.c_attn.bias"], 3, axis=-1)
        q, k, v = [t.reshape(n, 12, 64).transpose(1, 0, 2) for t in (q, k, v)]
        att = q @ k.transpose(0, 2, 1) / 8.0
        if causal: att = att + np.triu(np.full((n, n), -1e10), 1)
        att = np.exp(att - att.max(-1, keepdims=True)); att /= att.sum(-1, keepdims=True)
        x = x + (att @ v).transpose(1, 0, 2).reshape(n, 768) @ T[h+"attn.c_proj.weight"] + T[h+"attn.c_proj.bias"]
        m = ln(x, T[h+"ln_2.weight"], T[h+"ln_2.bias"])
        x = x + gelu(m @ T[h+"mlp.c_fc.weight"] + T[h+"mlp.c_fc.bias"]) @ T[h+"mlp.c_proj.weight"] + T[h+"mlp.c_proj.bias"]
    return x
enc = tiktoken.get_encoding("gpt2")
def dist(text):
    x = forward(enc.encode(text))
    return (ln(x, T["ln_f.weight"], T["ln_f.bias"])[-1] @ T["wte.weight"].T).astype(np.float64)
def soft(z, t=1.0):
    z = z / t; z = z - z.max(); p = np.exp(z); return p / p.sum()
for P in ["The United States of", "Barack", "1, 2, 3, 4,", "Once upon a", "Thank you very"]:
    z = dist(P); p = soft(z); o = np.argsort(-p); cs = np.cumsum(p[o])
    print("%-28r top: %s" % (P, ", ".join("%r %.3f" % (enc.decode([i]), p[i]) for i in o[:3])))
    print("      top-p 0.9 keeps %d tokens;  top-k 40 holds %.3f" % (int(np.searchsorted(cs, 0.9)) + 1, cs[39]))
