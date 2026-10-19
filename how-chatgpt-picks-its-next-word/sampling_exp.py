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
def probs(text, T=1.0):
    x = forward(enc.encode(text)); logits = ln(x, T["ln_f.weight"] if False else TT["ln_f.weight"], TT["ln_f.bias"])[-1] @ TT["wte.weight"].T
    return logits
TT = T
def dist(text):
    x = forward(enc.encode(text)); return (ln(x, TT["ln_f.weight"], TT["ln_f.bias"])[-1] @ TT["wte.weight"].T).astype(np.float64)
def soft(z, t=1.0):
    z = z / t; z = z - z.max(); p = np.exp(z); return p / p.sum()
P = "The cat sat on the"
z = dist(P); p = soft(z); o = np.argsort(-p)
print("vocab", len(p))
print("top 10 after %r:" % P)
for i in o[:10]: print("   %-12r logit %6.2f  p %.4f" % (enc.decode([i]), z[i], p[i]))
for w in [" mat", " floor", " roof", " windowsill", " carousel"]:
    ids = enc.encode(w); 
    if len(ids)==1:
        r = int(np.where(o == ids[0])[0][0]) + 1
        print("   %-12r rank %5d  p %.5f  logit %.2f" % (w, r, p[ids[0]], z[ids[0]]))
    else: print("   %r is %d tokens" % (w, len(ids)))
for t in (0.5, 1.0, 1.5):
    q = soft(z, t); print("T=%.1f  top1 %r %.3f   top10 mass %.3f" % (t, enc.decode([o[0]]), q[o[0]], q[o[:10]].sum()))
cs = np.cumsum(p[o])
for tp in (0.5, 0.9, 0.95): print("top-p %.2f keeps %d tokens" % (tp, int(np.searchsorted(cs, tp)) + 1))
print("top-k 40 keeps %.3f of the probability" % cs[39])
# greedy continuation
ids = enc.encode(P)
for _ in range(25):
    ids.append(int(np.argmax(dist(enc.decode(ids)))))
print("
GREEDY:", repr(enc.decode(ids)))
rng = np.random.default_rng(7)
for k in range(3):
    ids = enc.encode(P)
    for _ in range(10):
        q = soft(dist(enc.decode(ids)), 1.0); ids.append(int(rng.choice(len(q), p=q)))
    print("SAMPLE %d:" % k, repr(enc.decode(ids)))
