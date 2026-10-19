"""
pipeline.py - extra measurements for the architecture-walkthrough rework.
Imports experiment.py's model (pass the weights path the same way).

    python pipeline.py ../models/gpt2/model.safetensors
"""
import sys, io, contextlib, numpy as np
sys.argv = sys.argv[:2]
class _Sink(io.StringIO):
    def reconfigure(self, **kw):
        pass


sys.stdout.reconfigure(encoding="utf-8")
with contextlib.redirect_stdout(_Sink()):
    import experiment as E          # runs its own report silently

enc, T = E.enc, E.T
text = "The Eiffel Tower is in the city of"
ids = enc.encode(text)
toks = [enc.decode([i]) for i in ids]
print("tokens:", list(zip(toks, ids)))

print("\n=== embedding: the first 6 of 768 numbers ===")
for t, i in zip(toks, ids):
    print("  %-8r id %-6d %s" % (t, i, " ".join("%+.2f" % v for v in T["wte.weight"][i][:6])))
print("  repeated tokens in this prompt:", sorted({t for t in ids if ids.count(t) > 1}) or "none")
print("position 0 vector first 6:", " ".join("%+.2f" % v for v in T["wpe.weight"][0][:6]))

print("\n=== what each word attends to (head average, layers 1-2, ignoring token 0) ===")
st = E.run(ids)
from experiment import attn_weights
for L in range(3):
    pass
    w = attn_weights(L, st[L])
    for q in range(1, len(ids)):
        row = w[:, q, :q + 1].mean(0)
        row[0] = 0
        j = int(np.argmax(row))
        best_head = int(np.argmax(w[:, q, j]))
        print("  layer %d  %-8r -> %-8r head-avg %.0f%%  best head %d: %.0f%%"
              % (L + 1, toks[q], toks[j], 100 * row[j], best_head, 100 * w[best_head, q, j]))

print("\n=== the output step ===")
p = E.probs(st[-1][-1])
o = np.argsort(-p)[:8]
print("vocab", len(p))
for i in o:
    print("  %-12r %.2f%%" % (enc.decode([int(i)]), 100 * p[i]))

print("\n=== generate: append the top word, run again ===")
cur = list(ids)
for step in range(6):
    p = E.probs(E.run(cur)[-1][-1])
    nxt = int(np.argmax(p))
    print("  step %d: %r (%.1f%%)" % (step + 1, enc.decode([nxt]), 100 * p[nxt]))
    cur.append(nxt)
print("  text:", enc.decode(cur))

print("\n=== attention arithmetic, layer 3 head 4, query = ' of' ===")
L, H = 2, 4
x = E.run(ids)[L]
h = "h.%d." % L
a = E.ln(x, T[h + "ln_1.weight"], T[h + "ln_1.bias"])
q, k, v = np.split(a @ T[h + "attn.c_attn.weight"] + T[h + "attn.c_attn.bias"], 3, axis=-1)
q, k = q[:, H * 64:(H + 1) * 64], k[:, H * 64:(H + 1) * 64]
s = k @ q[-1]
p = np.exp(s / 8 - (s / 8).max()); p /= p.sum()
for t, sc, pp in zip(toks, s, p):
    print("  %-8r  q.k %8.1f   /8 %7.2f   softmax %5.1f%%" % (t, sc, sc / 8, 100 * pp))

print("\n=== layer 3, two heads side by side (every query's top key, head avg excluded) ===")
w = attn_weights(2, E.run(ids)[2])
for H in (2, 4):
    print("  head %d:" % H, ", ".join("%s->%s %.0f%%" % (toks[qi].strip(), toks[int(np.argmax(w[H, qi, 1:qi + 1])) + 1].strip() if qi > 0 else "-", 100 * w[H, qi, 1:qi + 1].max() if qi > 0 else 0) for qi in range(1, len(ids))))
