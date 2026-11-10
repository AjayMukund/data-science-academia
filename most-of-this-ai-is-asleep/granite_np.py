"""
granite_np.py - IBM Granite 3.0 1B-A400M (a mixture-of-experts model) in plain numpy.

Weights: ibm-granite/granite-3.0-1b-a400m-base (Apache-2.0), https://huggingface.co/ibm-granite/granite-3.0-1b-a400m-base
24 layers; each has attention plus 32 "experts" (small feed-forward networks). For every token, a router scores
all 32 and only the top 8 run; their outputs are mixed by softmax weights over those 8 scores.

route_k overrides how many experts run (the model was trained with 8); TRACE records which experts each token used.
"""
import json, os, struct
import numpy as np
from tokenizers import Tokenizer

T, CFG = {}, {}
tok = None
TRACE = []                                   # per forward call: list over layers of (n_tokens, k) expert ids
GATES = []                                   # per layer: (router scores (n, 32), chosen ids, blend weights)


def _read(path, want=None):
    with open(path, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        head = json.loads(f.read(hlen))
        base = 8 + hlen
        for k, m in head.items():
            if k == "__metadata__":
                continue
            s, e = m["data_offsets"]
            f.seek(base + s)
            raw = f.read(e - s)
            if m["dtype"] == "BF16":
                a = (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)
            elif m["dtype"] == "F32":
                a = np.frombuffer(raw, dtype=np.float32)
            else:
                raise ValueError(m["dtype"])
            T[k.replace("model.", "")] = a.reshape(m["shape"])


def load(folder):
    global tok
    CFG.update(json.load(open(os.path.join(folder, "config.json"))))
    for fn in sorted(os.listdir(folder)):
        if fn.endswith(".safetensors"):
            _read(os.path.join(folder, fn))
    tok = Tokenizer.from_file(os.path.join(folder, "tokenizer.json"))
    hd = CFG["hidden_size"] // CFG["num_attention_heads"]
    CFG["head_dim"] = hd
    CFG["inv_freq"] = 1.0 / (CFG["rope_theta"] ** (np.arange(0, hd, 2) / hd))


def params():
    """total parameters, and how many one token uses with k experts per layer."""
    L, E = CFG["num_hidden_layers"], CFG["num_local_experts"]
    expert = T["layers.0.block_sparse_moe.input_linear.weight"][0].size + T["layers.0.block_sparse_moe.output_linear.weight"][0].size
    total = sum(v.size for k, v in T.items() if k != "lm_head.weight")          # head is tied to the embedding
    return total, expert, L, E


def _rms(x, w):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + np.float32(CFG["rms_norm_eps"])) * w


def _rope(x, pos):
    f = np.outer(pos, CFG["inv_freq"])
    cos = np.cos(np.concatenate([f, f], -1)).astype(np.float32)
    sin = np.sin(np.concatenate([f, f], -1)).astype(np.float32)
    h = x.shape[-1] // 2
    return x * cos + np.concatenate([-x[..., h:], x[..., :h]], -1) * sin


def _moe(x, L, k, trace):
    p = "layers.%d.block_sparse_moe." % L
    logits = x @ T[p + "router.layer.weight"].T                       # (n, 32)
    top = np.argsort(-logits, -1)[:, :k]                               # the k best-scoring experts per token
    tl = np.take_along_axis(logits, top, -1)
    g = np.exp(tl - tl.max(-1, keepdims=True))
    g /= g.sum(-1, keepdims=True)                                      # softmax over the chosen k only
    out = np.zeros_like(x)
    Wi, Wo = T[p + "input_linear.weight"], T[p + "output_linear.weight"]
    I = CFG["intermediate_size"]
    for e in np.unique(top):
        rows, slot = np.where(top == e)
        h = x[rows] @ Wi[e].T                                          # (m, 2I): gate half, up half
        gt = h[:, :I]
        h = gt * 0.5 * (1 + np.tanh(gt / 2)) * h[:, I:]                 # SiLU(gate) * up (tanh form: no overflow)
        out[rows] += g[rows, slot][:, None] * (h @ Wo[e].T)
    trace.append(top)
    GATES.append((logits, top, g))
    return out


def forward(ids, cache, route_k=None, every=False):
    """Run new token ids, extending cache (list of per-layer [k, v]); return the last token's logits
    (every=True: the logits at every position, shape (n, vocab))."""
    nh, nkv, hd = CFG["num_attention_heads"], CFG["num_key_value_heads"], CFG["head_dim"]
    k_exp = route_k or CFG["num_experts_per_tok"]
    rm = np.float32(CFG["residual_multiplier"])
    start = cache[0][0].shape[1] if cache else 0
    n = len(ids)
    pos = np.arange(start, start + n)
    x = T["embed_tokens.weight"][ids].astype(np.float32) * np.float32(CFG["embedding_multiplier"])
    trace = []
    for L in range(CFG["num_hidden_layers"]):
        p = "layers.%d." % L
        a = _rms(x, T[p + "input_layernorm.weight"])
        q = _rope((a @ T[p + "self_attn.q_proj.weight"].T).reshape(n, nh, hd).transpose(1, 0, 2), pos)
        kk = _rope((a @ T[p + "self_attn.k_proj.weight"].T).reshape(n, nkv, hd).transpose(1, 0, 2), pos)
        v = (a @ T[p + "self_attn.v_proj.weight"].T).reshape(n, nkv, hd).transpose(1, 0, 2)
        if len(cache) <= L:
            cache.append([kk, v])
        else:
            cache[L] = [np.concatenate([cache[L][0], kk], 1), np.concatenate([cache[L][1], v], 1)]
        K, V = cache[L]
        rep = nh // nkv
        K, V = np.repeat(K, rep, 0), np.repeat(V, rep, 0)
        s = q @ K.transpose(0, 2, 1) * np.float32(CFG["attention_multiplier"])
        s = s + np.triu(np.full((n, K.shape[1]), -1e9, dtype=np.float32), start + 1)
        s = np.exp(s - s.max(-1, keepdims=True))
        s /= s.sum(-1, keepdims=True)
        o = (s @ V).transpose(1, 0, 2).reshape(n, nh * hd)
        x = x + (o @ T[p + "self_attn.o_proj.weight"].T) * rm
        x = x + _moe(_rms(x, T[p + "post_attention_layernorm.weight"]), L, k_exp, trace) * rm
    TRACE.append(trace)
    x = _rms(x if every else x[-1:], T["norm.weight"])
    lg = (x @ T["embed_tokens.weight"].T) / np.float32(CFG["logits_scaling"])
    return lg if every else lg[0]


def generate(prompt, max_new=12, route_k=None, stop="\n"):
    ids = tok.encode(prompt).ids
    cache = []
    logits = forward(ids, cache, route_k)
    out = []
    for _ in range(max_new):
        nxt = int(np.argmax(logits))
        if nxt == CFG["eos_token_id"]:
            break
        out.append(nxt)
        text = tok.decode(out)
        if stop and stop in text:
            break
        logits = forward([nxt], cache, route_k)
    return tok.decode(out)
