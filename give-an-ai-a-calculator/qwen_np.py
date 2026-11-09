"""
qwen_np.py - Qwen2.5 Instruct (0.5B / 1.5B) in plain numpy: the forward pass, a KV cache, greedy decoding.

Weights: Qwen/Qwen2.5-0.5B-Instruct (Apache-2.0), https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct
model.safetensors (bfloat16, widened to float32 here) + tokenizer.json, in a folder you choose.
"""
import json, os, struct
import numpy as np
from tokenizers import Tokenizer

T, CFG = {}, {}
tok = None


def load(folder):
    """folder holds model.safetensors, tokenizer.json and config.json."""
    global tok
    CFG.update(json.load(open(os.path.join(folder, "config.json"))))
    with open(os.path.join(folder, "model.safetensors"), "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        head = json.loads(f.read(hlen))
        base = 8 + hlen
        for k, m in head.items():
            if k == "__metadata__":
                continue
            s, e = m["data_offsets"]
            f.seek(base + s)
            raw = f.read(e - s)
            if m["dtype"] == "BF16":         # bfloat16 is the top half of a float32
                a = (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)
            else:
                a = np.frombuffer(raw, dtype=np.float32)
            T[k.replace("model.", "")] = a.reshape(m["shape"])
    tok = Tokenizer.from_file(os.path.join(folder, "tokenizer.json"))
    hd = CFG["hidden_size"] // CFG["num_attention_heads"]
    CFG["head_dim"] = hd
    CFG["inv_freq"] = 1.0 / (CFG["rope_theta"] ** (np.arange(0, hd, 2) / hd))   # float64 angles, cast below


def _rms(x, w):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + np.float32(CFG["rms_norm_eps"])) * w


def _rope(x, pos):
    """x: (heads, n, hd); rotate each half-pair by its position's angle."""
    f = np.outer(pos, CFG["inv_freq"])                     # (n, hd/2)
    cos = np.cos(np.concatenate([f, f], -1)).astype(np.float32)
    sin = np.sin(np.concatenate([f, f], -1)).astype(np.float32)   # keep float32: one float64 array
    # here would widen every weight matrix to float64 on every call
    h = x.shape[-1] // 2
    rot = np.concatenate([-x[..., h:], x[..., :h]], -1)
    return x * cos + rot * sin


def forward(ids, cache):
    """Run new token ids through the model, extending cache (a list of per-layer [k, v]); return last logits."""
    nh, nkv, hd = CFG["num_attention_heads"], CFG["num_key_value_heads"], CFG["head_dim"]
    start = cache[0][0].shape[1] if cache else 0
    n = len(ids)
    pos = np.arange(start, start + n)
    x = T["embed_tokens.weight"][ids].astype(np.float32)
    for L in range(CFG["num_hidden_layers"]):
        p = "layers.%d." % L
        a = _rms(x, T[p + "input_layernorm.weight"])
        q = a @ T[p + "self_attn.q_proj.weight"].T + T[p + "self_attn.q_proj.bias"]
        k = a @ T[p + "self_attn.k_proj.weight"].T + T[p + "self_attn.k_proj.bias"]
        v = a @ T[p + "self_attn.v_proj.weight"].T + T[p + "self_attn.v_proj.bias"]
        q = _rope(q.reshape(n, nh, hd).transpose(1, 0, 2), pos)
        k = _rope(k.reshape(n, nkv, hd).transpose(1, 0, 2), pos)
        v = v.reshape(n, nkv, hd).transpose(1, 0, 2)
        if len(cache) <= L:
            cache.append([k, v])
        else:
            cache[L] = [np.concatenate([cache[L][0], k], 1), np.concatenate([cache[L][1], v], 1)]
        K, V = cache[L]
        rep = nh // nkv                                     # grouped-query: each key/value head serves 7 query heads
        K, V = np.repeat(K, rep, 0), np.repeat(V, rep, 0)
        s = q @ K.transpose(0, 2, 1) / np.float32(np.sqrt(hd))
        tot = K.shape[1]
        s = s + np.triu(np.full((n, tot), -1e9, dtype=np.float32), start + 1)
        s = np.exp(s - s.max(-1, keepdims=True)); s /= s.sum(-1, keepdims=True)
        o = (s @ V).transpose(1, 0, 2).reshape(n, nh * hd)
        x = x + o @ T[p + "self_attn.o_proj.weight"].T
        m = _rms(x, T[p + "post_attention_layernorm.weight"])
        g = m @ T[p + "mlp.gate_proj.weight"].T
        u = m @ T[p + "mlp.up_proj.weight"].T
        x = x + ((g / (1 + np.exp(-g))) * u) @ T[p + "mlp.down_proj.weight"].T
    z = _rms(x[-1], T["norm.weight"]) @ T["embed_tokens.weight"].T   # tied output layer
    return z.astype(np.float64)


def probs(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def chat(user, system="You are a helpful assistant.", n=40):
    """The model's chat format, then greedy decoding to the end-of-turn token.
    Returns (answer, probability of each answer token, the token strings)."""
    prompt = ("<|im_start|>system\n%s<|im_end|>\n<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n"
              % (system, user))
    ids = tok.encode(prompt).ids
    cache = []
    z = forward(ids, cache)
    out, ps = [], []
    end = tok.token_to_id("<|im_end|>")
    for _ in range(n):
        p = probs(z)
        i = int(np.argmax(p))
        if i == end:
            break
        out.append(i); ps.append(float(p[i]))
        z = forward([i], cache)
    return tok.decode(out), ps, [tok.decode([i]) for i in out]


def first_word_scores(user, system="You are a helpful assistant.", top=10):
    """The scoreboard for the answer's first token."""
    prompt = ("<|im_start|>system\n%s<|im_end|>\n<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n"
              % (system, user))
    p = probs(forward(tok.encode(prompt).ids, []))
    o = np.argsort(-p)[:top]
    return [(tok.decode([int(i)]), float(p[i])) for i in o]
