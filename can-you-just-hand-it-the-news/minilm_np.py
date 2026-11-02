"""all-MiniLM-L6-v2 in plain numpy (torch/onnxruntime DLLs fail on this box)."""
import os, sys, numpy as np
from scipy.special import erf
from huggingface_hub import hf_hub_download
from safetensors.numpy import load_file
from tokenizers import Tokenizer
D = os.environ.get("MINILM_CACHE", os.path.join(os.path.expanduser("~"), "models", "minilm"))
REPO = "sentence-transformers/all-MiniLM-L6-v2"
W = load_file(hf_hub_download(REPO, "model.safetensors", cache_dir=D))
W = {k.replace("bert.", ""): v.astype(np.float64) for k, v in W.items()}
TOK = Tokenizer.from_file(hf_hub_download(REPO, "tokenizer.json", cache_dir=D))
TOK.no_padding(); TOK.no_truncation()
H, NH = 384, 12

def ln(x, p, eps=1e-12):
    m = x.mean(-1, keepdims=True); v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps) * W[p + ".weight"] + W[p + ".bias"]
def lin(x, p): return x @ W[p + ".weight"].T + W[p + ".bias"]

def embed(text):
    ids = np.array(TOK.encode(text).ids)
    n = len(ids)
    x = (W["embeddings.word_embeddings.weight"][ids]
         + W["embeddings.position_embeddings.weight"][:n]
         + W["embeddings.token_type_embeddings.weight"][0])
    x = ln(x, "embeddings.LayerNorm")
    for L in range(6):
        p = "encoder.layer.%d." % L
        q, k, v = (lin(x, p + "attention.self." + s).reshape(n, NH, -1).transpose(1, 0, 2)
                   for s in ("query", "key", "value"))
        a = q @ k.transpose(0, 2, 1) / np.sqrt(H // NH)
        a = np.exp(a - a.max(-1, keepdims=True)); a /= a.sum(-1, keepdims=True)
        c = (a @ v).transpose(1, 0, 2).reshape(n, H)
        x = ln(x + lin(c, p + "attention.output.dense"), p + "attention.output.LayerNorm")
        h = lin(x, p + "intermediate.dense"); h = 0.5 * h * (1 + erf(h / np.sqrt(2)))
        x = ln(x + lin(h, p + "output.dense"), p + "output.LayerNorm")
    e = x.mean(0)
    return e / np.linalg.norm(e)

def cos(a, b): return float(embed(a) @ embed(b))

if __name__ == "__main__":
    s = ["The weather is lovely today.", "It's so sunny outside!", "He drove to the stadium."]
    e = [embed(t) for t in s]
    print("validation (docs publish 0.6660, 0.1046, 0.1411):",
          round(float(e[0] @ e[1]), 4), round(float(e[0] @ e[2]), 4), round(float(e[1] @ e[2]), 4))
