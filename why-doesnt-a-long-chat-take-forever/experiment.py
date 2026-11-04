"""
experiment.py - why a long chat doesn't take forever: the KV cache, measured on a real chat model.

Qwen2.5-0.5B-Instruct (Apache-2.0) in plain numpy (qwen_np.py), on our laptop. One real 50-message chat
about a hike with a dog called Biscuit. The model writes every reply itself (greedy, so a re-run gives the
same words). For each turn we time:
  - the reply as chat servers make it, keeping every word's key and value (the KV cache), and
  - one full re-read of the chat so far: what EVERY new word would cost without the cache.
Then: what the notes cost in memory (this model, and two bigger ones from their config files), whether the
cache changes the answer (it must not), and whether the model still knows the dog's name - with the whole
chat, with the first message dropped, and with a book pasted in between.

  python experiment.py --models C:/models/hf --out out
"""
import argparse, glob, json, os, sys, time
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--models", default=os.path.expanduser("~/models/hf"), help="Hugging Face cache folder")
ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"))
ap.add_argument("--book", default=None, help="a long plain-text book (we used Alice in Wonderland, Project Gutenberg #11)")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qwen_np as Q
from huggingface_hub import snapshot_download, hf_hub_download

Q.load(snapshot_download("Qwen/Qwen2.5-0.5B-Instruct", cache_dir=a.models,
                         allow_patterns=["config.json", "tokenizer.json", "model.safetensors"]))
END = Q.tok.token_to_id("<|im_end|>")
SYSTEM = "You are a helpful assistant. Keep every answer under 50 words."
USER = [
    "Hi! I'm planning a hike in the Lake District with my dog, Biscuit.",
    "Which month has the best weather there?",
    "We'd like a walk of about 10 miles. Any ideas?",
    "Is that route steep?",
    "Where can we park near the start?",
    "Are dogs allowed on that path?",
    "What should I pack for the day?",
    "How much water should I carry?",
    "Should I bring water for the dog too?",
    "What if it rains halfway?",
    "Are there places to eat along the way?",
    "How early should we set off?",
    "What boots do you recommend?",
    "Is there phone signal on the hills?",
    "What should I do if we get lost?",
    "Are there sheep on the fells?",
    "How do I keep the dog calm around sheep?",
    "Can we swim in the lakes?",
    "What's a good spot for a photo?",
    "How long does the walk usually take?",
    "Is it busy at weekends?",
    "Where could we stay the night?",
    "Are pubs there dog friendly?",
    "Any tips for the drive back?",
]
RECALL = ["What is my dog's name?", "What did I say my dog is called?", "Remind me, what's my dog called?"]


def turn(role, text):
    return "<|im_start|>%s\n%s<|im_end|>\n" % (role, text)


def head():
    return turn("system", SYSTEM)


def feed(ids, cache, chunk=512):
    """Read ids into the cache (in chunks, to bound memory); return the last logits."""
    z = None
    for i in range(0, len(ids), chunk):
        z = Q.forward(ids[i:i + chunk], cache)
    return z


def reply(cache, z, n=90):
    out = []
    for _ in range(n):
        i = int(np.argmax(z))
        if i == END:
            break
        out.append(i)
        z = Q.forward([i], cache)
    return out


def mb(cache):
    return sum(k.nbytes + v.nbytes for k, v in cache) / 1e6


# 1. the chat, on one running cache (what a chat server keeps between your messages)
res = {"model": "Qwen/Qwen2.5-0.5B-Instruct", "system": SYSTEM, "turns": []}
cache, text = [], head()
feed(Q.tok.encode(text).ids, cache)
for u in USER:
    new = turn("user", u) + "<|im_start|>assistant\n"
    text += new
    t0 = time.perf_counter()
    z = feed(Q.tok.encode(new).ids, cache)
    out = reply(cache, z)
    t_cache = time.perf_counter() - t0
    ans = Q.tok.decode(out)
    Q.forward([END], cache)                                   # the end-of-turn token is part of the chat
    text += ans + "<|im_end|>\n"
    n_ctx = cache[0][0].shape[1]
    t0 = time.perf_counter()
    feed(Q.tok.encode(text).ids, [])                          # one full re-read = one word without the cache
    t_reread = time.perf_counter() - t0
    row = {"user": u, "reply": ans, "reply_words": len(out), "chat_tokens": n_ctx,
           "secs_reply_with_cache": round(t_cache, 2), "secs_per_word_without_cache": round(t_reread, 2),
           "secs_reply_without_cache_est": round(t_reread * len(out), 1), "cache_mb_fp32": round(mb(cache), 1)}
    res["turns"].append(row)
    print("turn %2d  %5d tok  reply %2d words  cache %.1fs  no-cache %.1fs/word (~%.0fs)  %s"
          % (len(res["turns"]), n_ctx, len(out), t_cache, t_reread, t_reread * len(out), ans[:60].replace("\n", " ")),
          flush=True)
    json.dump(res, open(os.path.join(a.out, "results.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)

# 2. does the cache change anything? same logits both ways
ids = Q.tok.encode(text + turn("user", RECALL[0]) + "<|im_start|>assistant\n").ids
c = []
feed(ids[:-40], c)
z1 = feed(ids[-40:], c)
z2 = feed(ids, [], chunk=len(ids))
res["cache_vs_reread_max_logit_diff"] = float(np.abs(z1 - z2).max())
res["same_top_word"] = bool(np.argmax(z1) == np.argmax(z2))
print("cache vs re-read: max logit diff", res["cache_vs_reread_max_logit_diff"], "same top word", res["same_top_word"])


# 3. does it still know the dog's name?
def ask(chat_text, q):
    ids = Q.tok.encode(chat_text + turn("user", q) + "<|im_start|>assistant\n").ids
    c = []
    return Q.tok.decode(reply(c, feed(ids, c), n=30)), len(ids)


full = text
first = turn("user", USER[0]) + turn("assistant", res["turns"][0]["reply"])
dropped = full.replace(first, "", 1)
res["recall"] = {}
for lab, t in (("whole chat", full), ("first message dropped", dropped)):
    for q in RECALL:
        ans, n = ask(t, q)
        res["recall"]["%s | %s" % (lab, q)] = {"answer": ans, "tokens": n, "says_biscuit": "Biscuit" in ans}
        print("recall | %-22s %5d tok | %s -> %s" % (lab, n, q, ans.replace("\n", " ")), flush=True)
if a.book:
    book = open(a.book, encoding="utf-8", errors="replace").read()
    for chars in (2000, 6000, 12000):
        t = head() + first + turn("user", "Here is part of a book I'm reading on the trip:\n" + book[:chars]) \
            + turn("assistant", "Thanks for sharing!")
        ans, n = ask(t, RECALL[0])
        res["recall"]["book %d chars" % chars] = {"answer": ans, "tokens": n, "says_biscuit": "Biscuit" in ans}
        print("recall | book %5d chars    %5d tok -> %s" % (chars, n, ans.replace("\n", " ")), flush=True)

# 4. what one word's notes cost: layers x key/value heads x head size x 2 (key and value) x 2 bytes (bf16)
res["notes_per_word"] = {}
for rid in ("Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-72B-Instruct"):
    cfg = json.load(open(hf_hub_download(rid, "config.json", cache_dir=a.models)))
    hd = cfg["hidden_size"] // cfg["num_attention_heads"]
    b = cfg["num_hidden_layers"] * cfg["num_key_value_heads"] * hd * 2 * 2
    win = cfg.get("max_position_embeddings")
    res["notes_per_word"][rid] = {"layers": cfg["num_hidden_layers"], "kv_heads": cfg["num_key_value_heads"],
                                  "head_dim": hd, "bytes_per_token_bf16": b, "config_window": win,
                                  "gb_at_32768": round(b * 32768 / 1e9, 2)}
    print(rid, b, "bytes/token;", round(b * 32768 / 1e9, 2), "GB at 32,768 tokens")

json.dump(res, open(os.path.join(a.out, "results.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("DONE")
