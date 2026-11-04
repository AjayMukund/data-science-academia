# KV-cache master (11-07): why doesn't a long chat take forever? — measured findings (2026-11-03)

**Model:** Qwen2.5-0.5B-Instruct (Apache-2.0, Sept 2024), run by us in plain numpy (`qwen_np.py`) on our
laptop (8 cores, 16 GB). Greedy decoding, so a re-run writes the same words.
`python experiment.py --models <hf cache> --book <alice.txt>` writes `out/results.json` (log: `out_run.log`).

**The chat:** 24 user messages about a Lake District hike "with my dog, Biscuit" (message 1), each answered
by the model itself (system prompt "Keep every answer under 50 words", replies capped at 90 tokens).
48 messages in all, then "what's my dog called?".

## 1. Speed: with the cache vs re-reading everything

To pick each next token the model must look back at every token so far. Two ways to do it:
- **re-read**: run the whole chat through all 24 layers again, for every new token;
- **KV cache**: keep each token's key and value from every layer (its "notes"), so a new token is the only
  thing that runs through the layers, and it looks up everyone else's notes.

| | chat so far | reply | with the cache | re-read, per token | re-read, whole reply (est.) |
|---|---|---|---|---|---|
| turn 1 | 126 tokens | 78 tokens | 12.7 s (0.16 s/token) | 0.86 s | 67 s |
| turn 24 | 1,916 tokens | 48 tokens | 15.9 s (0.33 s/token) | **20.9 s** | **1,005 s ≈ 17 min** |

- Whole 24-turn chat: **6.2 minutes** with the cache, **about 4 hours** re-reading (39x).
- At turn 24 one token costs 20.9 s re-read vs 0.33 s with the cache (**63x**).
- With the cache a token still gets slower as the chat grows (0.16 → 0.33 s): it has more notes to look up.
- The re-read figure is measured (one full pass over the chat at that turn); the whole-reply figure is that
  pass times the reply's tokens (it would grow slightly with each token: an underestimate).
- **Same answer both ways:** max logit difference 0.00003, same top token. The cache is not an approximation.
- First probe (non-chat text): 4,000 tokens re-read 50.8 s vs 0.44 s per token with the cache (115x).

## 2. Memory: what the notes cost

Per token: layers × key/value heads × head size × 2 (a key and a value) × 2 bytes (bf16).

| model | layers | kv heads | head size | per token | at 32,768 tokens |
|---|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct | 24 | 2 | 64 | 12,288 B (12 KB) | 0.40 GB |
| Qwen2.5-7B-Instruct | 28 | 4 | 128 | 57,344 B (56 KB) | 1.88 GB |
| Qwen2.5-72B-Instruct | 80 | 8 | 128 | 327,680 B (320 KB) | 10.74 GB |

- From each model's own `config.json`; 32,768 is the `max_position_embeddings` in all three configs.
- At turn 24 our chat's notes: 1,916 × 12,288 B = **23.5 MB** (47.1 MB in our float32 copy).
- The chat grew ~78 tokens a turn: 32,768 tokens is **about 410 turns** of this chat.
- The 0.5B model's own weights are ~1 GB in bf16, so a full window of notes is 40% of the model again.

## 3. Does it still know the dog's name?

Same chat, three phrasings, whole chat vs the first message (the Biscuit message and its reply) dropped,
as an app must drop something when a chat outgrows the window:

| question | whole chat (1,955 tokens) | first message dropped |
|---|---|---|
| "What did I say my dog is called?" | **"Your dog is called Biscuit."** | **"Your dog is called a "poodle.""** |
| "Remind me, what's my dog called?" | **"Your dog is named Biscuit."** | **"Your dog is called "Buddy.""** |
| "What is my dog's name?" | "Your dog's name is important for identification…" (missed) | "Your dog's name is your responsibility…" |

- With the note: 2 of 3 phrasings say Biscuit; in shorter chats (1, 4, 12 turns) all 3 do (probe, scratch).
- Note dropped: 0 of 3, and 2 of 3 **invent** a name (Buddy, a poodle): the hallucination master's
  behaviour, now caused by forgetting.
- With a book pasted between the note and the question (2,000 / 6,000 / 12,000 characters, up to 3,164
  tokens): Biscuit every time. A 2,600-token paste in an earlier probe with no length rule in the system
  prompt missed it: small models do not always use what is in their notes.

## Cautions for the script
- "Token", not "word" on screen (a token is a word or a piece of one: the tokenization master).
- Timings are "on our laptop, in numpy": real servers use GPUs and are far faster; the RATIO is the point.
- The model invented a trail name ("Waynflete Loop") in its replies: never show the chat's route as fact.
- What a given app drops (oldest messages, a summary) varies and is not public for most: say "has to drop
  something"; we dropped the first message.
- Real chat services also reuse notes across turns (prompt caching); our run kept one cache for the whole chat.
