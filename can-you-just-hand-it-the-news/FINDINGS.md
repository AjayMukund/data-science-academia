# RAG master (11-06): giving a model a library — measured findings (2026-10-29)

**Reader:** Qwen2.5-0.5B-Instruct (Alibaba, released September 2024, Apache-2.0), run by us in plain numpy
(`qwen_np.py`), greedy decoding, its own chat format.
**Search:** all-MiniLM-L6-v2 (22,713,728 numbers; 384 per text), in plain numpy (`minilm_np.py`),
cosine similarity. The port reproduces the model card's example exactly (0.6660 / 0.1046 / 0.1411).
**Port check (Qwen):** torch and llama.cpp do not load on this machine, so the port was checked two ways:
the KV-cache path matches a full recompute (max logit difference 7e-5), and its perplexity on a real
Wikipedia paragraph is 7.0 against 236 for the same words shuffled. It writes fluent, coherent answers.
**Library:** the opening section of 24 Wikipedia pages (CC BY-SA 4.0), read at pinned revisions
(`REVS` in experiment.py): 12 that hold the answers and 12 look-alikes (the edition before), cut into
**106 chunks** of about three sentences (median 66 words).

**Sureness** here = the probability the model gave the first word of the *name* in its answer
("South" in "won by the South Korean team"), not the answer's first word ("The").

## The running example: "Who won the 2026 FIFA World Cup?"
(Spain won, beating defending champions Argentina in the final, 19 July 2026.)

| | Its answer | Sure of the name |
|---|---|---|
| No library | "won by the South Korean team ... defeated the United States in the final, which was played in Qatar" | 29% |
| Top 3 chunks pasted in | "Spain won the 2026 FIFA World Cup." | **89%** |
| Only the 2022 World Cup page (look-alike) | "The 2026 FIFA World Cup was won by Argentina." | **88%** |

Search scores for the question (cosine similarity):
- 2026 FIFA World Cup, chunk 1 ("...concluded on July 19 with Spain winning the championship for the second time"): **0.73**
- 2022 FIFA World Cup, chunk 1: **0.61**
- 2023 FIFA Club World Cup, chunk 1: **0.57**

The look-alike's chunk 3 says "Argentina were crowned the champions after winning the final against
the title holder France 4–2 on penalties".

## All 12 questions (2025–2026 events)

1. **No library: 0 of 12 right.** 4 refusals (it called a pope's election and the Oscars "political
   matters"), 8 confident inventions: South Korea (World Cup), Real Madrid 4-1 AC Milan (Champions
   League), Malala Yousafzai (Nobel Peace), Novak Djokovic (Wimbledon), New England Patriots 34-21
   (Super Bowl LIX), Lewis Hamilton (F1), South Africa (Champions Trophy), Real Madrid (Club World Cup).
2. **Search:** the chunk holding the answer was the top match **7 of 12** times, in the top 3 **8 of 12**.
3. **Top 3 chunks pasted in: 5 of 12 right** (Spain, Paris Saint-Germain, María Corina Machado,
   Jannik Sinner, the Eagles).
   - Answer on the desk (8): read right 5. Misread 3: "Pope Francis was elected" (2025);
     "Pakistan" (the 2017 look-alike was also in the top 3); "Lewis Hamilton".
   - Answer not on the desk (4): it answered anyway **4 of 4**, never "I don't know":
     Chennai Super Kings (IPL), 20 Days in Mariupol (Oscars), **Joe Biden** (2024 election),
     the United States (Club World Cup).
4. **Wrong drawer** (only the look-alike page, told to say "I don't know" if it can't answer):
   **answered 10 of 12**, right 0, said it didn't know 2. Argentina 88%, Real Madrid, Pakistan,
   Kolkata Knight Riders, Carlos Alcaraz 87%, the Chiefs, Joe Biden 92% ...

## Honesty limits
- One small model (0.5B) and one small library. Larger models read better; we did not measure them.
- Real systems add more: better search, re-ranking, citations, checks. We show the bare idea.
- "Sureness" is the name's first-token probability: a clue, not a guarantee.
- Qwen's training cutoff is not published; it was released September 2024, and every event asked
  about happened after that.

## What the script may say
- "a model released in September 2024"; "half a billion numbers"
- "South Korea ... beat the United States"; "twenty-nine percent sure"
- "zero of twelve"; "four times it refused, eight times it made something up"
- "twenty-four pages, a hundred and six pieces"; "three hundred and eighty-four numbers"
- "0.73 / 0.61"; "the top three pieces"
- "Spain, eighty-nine percent"; "five of twelve"
- "eight times the answer was on the desk; five it read right"
- "four times the answer wasn't there, and four times it answered anyway"
- "Argentina, eighty-eight percent"; "ten of twelve answered from the wrong page"
