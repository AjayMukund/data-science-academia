# Why does AI make things up? - measured findings

**Model:** GPT-2 small (124M parameters, OpenAI 2019), run by us in plain numpy (`gpt2_np.py`).
- Weights: openai-community/gpt2 `model.safetensors`; tokenizer: tiktoken "gpt2".
- Greedy decoding: always take the most likely next word.
- **Sureness** = the probability the model gave the first word of its answer.

**The question**, asked after two worked examples (Italy → Rome, Japan → Tokyo):
```
Q: What is the capital of <country>?
A:
```
- Asked of **50 real countries** and **50 invented names** (Brastovia, Quelloria, Veldoria …).
- An answer counts as right if it contains the real capital; "Kiev" counts for Kyiv.
- `python experiment.py --weights <path>` writes `out/results.json`.

## 1. Real countries (50)
- **39 right**, 11 wrong (e.g. Canada → Toronto, Australia → Sydney, India → Mumbai), **0 declined**.
- Average sureness: **63% when right**, 29% when wrong.
- Examples: France → Paris 68%; Kenya → Nairobi 11% (right, but unsure).

## 2. Invented countries (50)
- **Answered 50 of 50. Declined 0.**
- 46 of 50 answers repeat the invented name as its own capital ("Brastovia" → "Brastovia"). The other 4 name a real city: Veldoria → Venice, Orvenia → Rome, Pelloria → Rome, Ostravel → "Ota".
- Average sureness: **34%**.
- Brastovia → "Brastovia", **56%**. Quelloria → "Quelloria", **60%**, the surest invention.
- **7 inventions** are over 50% sure.
- **All 50** inventions are surer than the least-sure right answer (Kenya → Nairobi, 11%).
- The surest invention (60%) is surer than **16 of the 39 right answers**.

## 3. Show it how to decline
- One more worked example is added to the prompt: "Q: What is the capital of Atlantis? A: There is no such country."
- It still declines **0 of 50** inventions (and 0 real countries).
- Real-country accuracy drops from 39 to **33**.
- Effect on inventions: Halvestan → "The capital of Halvestan", 8%; the rest still name a city.

## 4. A confidence cut-off on the plain model
Rule: refuse to answer when the first word is less sure than *t*.

| *t* | Inventions caught (of 50) | Right answers thrown away (of 39) | Wrong real answers caught (of 11) |
|---|---|---|---|
| 20% | 5 | 3 | 2 |
| 30% | 23 | 3 | 7 |
| **40%** | **35** | **6** | 10 |
| 50% | 43 | 12 | 10 |

## Honesty limits (say these, don't overreach)
- This is a small 2019 model. Today's chat assistants are far larger and are trained, with human feedback, to say "I don't know". They decline far more often, but published evaluations still find them making things up.
- **We did not measure a modern model; claim no rate for one.**
- "Sureness" here is the first word's probability. It is a clue, not a guarantee.

## What the script may say
- "Brastovia"; "fifty-six percent"; "Paris, sixty-eight"
- "a hundred and twenty-four million"; "2019"
- "thirty-nine of fifty right"
- "fifty out of fifty"; "not once" did it say there is no such country
- "forty-six times the capital is the country's own name"
- "on average sixty-three percent sure when right, thirty-four when inventing"
- "Quelloria: sixty percent, surer than sixteen of its right answers"
- "Kenya: Nairobi, eleven percent"
- "an example of saying no: still zero of fifty, and six fewer right"
- "a cut-off at forty percent: thirty-five of fifty caught, six right answers lost, fifteen still slip through"
