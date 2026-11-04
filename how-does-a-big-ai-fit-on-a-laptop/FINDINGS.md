# Quantization master (11-09): how small can a model's numbers get? — measured findings (2026-11-04)

**Model:** Qwen2.5-0.5B-Instruct (Apache-2.0, 494 M numbers), run by us in plain numpy (`qwen_np.py`).
**Method:** every weight matrix (attention, feed-forward, and the shared embedding/output table) is rounded to a
signed integer grid in groups of 32 numbers, each group sharing one 16-bit scale (absmax / largest level).
Biases and norm weights are kept as they are. The rounded numbers are turned back into floats for the maths,
so the answers are exactly what the stored integers would give. Sizes are what the weights would take.
**Quiz:** the 50 real countries from the hallucination master, "What is the capital of X? Answer with the city
name only.", greedy, in chat format. Plus one free answer: "Describe the Eiffel Tower in one sentence."
`python experiment.py --models <hf cache>` writes `out/results.json` (log `out_run.log`).

| weights | size | capitals right | Eiffel Tower answer |
|---|---|---|---|
| 16-bit (original file) | 988 MB | **45 / 50** | "…a iconic iron lattice structure in Paris, France, that stands as a symbol of French culture and engineering." |
| 8-bit | 525 MB | **44 / 50** | same sentence, "history" for "engineering" |
| 4-bit | 278 MB | **38 / 50** | "…a iconic landmark that stands as a symbol of Paris, France, and is a must-see destination…" |
| 3-bit | 216 MB | **0 / 50** | "8.8.8.8.8.8.8.9.111" |
| 2-bit | 155 MB | **0 / 50** | mixed-script gibberish |

- 8-bit: half the size, one capital lost (Finland). 4-bit: 28% of the size, 7 lost.
- 4-bit got Australia RIGHT ("Canberra") where the original said "Sydney"; it also newly said Peru → "Buenos Aires".
  Rounding is noise: it flips answers either way, mostly wrong (gained 1, lost 8: net −7).
- Original misses: Australia → Sydney, Kenya → Kampala, Nigeria, Morocco, Bangladesh.
- One weight row, first 8 numbers (layer 0 query): original −0.0019, −0.0052, 0.0188, 0.0125, 0.0040, …;
  4-bit −0.0037, −0.0037, 0.0187, 0.0112, 0.0037, …; 2-bit 0, 0, 0.0262, 0, 0, … (most become zero).
- The cliff is between 4 and 3 bits for this simple rounding.
- Share of weights that round to exactly zero (every matrix in layers 0, 12 and 23; `out/zeros.json`):
  8-bit **0.8%**, 4-bit **14.5%**, 3-bit **32.6%** ("a third"), 2-bit **76.6%** ("three in four").
- Steps per level: 8-bit 255 (−127..127), 4-bit 15, 3-bit 7, 2-bit 3 (−1, 0, 1).
- A 70-billion-parameter model: 70e9 × 2 bytes = **140 GB** at 16 bits; at 4 bits with one 16-bit scale per 32
  numbers, 70e9 × (0.5 + 2/32) bytes = **39.4 GB** (arithmetic, not measured).

## Cautions for the script
- Our rounding is the simplest kind. Laptop formats people use (llama.cpp k-quants, GPTQ, AWQ) protect outliers
  and pick scales cleverly, and work well at lower bits than ours: say "simple rounding", never "3-bit is broken".
- Small models suffer more from rounding than big ones (widely reported); we measured only this one.
- The 8-bit "Finland" miss is one answer: don't over-read single flips.
