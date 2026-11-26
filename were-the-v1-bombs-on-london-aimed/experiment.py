"""Were the V-1 flying bombs aimed at particular streets? Clarke (1946), J. Inst. Actuaries 72, 481:
144 sq km of south London cut into 576 squares of 1/4 sq km; 537 hits. Here: the chance-only (Poisson)
prediction, a chi-square test, and random scatters that LOOK clustered. Plain Python, fixed seed."""
import math, random, json, os
OBS = [229, 211, 93, 35, 7, 1]                     # squares with 0, 1, 2, 3, 4, 5+ hits (Clarke's table)
N, HITS = 576, 537
lam = HITS / N
p = [math.exp(-lam) * lam ** k / math.factorial(k) for k in range(5)]; p.append(1 - sum(p))
EXP = [N * q for q in p]
chi = sum((o - e) ** 2 / e for o, e in zip(OBS, EXP))
R = {"squares": N, "hits": HITS, "mean_per_square": round(lam, 4), "observed": OBS,
     "expected_chance": [round(e, 2) for e in EXP], "chi2": round(chi, 2)}
# p-value, 4 degrees of freedom (6 bins - 1 - 1 fitted mean): survival of chi2(4) = e^-x/2 (1 + x/2)
R["p_value_df4"] = round(math.exp(-chi / 2) * (1 + chi / 2), 2)
R["hits_in_squares_with_3plus"] = 3 * 35 + 4 * 7 + 5 * 1     # the 5+ square counted as 5 (lower bound)
# random scatters: 537 hits dropped uniformly on a 24 x 24 grid
random.seed(1946)
maxes, empties, big = [], [], 0; example = None
for t in range(10000):
    c = [0] * N
    for _ in range(HITS): c[random.randrange(N)] += 1
    maxes.append(max(c)); empties.append(c.count(0))
    if max(c) >= 4: big += 1
    if t == 0: example = c
R["sim_runs"] = 10000
R["sim_pct_with_a_square_of_4plus"] = round(100 * big / 10000, 1)
R["sim_median_busiest_square"] = sorted(maxes)[5000]
R["sim_empty_squares_range"] = [min(empties), max(empties)]
R["sim_example_counts"] = [example.count(k) for k in range(5)] + [sum(1 for x in example if x >= 5)]
R["sim_example_grid"] = example
# where is the busiest 2x2 block of the example vs. an even spread (537/144 = 3.7 per block)
best = max(sum(example[(r + i) * 24 + (c + j)] for i in (0, 1) for j in (0, 1)) for r in range(23) for c in range(23))
R["sim_example_busiest_2x2_block"] = best
json.dump(R, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.json"), "w"), indent=1)
for k, v in R.items():
    if k != "sim_example_grid": print(k, ":", v)
