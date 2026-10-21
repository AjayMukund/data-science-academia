"""
experiment.py - "Why is a list so slow?" (reel, 2026-11-06).

A million ticket numbers, kept as a list and as a dictionary. Time 1,000
membership checks (`in`) against each, then the dictionary's cost: the time
to build it and the size of its table. Stdlib only. Timings depend on your
machine; ours are in FINDINGS.md.

    python experiment.py          -> out/results.json
    python snippet.py             (the exact code shown in the video)
"""
import os, sys, json, random, time, timeit

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)
N = 1_000_000

tickets = ["P%07d" % i for i in range(N)]
seats = {t: i for i, t in enumerate(tickets)}
checks = random.Random(0).sample(tickets, 1000)            # random tickets, not the worst case

t = time.perf_counter(); assert sum(c in tickets for c in checks) == 1000; t_list = time.perf_counter() - t
t = time.perf_counter(); assert sum(c in seats for c in checks) == 1000; t_dict = time.perf_counter() - t
build = min(timeit.repeat(lambda: {k: i for i, k in enumerate(tickets)}, number=1, repeat=5))
out = {"n": N, "list_1000_s": t_list, "dict_1000_s": t_dict, "dict_build_s": build,
       "list_table_mb": sys.getsizeof(tickets) / 1e6, "dict_table_mb": sys.getsizeof(seats) / 1e6,
       "mean_position": sum(tickets.index(c) for c in checks[:50]) / 50 / N,
       "python": sys.version.split()[0]}
out["break_even_lookups"] = build / (t_list / 1000)
json.dump(out, open(os.path.join(OUT, "results.json"), "w"), indent=1)
print("1,000 lookups in a list of a million: %.1f s" % t_list)
print("1,000 lookups in a dictionary:        %.4f s  (%.0fx faster)" % (t_dict, t_list / t_dict))
print("a random ticket sits %.0f%% of the way down the list" % (100 * out["mean_position"]))
print("building the dictionary: %.2f s; tables: list %.1f MB, dict %.1f MB (strings shared)"
      % (build, out["list_table_mb"], out["dict_table_mb"]))
print("the dictionary pays for itself after about %.0f lookups" % out["break_even_lookups"])
