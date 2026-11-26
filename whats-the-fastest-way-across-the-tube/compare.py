"""Fair comparison: for famous pairs, (1) fewest stops (ties broken by time), (2) fewest changes (ties by time),
(3) fastest - each with a 5-minute change cost; plus how the fastest route flips as the change cost goes 0 -> 10."""
import heapq, json, os, itertools
from experiment import ride, lines_at, nm, ST, describe
def search(src, dst, key, change=5.0):
    # key: weights (w_time, w_stop, w_change) combined lexicographically via a tuple cost
    pq = [((0, 0, 0) if key == "stops" else (0, 0, 0), src, ln, (src, ln, None)) for ln in lines_at[src]]
    heapq.heapify(pq); seen = set()
    while pq:
        c, s, ln, path = heapq.heappop(pq)
        if (s, ln) in seen: continue
        seen.add((s, ln))
        if s == dst:
            seq = []; p = path
            while p: seq.append((p[0], p[1])); p = p[2]
            return c, seq[::-1], len(seen)
        for (b, l2), m in ride[(s, ln)]:
            t = (c[0] + 1, c[1] + m, c[2]) if key == "stops" else (c[0], c[1] + m, c[2]) if key == "changes" else (c[0] + m, c[1] + 1, c[2])
            heapq.heappush(pq, (t, b, l2, (b, l2, path)))
        for l2 in lines_at[s]:
            if l2 != ln:
                t = (c[0], c[1] + change, c[2] + 1) if key == "stops" else (c[0] + 1, c[1] + change, c[2]) if key == "changes" else (c[0] + change, c[1], c[2] + 1)
                heapq.heappush(pq, (t, s, l2, (s, l2, path)))
def summ(seq, change=5.0):
    legs = describe(seq); stops = sum(k for *_, k in legs); ch = len(legs) - 1
    t = 0.0
    for (s, ln), (s2, l2) in zip(seq, seq[1:]):
        if s == s2: t += change
        else: t += [m for (b, l3), m in ride[(s, ln)] if b == s2 and l3 == l2][0]
    return {"mins": t, "stops": stops, "changes": ch, "legs": legs}
FAM = ["King's Cross St. Pancras", "Waterloo", "Heathrow Terminals 2 & 3", "Wimbledon", "Bank", "Oxford Circus",
       "Canary Wharf", "Wembley Park", "Stratford", "Brixton", "Paddington", "Liverpool Street", "London Bridge",
       "Victoria", "Camden Town", "Euston", "Baker Street", "Westminster", "Richmond", "Walthamstow Central",
       "Leicester Square", "Covent Garden", "Greenwich", "Hammersmith", "Morden", "Ealing Broadway", "Green Park",
       "Bond Street", "Tottenham Court Road", "Holborn", "Angel", "Old Street", "Moorgate", "Highbury & Islington",
       "Finsbury Park", "Elephant & Castle", "Stockwell", "Clapham Common", "Notting Hill Gate", "Earl's Court",
       "South Kensington", "Wembley Central", "Harrow-on-the-Hill", "Stanmore", "Seven Sisters", "Tower Hill"]
ids = {}
for s in ST:
    for f in FAM:
        if nm(s) == f or nm(s).startswith(f + " ") : ids.setdefault(f, s)
rows = []
for a, b in itertools.permutations(ids, 2):
    s1 = summ(search(ids[a], ids[b], "stops")[1]); f = search(ids[a], ids[b], "time"); s3 = summ(f[1])
    if s3["mins"] < s1["mins"] - 0.01:
        rows.append((s1["mins"] - s3["mins"], a, b, s1, s3, f[2]))
rows.sort(key=lambda r: -r[0])
R = {"famous_matched": len(ids), "famous_pairs": len(ids) * (len(ids) - 1), "pairs_fewest_stops_not_fastest": len(rows),
     "top": [(a, b, round(g), s1, s3) for g, a, b, s1, s3, _ in rows[:10]]}
# change cost flip: pairs where the fastest route differs between change=1 and change=10
flips = []
for a, b in itertools.permutations(list(ids)[:30], 2):
    r1 = summ(search(ids[a], ids[b], "time", 1.0)[1], 1.0); r10 = summ(search(ids[a], ids[b], "time", 10.0)[1], 10.0)
    if r1["legs"] != r10["legs"] and r1["changes"] > r10["changes"]:
        flips.append((a, b, r1, r10))
R["change_cost_flips"] = len(flips); R["flip_examples"] = flips[:8]
json.dump(R, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "compare_results.json"), "w"), indent=1)
print("matched", len(ids), "pairs", R["famous_pairs"], "fewest-stops slower", len(rows))
for g, a, b, s1, s3, n in rows[:10]:
    print("%-26s -> %-24s  stops-route %2d stops %2d ch %3.0f min %s\n%52s fastest %2d stops %2d ch %3.0f min %s  (explored %d)" %
          (a, b, s1["stops"], s1["changes"], s1["mins"], s1["legs"], "", s3["stops"], s3["changes"], s3["mins"], s3["legs"], n))
print("flips", len(flips))
for a, b, r1, r10 in flips[:8]:
    print(a, "->", b, "| change 1 min:", r1["legs"], round(r1["mins"]), "| change 10 min:", r10["legs"], round(r10["mins"]))
