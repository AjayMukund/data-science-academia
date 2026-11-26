"""What's the fastest way across the Tube? Shortest-path search (Dijkstra) on TfL's own timetable minutes.
A trip is a chain of (station, line) stops; riding costs the timetable minutes, changing line costs CHANGE minutes
(an assumption we state: walk + wait). Compares the fewest-stops route with the fastest. Plain Python."""
import json, os, heapq, collections, itertools, sys
ROOT = os.path.dirname(os.path.abspath(__file__)); DL = os.path.join(ROOT, "download")
if not os.path.exists(os.path.join(DL, "edges.json")):
    import runpy; runpy.run_path(os.path.join(ROOT, "fetch.py"))
E = json.load(open(os.path.join(DL, "edges.json"))); N = json.load(open(os.path.join(DL, "stations.json")))
nm = lambda s: N.get(s, s).replace(" Underground Station", "").replace(" (Underground)", "")
ride = collections.defaultdict(list); lines_at = collections.defaultdict(set); adj = collections.defaultdict(set)
for e in E:
    ride[(e["a"], e["line"])].append(((e["b"], e["line"]), e["mins"]))
    lines_at[e["a"]].add(e["line"]); lines_at[e["b"]].add(e["line"]); adj[e["a"]].add(e["b"])
ST = sorted(lines_at)
def fastest(src, dst, change=5.0, log=None):
    pq = [(0.0, 0, src, ln, (src, ln, None)) for ln in lines_at[src]]; heapq.heapify(pq)
    best = {}; order = []
    while pq:
        t, ch, s, ln, path = heapq.heappop(pq)
        if (s, ln) in best: continue
        best[(s, ln)] = (t, path)
        if s not in order: order.append(s)
        if s == dst:
            seq = []; p = path
            while p: seq.append((p[0], p[1])); p = p[2]
            return t, ch, seq[::-1], order
        for (b, l2), m in ride[(s, ln)]:
            if (b, l2) not in best: heapq.heappush(pq, (t + m, ch, b, l2, (b, l2, path)))
        for l2 in lines_at[s]:
            if l2 != ln and (s, l2) not in best: heapq.heappush(pq, (t + change, ch + 1, s, l2, (s, l2, path)))
    return None
def fewest_stops(src, dst):
    prev = {src: None}; q = collections.deque([src])
    while q:
        s = q.popleft()
        if s == dst: break
        for b in sorted(adj[s]):
            if b not in prev: prev[b] = s; q.append(b)
    if dst not in prev: return None
    p = [dst]
    while prev[p[-1]]: p.append(prev[p[-1]])
    return p[::-1]
def time_of_station_path(p, change=5.0):          # cheapest way to ride exactly this station sequence
    states = {ln: 0.0 for ln in lines_at[p[0]]}
    for a, b in zip(p, p[1:]):
        nxt = {}
        for ln, t in states.items():
            for l2 in lines_at[a]:
                for (bb, l3), m in ride[(a, l2)]:
                    if bb == b and l3 == l2:
                        c = t + m + (change if l2 != ln else 0)
                        if c < nxt.get(l2, 1e9): nxt[l2] = c
        states = nxt
    return min(states.values()) if states else None
def describe(seq):
    out = []; 
    for s, ln in seq:
        if not out or out[-1][0] != ln: out.append([ln, nm(s), nm(s), 0])
        else: out[-1][2] = nm(s); out[-1][3] += 1
    return [(ln, a, b, k) for ln, a, b, k in out if k > 0]
if __name__ == "__main__":
    R = {"stations_with_service": len(ST), "links": len(E), "change_minutes_assumed": 5}
    FAM = ["King's Cross St. Pancras", "Waterloo", "Heathrow Terminals 2 & 3", "Wimbledon", "Bank", "Oxford Circus",
           "Canary Wharf", "Wembley Park", "Stratford", "Brixton", "Paddington", "Liverpool Street", "London Bridge",
           "Victoria", "Camden Town", "Euston", "Baker Street", "Westminster", "Richmond", "Walthamstow Central",
           "Leicester Square", "Covent Garden", "Greenwich", "Hammersmith", "Morden", "Ealing Broadway"]
    ids = {}
    for s in ST:
        for f in FAM:
            if nm(s).startswith(f) and f not in ids: ids[f] = s
    R["matched_famous"] = len(ids)
    diffs = []; npair = 0; slower = 0
    import random; random.seed(2026)
    allp = list(itertools.permutations(ST, 2)); fam = set(ids.values())
    sample = random.sample(allp, 4000) + [(a, b) for a, b in allp if a in fam and b in fam]
    for a, b in sample:
        p = fewest_stops(a, b)
        if p is None: continue
        tf = time_of_station_path(p); r = fastest(a, b)
        if tf is None or r is None: continue
        npair += 1
        if tf > r[0] + 0.01: slower += 1
        if a in ids.values() and b in ids.values():
            diffs.append((tf - r[0], nm(a), nm(b), len(p) - 1, tf, len(r[2]), r[0], r[1]))
    R["pairs_checked"] = npair; R["pairs_where_fewest_stops_is_slower_pct"] = round(100 * slower / npair, 1)
    diffs.sort(reverse=True)
    R["famous_biggest_gaps"] = [(a, b, "stops-route", k, "stops", round(tf), "min | fastest", round(t), "min,", c, "changes") for d, a, b, k, tf, n, t, c in diffs[:12]]
    json.dump(R, open(os.path.join(ROOT, "results.json"), "w"), indent=1)
    for k, v in R.items():
        if isinstance(v, list): print(k); [print("  ", x) for x in v]
        else: print(k, ":", v)
