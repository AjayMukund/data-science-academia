"""Who is the best batter in IPL history? Small samples top every leaderboard.
Data: Cricsheet IPL JSON (ODC-By), research_data/ipl_toss/download/ipl. Plain Python."""
import json, glob, os, collections, math
ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "..", "ipl_toss", "download", "ipl")          # shared with the toss reel in the video project
if not os.path.isdir(SRC):                                              # a fresh copy: fetch Cricsheet's IPL archive
    import urllib.request, zipfile, io
    SRC = os.path.join(ROOT, "download", "ipl")
    if not os.path.isdir(SRC):
        rq = urllib.request.Request("https://cricsheet.org/downloads/ipl_json.zip", headers={"User-Agent": "research-script"})
        zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(rq, timeout=300).read())).extractall(SRC)
runs = collections.Counter(); balls = collections.Counter(); outs = collections.Counter()
for f in glob.glob(os.path.join(SRC, "*.json")):
    d = json.load(open(f, encoding="utf-8"))
    for inn in d["innings"]:
        if inn.get("super_over"): continue
        for ov in inn["overs"]:
            for dl in ov["deliveries"]:
                b = dl["batter"]; runs[b] += dl["runs"]["batter"]
                if "wides" not in dl.get("extras", {}): balls[b] += 1
                for w in dl.get("wickets", []): outs[w["player_out"]] += 1
P = [p for p in balls if balls[p] > 0]
sr = {p: 100 * runs[p] / balls[p] for p in P}
tot_r = sum(runs.values()); tot_b = sum(balls.values())
print("batters", len(P), "league SR", round(100 * tot_r / tot_b, 1))
top = sorted(P, key=lambda p: -sr[p])
print("TOP SR, no minimum:"); [print(" ", p, runs[p], balls[p], round(sr[p], 1)) for p in top[:8]]
print("BOTTOM SR:"); [print(" ", p, runs[p], balls[p], round(sr[p], 1)) for p in top[-6:]]
for m in (100, 500, 1000):
    q = [p for p in P if balls[p] >= m]; t = sorted(q, key=lambda p: -sr[p])
    print(f"min {m} balls: n={len(q)} top", [(p, runs[p], balls[p], round(sr[p], 1)) for p in t[:4]])
# where do the extremes come from: balls faced of top/bottom 20
for k in (20,):
    print("median balls top20", sorted(balls[p] for p in top[:k])[k // 2], "bottom20", sorted(balls[p] for p in top[-k:])[k // 2], "all", sorted(balls[p] for p in P)[len(P) // 2])
# spread by sample size band
for lo, hi in ((1, 10), (10, 50), (50, 200), (200, 1000), (1000, 99999)):
    s = [sr[p] for p in P if lo <= balls[p] < hi]
    if s: s.sort(); print(f"balls {lo}-{hi}: n={len(s)} min {s[0]:.0f} max {s[-1]:.0f}")
# shrinkage: add K league-average balls
K = 100; avg = tot_r / tot_b
sh = {p: 100 * (runs[p] + K * avg) / (balls[p] + K) for p in P}
t = sorted(P, key=lambda p: -sh[p])
print("shrunk K=100 top", [(p, runs[p], balls[p], round(sr[p], 1), round(sh[p], 1)) for p in t[:6]])
print("rank of no-min #1 after shrink", t.index(top[0]) + 1)

import json
res = dict(batters=len(P), league_sr=round(100*tot_r/tot_b,1),
  top_no_min=[(p,runs[p],balls[p],round(sr[p],1)) for p in top[:5]],
  zero_sr=sum(1 for p in P if sr[p]==0),
  russell=(runs["AD Russell"],balls["AD Russell"],round(sr["AD Russell"],1),round(sh["AD Russell"],1),t.index("AD Russell")+1),
  maphaka=(runs["KT Maphaka"],balls["KT Maphaka"],round(sr["KT Maphaka"],1),round(sh["KT Maphaka"],1),t.index("KT Maphaka")+1),
  suryavanshi=(runs["V Suryavanshi"],balls["V Suryavanshi"],round(sr["V Suryavanshi"],1),round(sh["V Suryavanshi"],1),t.index("V Suryavanshi")+1),
  K=K,
  players=sorted([balls[p], round(sr[p], 1)] for p in P),
  shrunk_top=[(p,runs[p],balls[p],round(sr[p],1),round(sh[p],1)) for p in t[:5]],
  under10=[min(sr[p] for p in P if balls[p]<10), max(sr[p] for p in P if balls[p]<10)],
  over1000=[min(sr[p] for p in P if balls[p]>=1000), max(sr[p] for p in P if balls[p]>=1000)])
json.dump(res, open(os.path.join(ROOT,"results.json"),"w"), indent=1); print(res)

# Luck test: an average batter = balls drawn at random from every real IPL delivery (wides excluded)
import random
R = []
for f in glob.glob(os.path.join(SRC, "*.json")):
    d = json.load(open(f, encoding="utf-8"))
    for inn in d["innings"]:
        if inn.get("super_over"): continue
        for ov in inn["overs"]:
            for dl in ov["deliveries"]:
                if "wides" not in dl.get("extras", {}): R.append(dl["runs"]["batter"])
rng = random.Random(2026); N = 100000; best = 0; hit = 0; two = 0
for _ in range(N):
    s = sum(rng.choice(R) for _ in range(449)); best = max(best, s); hit += s >= 1028
    two += rng.choice(R) >= 4 and rng.choice(R) >= 4
res.update(luck_balls=len(R), luck_two_boundaries=two / N, luck_449_hits=hit, luck_449_best_sr=round(100 * best / 449, 1))
json.dump(res, open(os.path.join(ROOT, "results.json"), "w"), indent=1); print("luck", res["luck_two_boundaries"], hit, res["luck_449_best_sr"])
