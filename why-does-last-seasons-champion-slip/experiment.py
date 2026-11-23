"""
experiment.py - why does last season's top team usually slip? (regression to the mean)

Data: openfootball/football.json (CC0 1.0, public domain), English Premier League match results,
every complete season available. Tables are rebuilt from results: 3 points a win, 1 a draw.

For each pair of consecutive seasons, every team in both: points this season -> points next season.
  python experiment.py      (downloads to cache/, writes out/results.json)
"""
import json, os, statistics as st, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE, OUT = os.path.join(HERE, "cache"), os.path.join(HERE, "out")
os.makedirs(CACHE, exist_ok=True); os.makedirs(OUT, exist_ok=True)
URL = "https://raw.githubusercontent.com/openfootball/football.json/master/%s/en.1.json"


def norm(t):
    """Team names drift across files ('Manchester City' / 'Manchester City FC'); key on the bare name."""
    t = t.replace("&", "and").strip()
    for suf in (" FC", " AFC"):
        if t.endswith(suf):
            t = t[: -len(suf)]
    return t.replace("AFC ", "").strip()


def season(y):
    p = os.path.join(CACHE, "en1_%s.json" % y)
    if not os.path.exists(p):
        try:
            req = urllib.request.Request(URL % y, headers={"User-Agent": "research-script"})
            data = urllib.request.urlopen(req, timeout=60).read()
        except Exception:
            return None                               # no file for that season
        open(p, "wb").write(data)
    ms = json.load(open(p, encoding="utf-8"))["matches"]
    pts, gp, gd = {}, {}, {}
    for m in ms:
        sc = m.get("score")
        s = sc.get("ft") if isinstance(sc, dict) else sc      # 2025-26 stores [h, a]
        if not s:
            continue
        a, b = norm(m["team1"]), norm(m["team2"])
        for t in (a, b):
            pts.setdefault(t, 0); gp.setdefault(t, 0); gd.setdefault(t, 0)
        gp[a] += 1; gp[b] += 1
        gd[a] += s[0] - s[1]; gd[b] += s[1] - s[0]
        if s[0] > s[1]: pts[a] += 3
        elif s[0] < s[1]: pts[b] += 3
        else: pts[a] += 1; pts[b] += 1
    if len(pts) != 20 or any(g != 38 for g in gp.values()):
        return None                                   # incomplete season
    return {t: dict(pts=pts[t], gd=gd[t]) for t in pts}


years = ["%d-%02d" % (y, (y + 1) % 100) for y in range(1990, 2026)]
S = {}
for y in years:
    t = season(y)
    if t:
        S[y] = t
ys = sorted(S)
print("complete seasons:", ys[0], "to", ys[-1], len(ys))

res = dict(seasons=ys, champions=[], pairs=[])
for a, b in zip(ys, ys[1:]):
    ta, tb = S[a], S[b]
    rank = sorted(ta, key=lambda t: (-ta[t]["pts"], -ta[t]["gd"]))
    champ = rank[0]
    res["champions"].append(dict(season=a, team=champ, pts=ta[champ]["pts"], next=tb[champ]["pts"],
                                 next_rank=1 + sorted(tb, key=lambda t: (-tb[t]["pts"], -tb[t]["gd"])).index(champ)))
    for pos, t in enumerate(rank, 1):
        if t in tb:
            res["pairs"].append(dict(season=a, team=t, pos=pos, pts=ta[t]["pts"], next=tb[t]["pts"]))

ch = res["champions"]
drops = [c["pts"] - c["next"] for c in ch]
res["champ_summary"] = dict(n=len(ch), fell=sum(d > 0 for d in drops), rose=sum(d < 0 for d in drops),
                            same=sum(d == 0 for d in drops), mean_drop=st.mean(drops),
                            mean_pts=st.mean(c["pts"] for c in ch), mean_next=st.mean(c["next"] for c in ch),
                            retained=sum(c["next_rank"] == 1 for c in ch))
print("champions next season:", res["champ_summary"])
for c in ch:
    print("  %s %-28s %3d -> %3d  (next rank %d)" % (c["season"], c["team"], c["pts"], c["next"], c["next_rank"]))

# all teams by finishing position: mean points now vs next season (only teams that stay up)
pp = res["pairs"]
mean_all = st.mean(p["pts"] for p in pp)
bypos = {}
for p in pp:
    bypos.setdefault(p["pos"], []).append(p)
res["by_position"] = {k: dict(n=len(v), pts=st.mean(x["pts"] for x in v), next=st.mean(x["next"] for x in v))
                      for k, v in sorted(bypos.items())}
print("by position (teams in both seasons):")
for k, v in res["by_position"].items():
    print("  %2d  n=%2d  %5.1f -> %5.1f  (%+.1f)" % (k, v["n"], v["pts"], v["next"], v["next"] - v["pts"]))

# the 'lucky' group: teams that beat their own goal difference? keep simple: top 4 and bottom of the survivors
def grp(lo, hi):
    g = [p for p in pp if lo <= p["pos"] <= hi]
    return dict(n=len(g), pts=st.mean(p["pts"] for p in g), next=st.mean(p["next"] for p in g),
                fell=sum(p["next"] < p["pts"] for p in g))
res["top4"] = grp(1, 4)
res["pos5_8"] = grp(5, 8)
res["pos13_17"] = grp(13, 17)
print("top 4:", res["top4"]); print("5-8:", res["pos5_8"]); print("13-17:", res["pos13_17"])

# correlation season to season
xs = [p["pts"] for p in pp]; ys2 = [p["next"] for p in pp]
mx, my = st.mean(xs), st.mean(ys2)
r = sum((x - mx) * (y - my) for x, y in zip(xs, ys2)) / (sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys2)) ** 0.5
slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys2)) / sum((x - mx) ** 2 for x in xs)
res["corr"] = dict(r=r, slope=slope, mean_pts=mx, mean_next=my, n=len(pp))
print("r = %.3f  slope = %.3f  (n=%d; mean %.1f -> %.1f)" % (r, slope, len(pp), mx, my))

# the biggest single-season jumps and what happened next (teams that over-performed)
up = []
for a, b, c in zip(ys, ys[1:], ys[2:]):
    for t in S[b]:
        if t in S[a] and t in S[c]:
            up.append(dict(team=t, before=S[a][t]["pts"], season=b, pts=S[b][t]["pts"], after=S[c][t]["pts"]))
up.sort(key=lambda u: -(u["pts"] - u["before"]))
res["big_jumps"] = up[:10]
print("biggest jumps (before -> jump season -> after):")
for u in up[:10]:
    print("  %-28s %s  %3d -> %3d -> %3d" % (u["team"], u["season"], u["before"], u["pts"], u["after"]))
json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=1)
print("wrote out/results.json")

# ---- a pure-luck league: 20 identical teams, each match drawn at the real home/draw/away rates
import random
hw = dr = aw = 0
for y in ys:
    p = os.path.join(CACHE, "en1_%s.json" % y)
    for m in json.load(open(p, encoding="utf-8"))["matches"]:
        sc = m.get("score"); s = sc.get("ft") if isinstance(sc, dict) else sc
        if not s:
            continue
        hw += s[0] > s[1]; dr += s[0] == s[1]; aw += s[0] < s[1]
tot = hw + dr + aw
rates = (hw / tot, dr / tot, aw / tot)
random.seed(11)


def luck_season():
    pts = [0] * 20
    for h in range(20):
        for a in range(20):
            if h == a:
                continue
            u = random.random()
            if u < rates[0]: pts[h] += 3
            elif u < rates[0] + rates[1]: pts[h] += 1; pts[a] += 1
            else: pts[a] += 3
    return pts


R = 2000
champ_pts, champ_next, spread = [], [], []
for _ in range(R):
    a, b = luck_season(), luck_season()
    c = max(range(20), key=lambda i: a[i])
    champ_pts.append(a[c]); champ_next.append(b[c]); spread.append(max(a) - min(a))
res["luck_league"] = dict(rates=rates, seasons=R, champ_pts=st.mean(champ_pts), champ_next=st.mean(champ_next),
                          mean_spread=st.mean(spread), mean_pts=st.mean(luck_season()))
print("real result rates home/draw/away: %.3f %.3f %.3f" % rates)
print("pure-luck league, %d seasons: champion %.1f pts -> next season %.1f; top-to-bottom spread %.1f"
      % (R, st.mean(champ_pts), st.mean(champ_next), st.mean(spread)))
real_spread = st.mean(max(v["pts"] for v in S[y].values()) - min(v["pts"] for v in S[y].values()) for y in ys)
res["real_spread"] = real_spread
print("real top-to-bottom spread %.1f" % real_spread)
jumps = up[:10]
res["jumps10"] = dict(gain=st.mean(u["pts"] - u["before"] for u in jumps), kept=st.mean(u["after"] - u["before"] for u in jumps))
print("10 biggest jumps: gained %.1f, next season kept %.1f of it" % (res["jumps10"]["gain"], res["jumps10"]["kept"]))
json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=1)

# ---- for the reel's pictures: every rebuilt table, and one dice season pair that matches the averages
res["tables"] = {y: {t: S[y][t]["pts"] for t in S[y]} for y in ys}
rng = random.Random(2026)


def luck_season_r():
    pts = [0] * 20
    for h in range(20):
        for a in range(20):
            if h == a:
                continue
            u = rng.random()
            if u < rates[0]: pts[h] += 3
            elif u < rates[0] + rates[1]: pts[h] += 1; pts[a] += 1
            else: pts[a] += 3
    return pts


for k in range(5000):
    a, b = luck_season_r(), luck_season_r()
    c = max(range(20), key=lambda i: a[i])
    if a[c] == 68 and b[c] == 53 and sorted(a)[-2] < 68:      # a clear champion on 68, then 53
        res["dice_example"] = dict(tries=k + 1, season=a, next=b, champ=c)
        print("dice example after %d tries: champion %d -> %d" % (k + 1, a[c], b[c]))
        break
json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=1)
