"""Everything the reel draws, from the same search code: routes as (station id, line) chains, the search's
settle order from Tower Hill with each station's quickest time, and the Angel -> Clapham Common flip.
Writes scene_data.json."""
import json, os, io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    from experiment import fastest, nm, lines_at
    from compare import search, summ, ids, rows, FAM
ROOT = os.path.dirname(os.path.abspath(__file__))
D = {}
a, b = ids["Tower Hill"], ids["Earl's Court"]
st = search(a, b, "stops", 5.0)[1]; ft = search(a, b, "time", 5.0)[1]
D["fewest"] = {"seq": st, **summ(st, 5.0)}
D["fastest"] = {"seq": ft, **summ(ft, 5.0)}
D["fewest_at_1min"] = summ(search(a, b, "stops", 1.0)[1], 1.0)["mins"]
t, c, seq, order = fastest(a, b, 5.0)
# each settled station's quickest time (5-min changes): rerun the search to each, cheap enough for 71
D["settle_order"] = [(s, nm(s), fastest(a, s, 5.0)[0]) for s in order]
x, y = ids["Angel"], ids["Clapham Common"]
D["angel_1"] = {"seq": search(x, y, "time", 1.0)[1], **summ(search(x, y, "time", 1.0)[1], 1.0)}
D["angel_3"] = {"seq": search(x, y, "time", 3.0)[1], **summ(search(x, y, "time", 3.0)[1], 3.0)}
D["big_stations"] = len(ids); D["pairs"] = len(ids) * (len(ids) - 1); D["pairs_fewest_slower"] = len(rows)
D["names"] = {s: nm(s) for s in lines_at}
json.dump(D, open(os.path.join(ROOT, "scene_data.json"), "w"), indent=0)
print({k: v for k, v in D.items() if k in ("big_stations", "pairs", "pairs_fewest_slower", "fewest_at_1min")},
      D["fewest"]["mins"], D["fastest"]["mins"], len(D["settle_order"]), D["settle_order"][-1], D["angel_1"]["mins"], D["angel_3"]["mins"])
