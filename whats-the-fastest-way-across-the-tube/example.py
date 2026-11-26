"""The running example: Ealing Broadway -> Liverpool Street. Numbers for the script."""
import json, os
from experiment import fastest, nm
from compare import search, summ, ids
R = {}
a, b = ids["Ealing Broadway"], ids["Liverpool Street"]
for ch in (0.0, 2.0, 5.0, 10.0):
    st = summ(search(a, b, "stops", ch)[1], ch); ft = summ(search(a, b, "time", ch)[1], ch)
    R["change %g" % ch] = {"fewest_stops": st, "fastest": ft}
t, c, seq, order = fastest(a, b, 5.0)
R["dijkstra_stations_settled_before_arrival"] = len(order)
R["dijkstra_first_20"] = [nm(s) for s in order[:20]]
a2, b2 = ids["Angel"], ids["Clapham Common"]
for ch in (1.0, 3.0, 5.0, 10.0):
    R["angel->clapham change %g" % ch] = summ(search(a2, b2, "time", ch)[1], ch)
json.dump(R, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "example_results.json"), "w"), indent=1)
for k, v in R.items(): print(k, ":", v)
