"""Build the London Underground network from TfL's Unified API (Powered by TfL Open Data; OGL v2 terms):
every tube line, both directions, each branch's timetable from its first station -> minutes between neighbouring
stations. Anonymous API use, paced at one call every 1.5 s. Writes download/edges.json + download/stations.json."""
import json, os, time, urllib.request, collections, statistics
ROOT = os.path.dirname(os.path.abspath(__file__)); DL = os.path.join(ROOT, "download"); os.makedirs(DL, exist_ok=True)
LINES = ["bakerloo", "central", "circle", "district", "hammersmith-city", "jubilee", "metropolitan", "northern",
         "piccadilly", "victoria", "waterloo-city"]
def get(path):
    for a in range(5):
        try:
            time.sleep(1.5)
            rq = urllib.request.Request("https://api.tfl.gov.uk" + path, headers={"User-Agent": "research-script"})
            return json.load(urllib.request.urlopen(rq, timeout=60))
        except Exception as e:
            print("retry", path, e); time.sleep(20 * (a + 1))
    raise SystemExit("failed " + path)
if __name__ == "__main__":
    names, edges = {}, collections.defaultdict(list)
    for ln in LINES:
        origins = set()
        for d in ("outbound", "inbound"):
            seq = get("/Line/%s/Route/Sequence/%s" % (ln, d))
            for st in seq["stations"]: names[st["id"]] = st["name"]
            for r in seq["orderedLineRoutes"]: origins.add(r["naptanIds"][0])
        for o in sorted(origins):
            tt = get("/Line/%s/Timetable/%s" % (ln, o))
            for s in tt.get("stops", []): names.setdefault(s["id"], s["name"])
            for route in tt.get("timetable", {}).get("routes", []):
                for iv in route["stationIntervals"]:
                    prev, pt = o, 0.0
                    for x in iv["intervals"]:
                        dt = x["timeToArrival"] - pt
                        if dt > 0: edges[(prev, x["stopId"], ln)].append(dt)
                        prev, pt = x["stopId"], x["timeToArrival"]
        print(ln, "origins", len(origins), "edges so far", len(edges), flush=True)
    E = [{"a": a, "b": b, "line": ln, "mins": statistics.median(v), "n": len(v)} for (a, b, ln), v in edges.items()]
    json.dump(E, open(os.path.join(DL, "edges.json"), "w"), indent=0)
    json.dump(names, open(os.path.join(DL, "stations.json"), "w"), indent=0)
    print("stations", len(names), "edges", len(E))
