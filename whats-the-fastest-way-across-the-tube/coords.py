"""Station positions (lat/lon) from TfL's route sequences, for drawing the network geographically
(never TfL's diagram). Writes download/coords.json. Paced like fetch.py."""
import json, os, time, urllib.request
ROOT = os.path.dirname(os.path.abspath(__file__)); DL = os.path.join(ROOT, "download")
from fetch import LINES
out = {}
for ln in LINES:
    for d in ("outbound", "inbound"):
        time.sleep(1.5)
        rq = urllib.request.Request("https://api.tfl.gov.uk/Line/%s/Route/Sequence/%s" % (ln, d), headers={"User-Agent": "research-script"})
        seq = json.load(urllib.request.urlopen(rq, timeout=60))
        for sps in seq["stopPointSequences"]:
            for sp in sps["stopPoint"]:
                for k in ("id", "stationId", "topMostParentId"):
                    if sp.get(k):
                        out[sp[k]] = (sp["lat"], sp["lon"])
json.dump(out, open(os.path.join(DL, "coords.json"), "w"), indent=0)
print("positions", len(out))
