"""
fetch.py - NYC 2015 Street Tree Census (NYC Parks / NYC Open Data, dataset uvpi-gqnh): every street tree the city
counted, with its status (alive, dead, stump), health, species and neighbourhood. Downloads the columns we use to
data/trees.csv (not committed; ~40 MB).   python fetch.py
"""
import csv, io, os, time, requests
HERE = os.path.dirname(os.path.abspath(__file__)); os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
URL = "https://data.cityofnewyork.us/resource/uvpi-gqnh.csv"
COLS = "tree_id,status,health,spc_common,boroname,nta_name,tree_dbh,latitude,longitude"
S = requests.Session(); S.headers["User-Agent"] = "dsa-research/1.0 (educational explainer)"
rows, off = [], 0
while True:
    for k in range(6):
        try:
            r = S.get(URL, params={"$select": COLS, "$order": "tree_id", "$limit": 50000, "$offset": off}, timeout=120)
            r.raise_for_status(); break
        except Exception: time.sleep(5 * (k + 1))
    part = list(csv.reader(io.StringIO(r.text)))[1:]
    if not part: break
    rows += part; off += len(part); print(off, flush=True)
with open(os.path.join(HERE, "data", "trees.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(COLS.split(",")); w.writerows(rows)
print("trees", len(rows))
