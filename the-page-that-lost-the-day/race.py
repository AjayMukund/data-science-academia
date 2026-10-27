"""race.py - hour-by-hour views of the day's top two pages (English Wikipedia, desktop + mobile).
    python race.py --data <folder with the 24 .gz files> --out out"""
import os, gzip, json, argparse
from multiprocessing import Pool
PAGES = (b"Kim_Jong_Un", b"Jailer_2")


def hour(path):
    got = dict.fromkeys(PAGES, 0)
    with gzip.open(path, "rb") as f:
        for line in f:
            p = line.split(b" ")
            if len(p) >= 3 and p[0] in (b"en", b"en.m") and p[1] in got:
                got[p[1]] += int(p[2])
    return [got[t] for t in PAGES]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="out")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    files = sorted(os.path.join(a.data, f) for f in os.listdir(a.data) if f.startswith("pageviews-") and f.endswith(".gz"))
    with Pool(8) as pool:
        rows = pool.map(hour, files)
    out = {"pages": [t.decode() for t in PAGES], "hours_utc": [os.path.basename(f)[19:21] for f in files], "views": rows}
    json.dump(out, open(os.path.join(a.out, "race.json"), "w"), indent=1)
    ck = cj = 0
    for h, (k, j) in zip(out["hours_utc"], rows):
        ck += k; cj += j
        print(h, k, j, "| running", ck, cj, "lead", ck - cj)
