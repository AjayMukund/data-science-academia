"""download.py - fetch one day of Wikipedia's public hourly pageview files (24 x ~60 MB).

    python download.py --day 2026-10-20 --dest pageviews/

Source: https://dumps.wikimedia.org/other/pageviews/ (Wikimedia Foundation, CC0).
Already-downloaded files are skipped, so it can be re-run after a dropped connection."""
import os, argparse, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--day", default="2026-10-20")
ap.add_argument("--dest", default="pageviews")
a = ap.parse_args()
y, m, d = a.day.split("-")
os.makedirs(a.dest, exist_ok=True)
for h in range(24):
    name = "pageviews-%s%s%s-%02d0000.gz" % (y, m, d, h)
    out = os.path.join(a.dest, name)
    if os.path.exists(out) and os.path.getsize(out) > 0:
        continue
    url = "https://dumps.wikimedia.org/other/pageviews/%s/%s-%s/%s" % (y, y, m, name)
    print("downloading", name, flush=True)
    urllib.request.urlretrieve(url, out + ".part")
    os.replace(out + ".part", out)
print("done:", a.dest)
