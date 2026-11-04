"""Extrapolation - the men's mile world record, 1913 to today.

Data: Wikipedia, "Mile run world record progression", IAAF / World Athletics era
table (ratified rows only), re-downloaded if missing. Fits a straight line to
the records and asks it the future: the record it expects in 2025, the year it
reaches 3:30, 3:00 and a mile in no time at all - against what really happened.

    python experiment.py
"""
import datetime as dt
import json
import os
import re
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(HERE, "download"), exist_ok=True)
os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
RAW = os.path.join(HERE, "download", "mile_raw.wiki")
if not os.path.exists(RAW):
    req = urllib.request.Request("https://en.wikipedia.org/w/index.php?title=Mile_run_world_record_progression&action=raw",
                                 headers={"User-Agent": "DSA-research/1.0"})
    open(RAW, "wb").write(urllib.request.urlopen(req).read())
txt = open(RAW, encoding="utf-8").read()
sec = txt.split("=== IAAF / World Athletics era ===")[1].split("== Men's indoor ==")[0]

MONTHS = {m: i + 1 for i, m in enumerate("January February March April May June July August September October November December".split())}
recs = []
for row in sec.split("|-")[1:]:
    row = row.strip()
    pending = row.startswith('bgcolor="#cef6f5"')      # awaiting ratification (Kerr, July 2026)
    if pending:
        row = row.split("\n", 1)[1].strip()
    if not row.startswith("|") or "background" in row.split("||")[0] or "bgcolor" in row.split("\n")[0]:
        continue                       # header rows and not-ratified / rescinded rows
    cells = [c.strip() for c in row.lstrip("|").split("||")]
    if len(cells) < 5:
        continue
    t = re.search(r"(\d):(\d\d\.\d+)", cells[0])
    d = re.search(r"(\d{1,2}) (\w+) (\d{4})", cells[4])
    if not (t and d):
        continue
    name = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]", r"\1", cells[2]).strip()
    date = dt.date(int(d.group(3)), MONTHS[d.group(2)], int(d.group(1)))
    recs.append(dict(sec=60 * int(t.group(1)) + float(t.group(2)), name=name, date=str(date),
                     year=date.year + (date.timetuple().tm_yday - 1) / 365.25, pending=pending))

# the line is fitted on the records ratified up to 1999 (El Guerrouj); anything later -
# Kerr's 3:42.66, pending in July 2026 - is what the line is tested against. Dated, not
# flagged, so the numbers hold once Kerr's record is ratified and loses its flag.
new = [r for r in recs if r["date"] >= "2000"]
recs = [r for r in recs if r["date"] < "2000" and not r["pending"]]

yr = np.array([r["year"] for r in recs])
s = np.array([r["sec"] for r in recs])
fmt = lambda x: "%d:%05.2f" % (x // 60, x % 60) if x >= 0 else "-%d:%05.2f" % (-x // 60, -x % 60)
out = {"records": recs, "n": len(recs)}
for tag, upto in (("all", 3000), ("to_bannister", 1954.35), ("to_1975", 1975.99)):   # 1954.35: Bannister (6 May), not Landy (21 June)
    m = yr <= upto
    b, a = np.polyfit(yr[m], s[m], 1)
    when = lambda target: (target - a) / b
    out[tag] = dict(n=int(m.sum()), sec_per_decade=round(-10 * b, 2),
                    pred_2025=fmt(a + b * 2025), pred_1999=fmt(a + b * 1999),
                    year_3_30=round(when(210)), year_3_00=round(when(180)), year_zero=round(when(0)),
                    # evaluated on the day of the last ratified record (El Guerrouj, 7 July 1999), not 1 Jan
                    pred_last_record_day=fmt(a + b * recs[-1]["year"]))
out["last"] = recs[-1]
b, a = np.polyfit(yr, s, 1)
for r in new:
    r["line_expected"] = fmt(a + b * r["year"])
    r["years_after_last"] = round(r["year"] - recs[-1]["year"], 1)
    r["seconds_cut"] = round(recs[-1]["sec"] - r["sec"], 2)
out["pending"] = new
for k, v in out.items():
    if k != "records":
        print(k, v)
for r in recs:
    print("  %s  %s  %s" % (r["date"], fmt(r["sec"]), r["name"]))
json.dump(out, open(os.path.join(HERE, "out", "results.json"), "w"), indent=1)
