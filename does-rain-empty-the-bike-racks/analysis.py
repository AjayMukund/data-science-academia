"""Group-by in pandas - Seoul public bike rentals (UCI 560, Dec 2017 - Nov 2018).

Every snippet in SNIPPETS is executed exactly as written (the video shows them
verbatim); its printed output and time (fastest of 5 runs on our laptop) go to
out/results.json. Needs pandas (pip install pandas):

    python analysis.py
"""
import contextlib
import io
import json
import os
import time
import urllib.request

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(HERE, "download"), exist_ok=True)
os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
CSV = os.path.join(HERE, "download", "SeoulBikeData.csv")
if not os.path.exists(CSV):
    urllib.request.urlretrieve("https://archive.ics.uci.edu/ml/machine-learning-databases/00560/SeoulBikeData.csv", CSV)
os.chdir(os.path.join(HERE, "download"))

SNIPPETS = {   # short lines: the video shows them verbatim on a phone-shaped frame
    "load": """bikes = pd.read_csv("SeoulBikeData.csv",
                    encoding="latin-1")
running = bikes["Functioning Day"] == "Yes"
bikes = bikes[running]
bikes["rented"] = bikes["Rented Bike Count"]
print(len(bikes), "hours")""",
    "overall": """print(bikes["rented"].mean().round())""",
    "rain": """bikes["raining"] = bikes["Rainfall(mm)"] > 0
print(bikes.groupby("raining")["rented"]
      .mean().round())""",
    "hour": """by_hour = bikes.groupby("Hour")["rented"]
print(by_hour.mean().round()
      .loc[[4, 8, 13, 18, 23]])""",
    "rain_hour": """both = bikes.groupby(["Hour", "raining"])
table = both["rented"].mean().unstack()
print(table.round().loc[[8, 18]])""",
    "season": """print(bikes.groupby("Seasons")["rented"]
      .mean().round())""",
}

ns = {"pd": pd}
out = {}
for k, code in SNIPPETS.items():
    best = 1e9
    for i in range(5):
        env = dict(ns) if i else ns          # first run keeps its variables for later snippets
        buf = io.StringIO()
        t0 = time.perf_counter()
        with contextlib.redirect_stdout(buf):
            exec(code, env)
        best = min(best, time.perf_counter() - t0)
        if i == 0:
            text = buf.getvalue()
    out[k] = dict(code=code, output=text, seconds=round(best, 4))
    print("### %s  (%.4f s)\n%s\n%s" % (k, best, code, text))

# extra checks for the script (not shown as code)
b = ns["bikes"]
out["extra"] = dict(
    hours=int(len(b)), rainy_hours=int(b["raining"].sum()),
    rainy_share=float(b["raining"].mean()),
    rainy_by_hour_share=b.groupby("Hour")["raining"].mean().round(3).to_dict(),
    dropped_non_functioning=int(8760 - len(b)),
    first=str(b["Date"].iloc[0]), last=str(b["Date"].iloc[-1]),
    max_hour=b.loc[b["rented"].idxmax(), ["Date", "Hour", "rented"]].astype(str).to_dict(),
)
print(json.dumps(out["extra"], indent=1))
os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
json.dump(out, open(os.path.join(HERE, "out", "results.json"), "w"), indent=1)
