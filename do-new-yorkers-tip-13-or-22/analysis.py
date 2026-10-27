"""
analysis.py - do New Yorkers tip 13% or 22%? One month of NYC taxi trips, in pandas.

Data: NYC Taxi & Limousine Commission, Yellow Taxi Trip Records, August 2026
(https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page, free public data):
https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2026-08.parquet

Every snippet is in SNIPPETS below, run exactly as written, and its printed output saved. Timings are the fastest of 5 runs on our laptop.

    python download.py
    python analysis.py --data yellow_tripdata_2026-08.parquet --out out
"""
import os, json, time, argparse, contextlib, io

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--out", default="out")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

SNIPPETS = [
    ("load", 'import pandas as pd\n'
             'trips = pd.read_parquet("yellow_tripdata_2026-08.parquet")\n'
             'print(len(trips), "trips,", len(trips.columns), "columns")'),
    ("look", 'print(trips[["tpep_pickup_datetime", "fare_amount", "tip_amount", "payment_type"]].head(4))'),
    ("clean", 'august = trips.tpep_pickup_datetime.between("2026-08-01", "2026-09-01")\n'
              'print((~august).sum(), "trips not in August")\n'
              'trips = trips[august & (trips.fare_amount > 0)]'),
    ("naive", 'tip = trips.tip_amount.sum() / trips.fare_amount.sum()\n'
              'print(f"Tips: {tip:.1%} of the fare")'),
    ("by_pay", 'pay = trips.groupby("payment_type")[["tip_amount", "fare_amount"]].sum()\n'
               'print((pay.tip_amount / pay.fare_amount).round(3))'),
    ("card", 'card = trips[trips.payment_type == 1]\n'
             'tip = card.tip_amount.sum() / card.fare_amount.sum()\n'
             'print(f"Card trips: {tip:.1%} of the fare")'),
    ("by_hour", 'hour = card.tpep_pickup_datetime.dt.hour\n'
                'h = card.groupby(hour)[["tip_amount", "fare_amount"]].sum()\n'
                'print((h.tip_amount / h.fare_amount).round(3).sort_values())'),
]

R = {"source": "NYC TLC Yellow Taxi Trip Records, August 2026", "snippets": {}}
env = {}
for name, src in SNIPPETS:
    src_run = src.replace('"yellow_tripdata_2026-08.parquet"', repr(a.data))
    times = []
    for i in range(5 if name != "clean" else 1):          # clean mutates `trips`: run once
        local = dict(env)
        buf = io.StringIO()
        t = time.perf_counter()
        with contextlib.redirect_stdout(buf):
            exec(src_run, local)
        times.append(time.perf_counter() - t)
    env = local
    R["snippets"][name] = {"code": src, "output": buf.getvalue(), "secs": round(min(times), 3)}
    print("== %s (%.3f s)\n%s%s" % (name, min(times), src, "\n" + buf.getvalue()))

# the numbers behind the story (not shown as code)
import pandas as pd
t = env["trips"]
card = env["card"]
R["rows_raw"] = int(len(pd.read_parquet(a.data, columns=["payment_type"])))
R["rows_clean"] = int(len(t))
R["spreadsheets"] = round(R["rows_raw"] / 1048576, 2)
R["pay_counts"] = {int(k): int(v) for k, v in t.payment_type.value_counts().items()}
cash = t[t.payment_type == 2]
R["cash"] = {"trips": int(len(cash)), "tips_total": round(float(cash.tip_amount.sum()), 2),
             "zero_tip_share": round(float((cash.tip_amount == 0).mean()), 5)}
app = t[t.payment_type == 0]
R["app_booked"] = {"trips": int(len(app)), "tip_pct": round(100 * float(app.tip_amount.sum() / app.fare_amount.sum()), 1),
                   "request_sources": {str(k): int(v) for k, v in app.request_source.value_counts().items()}}
R["tip_pct_all"] = round(100 * float(t.tip_amount.sum() / t.fare_amount.sum()), 1)
R["tip_pct_card"] = round(100 * float(card.tip_amount.sum() / card.fare_amount.sum()), 1)
hr = t.tpep_pickup_datetime.dt.hour
g = t.groupby(hr)
R["by_hour"] = []
for h, grp in g:
    c = grp[grp.payment_type == 1]
    R["by_hour"].append({"hour": int(h), "trips": int(len(grp)),
                         "tip_pct_all": round(100 * float(grp.tip_amount.sum() / grp.fare_amount.sum()), 1),
                         "tip_pct_card": round(100 * float(c.tip_amount.sum() / c.fare_amount.sum()), 1),
                         "card_share": round(float((grp.payment_type == 1).mean()), 3),
                         "cash_share": round(float((grp.payment_type == 2).mean()), 3),
                         "app_share": round(float((grp.payment_type == 0).mean()), 3)})
R["total_secs"] = round(sum(s["secs"] for s in R["snippets"].values()), 2)
json.dump(R, open(os.path.join(a.out, "results.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
for r in R["by_hour"]:
    print(r)
print({k: v for k, v in R.items() if k not in ("snippets", "by_hour")})
