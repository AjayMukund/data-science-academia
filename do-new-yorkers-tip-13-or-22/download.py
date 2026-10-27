"""download.py - one month of NYC yellow-taxi trips (NYC TLC, free public data, ~59 MB).

    python download.py --month 2026-08
"""
import argparse, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--month", default="2026-08")
a = ap.parse_args()
name = "yellow_tripdata_%s.parquet" % a.month
urllib.request.urlretrieve("https://d37ci6vzurychx.cloudfront.net/trip-data/" + name, name)
print("saved", name)
