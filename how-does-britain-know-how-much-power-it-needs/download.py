"""download.py - fetch the data into download/: NESO half-hourly national demand 2021-2025 (NESO Open Data Licence,
"Supported by National Energy SO Open Data") and Open-Meteo daily mean temperatures (CC BY 4.0) for four cities.
  python download.py"""
import json, os, urllib.request
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "download")
os.makedirs(D, exist_ok=True)
pkg = json.load(urllib.request.urlopen("https://api.neso.energy/api/3/action/package_show?id=historic-demand-data"))["result"]
for r in pkg["resources"]:
    for y in range(2021, 2026):
        if r["name"].endswith(str(y)):
            urllib.request.urlretrieve(r["url"], os.path.join(D, "demand_%d.csv" % y))
            print("demand", y)
for name, lat, lon in (("london", 51.51, -0.13), ("birmingham", 52.49, -1.89), ("manchester", 53.48, -2.24), ("glasgow", 55.86, -4.25)):
    url = ("https://archive-api.open-meteo.com/v1/archive?latitude=%s&longitude=%s&start_date=2021-01-01&end_date=2025-10-31"
           "&daily=temperature_2m_mean&timezone=Europe%%2FLondon" % (lat, lon))
    urllib.request.urlretrieve(url, os.path.join(D, "temp_%s.json" % name))
    print("temperature", name)
