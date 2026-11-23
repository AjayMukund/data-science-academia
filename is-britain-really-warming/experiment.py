"""It's 3 degrees in November - so is Britain really warming? Signal vs noise in the
Central England Temperature series (Met Office HadCET, Open Government Licence v3).
Plain Python. Files in download/ (meantemp_monthly_totals.txt, meantemp_daily_totals.txt)."""
import os, statistics as st, collections, json
ROOT = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(ROOT, "download")
import urllib.request
os.makedirs(D, exist_ok=True)
for _f in ("meantemp_monthly_totals.txt", "meantemp_daily_totals.txt"):      # first run: fetch from the Met Office
    if not os.path.exists(os.path.join(D, _f)):
        _rq = urllib.request.Request("https://www.metoffice.gov.uk/hadobs/hadcet/data/" + _f, headers={"User-Agent": "research-script"})
        open(os.path.join(D, _f), "wb").write(urllib.request.urlopen(_rq, timeout=120).read())
ann = {}
for ln in open(os.path.join(D, "meantemp_monthly_totals.txt")):
    t = ln.split()
    if len(t) == 14 and t[0].isdigit() and float(t[-1]) > -99: ann[int(t[0])] = float(t[-1])
day = {}
for ln in open(os.path.join(D, "meantemp_daily_totals.txt")):
    t = ln.split()
    if len(t) == 2 and t[0][:1].isdigit() and float(t[1]) > -99: day[t[0]] = float(t[1])
res = {}
Y = sorted(ann); res["years"] = [Y[0], Y[-1]]
def m(a, b): return round(st.mean(ann[y] for y in range(a, b + 1)), 2)
res["mean_1659_1900"] = m(1659, 1900); res["mean_1901_1950"] = m(1901, 1950); res["mean_1961_1990"] = m(1961, 1990)
res["mean_2016_2025"] = m(2016, 2025)
rk = sorted(Y, key=lambda y: -ann[y]); res["warmest10"] = [(y, ann[y]) for y in rk[:10]]
res["coldest5"] = [(y, ann[y]) for y in rk[-5:]]
res["n_warmest10_since_2000"] = sum(y >= 2000 for y in rk[:10])
res["year_to_year_sd_1961_2025"] = round(st.stdev(ann[y] for y in range(1961, 2026)), 2)
# one date across years: 22 November
d22 = {int(k[:4]): v for k, v in day.items() if k[5:] == "11-22"}
res["nov22_2026"] = d22.get(2026); res["nov22_mean_1991_2020"] = round(st.mean(d22[y] for y in range(1991, 2021)), 2)
res["nov22_sd_1991_2020"] = round(st.stdev(d22[y] for y in range(1991, 2021)), 2)
res["nov22_range_1991_2025"] = (min(d22[y] for y in range(1991, 2026)), max(d22[y] for y in range(1991, 2026)))
# 2026 so far: days colder than the 1961-90 average for that date
clim = collections.defaultdict(list)
for k, v in day.items():
    if 1991 <= int(k[:4]) <= 2020: clim[k[5:]].append(v)
clim = {k: st.mean(v) for k, v in clim.items()}
for yr in (2025, 2026):
    ks = [k for k in day if k.startswith(str(yr))]
    res[f"days_{yr}"] = len(ks); res[f"days_colder_than_9120_{yr}"] = sum(day[k] < clim[k[5:]] for k in ks)
    res[f"mean_anomaly_{yr}"] = round(st.mean(day[k] - clim[k[5:]] for k in ks), 2)
# decade means
res["decades"] = {f"{a}s": m(a, a + 9) for a in range(1660, 2020, 10)}
res["nov22_by_year"] = {y: d22[y] for y in range(1991, 2026)}
res["annual"] = {y: ann[y] for y in Y}
res["days_2025_colder"] = [int(day[k] < clim[k[5:]]) for k in sorted(k for k in day if k.startswith("2025"))]
res["days_2025"] = [day[k] for k in sorted(k for k in day if k.startswith("2025"))]
res["year_sd_1991_2025_detrended_note"] = "year-to-year sd quoted over 1961-2025"
# noise shrinks: sd of single days vs year means vs decade means (detrended within 1961-1990)
json.dump(res, open(os.path.join(ROOT, "out_results.json"), "w"), indent=1)
for k, v in res.items():
    if k != "decades": print(k, v)
print({k: v for k, v in res["decades"].items() if k[:2] in ("16", "17") or k >= "1900s"})
