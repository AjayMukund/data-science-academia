"""Why do scientists squash their charts? Log scales and doubling times, on two real series:
transistors per microprocessor (Rupp 2022, via Our World in Data) and AI training compute (Epoch AI 2026, via OWID).
Both CC BY 4.0. Plain Python; downloads on first run."""
import os, csv, math, json, urllib.request
ROOT = os.path.dirname(os.path.abspath(__file__)); DL = os.path.join(ROOT, "download"); os.makedirs(DL, exist_ok=True)
def get(slug):
    p = os.path.join(DL, slug + ".csv")
    if not os.path.exists(p):
        rq = urllib.request.Request("https://ourworldindata.org/grapher/%s.csv?v=1&csvType=full&useColumnShortNames=false" % slug,
                                    headers={"User-Agent": "research-script"})
        open(p, "wb").write(urllib.request.urlopen(rq, timeout=120).read())
    return list(csv.DictReader(open(p, encoding="utf-8")))
def fit(xs, ys):                                     # least squares on log2(y): slope = doublings per year
    ly = [math.log2(y) for y in ys]; mx = sum(xs) / len(xs); my = sum(ly) / len(ly)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ly)) / sum((x - mx) ** 2 for x in xs)
    r2 = 1 - sum((y - (my + b * (x - mx))) ** 2 for x, y in zip(xs, ly)) / sum((y - my) ** 2 for y in ly)
    return b, r2
R = {}
T = [(int(r["Year"]), float(r["Transistors per microprocessor"])) for r in get("transistors-per-microprocessor") if r["Entity"] == "World"]
R["transistors_years"] = [T[0][0], T[-1][0]]; R["transistors_first_last"] = [T[0][1], T[-1][1]]
R["transistors_growth_x"] = round(T[-1][1] / T[0][1])
b, r2 = fit([x for x, _ in T], [y for _, y in T]); R["transistor_doubling_years"] = round(1 / b, 2); R["transistor_fit_r2"] = round(r2, 3)
mx = T[-1][1]
R["share_of_peak_by_year"] = {y: round(100 * v / mx, 3) for y, v in T if y in (1971, 1980, 1990, 2000, 2005, 2010, 2015, 2019, 2020, 2021)}
R["transistors_by_year"] = {y: v for y, v in T if y in (1971, 1980, 1990, 2000, 2010, 2020, 2021)}
# log-scale straightness by decade: doubling time fitted within each decade
R["doubling_by_decade"] = {}
for d in (1970, 1980, 1990, 2000, 2010):
    pts = [(x, y) for x, y in T if d <= x < d + 10]
    if len(pts) >= 3: R["doubling_by_decade"][d] = round(1 / fit([x for x, _ in pts], [y for _, y in pts])[0], 2)
A = []
for r in get("artificial-intelligence-training-computation"):
    try: A.append((r["Entity"], r["Day"], float(r["Training computation (petaFLOP)"])))
    except ValueError: pass
yr = lambda d: int(d[:4]) + (int(d[5:7]) - 1) / 12 + (int(d[8:10]) - 1) / 365
R["ai_models"] = len(A); R["ai_days"] = [min(a[1] for a in A), max(a[1] for a in A)]
dl = [(yr(d), c) for _, d, c in A if yr(d) >= 2010]
b, r2 = fit([x for x, _ in dl], [y for _, y in dl]); R["ai_doubling_months_2010on_all"] = round(12 / b, 1); R["ai_fit_r2"] = round(r2, 3)
# the frontier: the largest model so far, record by record
fr, best = [], 0
for nm, d, c in sorted(A, key=lambda a: a[1]):
    if c > best: best = c; fr.append((nm, d, c))
fr10 = [(n, d, c) for n, d, c in fr if yr(d) >= 2010]
b2, r22 = fit([yr(d) for _, d, _ in fr10], [c for _, _, c in fr10]); R["ai_frontier_doubling_months_2010on"] = round(12 / b2, 1)
R["ai_frontier_records_2010on"] = [(n, d, c) for n, d, c in fr10]
for key in ("AlexNet", "GPT-2", "GPT-3 175B (davinci)", "GPT-4", "Gemini 1.0 Ultra", "Grok 4", "GPT-4o"):
    m = [a for a in A if a[0] == key]
    if m: R["model " + key] = m[0]
R["largest"] = max(A, key=lambda a: a[2])
json.dump(R, open(os.path.join(ROOT, "results.json"), "w"), indent=1)
for k, v in R.items(): print(k, ":", v)
