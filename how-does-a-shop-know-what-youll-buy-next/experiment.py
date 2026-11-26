"""How does a shop know what you'll buy next? Market-basket analysis (support, confidence, lift) on
UCI Online Retail (Chen, Sain & Guo 2012; CC BY 4.0): a UK online gift shop, Dec 2010 - Dec 2011.
Plain Python: reads the .xlsx directly (zip + XML), downloads it on first run."""
import os, io, re, zipfile, json, collections, itertools, urllib.request
import xml.etree.ElementTree as ET
ROOT = os.path.dirname(os.path.abspath(__file__)); DL = os.path.join(ROOT, "download"); os.makedirs(DL, exist_ok=True)
X = os.path.join(DL, "Online Retail.xlsx")
if not os.path.exists(X):
    rq = urllib.request.Request("https://archive.ics.uci.edu/static/public/352/online+retail.zip", headers={"User-Agent": "research-script"})
    zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(rq, timeout=300).read())).extractall(DL)
CSV = os.path.join(DL, "rows.tsv")
if not os.path.exists(CSV):                         # xlsx -> tsv once (shared strings + sheet1, streamed)
    z = zipfile.ZipFile(X); ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    ss = ["".join(t.text or "" for t in si.iter(ns + "t")) for si in ET.parse(z.open("xl/sharedStrings.xml")).getroot()]
    out = open(CSV, "w", encoding="utf-8")
    for ev, el in ET.iterparse(z.open("xl/worksheets/sheet1.xml")):
        if el.tag == ns + "row":
            vals = {}
            for c in el.iter(ns + "c"):
                col = re.match(r"[A-Z]+", c.get("r")).group(); v = c.find(ns + "v")
                v = "" if v is None else v.text
                if c.get("t") == "s": v = ss[int(v)]
                vals[col] = v.replace("\t", " ").strip()
            out.write("\t".join(vals.get(k, "") for k in "ABCDEFGH") + "\n"); el.clear()
    out.close()
rows = [l.rstrip("\n").split("\t") for l in open(CSV, encoding="utf-8")][1:]
R = {"rows": len(rows)}
# InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country
baskets = collections.defaultdict(set); name = {}
for inv, code, desc, qty, *_ , country in rows:
    if inv.startswith("C") or not code or not desc: continue             # cancellations out
    try:
        if float(qty) <= 0: continue
    except ValueError: continue
    if code.upper() in ("POST", "DOT", "M", "D", "C2", "BANK CHARGES", "AMAZONFEE", "CRUK", "PADS", "B"): continue  # postage and fees
    baskets[inv].add(code); name.setdefault(code, desc)
B = list(baskets.values()); n = len(B)
R["baskets"] = n; R["products"] = len(name); R["median_items_per_basket"] = sorted(len(b) for b in B)[n // 2]
one = collections.Counter(c for b in B for c in b)
pair = collections.Counter()
for b in B:
    if len(b) <= 60:                                 # skip giant wholesale orders when counting pairs
        for a, c in itertools.combinations(sorted(b), 2): pair[(a, c)] += 1
R["baskets_used_for_pairs"] = sum(1 for b in B if len(b) <= 60)
def stats(a, c):
    s = pair[tuple(sorted((a, c)))]
    return {"both": s, "a": one[a], "c": one[c], "support_pct": round(100 * s / n, 2),
            "conf_a_to_c_pct": round(100 * s / one[a], 1), "base_c_pct": round(100 * one[c] / n, 1),
            "lift": round((s / one[a]) / (one[c] / n), 1)}
R["best_sellers"] = [(name[c], one[c], round(100 * one[c] / n, 1)) for c, _ in one.most_common(8)]
R["top_pairs_by_count"] = [(name[a], name[c], k) for (a, c), k in pair.most_common(8)]
# rules a -> c with enough data: highest confidence, and highest lift
rules = []
for (a, c), k in pair.items():
    if k < 100: continue
    for x, y in ((a, c), (c, a)):
        rules.append((k / one[x], (k / one[x]) / (one[y] / n), x, y, k))
rules.sort(reverse=True)
R["top_rules_by_confidence"] = [(name[x], "->", name[y], k, round(100 * cf, 1), round(lf, 1)) for cf, lf, x, y, k in rules[:10]]
rl = sorted(rules, key=lambda r: -r[1])
R["top_rules_by_lift"] = [(name[x], "->", name[y], k, round(100 * cf, 1), round(lf, 1)) for cf, lf, x, y, k in rl[:10]]
# the trap: a rule with decent confidence only because the consequent is a best-seller (lift ~1)
top = one.most_common(1)[0][0]
trap = sorted(((k / one[x], x, k) for (a, c), k in pair.items() for x, y in ((a, c), (c, a)) if y == top and k >= 50 and one[x] >= 300),
              key=lambda t: (t[0] / (one[top] / n)))
R["best_seller"] = name[top]
R["rules_into_best_seller_lowest_lift"] = [(name[x], k, round(100 * cf, 1), round(cf / (one[top] / n), 2)) for cf, x, k in trap[:6]]
json.dump(R, open(os.path.join(ROOT, "results.json"), "w"), indent=1)
for k, v in R.items():
    if isinstance(v, list): print(k); [print("   ", r) for r in v]
    else: print(k, ":", v)
# the running example: a customer puts the GREEN REGENCY TEACUP AND SAUCER in the basket. What else to show?
code = {v: k for k, v in name.items()}
g = code["GREEN REGENCY TEACUP AND SAUCER"]
withg = [(c, pair[tuple(sorted((g, c)))]) for c in one if c != g and pair[tuple(sorted((g, c)))]]
R["teacup"] = {"green_baskets": one[g], "green_pct_of_all": round(100 * one[g] / n, 1)}
R["teacup_most_often_with"] = [(name[c], k, round(100 * k / one[g], 1), round(100 * one[c] / n, 1),
                                round((k / one[g]) / (one[c] / n), 1)) for c, k in sorted(withg, key=lambda t: -t[1])[:8]]
for nm in ("ROSES REGENCY TEACUP AND SAUCER", "PINK REGENCY TEACUP AND SAUCER", "WHITE HANGING HEART T-LIGHT HOLDER",
           "JUMBO BAG RED RETROSPOT", "REGENCY CAKESTAND 3 TIER", "PARTY BUNTING"):
    R["teacup -> " + nm] = stats(g, code[nm])
# among items bought with the teacup at least 30 times, lowest lift (bought together only because both are popular)
low = sorted(((k / one[g]) / (one[c] / n), c, k) for c, k in withg if k >= 30)
R["teacup_lowest_lift"] = [(name[c], k, round(100 * k / one[g], 1), round(100 * one[c] / n, 1), round(lf, 2)) for lf, c, k in low[:5]]
json.dump(R, open(os.path.join(ROOT, "results.json"), "w"), indent=1)
for k in [k for k in R if k.startswith("teacup")]: print(k, ":", R[k])
