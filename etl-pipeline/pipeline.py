"""
pipeline.py - a small, real ETL pipeline, and how it fails without an error.

A real, small Extract -> Transform -> Load pipeline on the Titanic passenger
list: download the CSV, clean and
reshape it, load it into SQLite, and answer a question with one SQL query.
Then the honest turn: feed it a source that quietly changed (one column
renamed) and compare a pipeline WITHOUT checks to one WITH checks.

    python pipeline.py                 -> out/results.json, out/titanic.db
    python pipeline.py --out some/dir
"""
import os, sys, csv, io, re, json, time, sqlite3, statistics, urllib.request, argparse

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="out", help="folder for the results, the raw file and the databases")
OUT = ap.parse_args().out
os.makedirs(OUT, exist_ok=True)
sys.stdout.reconfigure(encoding="utf-8")
URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
R = {}


def timed(fn, *a, runs=1):
    """Run fn `runs` times; return its output and the FASTEST time in ms, as Python's
    timeit does (a single first run is noisy: transform measured 8 ms once and
    3.1-5.4 ms over 30 runs - the slow ones are the machine doing other work)."""
    ts = []
    for _ in range(runs):
        t = time.perf_counter()
        out = fn(*a)
        ts.append((time.perf_counter() - t) * 1000)
    return out, round(min(ts), 1 if runs == 1 else 2)


# ---- EXTRACT -----------------------------------------------------------------
def extract():
    raw = urllib.request.urlopen(URL, timeout=30).read()
    return raw


raw, ms = timed(extract, runs=5)
text = raw.decode("utf-8")
open(os.path.join(OUT, "titanic.csv"), "wb").write(raw)          # the exact bytes extracted (shown on screen)
rows = list(csv.DictReader(io.StringIO(text)))
R["extract"] = {"bytes": len(raw), "rows": len(rows), "columns": list(rows[0].keys()), "ms": ms,
                "sex_counts": {s: sum(1 for r in rows if r["Sex"] == s) for s in ("female", "male")}}


# ---- TRANSFORM ---------------------------------------------------------------
def title(name):
    m = re.search(r",\s*([^.]+)\.", name)
    t = m.group(1).strip() if m else ""
    return t if t in ("Mr", "Mrs", "Miss", "Master") else "Other"


NEED = {"Survived", "Pclass", "Name", "Sex", "Age", "Fare", "Cabin", "Embarked"}


def transform(rows, checks=True):
    log = {"age_filled": 0, "port_filled": 0, "cabin_flag": 0, "titles": 0, "dropped": 0}
    if checks:                                            # check 1, before transforming
        missing = NEED - set(rows[0])
        if missing:
            raise ValueError(f"missing column(s) {sorted(missing)}")
    ages = {}
    for r in rows:
        if r.get("Age", "").strip():
            ages.setdefault(title(r["Name"]), []).append(float(r["Age"]))
    med = {t: statistics.median(v) for t, v in ages.items()}
    out = []
    for r in rows:
        sex = r.get("Sex", "").strip().lower()             # without checks a renamed column reads as ""
        age = r.get("Age", "").strip()
        if not age:
            age = med.get(title(r["Name"]), 28.0)
            log["age_filled"] += 1
        port = r.get("Embarked", "").strip() or "S"
        if not r.get("Embarked", "").strip():
            log["port_filled"] += 1
        has_cabin = 1 if r.get("Cabin", "").strip() else 0
        log["cabin_flag"] += has_cabin
        t = title(r["Name"])
        log["titles"] += 1
        out.append({"survived": int(r["Survived"]), "pclass": int(r["Pclass"]), "sex": sex, "age": float(age),
                    "fare": float(r["Fare"]), "port": port, "has_cabin": has_cabin, "title": t})
    if checks:                                            # check 2, after transforming
        bad = [o for o in out if o["sex"] not in ("male", "female")]
        if bad:
            raise ValueError(f"{len(bad)} rows with an unknown sex")
    return out, log


clean, ms = timed(transform, rows, runs=15)
clean, log = clean
R["transform"] = {"rows_in": len(rows), "rows_out": len(clean), "ms": ms, **log}


# ---- LOAD --------------------------------------------------------------------
def load(clean, path):
    if os.path.exists(path):
        os.remove(path)
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE passengers (survived INT, pclass INT, sex TEXT, age REAL, fare REAL, port TEXT,"
               " has_cabin INT, title TEXT)")
    db.executemany("INSERT INTO passengers VALUES (:survived,:pclass,:sex,:age,:fare,:port,:has_cabin,:title)", clean)
    db.commit()
    return db


DB = os.path.join(OUT, "titanic.db")
db, ms = timed(load, clean, DB)
R["load"] = {"rows": db.execute("SELECT COUNT(*) FROM passengers").fetchone()[0], "ms": ms,
             "db_bytes": os.path.getsize(DB)}

# ---- QUESTION: one SQL query --------------------------------------------------
Q = ("SELECT pclass, sex, ROUND(100.0 * AVG(survived), 1) AS pct, COUNT(*) AS n "
     "FROM passengers GROUP BY pclass, sex ORDER BY pclass, sex")
res, ms = timed(lambda: db.execute(Q).fetchall(), runs=15)
R["query"] = {"sql": Q, "ms": ms, "rows": [list(r) for r in res]}
db.close()

# ---- THE TURN: the source quietly changes -------------------------------------
# Simulated: the provider renames one column, Sex -> Gender.
changed = [{("Gender" if k == "Sex" else k): v for k, v in r.items()} for r in rows]
nocheck, _ = transform(changed, checks=False)
db2 = load(nocheck, os.path.join(OUT, "titanic_nocheck.db"))
women = db2.execute("SELECT COUNT(*) FROM passengers WHERE sex = 'female'").fetchone()[0]
pct_w = db2.execute("SELECT ROUND(100.0*AVG(survived),1) FROM passengers WHERE sex='female'").fetchone()[0]
total = db2.execute("SELECT COUNT(*) FROM passengers").fetchone()[0]
db2.close()
try:
    transform(changed, checks=True)
    stopped = None
except ValueError as e:
    stopped = "%s: %s" % (type(e).__name__, e)
R["changed_source"] = {"simulated": "column Sex renamed to Gender",
                       "no_checks": {"loaded_rows": total, "women_found": women, "women_survival": pct_w,
                                     "error": None},
                       "with_checks": {"stopped_with": stopped}}

json.dump(R, open(os.path.join(OUT, "results.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in R.items()}, indent=1))
