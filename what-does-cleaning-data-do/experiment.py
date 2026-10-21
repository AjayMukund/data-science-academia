"""
experiment.py - "What does cleaning data actually do?"

The Titanic passenger list (Kaggle training set, 891 passengers; mirror
https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv).
A model has to learn who survived from the raw list. What does each cleaning
choice do - to WHO is left in the data, and to how well the model does on
passengers it never saw? Plain numpy: logistic regression (Newton / IRLS),
scored with 10 x 5-fold cross-validation so a lucky split cannot decide it.

    python experiment.py          -> out/results.json
"""
import os, sys, csv, re, json, collections, urllib.request
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)
sys.stdout.reconfigure(encoding="utf-8")
SRC = os.path.join(HERE, "download", "titanic.csv")
if not os.path.exists(SRC):
    os.makedirs(os.path.dirname(SRC), exist_ok=True)
    urllib.request.urlretrieve("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv", SRC)

rows = list(csv.DictReader(open(SRC, encoding="utf-8")))
N = len(rows)
y_all = np.array([int(r["Survived"]) for r in rows])
R = {"n": N, "survival": round(float(y_all.mean()), 4),
     "class_mix": {c: sum(r["Pclass"] == c for r in rows) for c in "123"}}

# ---- 1. what is messy -------------------------------------------------------
R["blank"] = {k: sum(1 for r in rows if not r[k].strip()) for k in rows[0]}
R["blank"] = {k: v for k, v in R["blank"].items() if v}
fares = np.array([float(r["Fare"]) for r in rows])
zero = [r for r in rows if float(r["Fare"]) == 0]
R["fare"] = {"zero": len(zero), "zero_line_tickets": sum(r["Ticket"] == "LINE" for r in zero),
             "zero_survived": sum(int(r["Survived"]) for r in zero),
             "max": round(float(fares.max()), 2), "at_max": int((fares == fares.max()).sum()),
             "at_max_survived": int(y_all[fares == fares.max()].sum()),
             "median": round(float(np.median(fares)), 2)}
emb = [r for r in rows if not r["Embarked"]]
R["embarked_blank"] = {"n": len(emb), "same_ticket": len({r["Ticket"] for r in emb}) == 1,
                       "same_cabin": len({r["Cabin"] for r in emb}) == 1, "ticket": emb[0]["Ticket"]}


def title(name):
    t = re.search(r",\s*([^.]+)\.", name).group(1).strip()
    return t if t in ("Mr", "Mrs", "Miss", "Master") else "Other"


# ---- 2. delete every incomplete row: who is left? ---------------------------
complete = [r for r in rows if all(r[k].strip() for k in ("Age", "Cabin", "Embarked"))]
R["drop_incomplete"] = {"kept": len(complete),
                        "class_mix": {c: sum(r["Pclass"] == c for r in complete) for c in "123"},
                        "survival": round(float(np.mean([int(r["Survived"]) for r in complete])), 4)}
has_age = [r for r in rows if r["Age"].strip()]
R["drop_age_blank"] = {"kept": len(has_age), "survival": round(float(np.mean([int(r["Survived"]) for r in has_age])), 4)}

# ---- 3. the emptiest column is a clue ------------------------------------
cab = np.array([1 if r["Cabin"].strip() else 0 for r in rows])
R["cabin"] = {"known": int(cab.sum()), "blank": int(N - cab.sum()),
              "surv_known": round(float(y_all[cab == 1].mean()), 4),
              "surv_blank": round(float(y_all[cab == 0].mean()), 4),
              "known_by_class": {c: int(sum(cab[i] for i, r in enumerate(rows) if r["Pclass"] == c)) for c in "123"}}


# ---- 4. a model, under each cleaning choice -------------------------------
def build(rs, age_fill, cabin_flag=False, use_title=False):
    """Feature matrix for rows `rs`; age_fill(row) supplies a missing age."""
    X = []
    for r in rs:
        age = float(r["Age"]) if r["Age"].strip() else age_fill(r)
        e = r["Embarked"] or "S"                 # 2 blanks; see R["embarked_blank"]
        f = [float(r["Pclass"] == "1"), float(r["Pclass"] == "2"), float(r["Sex"] == "female"), age,
             float(r["SibSp"]), float(r["Parch"]), np.log1p(float(r["Fare"])), float(e == "C"), float(e == "Q")]
        if cabin_flag:
            f.append(float(bool(r["Cabin"].strip())))
        if use_title:
            t = title(r["Name"])
            f += [float(t == "Master"), float(t == "Miss"), float(t == "Mrs")]
        X.append(f)
    return np.array(X)


def fit_predict(Xtr, ytr, Xte, lam=1e-2):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    A = np.c_[np.ones(len(Xtr)), (Xtr - mu) / sd]
    B = np.c_[np.ones(len(Xte)), (Xte - mu) / sd]
    w = np.zeros(A.shape[1])
    for _ in range(30):                          # Newton steps (IRLS), light L2
        p = 1 / (1 + np.exp(-A @ w))
        g = A.T @ (p - ytr) + lam * np.r_[0, w[1:]]
        H = (A * (p * (1 - p))[:, None]).T @ A + lam * np.diag(np.r_[0, np.ones(len(w) - 1)])
        w -= np.linalg.solve(H, g)
    return (1 / (1 + np.exp(-B @ w))) > 0.5


def cv(rs, y, make, reps=10, k=5, seed=0):
    accs = []
    for rep in range(reps):
        idx = np.random.default_rng(seed + rep).permutation(len(rs))
        for f in range(k):
            te = idx[f::k]
            tr = np.setdiff1d(idx, te)
            Xtr, Xte = make([rs[i] for i in tr], [rs[i] for i in te])
            accs.append(float((fit_predict(Xtr, y[tr], Xte) == y[te]).mean()))
    return round(float(np.mean(accs)), 4), round(float(np.std(accs)), 4)


def mean_fill(train):
    m = np.mean([float(r["Age"]) for r in train if r["Age"].strip()])
    return lambda r: m


def title_fill(train):
    by = collections.defaultdict(list)
    for r in train:
        if r["Age"].strip():
            by[title(r["Name"])].append(float(r["Age"]))
    med = {t: float(np.median(v)) for t, v in by.items()}
    allm = float(np.median([float(r["Age"]) for r in train if r["Age"].strip()]))
    return lambda r: med.get(title(r["Name"]), allm)


def variant(fill, cabin_flag=False, use_title=False):
    def make(tr, te):
        f = fill(tr)                             # fill values learned from the TRAINING fold only
        return build(tr, f, cabin_flag, use_title), build(te, f, cabin_flag, use_title)
    return make


Y = y_all
R["baseline_everyone_dies"] = round(float(1 - Y.mean()), 4)
ya = np.array([int(r["Survived"]) for r in has_age])
R["model"] = {
    "drop_age_rows": dict(zip(("acc", "sd"), cv(has_age, ya, variant(mean_fill)))),
    "mean_fill": dict(zip(("acc", "sd"), cv(rows, Y, variant(mean_fill)))),
    "title_fill": dict(zip(("acc", "sd"), cv(rows, Y, variant(title_fill)))),
    "title_fill_cabin": dict(zip(("acc", "sd"), cv(rows, Y, variant(title_fill, cabin_flag=True)))),
    "title_fill_cabin_title": dict(zip(("acc", "sd"), cv(rows, Y, variant(title_fill, True, True)))),
}

# the 183 "complete" rows as training data: how does it do on everyone else?
cset = set(id(r) for r in complete)
rest = [r for r in rows if id(r) not in cset]
f = title_fill(complete)
Xc, Xr = build(complete, f, True), build(rest, f, True)
yc = np.array([int(r["Survived"]) for r in complete])
yr = np.array([int(r["Survived"]) for r in rest])
pr = fit_predict(Xc, yc, Xr)
R["train_on_complete_only"] = {"test_on_rest_acc": round(float((pr == yr).mean()), 4),
                               "rest_n": len(rest), "rest_survival": round(float(yr.mean()), 4),
                               "predicted_survival_rest": round(float(pr.mean()), 4)}

json.dump(R, open(os.path.join(OUT, "results.json"), "w"), indent=1)
print("survival, all 891:              %.1f%%" % (100 * R["survival"]))
print("delete every row with a blank:   %d kept, %s by class, survival %.1f%%" % (
    R["drop_incomplete"]["kept"], R["drop_incomplete"]["class_mix"], 100 * R["drop_incomplete"]["survival"]))
print("cabin on record / blank:         %.1f%% / %.1f%% survived" % (100 * R["cabin"]["surv_known"], 100 * R["cabin"]["surv_blank"]))
print("model trained on the 183 only:   %.1f%% right on the other 708" % (100 * R["train_on_complete_only"]["test_on_rest_acc"]))
for k, v in R["model"].items():
    print("model, %-24s %.1f%% (sd %.1f)" % (k + ":", 100 * v["acc"], 100 * v["sd"]))
print("guess everyone died:             %.1f%%" % (100 * R["baseline_everyone_dies"]))
