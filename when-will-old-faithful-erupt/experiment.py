"""
experiment.py - linear regression from scratch: when will Old Faithful erupt next?

Data: R's `faithful` (Haerdle 1991): 272 eruptions of Old Faithful, Yellowstone, each with its
length (minutes) and the wait until the next eruption (minutes). Fetched from the Rdatasets
mirror of the R datasets package.

Plain numpy, no fitting library:
  1. the problem: how much the wait varies; guessing the average, how far off
  2. a line, scored by its misses: sum of squared misses for a few hand-picked lines
  3. the line finds itself: gradient descent from a flat line, step by step
  4. learn on 200, test on 72: average miss, share within +-10 minutes
  5. the honest turn: two kinds of eruption - is the line really just "short or long"?

    python experiment.py --out out
"""
import os, json, argparse, urllib.request
import numpy as np

URL = "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/datasets/faithful.csv"
ap = argparse.ArgumentParser()
ap.add_argument("--data", default="download/faithful.csv")
ap.add_argument("--out", default="out")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
if not os.path.exists(a.data):
    os.makedirs(os.path.dirname(a.data) or ".", exist_ok=True)
    urllib.request.urlretrieve(URL, a.data)

rows = [l.strip().split(",") for l in open(a.data)][1:]
x = np.array([float(r[1]) for r in rows])          # eruption length, minutes
y = np.array([float(r[2]) for r in rows])          # wait until the next one, minutes
N = len(x)
R = {"eruptions": N, "length_min_max": [float(x.min()), float(x.max())],
     "wait_min_max": [float(y.min()), float(y.max())], "wait_mean": round(float(y.mean()), 2),
     "share_wait_ge_90": round(float((y >= 90).mean()), 4), "waits_ge_90": int((y >= 90).sum())}

rng = np.random.default_rng(a.seed)
p = rng.permutation(N)
tr, te = p[:200], p[200:]
R["split"] = {"learn": len(tr), "test": len(te), "seed": a.seed}


def sse(m, c, idx):
    return float(((y[idx] - (m * x[idx] + c)) ** 2).sum())


def mae(pred, idx):
    return float(np.abs(y[idx] - pred).mean())


# --- 1. guess the average -------------------------------------------------------------------
avg = y[tr].mean()
R["average_guess"] = {"guess": round(float(avg), 2), "test_avg_miss": round(mae(avg, te), 2),
                      "test_within_10": round(float((np.abs(y[te] - avg) <= 10).mean()), 4),
                      "test_worst_miss": round(float(np.abs(y[te] - avg).max()), 1)}

# --- 2. a few lines, scored by their misses (learning eruptions) ------------------------------
# the best line, closed form, for reference
xm, ym = x[tr].mean(), y[tr].mean()
m_best = float(((x[tr] - xm) * (y[tr] - ym)).sum() / ((x[tr] - xm) ** 2).sum())
c_best = float(ym - m_best * xm)
lines = {"flat (the average)": (0.0, float(avg)), "too gentle": (5.0, float(ym - 5.0 * xm)),
         "too steep": (20.0, float(ym - 20.0 * xm)), "best": (m_best, c_best)}
R["lines"] = {k: {"slope": round(m, 3), "start": round(c, 2), "sum_sq_miss": round(sse(m, c, tr)),
                  "avg_miss": round(mae(m * x[tr] + c, tr), 2)} for k, (m, c) in lines.items()}
# the bowl: total squared miss as the slope changes (intercept re-centred through the means)
R["bowl"] = [[s, round(sse(s, float(ym - s * xm), tr))] for s in np.arange(0, 21, 1.0).tolist()]

# --- 3. the line finds itself: gradient descent from flat ------------------------------------
m, c = 0.0, float(avg)
lr = 0.02
steps = []
for k in range(1, 2001):
    err = (m * x[tr] + c) - y[tr]
    gm, gc = 2 * (err * x[tr]).mean(), 2 * err.mean()
    m, c = m - lr * gm, c - lr * gc
    if k in (1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000, 2000):
        steps.append({"step": k, "slope": round(m, 3), "start": round(c, 2), "sum_sq_miss": round(sse(m, c, tr))})
R["descent"] = {"learning_rate": lr, "steps": steps}
# how many steps until the slope is within 0.1 min/min of the best (re-run, cheap)
m, c = 0.0, float(avg)
for k in range(1, 200001):
    err = (m * x[tr] + c) - y[tr]
    m, c = m - lr * 2 * (err * x[tr]).mean(), c - lr * 2 * err.mean()
    if abs(m - m_best) < 0.1:
        R["descent"]["steps_to_within_0.1_of_best_slope"] = k
        break

# --- 4. learn on 200, test on 72 --------------------------------------------------------------
pred = m_best * x[te] + c_best
R["line"] = {"slope": round(m_best, 3), "start": round(c_best, 2),
             "r2_learn": round(1 - sse(m_best, c_best, tr) / float(((y[tr] - ym) ** 2).sum()), 3),
             "test_avg_miss": round(mae(pred, te), 2),
             "test_within_10": round(float((np.abs(y[te] - pred) <= 10).mean()), 4),
             "test_worst_miss": round(float(np.abs(y[te] - pred).max()), 1),
             "predict": {str(L): round(m_best * L + c_best, 1) for L in (2.0, 3.0, 4.0, 4.5, 5.0)}}

# --- 5. the honest turn: two kinds of eruption -----------------------------------------------
cut = 3.0
short, long_ = x < cut, x >= cut
R["clusters"] = {"cut_minutes": cut, "short": int(short.sum()), "long": int(long_.sum()),
                 "between_2.5_and_3.5": int(((x > 2.5) & (x < 3.5)).sum()),
                 "short_wait_mean": round(float(y[short].mean()), 1), "long_wait_mean": round(float(y[long_].mean()), 1)}
# two-guess model: average wait after a short eruption, after a long one (learned on tr)
gs, gl = y[tr][x[tr] < cut].mean(), y[tr][x[tr] >= cut].mean()
two = np.where(x[te] < cut, gs, gl)
R["two_guesses"] = {"after_short": round(float(gs), 1), "after_long": round(float(gl), 1),
                    "test_avg_miss": round(mae(two, te), 2),
                    "test_within_10": round(float((np.abs(y[te] - two) <= 10).mean()), 4)}
# slope INSIDE each group (learned on tr)
for name, msk in (("short", x[tr] < cut), ("long", x[tr] >= cut)):
    xx, yy = x[tr][msk], y[tr][msk]
    s = float(((xx - xx.mean()) * (yy - yy.mean())).sum() / ((xx - xx.mean()) ** 2).sum())
    R["clusters"]["slope_inside_" + name] = round(s, 2)
json.dump(R, open(os.path.join(a.out, "results.json"), "w"), indent=1)
print(json.dumps(R, indent=1))
