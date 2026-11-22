"""
experiment.py - are more babies born on a full moon? Every US birth, by day, 2000-2014 (5,479 days).

Data: US births by date, Social Security Administration, via FiveThirtyEight (CC BY 4.0):
https://github.com/fivethirtyeight/data/tree/master/births  ->  data/births.csv
Full moons: computed with Meeus' algorithm (Astronomical Algorithms, ch. 49), accurate to minutes; a birth "date" is
the US calendar date, so the full-moon day is the US Eastern date (UTC-5) holding the full moon.

  1. Every day's births against its expected value: the median of the same weekday 1, 2 and 3 weeks either side
     (that removes weekends, seasons and the trend). US federal holidays and the days around Christmas and New Year
     are left out of every test (a first version without that found a false +1.4% "full-moon effect": Thanksgiving
     dragged its week's baseline down). Residual = births / expected - 1.
  2. Full-moon days: their average residual, and how often 10,000 random sets of days do as well (every date
     moved by its own random 3-26 days).
  3. The real effects: weekdays; Friday the 13th (vs other Fridays); Christmas Day; Valentine's Day; Halloween.
  4. Fishing: test each of the 30 days of the moon's cycle on its own; how many come out "significant" (p < 0.05).

Plain numpy, seeded (0). Writes out/results.json.   python experiment.py
"""
import csv, datetime as dt, json, math, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
rows = list(csv.DictReader(open(os.path.join(HERE, "data", "births.csv"))))
days = [dt.date(int(r["year"]), int(r["month"]), int(r["date_of_month"])) for r in rows]
births = np.array([int(r["births"]) for r in rows], float)
wd = np.array([d.weekday() for d in days])            # 0 Monday .. 6 Sunday
ym = np.array([d.year * 12 + d.month for d in days])
N = len(days)
index = {d: i for i, d in enumerate(days)}


def full_moon_jde(k):
    """Meeus ch. 49: Julian Ephemeris Day of full moon number k (k = n + 0.5)"""
    T = k / 1236.85
    jde = 2451550.09766 + 29.530588861 * k + 0.00015437 * T ** 2 - 0.00000015 * T ** 3
    E = 1 - 0.002516 * T - 0.0000074 * T ** 2
    r = math.radians
    M = r(2.5534 + 29.10535670 * k - 0.0000014 * T ** 2)
    Mp = r(201.5643 + 385.81693528 * k + 0.0107582 * T ** 2)
    F = r(160.7108 + 390.67050284 * k - 0.0016118 * T ** 2)
    Om = r(124.7746 - 1.56375588 * k + 0.0020672 * T ** 2)
    s = math.sin
    corr = (-0.40614 * s(Mp) + 0.17302 * E * s(M) + 0.01614 * s(2 * Mp) + 0.01043 * s(2 * F)
            + 0.00734 * E * s(Mp - M) - 0.00515 * E * s(Mp + M) + 0.00209 * E * E * s(2 * M)
            - 0.00111 * s(Mp - 2 * F) - 0.00057 * s(Mp + 2 * F) + 0.00056 * E * s(2 * Mp + M)
            - 0.00042 * s(3 * Mp) + 0.00042 * E * s(M + 2 * F) + 0.00038 * E * s(M - 2 * F)
            - 0.00024 * E * s(2 * Mp - M) - 0.00017 * s(Om) - 0.00007 * s(Mp + 2 * M)
            + 0.00004 * s(2 * Mp - 2 * F) + 0.00004 * s(3 * M) + 0.00003 * s(Mp + M - 2 * F)
            + 0.00003 * s(2 * Mp + 2 * F) - 0.00003 * s(Mp + M + 2 * F) + 0.00003 * s(Mp - M + 2 * F)
            - 0.00002 * s(Mp - M - 2 * F) - 0.00002 * s(3 * Mp + M) + 0.00002 * s(4 * Mp))
    return jde + corr


def jd_to_datetime(jd):
    return dt.datetime(2000, 1, 1, 12) + dt.timedelta(days=jd - 2451545.0)


full = []
for n in range(-2, 190):
    t = jd_to_datetime(full_moon_jde(n + 0.5) - 66 / 86400)      # TT -> UTC (delta T ~ 66 s)
    us = (t - dt.timedelta(hours=5)).date()
    if us in index:
        full.append((us, t))
FM = np.array([index[d] for d, _ in full])
# check against two published full moons (USNO): 2000-01-21 04:40 UTC, 2014-12-06 12:27 UTC
chk = {str(t.date()): t.strftime("%H:%M") for _, t in full if str(t.date()) in ("2000-01-21", "2014-12-06")}
print("full moons:", len(FM), "check", chk)

# US federal holidays (and the days around Christmas / New Year) distort their week: leave them out of every test
def nth(y, m, wday, n):
    d = dt.date(y, m, 1); d += dt.timedelta((wday - d.weekday()) % 7)
    return d + dt.timedelta(7 * (n - 1))
def last(y, m, wday):
    d = dt.date(y, m + 1, 1) - dt.timedelta(1) if m < 12 else dt.date(y, 12, 31)
    return d - dt.timedelta((d.weekday() - wday) % 7)
hol = set()
for y in range(1999, 2016):
    tg = nth(y, 11, 3, 4)
    hol |= {dt.date(y, 1, 1), dt.date(y, 7, 4), dt.date(y, 11, 11), nth(y, 1, 0, 3), nth(y, 2, 0, 3), last(y, 5, 0),
            nth(y, 9, 0, 1), nth(y, 10, 0, 2), tg, tg + dt.timedelta(1), tg - dt.timedelta(1)}
    hol |= {dt.date(y, 12, d) for d in range(22, 32)} | {dt.date(y, 1, d) for d in range(2, 4)}
    hol |= {dt.date(y, 7, 3), dt.date(y, 7, 5)}
ok = np.array([d not in hol for d in days])
# expected = median births on the same weekday 1, 2 and 3 weeks either side (non-holiday days only)
exp_ = np.full(N, np.nan)
for i in range(N):
    nb = [i + 7 * k for k in (-3, -2, -1, 1, 2, 3) if 0 <= i + 7 * k < N and ok[i + 7 * k]]
    if len(nb) >= 3:
        exp_[i] = np.median(births[nb])
res = births / exp_ - 1
use = ok & ~np.isnan(res)
FM = np.array([i for i in FM if use[i]])
rng = np.random.default_rng(0)


def shift_test(idx, reps=10000):
    idx = np.array([i for i in idx if use[i]])
    obs = res[idx].mean()
    null = []
    for _ in range(reps):
        j = idx + rng.integers(3, 27, len(idx))      # every date moved by its own random 3-26 days
        j = j[(j < N)]
        j = j[use[j]]
        null.append(res[j].mean())
    null = np.array(null)
    return float(obs), float(np.mean(np.abs(null) >= abs(obs))), null


obs, p, null = shift_test(FM)
R = {"days": N, "births": int(births.sum()), "per_day": float(births.mean()),
     "full_moons": len(FM), "full_moon_check_utc": chk,
     "full_moon": dict(mean_births=float(births[FM].mean()), residual=obs, p=p,
                       null_95=[float(np.percentile(null, 2.5)), float(np.percentile(null, 97.5))],
                       extra_babies_per_day=float((births[FM] - exp_[FM]).mean()))}
print("full moon residual %.3f%%  p=%.3f  extra/day %.1f" % (100 * obs, p, R["full_moon"]["extra_babies_per_day"]))

names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
R["weekday"] = {names[k]: float(births[wd == k].mean()) for k in range(7)}
print({k: round(v) for k, v in R["weekday"].items()})

f13 = np.array([i for i, d in enumerate(days) if d.day == 13 and d.weekday() == 4])
o, pp, _ = shift_test(f13)
o_f6 = np.array([i for i, d in enumerate(days) if d.day == 6 and d.weekday() == 4])
R["friday13"] = dict(n=len(f13), residual=o, p=pp, mean_births=float(births[f13].mean()),
                     residual_friday_6th=float(res[o_f6].mean()))
print("Friday 13th: n=%d residual %.2f%% p=%.4f" % (len(f13), 100 * o, pp))
for name, (mo, da) in {"christmas": (12, 25), "valentine": (2, 14), "halloween": (10, 31), "july4": (7, 4)}.items():
    idx = np.array([i for i, d in enumerate(days) if d.month == mo and d.day == da])
    R[name] = dict(residual=float(np.nanmean(res[idx])), mean_births=float(births[idx].mean()))
    print(name, "%.1f%%" % (100 * np.nanmean(res[idx])))

# 4. fishing: the 30 days of the moon's cycle (days since the full moon, -14 .. +15), one test each
fm_dates = [d for d, _ in full]
since = np.full(N, 99)
for i, d in enumerate(days):
    best = min(fm_dates, key=lambda f: abs((d - f).days))
    since[i] = (d - best).days
tests = []
for k in range(-14, 16):
    idx = np.where((since == k) & use)[0]
    o, pk, _ = shift_test(idx, reps=4000)
    tests.append(dict(day=int(k), n=len(idx), residual=o, p=pk))
sig = [t for t in tests if t["p"] < 0.05]
R["cycle_tests"] = tests
R["cycle_significant"] = [t["day"] for t in sig]
print("cycle tests: %d of %d with p<0.05: %s" % (len(sig), len(tests), [(t["day"], round(100 * t["residual"], 2), t["p"]) for t in sig]))
json.dump(R, open(os.path.join(HERE, "out", "results.json"), "w"), indent=1)

# for the scene: every day's residual, the full moons, the shuffled results, the Friday 13ths, the cycle tests
obs, p_, null_fm = shift_test(FM)
np.savez_compressed(os.path.join(HERE, "out", "scene.npz"), births=births, expected=exp_, res=res, use=use, fm=FM,
                    f13=np.array([i for i in f13 if use[i]]), null=null_fm, weekday=wd,
                    date=np.array([d.isoformat() for d in days]),
                    cycle_day=np.array([t["day"] for t in tests]), cycle_res=np.array([t["residual"] for t in tests]),
                    cycle_p=np.array([t["p"] for t in tests]))
