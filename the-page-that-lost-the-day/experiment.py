"""
experiment.py - one day of Wikipedia, too big to hold.

Question: which English Wikipedia page was read most on one day?
Data: Wikimedia's public hourly pageview files for 2026-10-20 (24 files,
https://dumps.wikimedia.org/other/pageviews/), one line per (wiki, page, hour):
    "en Some_Page 123 0"   -> project, title, views, (bytes, always 0)
English Wikipedia = project "en" (desktop) + "en.m" (mobile), summed per title.

Four measurements, all on this laptop (16 GB RAM, 4 cores / 8 threads):
  A  size        rows, unpacked bytes, and the time to read everything once
  B  naive       hold the raw lines in memory, hour by hour, until a safety
                 limit; memory per hour, extrapolated to 24 (said as such)
  C  one worker  stream every file, keep a running count per page
  D  map-reduce  8 workers: MAP each file to partial counts split into 8
                 buckets by page name; REDUCE each bucket across all files;
                 merge the 8 top lists
C and D must give the identical answer (asserted).

    python experiment.py --data <folder with the 24 .gz files> --out out
"""
import os, sys, gzip, json, time, pickle, argparse, ctypes, heapq, zlib
from ctypes import wintypes
from multiprocessing import Pool

SKIP_PREFIX = ("Special:", "Main_Page", "-", "Wikipedia:", "Portal:", "Help:", "File:", "Category:",
               "Template:", "User:", "Talk:")
EN = {b"en", b"en.m"}
BUCKETS = 8


def rss_gb():
    """This process's current and peak working set (Windows), in GB."""
    class PMC(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
    c = PMC()
    c.cb = ctypes.sizeof(PMC)
    k32, psapi = ctypes.WinDLL("kernel32"), ctypes.WinDLL("psapi")
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    assert psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb)
    return c.WorkingSetSize / 2 ** 30, c.PeakWorkingSetSize / 2 ** 30


def keep(title):
    return not title.startswith(SKIP_PREFIX)


def count_file(path, counts):
    """Add one hourly file's English page views into `counts`; return rows read."""
    rows = 0
    with gzip.open(path, "rb") as f:
        for line in f:
            rows += 1
            p = line.split(b" ")
            if len(p) >= 3 and p[0] in EN:
                t = p[1]
                counts[t] = counts.get(t, 0) + int(p[2])
    return rows


def top(counts, n=10):
    good = ((v, t) for t, v in counts.items() if keep(t.decode("utf-8", "replace")))
    return [(t.decode("utf-8", "replace"), v) for v, t in heapq.nlargest(n, good)]


# ---- D: map and reduce, each a separate process ------------------------------
def map_one(args):
    path, tmp = args
    t0 = time.perf_counter()
    counts = {}
    rows = count_file(path, counts)
    parts = [{} for _ in range(BUCKETS)]
    lead = top(counts, 1)[0]                 # this hour's most-read page
    for t, v in counts.items():
        parts[zlib.crc32(t) % BUCKETS][t] = v
    base = os.path.join(tmp, os.path.basename(path))
    for k, part in enumerate(parts):
        with open("%s.b%d" % (base, k), "wb") as f:
            pickle.dump(part, f, protocol=pickle.HIGHEST_PROTOCOL)
    return rows, time.perf_counter() - t0, rss_gb()[1], lead


def reduce_one(args):
    k, tmp = args
    t0 = time.perf_counter()
    total = {}
    for name in sorted(os.listdir(tmp)):
        if name.endswith(".b%d" % k):
            with open(os.path.join(tmp, name), "rb") as f:
                for t, v in pickle.load(f).items():
                    total[t] = total.get(t, 0) + v
    return top(total, 10), len(total), time.perf_counter() - t0, rss_gb()[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="out")
    ap.add_argument("--repeat", type=int, default=3, help="time C and D this many times, keep the fastest")
    ap.add_argument("--naive-limit-gb", type=float, default=3.0,
                    help="stop the naive load before this much memory (the machine has 16 GB)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    files = sorted(os.path.join(a.data, f) for f in os.listdir(a.data)
                   if f.startswith("pageviews-") and f.endswith(".gz"))
    assert len(files) == 24, "need all 24 hourly files, found %d" % len(files)
    R = {"files": [os.path.basename(f) for f in files],
         "gz_bytes": sum(os.path.getsize(f) for f in files)}

    # A: size, and one plain pass over everything
    t0 = time.perf_counter()
    rows = raw = 0
    for p in files:
        with gzip.open(p, "rb") as f:
            for line in f:
                rows += 1
                raw += len(line)
    R["A_size"] = {"rows": rows, "unpacked_bytes": raw, "read_secs": round(time.perf_counter() - t0, 1)}
    print("A", R["A_size"], flush=True)

    # B: naive - keep every raw line in memory, hour by hour
    held, steps = [], []
    base = rss_gb()[0]
    for i, p in enumerate(files):
        with gzip.open(p, "rb") as f:
            held.extend(f.read().splitlines())
        now = rss_gb()[0]
        steps.append({"hours": i + 1, "rows_held": len(held), "gb": round(now - base, 2)})
        print("B", steps[-1], flush=True)
        if now - base > a.naive_limit_gb:
            break
    per_row = steps[-1]["gb"] / steps[-1]["rows_held"]          # later hours are busier: scale by rows
    R["B_naive"] = {"steps": steps, "bytes_per_row": round(per_row * 2 ** 30, 1),
                    "projected_gb_for_all_rows": round(per_row * rows, 1), "stopped_at_hours": steps[-1]["hours"],
                    "limit_gb": a.naive_limit_gb}
    del held

    # C: one worker, streaming
    import gc
    gc.collect()
    base = rss_gb()[0]                       # the peak counter still remembers B, so measure the growth
    c_runs = []
    for _ in range(a.repeat):              # fastest of N, as Python's timeit does: the slow runs
        counts = {}                        # are this machine doing other work
        t0 = time.perf_counter()
        for p in files:
            count_file(p, counts)
        c_runs.append(time.perf_counter() - t0)
    c_secs = min(c_runs)
    c_top = top(counts, 10)
    R["C_one_worker"] = {"secs": round(c_secs, 1), "runs": [round(x, 1) for x in c_runs], "pages": len(counts), "top10": c_top,
                         "tally_gb": round(rss_gb()[0] - base, 2)}
    print("C", R["C_one_worker"]["secs"], c_top[:3], flush=True)
    del counts

    # D: map-reduce on 8 workers
    tmp = os.path.join(a.out, "shuffle")
    os.makedirs(tmp, exist_ok=True)
    for f in os.listdir(tmp):
        os.remove(os.path.join(tmp, f))
    d_runs = []
    for _ in range(a.repeat):
        for f in os.listdir(tmp):
            os.remove(os.path.join(tmp, f))
        t0 = time.perf_counter()
        with Pool(8) as pool:
            maps = pool.map(map_one, [(p, tmp) for p in files], chunksize=1)
            t1 = time.perf_counter()
            reds = pool.map(reduce_one, [(k, tmp) for k in range(BUCKETS)])
        t2 = time.perf_counter()
        d_runs.append((t2 - t0, t1 - t0, t2 - t1))
    best = min(d_runs)
    t0, t1, t2 = 0.0, best[1], best[0]
    d_top = heapq.nlargest(10, [x for r in reds for x in r[0]], key=lambda x: x[1])
    R["D_map_reduce"] = {"workers": 8, "runs": [round(x[0], 1) for x in d_runs], "map_secs": round(t1 - t0, 1), "reduce_secs": round(t2 - t1, 1),
                         "total_secs": round(t2 - t0, 1),
                         "map_worker_peak_gb_max": round(max(m[2] for m in maps), 2),
                         "reduce_worker_peak_gb_max": round(max(r[3] for r in reds), 2),
                         "pages_per_bucket": [r[1] for r in reds],
                         "shuffle_bytes": sum(os.path.getsize(os.path.join(tmp, f)) for f in os.listdir(tmp)),
                         "top10": d_top,
                         "hourly_leaders": [[os.path.basename(p)[19:21], m[3][0], m[3][1]] for p, m in zip(files, maps)]}
    assert [t for t, v in d_top] == [t for t, v in c_top] and d_top == c_top, "map-reduce disagrees"
    R["speedup"] = round(c_secs / (t2 - t0), 2)
    print("D", R["D_map_reduce"]["total_secs"], "speedup", R["speedup"], flush=True)
    for f in os.listdir(tmp):
        os.remove(os.path.join(tmp, f))

    json.dump(R, open(os.path.join(a.out, "results.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(json.dumps(R, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
