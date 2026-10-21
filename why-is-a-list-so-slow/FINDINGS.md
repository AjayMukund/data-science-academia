# Findings: why is a list so slow?

On our laptop (Intel i3-10105, Windows, CPython 3.10), machine otherwise idle.

- `snippet.py`, three runs: 1,000 lookups in a list of a million took
  **9.7 / 10.2 / 13.3 s**; in a dictionary **0.0004-0.0005 s**. The video
  shows the middle run: 10.2499 s vs 0.0005 s, over ten thousand times faster.
  (`random.sample` is unseeded there, so runs vary.)
- A random ticket sits about 54% of the way down the list: roughly half a
  million comparisons per lookup. A dictionary lookup is one hash and one jump.
- The catch: building the dictionary took **0.28 s**, and its table is
  **41.9 MB** against the list's **8.4 MB** (the ticket strings are shared).
  It pays for itself after about **20** list lookups.
