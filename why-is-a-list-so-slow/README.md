# Why is a list so slow? Lists vs dictionaries in Python

A thousand membership checks (`in`) against a million ticket numbers: about
ten seconds with a list, under a millisecond with a dictionary. A list walks
until it finds a match; a dictionary hashes the key and jumps to its slot.

```bash
python snippet.py         # the exact code shown in the video
python experiment.py      # the fuller measurement, incl. build time and memory
```

Timings depend on your machine. See `FINDINGS.md` for ours. Try: use a `set`
instead of a dictionary, or look up tickets that are not in the list.
