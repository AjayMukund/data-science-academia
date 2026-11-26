# What's the fastest way across the Tube? How route search works

The London Underground built from TfL's own timetables (TfL Unified API; Powered by TfL Open Data, contains OS data © Crown copyright and database rights): 272 stations, 759 one-way links in minutes. Changing line costs a fixed 5 minutes (our assumption). **Tower Hill to Earl's Court: the fewest stops is 11 with four changes, 45.5 minutes; staying on the District line is 13 stops, 23 minutes.** Shortest-path search (Dijkstra) settles 71 stations before it reaches Earl's Court. Between 45 big stations, the fewest-stops route is slower for 778 of 1,980 trips.

```bash
python fetch.py           # ~45 paced calls to the TfL API -> download/edges.json, stations.json
python coords.py          # station positions, for drawing
python experiment.py      # fewest stops vs fastest on a sample of trips
python compare.py         # 45 big stations, and how the change cost flips routes
python example.py ; python scene_data.py
```

See `FINDINGS.md`. TfL changes its timetables, so minutes can shift by one. Try: set the change cost to 0 or 10 in `compare.py`.
