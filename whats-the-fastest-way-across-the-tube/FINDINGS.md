# What's the fastest way across the Tube? (shortest-path search, Dijkstra)

Reel 12-08. Run: `python fetch.py` (TfL API, ~45 paced calls) then `python experiment.py`, `python compare.py`,
`python example.py` -> results.json, compare_results.json, example_results.json. Plain Python.

## Data
TfL Unified API (Powered by TfL Open Data; contains OS data © Crown copyright and database rights), fetched 2026-11-25:
for each of the 11 tube lines, both directions, every branch's timetable from its first station gives the minutes
to each later stop. Differences = minutes between neighbouring stations (median across branches).
**272 stations, 759 one-way links.** Times are whole minutes in TfL's data (a link is 1-4 min); treat totals as ±1 min.
Not in the model: the Elizabeth line, Overground and DLR (not tube lines), walking, waiting for the first train.
**Assumption we state on screen:** changing line costs a fixed number of minutes (walk + wait), 5 by default.

## Running example: Tower Hill -> Earl's Court
| route | stops | changes | minutes (5-min changes) |
|---|---|---|---|
| **fewest stops**: Circle -> Central -> Waterloo & City -> Jubilee -> District | **11** | **4** | **45.5** |
| **fastest**: stay on the District line | 13 | 0 | **23** |
- Even if a change took only 1 minute, the fewest-stops route takes 29.5 vs 23. At 10 minutes a change: 65.5.
- Shortest-path search (Dijkstra) from Tower Hill, always extending the stop it can reach soonest: Monument,
  Aldgate East, Cannon Street, Mansion House, Whitechapel, Aldgate, Blackfriars, ... it is sure about **71** stations
  before it reaches Earl's Court - and then it knows no route can beat 23 minutes.

## How often it matters
- 45 well-known stations, 1,980 ordered pairs: for **778 (39%)** the fewest-stops route (ties broken by time) is
  slower than the fastest route. A random sample of all pairs: 66% (with ties broken arbitrarily - don't quote).
- Biggest gaps: Tower Hill -> Richmond (17 stops / 5 changes / 70 min vs District 22 stops / 43 min);
  Ealing Broadway -> Liverpool Street (14 stops / 4 changes / 58 min vs Central line 18 stops / 36 min - but the
  Elizabeth line is faster than both, so not the example).

## The answer depends on the price of a change
Angel -> Clapham Common: with 1-minute changes the fastest is Northern -> Victoria at King's Cross -> Northern at
Stockwell (2 changes, 21 min); at 3 minutes or more it stays on the Northern line (11 stops, 22 min).
The search does not know the map; it knows only the numbers it is given.
