# Could 1,000 trees have told New York the answer? Sampling and margin of error

In 2015-16 New York counted every street tree: 683,788 records (dead trees and stumps included), 77.3% in good
health. We treat that census as the truth and ask what a survey would have said. 1,000 random trees give 78.2%;
10,000 random samples of 1,000 land within ±2.6 points 95% of the time. 10 trees give anything from 50% to 100%,
4,000 give ±1.3, and the size of the place barely matters (Manhattan ±2.8, the whole city ±2.6). But 1,000 trees
from the Rockaways say 52%, and checking all 4,351 of them still says 52%: who you ask beats how many.

```bash
pip install numpy requests pillow
python fetch.py        # downloads the census columns from NYC Open Data to data/trees.csv (~40 MB)
python experiment.py   # 10,000 samples at each size -> out/results.json, out/estimates.npz (about a minute)
python detail.py       # the tree map and the worked samples -> out/map_all.png, out/map_health.png, out/sample.json
```

See `FINDINGS.md`. Data: 2015 Street Tree Census, NYC Parks, via NYC Open Data (dataset `uvpi-gqnh`).
Things to try: ask a different question (share of London planetrees, of trees in poor health); sample by borough in
proportion to its size (stratified) and compare the margin; sample whole blocks instead of trees and watch the
margin grow.
