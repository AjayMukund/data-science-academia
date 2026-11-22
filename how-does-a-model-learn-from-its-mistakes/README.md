# How does a model learn from its mistakes? Gradient boosting from scratch

Gradient boosting in plain NumPy on California house prices (20,640 districts, 1990 census). Guessing the average
misses by $92,511 on districts the model never saw; one small tree (3 questions deep) by $58,600. Boosting fits each
new small tree to the mistakes of all the trees before it, at a tenth of its size: 10 trees miss by $62,572, 100 by
$36,130, 1,000 by $29,767 (about 6 seconds on a laptop CPU). Santa Monica (true value $450,000) climbs from $206,663
to $432,142 tree by tree. Too eager (learning rate 1.0) peaks at 340 trees ($32,043) and then memorises; a random
forest of 100 deep trees gets $32,707.

```bash
pip install numpy
curl -L -o cal_housing.tgz https://ndownloader.figshare.com/files/5976036
mkdir -p data && tar xzf cal_housing.tgz -C data          # -> data/CaliforniaHousing/cal_housing.data
python experiment.py                                       # writes out/results.json
```

Data: California Housing (Pace & Barry 1997), the copy scikit-learn uses. See `FINDINGS.md`.
Things to try: learning rates 0.03 and 0.3; deeper trees; stop when the held-out error stops falling.
