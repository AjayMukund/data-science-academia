"""pair.py - the minimal pairs: close the top of an open 4 (test digit 65) with a short stroke; remove a stray dot
from a 9 (test digit 9587). What does the network say, and the closest-average rule? Uses out/net.npz, averages.npy.
  python pair.py --data download"""
import argparse, gzip, json, os
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True)
ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")); a = ap.parse_args()
X = np.frombuffer(gzip.open(os.path.join(a.data, "t10k-images-idx3-ubyte.gz")).read(), np.uint8, offset=16).reshape(-1, 28, 28).astype(np.float32) / 255
n = np.load(os.path.join(a.out, "net.npz")); avg = np.load(os.path.join(a.out, "averages.npy"))


def read(img):
    x = img.reshape(1, 784)
    h = np.maximum(x @ n["W1"] + n["b1"], 0)
    z = h @ n["W2"] + n["b2"]; p = np.exp(z - z.max()); p /= p.sum()
    d = ((x - avg) ** 2).sum(1)
    return {"net": int(p.argmax()), "p": [round(float(v), 4) for v in p[0]], "rule": int(d.argmin()),
            "rule_dist_4": float(d[4]), "rule_dist_9": float(d[9]), "active_neurons": int((h > 0).sum())}


out = {}
four = X[65].copy(); out["four_65"] = read(four)
closed = four.copy(); closed[5:7, 12:17] = 1.0              # a short stroke across the top: 10 pixels
out["four_65_closed"] = read(closed); out["closed_pixels_changed"] = int((closed != four).sum())
nine = X[9587].copy(); out["nine_9587"] = read(nine)
clean = nine.copy(); clean[3:7, 5:10] = 0.0                 # rub out the stray dot
out["nine_9587_clean"] = read(clean); out["clean_pixels_changed"] = int((clean != nine).sum())
np.save(os.path.join(a.out, "four_closed.npy"), closed); np.save(os.path.join(a.out, "nine_clean.npy"), clean)
json.dump(out, open(os.path.join(a.out, "pair.json"), "w"), indent=1)
for k, v in out.items():
    print(k, v if not isinstance(v, dict) else {kk: (vv if kk != "p" else [round(x, 3) for x in vv]) for kk, vv in v.items()})
