"""
detail.py - pictures and the worked samples for the scene.
  out/map_all.png: every one of the 683,788 trees as a dot (green good, amber fair, red poor/dead/stump), on black.
  out/sample.json: one random 1,000 (seed 1), with map positions; one 1,000 from each of two neighbourhoods; the map
  projection, so the scene can place dots on the picture.
  python detail.py
"""
import csv, json, os
import numpy as np
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__))
rows = list(csv.DictReader(open(os.path.join(HERE, "data", "trees.csv"), encoding="utf-8")))
rows.sort(key=lambda r: int(r["tree_id"]))
lat = np.array([float(r["latitude"]) for r in rows]); lon = np.array([float(r["longitude"]) for r in rows])
health = np.array([r["health"] if r["status"] == "Alive" else r["status"] for r in rows])
good = health == "Good"; nta = np.array([r["nta_name"] for r in rows])
# equirectangular with cos(lat) correction, square canvas
k = np.cos(np.radians(40.7)); x = (lon - lon.min()) * k; y = lat.max() - lat
span = max(x.max(), y.max()); W = 1800; pad = 40
px = (pad + x / span * (W - 2 * pad)).astype(int); py = (pad + y / span * (W - 2 * pad)).astype(int)
img = np.zeros((W, W, 3), np.float32)
col = {"Good": (63, 185, 138), "Fair": (240, 192, 80), "Poor": (255, 92, 92), "Dead": (255, 92, 92), "Stump": (255, 92, 92)}
for h, c in col.items():
    m = health == h
    np.add.at(img, (py[m], px[m]), np.array(c, np.float32) * 0.55)
img = np.clip(img, 0, 255).astype(np.uint8)
Image.fromarray(img).save(os.path.join(HERE, "out", "map_all.png"))
Image.fromarray(np.where((img.sum(-1) > 0)[..., None], np.array([139, 135, 127], np.uint8), 0).astype(np.uint8)).save(
    os.path.join(HERE, "out", "map_grey.png"))
rng = np.random.default_rng(1)
pick = rng.choice(len(rows), 1000, replace=False)
def pts(idx): return [[round(float(px[i]) / W, 4), round(float(py[i]) / W, 4), bool(good[i])] for i in idx]
out = dict(W=W, random=dict(share=float(good[pick].mean()), pts=pts(pick)))
for name in ("Breezy Point-Belle Harbor-Rockaway Park-Broad Channel", "Westerleigh"):
    idx = np.where(nta == name)[0]; p = rng.choice(idx, 1000, replace=False)
    out[name] = dict(trees=int(len(idx)), truth=float(good[idx].mean()), share=float(good[p].mean()), pts=pts(p))
    print(name, out[name]["trees"], out[name]["truth"], out[name]["share"])
print("random 1000", out["random"]["share"])
json.dump(out, open(os.path.join(HERE, "out", "sample.json"), "w"))

# the first five 10-tree samples exactly as experiment.py drew them (same generator, same order)
r0 = np.random.default_rng(0)
ten = [r0.choice(len(rows), 10, replace=False) for _ in range(5)]
out["ten"] = [dict(share=float(good[t].mean()), pts=pts(t)) for t in ten]
print("ten-tree samples", [o["share"] for o in out["ten"]])
json.dump(out, open(os.path.join(HERE, "out", "sample.json"), "w"))

# map_health.png: each pixel coloured by the share of its trees in good health (red 0 -> green 1), brightness by count
cnt = np.zeros((W, W)); gd = np.zeros((W, W))
np.add.at(cnt, (py, px), 1); np.add.at(gd, (py, px), good.astype(float))
share = np.divide(gd, cnt, out=np.zeros_like(gd), where=cnt > 0)
red, green = np.array([255, 92, 92]), np.array([63, 185, 138])
rgb = (1 - share[..., None]) * red + share[..., None] * green
lum = np.clip(0.55 + 0.25 * np.log2(np.maximum(cnt, 1)), 0, 1)[..., None] * (cnt[..., None] > 0)
Image.fromarray((rgb * lum).clip(0, 255).astype(np.uint8)).save(os.path.join(HERE, "out", "map_health.png"))
