"""Convolution - finding crater rims in a NASA Moon image with one 3x3 grid.

Image: NASA Goddard, "Full Moon" (GSFC_20171208_Archive_e000868), built from Lunar
Reconnaissance Orbiter maps; public domain (NASA media guidelines). Re-downloaded
if missing.

Measures, on the grey image (0-255):
  - the edge grid (Sobel: -1 0 1 / -2 0 2 / -1 0 1, "right minus left") slid over
    every pixel, then the same grid turned on its side; the edge map is the two
    combined (hypot). The Laplacian was tried first: it drew speckle, not rims.
  - one 3x3 patch on a flat plain and one across a plain's edge, with the nine
    products and their sum, for the worked example
  - the share of the Moon the grid marks as edges (strength > 60), and the blur
    grid (all 1/9) for contrast
Writes moon_grey.png, moon_edges.png, moon_blur.png for viewing.

    python experiment.py
"""
import json
import os
import urllib.request

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(HERE, "download"), exist_ok=True)
os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
JPG = os.path.join(HERE, "download", "moon_large.jpg")
if not os.path.exists(JPG):
    urllib.request.urlretrieve("https://images-assets.nasa.gov/image/GSFC_20171208_Archive_e000868/"
                               "GSFC_20171208_Archive_e000868~large.jpg", JPG)
g = np.asarray(Image.open(JPG).convert("L").resize((960, 960), Image.LANCZOS), dtype=float)

EDGE = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)
BLUR = np.full((3, 3), 1 / 9)


def conv(img, k):
    out = np.zeros_like(img)
    H, W = img.shape
    for dy in range(3):
        for dx in range(3):
            out[1:H - 1, 1:W - 1] += k[dy, dx] * img[dy:H - 2 + dy, dx:W - 2 + dx]
    return out


e = conv(g, EDGE)                               # edges running up and down
e2 = conv(g, EDGE.T)                            # the grid on its side: edges running across
m = np.hypot(e, e2)
b = conv(g, BLUR)
disc = g > 20                                   # pixels on the Moon, not space
edge = m > 60
out = {"size": g.shape, "moon_pixels": int(disc.sum()), "edge_pixels": int((edge & disc).sum()),
       "edge_share_of_moon": float((edge & disc).sum() / disc.sum())}

# worked patches: the flattest 3x3 on the Moon, and the strongest rim
# inside the lit surface: every pixel within 12 px is on the Moon (not its rim against space)
# (a 25 x 25 sliding minimum, in plain NumPy)
_win = np.lib.stride_tricks.sliding_window_view(np.pad(g, 12, mode="edge"), (25, 25))
inner = _win.min(axis=(2, 3)) > 40
flat_y, flat_x = np.unravel_index(np.where(inner & (g > 60), m, 1e9).argmin(), e.shape)
rim_y, rim_x = np.unravel_index(np.where(inner, np.abs(e), -1e9).argmax(), e.shape)
for tag, (yy, xx) in (("flat", (flat_y, flat_x)), ("edge", (rim_y, rim_x))):
    p = g[yy - 1:yy + 2, xx - 1:xx + 2].astype(int)
    out[tag] = dict(at=(int(yy), int(xx)), patch=p.tolist(), products=(p * EDGE).astype(int).tolist(),
                    total=int((p * EDGE).sum()))
Image.fromarray(g.astype(np.uint8)).save(os.path.join(HERE, "out", "moon_grey.png"))
Image.fromarray(np.clip(m * 1.2, 0, 255).astype(np.uint8)).save(os.path.join(HERE, "out", "moon_edges.png"))
# the upright grid alone: only edges that run up and down
Image.fromarray(np.clip(np.abs(e) * 1.2, 0, 255).astype(np.uint8)).save(os.path.join(HERE, "out", "moon_edges_x.png"))
Image.fromarray(np.clip(b, 0, 255).astype(np.uint8)).save(os.path.join(HERE, "out", "moon_blur.png"))
print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(HERE, "out", "results.json"), "w"), indent=1)
