# How a computer finds edges - convolution on the Moon

Run: `python experiment.py` (2026-11-04).
Image: NASA Goddard, "Full Moon", GSFC_20171208_Archive_e000868 (images.nasa.gov),
**built from Lunar Reconnaissance Orbiter maps** (a NASA visualisation, not one
camera frame). Public domain under NASA's media
guidelines; credit "NASA Goddard / LRO". Greyscale, 960 x 960.

The grid (Sobel, "right minus left"):

```
-1  0  1
-2  0  2
-1  0  1
```

Slide it over every pixel: multiply the nine pixels under it by the nine
numbers, add up. Bright-left / dark-right gives a big negative; flat gives 0.

| Patch | Pixels (0-255) | Sum |
|---|---|---|
| flat plain (row 91, col 466) | 125 128 132 / 126 120 118 / 124 128 133 | **0** |
| across a dark plain's edge (row 455, col 852) | 236 178 129 / 227 179 122 / 192 148 120 | **-389** |

Edge patch products: -236 0 129 / -454 0 244 / -192 0 120.

- The same grid turned on its side finds edges running across; the two
  combined draw every crater ring and the outline of the dark plains
  (`moon_edges.png`). 28.9% of the Moon's pixels pass strength 60.
- The Laplacian (8 in the middle, -1 around) was tried first: it lit up
  speckle everywhere and its "best rim" was one bright pixel. Not used.
- Blur grid (all nine = 1/9) for contrast: `moon_blur.png`.
- Image networks learn grids like this on their own in their first layer
  (Krizhevsky et al. 2012, fig. 3) - the only claim beyond our own run.
