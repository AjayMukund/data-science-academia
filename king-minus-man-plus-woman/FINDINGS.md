# Findings: king minus man plus woman

GloVe, 400,000 words x 300 numbers. Cosine similarity.

- king·queen **0.634**, king·man 0.333, king·strawberry **0.044**.

| sum | over all 400,000 words | with the three input words removed |
|---|---|---|
| king − man + woman | **king 0.757**, queen 0.671, princess 0.543 | **queen 0.671**, princess 0.543, throne 0.539 |
| uncle − man + woman | **uncle 0.738**, aunt 0.737, niece 0.708 | aunt 0.737 |
| paris − france + japan | **tokyo 0.823**, japan 0.713 | tokyo 0.823 |
| bigger − big + small | larger 0.777, smaller 0.775 | same |

The 2D picture in `out/results.json` is a flat shadow of 300 numbers: in it
the king − man + woman point sits nearer queen, which is not true in the full
300 numbers. Use it to see the man→woman direction, not to find nearest words.
