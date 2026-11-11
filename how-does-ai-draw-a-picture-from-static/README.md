# How does AI draw a picture from static? Stable Diffusion, step by step

Stable Diffusion 1.5 driven by hand on a laptop CPU: 30 DDIM steps turn random noise into "a photograph of a red
bicycle leaning against a brick wall" (seed 7). At every step we save the noisy picture and the model's guess at the
finished one: the guess matches the final picture at 0.28 after step 1, 0.80 after step 10, 0.97 after step 20.
Also: fewer steps (1 to 50), guidance off (a dark bicycle), "red" to "blue" (the wall turns blue too), other seeds,
and `cat_noise.py`, which adds the model's own noise to any photo and asks it to guess the photo back.

```bash
pip install torch==2.5.1 diffusers==0.31.0 transformers==4.46.3 numpy pillow
python experiment.py --models path/to/hf-cache                      # about 8 s a step on CPU; ~40 min in all
python cat_noise.py --models path/to/hf-cache --photo your_photo.png
```

Model: stable-diffusion-v1-5/stable-diffusion-v1-5 (CreativeML OpenRAIL-M; about 4 GB, downloaded on first run).
See `FINDINGS.md`. Things to try: your own prompt; guidance 3 vs 15; the same prompt over ten seeds.
