# Diffusion master (11-18): how does an AI draw a picture from noise? — measured (2026-11-10)

**Model:** Stable Diffusion 1.5 (stable-diffusion-v1-5/stable-diffusion-v1-5, **CreativeML OpenRAIL-M**: outputs may be
used commercially; the licence's use restrictions apply). Half-precision weights loaded as float32, run on our laptop's
CPU with the DDIM scheduler, driven step by step in `experiment.py` (research venv: torch 2.5.1 CPU, diffusers 0.31.0,
transformers 4.46.3). Every image is real output, saved in `out/`.
**Prompt (the running example):** "a photograph of a red bicycle leaning against a brick wall", seed 7, 30 steps,
guidance 7.5.

## The parts
- The picture is drawn in a small "sketch" space: **4 × 64 × 64 = 16,384 numbers**, against 512 × 512 × 3 = 786,432
  for the finished image (**48x fewer**). A second network (the VAE decoder, 83.7 M parameters) paints it up to pixels.
- The denoiser (the U-Net): **859,520,964** parameters (~860 M). The text encoder (CLIP): 123 M.
- Per the model card, SD v1 was trained on LAION-5B subsets (laion2B-en, then "aesthetic" subsets): roughly two billion
  captioned pictures from the internet. (Cite the model card; we didn't measure this.)

## Thirty steps, watched (`s30_step*_noisy.png`, `s30_step*_guess.png`)
- The start: pure random noise, decoded = colourful static (`start_noise.png`).
- The working picture stays static-like until about step 20 of 30; at step 20 a bicycle shows through the noise.
- At every step the model also makes a guess at the finished picture. Similarity of that guess to the final picture
  (correlation, in the sketch space): **step 1: 0.28** (a red blur on a reddish wall), 2: 0.45, 3: 0.55, 5: 0.49 (a
  rough red bicycle), **10: 0.80** (a clear bicycle), **20: 0.97**, 30: 1.00.
- Time: **243 s for 30 steps = 8.1 s per step** on our laptop (two U-Net passes per step: with and without the words).

## Fewer steps, same noise (`steps01..50.png`)
- 1 step: still static. 2: a smeared texture, no bicycle. 5: a broken bicycle. **10: a good red bicycle.**
  20 and 50: almost the same picture as 30. Times: 7.0 / 15.1 / 36.5 / 72.4 / 145.1 / 363.6 s.

## Steering with words (guidance)
- Each step runs the denoiser twice: with the prompt and with an empty prompt, and pushes 7.5x further in the
  direction the words point.
- **Guidance off (1.0), same noise:** a bicycle against a grey brick wall, but **dark, not red** — the words barely
  steer (`final_red_seed7_guidance1.png`).
- **One word, same noise:** "red" → "blue": the same layout, a blue bicycle — and the wall turns blue too
  (`final_blue_seed7.png`).
- **Same words, other noise** (seeds 8, 9, 10): three different red bicycles in different scenes; two walls turn red.

## Cautions for the script
- "Similarity" is a correlation between guesses, not "percent finished"; say "already close to the final picture".
- No claims about how a given real product works (Midjourney, DALL·E etc. differ); this is SD 1.5, measured.
- Training data: model card, not our measurement; the ownership debate is real and unresolved — mention, don't judge.
- CPU timings: "on our laptop"; GPUs are far faster.
