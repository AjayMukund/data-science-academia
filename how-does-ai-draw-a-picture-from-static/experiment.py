"""
experiment.py - how does an AI draw a picture from noise? Stable Diffusion 1.5, measured step by step.

Model: stable-diffusion-v1-5/stable-diffusion-v1-5 (CreativeML OpenRAIL-M), half-precision weights loaded as float32,
run on CPU with the DDIM scheduler, driven by hand so every step can be looked at. Every image is saved to out/.
The running example: "a photograph of a red bicycle leaning against a brick wall", seed 7.

  1. The start: the random noise it begins from, decoded as it is.
  2. 30 steps: after steps 1, 2, 3, 5, 10, 20, 30, the current noisy picture and the model's guess at the finished one;
     how similar each guess is to the final picture (correlation, in the model's small sketch space).
  3. Fewer steps: the same noise finished in 1, 2, 5, 10, 20, 50 steps.
  4. One word: "red" -> "blue", same noise.  5. Same words, four noises (seeds 7, 8, 9, 10).
  6. No steering: guidance 1 (prompt ignored) vs 7.5.  7. Sizes and seconds per step.

  python experiment.py --models <folder> --out out
"""
import argparse, json, os, time
import numpy as np
import torch
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--models", required=True)
ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"))
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
torch.set_grad_enabled(False)

from diffusers import StableDiffusionPipeline, DDIMScheduler
pipe = StableDiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", cache_dir=a.models,
                                               variant="fp16", torch_dtype=torch.float32, safety_checker=None,
                                               requires_safety_checker=False)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
unet, vae, sched = pipe.unet, pipe.vae, pipe.scheduler
PROMPT = "a photograph of a red bicycle leaning against a brick wall"
res = {"prompt": PROMPT, "unet_params": int(sum(p.numel() for p in unet.parameters())),
       "vae_params": int(sum(p.numel() for p in vae.parameters())),
       "text_params": int(sum(p.numel() for p in pipe.text_encoder.parameters())),
       "latent_shape": [4, 64, 64], "image_shape": [512, 512, 3]}


def embed(prompt):
    ids = pipe.tokenizer([prompt, ""], padding="max_length", max_length=77, return_tensors="pt").input_ids
    e = pipe.text_encoder(ids)[0]
    return e[:1], e[1:]                                    # the prompt, and the empty prompt (for guidance)


def noise(seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randn((1, 4, 64, 64), generator=g)


def decode(lat):
    img = vae.decode(lat / vae.config.scaling_factor).sample
    img = ((img.clamp(-1, 1) + 1) / 2 * 255).round().byte()[0].permute(1, 2, 0).numpy()
    return Image.fromarray(img)


def run(prompt, seed, steps, guidance=7.5, keep=()):
    cond, uncond = embed(prompt)
    sched.set_timesteps(steps)
    lat = noise(seed) * sched.init_noise_sigma
    kept, guesses = {}, []
    t0 = time.time()
    for i, t in enumerate(sched.timesteps):
        if guidance == 1.0:
            eps = unet(lat, t, encoder_hidden_states=cond).sample
        else:
            both = unet(torch.cat([lat, lat]), t, encoder_hidden_states=torch.cat([uncond, cond])).sample
            eu, ec = both.chunk(2)
            eps = eu + guidance * (ec - eu)                # steer: push away from "anything" towards the words
        out = sched.step(eps, t, lat)
        lat = out.prev_sample
        guesses.append(out.pred_original_sample.clone())
        if (i + 1) in keep:
            kept[i + 1] = (lat.clone(), out.pred_original_sample.clone())
    secs = time.time() - t0
    return lat, kept, guesses, secs


def save(img, name):
    img.save(os.path.join(a.out, name + ".png"))
    return name + ".png"


# 1. the start
save(decode(noise(7)), "start_noise")

# 2. thirty steps, watched
KEEP = (1, 2, 3, 5, 10, 20, 30)
lat, kept, guesses, secs = run(PROMPT, 7, 30, keep=KEEP)
final = guesses[-1].flatten()
res["steps30"] = {"secs": round(secs, 1), "secs_per_step": round(secs / 30, 2), "frames": {}}
corr = [float(np.corrcoef(g.flatten().numpy(), final.numpy())[0, 1]) for g in guesses]
res["steps30"]["guess_vs_final_corr"] = corr
for k in KEEP:
    noisy, guess = kept[k]
    res["steps30"]["frames"][k] = {"noisy": save(decode(noisy), "s30_step%02d_noisy" % k),
                                   "guess": save(decode(guess), "s30_step%02d_guess" % k), "corr": corr[k - 1]}
save(decode(lat), "final_red_seed7")
print("30 steps: %.1fs; guess-vs-final corr at steps 1,2,3,5,10,20:" % secs, [round(corr[k - 1], 2) for k in KEEP], flush=True)

# 3. fewer steps, same noise
res["by_steps"] = {}
for n in (1, 2, 5, 10, 20, 50):
    lat_n, _, _, s = run(PROMPT, 7, n)
    res["by_steps"][n] = {"image": save(decode(lat_n), "steps%02d" % n), "secs": round(s, 1)}
    print("steps", n, round(s, 1), flush=True)

# 4. one word
lat_b, _, _, _ = run(PROMPT.replace("red", "blue"), 7, 30)
res["blue"] = save(decode(lat_b), "final_blue_seed7")
# 5. same words, other noise
res["seeds"] = {s: save(decode(run(PROMPT, s, 30)[0]), "final_red_seed%d" % s) for s in (8, 9, 10)}
# 6. no steering
lat_g1, _, _, _ = run(PROMPT, 7, 30, guidance=1.0)
res["no_guidance"] = save(decode(lat_g1), "final_red_seed7_guidance1")
json.dump(res, open(os.path.join(a.out, "results.json"), "w"), indent=1)
print("DONE", json.dumps({k: v for k, v in res.items() if k in ("unet_params", "vae_params", "text_params")}))
