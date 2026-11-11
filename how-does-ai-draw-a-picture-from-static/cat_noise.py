"""
cat_noise.py - "Take a real photo. Add a little static. Then a little more." The model's own forward process on one
real photo: encode it into the sketch space, add the scheduler's noise at levels 0..999, decode each. Then the one job
the network learned: at the halfway level, guess the noise, and from that guess the clean picture.

  python cat_noise.py --models <folder> --photo cat_photo.png --out out
"""
import argparse, json, os
import numpy as np
import torch
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--models", required=True)
ap.add_argument("--photo", required=True)
ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"))
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
torch.set_grad_enabled(False)
from diffusers import StableDiffusionPipeline, DDIMScheduler
pipe = StableDiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", cache_dir=a.models,
                                               variant="fp16", torch_dtype=torch.float32, safety_checker=None,
                                               requires_safety_checker=False)
sched = DDIMScheduler.from_config(pipe.scheduler.config)
vae, unet = pipe.vae, pipe.unet
img = Image.open(a.photo).convert("RGB").resize((512, 512), Image.LANCZOS)
x = torch.from_numpy(np.asarray(img)).float().permute(2, 0, 1)[None] / 127.5 - 1
lat = vae.encode(x).latent_dist.mean * vae.config.scaling_factor


def decode(l):
    im = vae.decode(l / vae.config.scaling_factor).sample
    return Image.fromarray(((im.clamp(-1, 1) + 1) / 2 * 255).round().byte()[0].permute(1, 2, 0).numpy())


g = torch.Generator().manual_seed(7)
eps = torch.randn(lat.shape, generator=g)
res = {"levels": {}}
img.save(os.path.join(a.out, "cat_original.png"))
for t in (0, 100, 250, 500, 750, 999):
    ab = float(sched.alphas_cumprod[t])
    noisy = sched.add_noise(lat, eps, torch.tensor([t])) if t > 0 else lat
    decode(noisy).save(os.path.join(a.out, "cat_t%03d.png" % t))
    res["levels"][t] = {"signal_share": ab, "image": "cat_t%03d.png" % t}       # how much of the photo is left
    print("t=%d: %.0f%% photo" % (t, 100 * ab), flush=True)

# the one job: look at the noisy cat (t=500) and guess the noise; from that guess, the clean picture
t = 500
noisy = sched.add_noise(lat, eps, torch.tensor([t]))
ids = pipe.tokenizer(["a photograph of a ginger cat on a windowsill"], padding="max_length", max_length=77,
                     return_tensors="pt").input_ids
cond = pipe.text_encoder(ids)[0]
pred = unet(noisy, t, encoder_hidden_states=cond).sample
ab = sched.alphas_cumprod[t]
x0 = (noisy - (1 - ab).sqrt() * pred) / ab.sqrt()
decode(x0).save(os.path.join(a.out, "cat_guess_t500.png"))
decode(pred * 1.0).save(os.path.join(a.out, "cat_guessed_noise_t500.png"))
res["guess_t500"] = {"noise_corr": float(np.corrcoef(pred.flatten().numpy(), eps.flatten().numpy())[0, 1]),
                     "image": "cat_guess_t500.png"}
json.dump(res, open(os.path.join(a.out, "cat.json"), "w"), indent=1)
print("noise guess vs true noise, correlation %.2f" % res["guess_t500"]["noise_corr"])
