"""
pictures.py - what the video shows: CLIP's real 512 numbers for the puppy photo and for its sentences, and real
photos from the test split (the failures in beat 7, a wall of one photo per breed, a pile of thumbnails).
Writes out/pictures.npz and out/photos/*.jpg.   python pictures.py --model <clip folder> --data <oxford-pets folder>
"""
import argparse, json, os
import numpy as np
import torch
from PIL import Image

ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--data", required=True)
a = ap.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "out"); PH = os.path.join(OUT, "photos")
os.makedirs(PH, exist_ok=True)
from transformers import CLIPModel, CLIPProcessor
model = CLIPModel.from_pretrained(a.model).eval(); proc = CLIPProcessor.from_pretrained(a.model)
R = json.load(open(os.path.join(OUT, "results.json")))
CL = R["classes"]
T = None


@torch.no_grad()
def te(s):
    e = model.get_text_features(**proc(text=s, return_tensors="pt", padding=True)); return e / e.norm(dim=-1, keepdim=True)


@torch.no_grad()
def ie(ims):
    e = model.get_image_features(**proc(images=ims, return_tensors="pt")); return e / e.norm(dim=-1, keepdim=True)


load = lambda s: Image.open(os.path.join(a.data, "images", s + ".jpg")).convert("RGB")


def save(stem, name, h=720, square=False):
    im = load(stem)
    if square:                                       # wall and pile thumbnails: centre-cropped square
        m = min(im.size); im = im.crop(((im.width - m) // 2, (im.height - m) // 2, (im.width + m) // 2, (im.height + m) // 2))
    im = im.resize((round(im.width * h / im.height), h), Image.LANCZOS)
    im.save(os.path.join(PH, name + ".jpg"), quality=92)


T = te(["a photo of a %s, a type of pet." % c for c in CL])
out = {}
pup = R["followed"]["stem"]
out["photo"] = ie([load(pup)])[0].numpy()
out["card"] = ie([Image.open(os.path.join(OUT, "followed_card.jpg"))])[0].numpy()
for k, s in (("t_shiba", "a photo of a shiba inu."), ("t_car", "a photo of a car."), ("t_dog", "a photo of a dog."),
             ("t_cat", "a photo of a cat.")):
    out[k] = te([s])[0].numpy()
save(pup, "puppy"); Image.open(os.path.join(OUT, "followed_card.jpg")).save(os.path.join(PH, "puppy_card.jpg"), quality=92)
save(R["search"]["a puppy wearing a hat"][0][0], "hat")

rows = [l.split() for l in open(os.path.join(a.data, "annotations", "test.txt")) if l.strip()]
by = {}
for stem, cid, _, _ in rows:
    by.setdefault(int(cid) - 1, []).append(stem)


def preds(stems):
    return (ie([load(s) for s in stems]) @ T.T).argmax(1).numpy()


picks = {}
for breed, wrong in (("Bombay", "Russian Blue"), ("Ragdoll", "Birman")):
    k, w = CL.index(breed), CL.index(wrong)
    p = preds(by[k])
    picks[breed + "_as_" + wrong] = next(s for s, q in zip(by[k], p) if q == w)
    picks[breed + "_right"] = next(s for s, q in zip(by[k], p) if q == k)
k = CL.index("Birman"); p = preds(by[k]); picks["Birman_right"] = next(s for s, q in zip(by[k], p) if q == k)
k = CL.index("Russian Blue"); p = preds(by[k]); picks["Russian_right"] = next(s for s, q in zip(by[k], p) if q == k)
for name, stem in picks.items():
    save(stem, name)
print(picks)

# left out of the wall and pile: a recognisable person (samoyed_191), writing in the photo (Maine_Coon_262)
SKIP = {"samoyed_191", "Maine_Coon_262"}
# one photo per breed, the first the model gets right (beat 5's sorted wall), and a pile of 48 for the shelter
wall = []
for k in range(37):
    p = preds(by[k])
    wall.append(next(s for s, q in zip(by[k], p) if q == k and s not in SKIP))
for i, s in enumerate(wall):
    save(s, "wall_%02d" % i, h=240, square=True)
rng = np.random.default_rng(1)
pile = sorted(rng.choice([r[0] for r in rows if r[0] not in SKIP], 48, replace=False).tolist())
for i, s in enumerate(pile):
    save(s, "pile_%02d" % i, h=240, square=True)
out["wall"] = np.array(wall); out["pile"] = np.array(pile)
json.dump(dict(picks=picks, wall=wall, pile=pile), open(os.path.join(OUT, "pictures.json"), "w"), indent=1)
np.savez_compressed(os.path.join(OUT, "pictures.npz"), **out)
print("cos photo/shiba %.3f card/shiba %.3f" % (out["photo"] @ out["t_shiba"], out["card"] @ out["t_shiba"]))
