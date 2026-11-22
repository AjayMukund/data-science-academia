"""
experiment.py - how does an AI know what's in a photo? CLIP, measured.

Model: openai/clip-vit-base-patch32 (MIT), on our laptop's CPU. It was never trained on our labels: it turns a
photo into 512 numbers, a sentence into 512 numbers, and picks the sentence whose numbers point the same way.
Data: Oxford-IIIT Pet (Parkhi et al. 2012, CC BY-SA 4.0): 37 breeds of cats and dogs, the 3,669-photo test split.
https://thor.robots.ox.ac.uk/datasets/pets/images.tar.gz  and  annotations.tar.gz

  1. Zero-shot: for each test photo, 37 sentences "a photo of a {breed}, a type of pet." - right if the nearest
     sentence is the photo's breed. Chance is 1 in 37. Also: cat or dog (2 sentences), and the bare breed name.
  2. One photo followed: its top 5 sentences, with the softmax probabilities CLIP itself reports.
  3. Minimal pairs on that photo: change one word of the sentence, watch the score.
  4. The catch: write a different breed's name on a white card in the photo's corner. How often does the
     answer follow the writing instead of the animal?
  5. Search the other way: a sentence finds the nearest photos among all 3,669.
Writes out/results.json.   python experiment.py --model <clip folder> --data <oxford-pets folder>
"""
import argparse, json, os, time
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--data", required=True)
ap.add_argument("--photo", default="")          # the followed photo (file stem); default picked below
a = ap.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "out"); os.makedirs(OUT, exist_ok=True)
torch.manual_seed(0); torch.set_num_threads(max(1, os.cpu_count() - 2))

from transformers import CLIPModel, CLIPProcessor
model = CLIPModel.from_pretrained(a.model).eval()
proc = CLIPProcessor.from_pretrained(a.model)

# ---- the test split and the 37 breed names
rows = [l.split() for l in open(os.path.join(a.data, "annotations", "test.txt")) if l.strip()]
names = {}
for stem, cid, species, _ in rows:
    names[int(cid)] = " ".join(stem.split("_")[:-1])
CLASSES = [names[i].replace("_", " ") for i in sorted(names)]
SPECIES = {int(cid): ("cat" if sp == "1" else "dog") for _, cid, sp, _ in rows}
test = [(stem, int(cid) - 1) for stem, cid, _, _ in rows]
print(len(test), "test photos,", len(CLASSES), "breeds")


def pretty(c):
    return " ".join(w if w.isupper() else w.capitalize() if i == 0 or c[0].isupper() else w
                    for i, w in enumerate(c.split()))


@torch.no_grad()
def text_emb(sents):
    t = proc(text=sents, return_tensors="pt", padding=True)
    e = model.get_text_features(**t)
    return e / e.norm(dim=-1, keepdim=True)


@torch.no_grad()
def img_emb(ims):
    p = proc(images=ims, return_tensors="pt")
    e = model.get_image_features(**p)
    return e / e.norm(dim=-1, keepdim=True)


def load(stem):
    return Image.open(os.path.join(a.data, "images", stem + ".jpg")).convert("RGB")


SCALE = float(model.logit_scale.exp())

# ---- 1. embed every test photo once
t0 = time.time()
E = []
B = 32
for i in range(0, len(test), B):
    E.append(img_emb([load(s) for s, _ in test[i:i + B]]))
    if i % 640 == 0:
        print("  embedded", i, "%.0fs" % (time.time() - t0))
E = torch.cat(E)
secs = time.time() - t0
y = np.array([c for _, c in test])

prompts = ["a photo of a %s, a type of pet." % c for c in CLASSES]
T = text_emb(prompts)
sim = (E @ T.T).numpy()
pred = sim.argmax(1)
acc = float((pred == y).mean())
Tb = text_emb(CLASSES)
acc_bare = float(((E @ Tb.T).numpy().argmax(1) == y).mean())
Tcd = text_emb(["a photo of a cat.", "a photo of a dog."])
sp_true = np.array([0 if SPECIES[c + 1] == "cat" else 1 for c in y])
acc_catdog = float(((E @ Tcd.T).numpy().argmax(1) == sp_true).mean())
per = {CLASSES[k]: float((pred[y == k] == k).mean()) for k in range(len(CLASSES))}
conf = np.zeros((37, 37), int)
for t_, p_ in zip(y, pred):
    conf[t_, p_] += 1
np.fill_diagonal(conf, 0)
worst_pairs = sorted(((int(conf[i, j]), CLASSES[i], CLASSES[j]) for i in range(37) for j in range(37) if conf[i, j]),
                     reverse=True)[:8]
print("zero-shot 37-way %.1f%%  bare name %.1f%%  cat/dog %.1f%%  (%.1f photos/s)"
      % (100 * acc, 100 * acc_bare, 100 * acc_catdog, len(test) / secs))

# ---- 2. one photo followed: a confident, correct dog photo of a breed people know
if a.photo:
    pick = next(i for i, (s, _) in enumerate(test) if s == a.photo)
else:
    want = CLASSES.index("shiba inu")
    probs_all = torch.softmax(torch.tensor(sim) * SCALE, 1).numpy()
    cand = [i for i in range(len(test)) if y[i] == want and pred[i] == want]
    pick = sorted(cand, key=lambda i: -probs_all[i, want])[len(cand) // 4]   # clearly right, not the easiest
stem, cls = test[pick]
img = load(stem)
pr = torch.softmax(torch.tensor(sim[pick]) * SCALE, 0).numpy()
top5 = [(CLASSES[k], float(pr[k])) for k in np.argsort(-pr)[:5]]
print("followed", stem, CLASSES[cls], top5)
img.save(os.path.join(OUT, "followed.jpg"))

# ---- 3. minimal pairs: raw cosine similarity of the photo against one-word changes
e1 = E[pick:pick + 1]
pairs = ["a photo of a dog.", "a photo of a cat.", "a photo of a fox.", "a photo of a shiba inu.",
         "a photo of a dog on a sofa.", "a photo of a dog in the snow.", "a drawing of a dog.", "a photo of a car."]
pair_sims = [(s, float((e1 @ text_emb([s]).T)[0, 0])) for s in pairs]
print(pair_sims)

# ---- 4. the catch: a breed's name written on a card in the corner
FONT = None
for f in ("arialbd.ttf", "Arial Bold.ttf", "DejaVuSans-Bold.ttf"):
    try:
        FONT = f; ImageFont.truetype(f, 20); break
    except OSError:
        FONT = None


def with_card(im, word):
    im = im.copy(); W, H = im.size
    size = max(14, int(min(W, H) * 0.09))
    f = ImageFont.truetype(FONT, size) if FONT else ImageFont.load_default()
    d = ImageDraw.Draw(im)
    tw, th = d.textbbox((0, 0), word, font=f)[2:]
    pad = int(size * 0.35)
    x0, y0 = W - tw - 3 * pad, H - th - 3 * pad
    d.rectangle([x0, y0, x0 + tw + 2 * pad, y0 + th + 2 * pad], fill="white")
    d.text((x0 + pad, y0 + pad * 0.6), word, fill="black", font=f)
    return im


rng = np.random.default_rng(0)
right = [i for i in range(len(test)) if pred[i] == y[i]]
sub = sorted(rng.choice(right, size=min(500, len(right)), replace=False).tolist())
targets = []
for i in sub:                         # a breed of the SAME species, so the picture alone can't settle it trivially
    same = [k for k in range(37) if SPECIES[k + 1] == SPECIES[y[i] + 1] and k != y[i]]
    targets.append(int(rng.choice(same)))
hij, follows = [], 0
for j in range(0, len(sub), B):
    ims = [with_card(load(test[i][0]), CLASSES[targets[j + n]]) for n, i in enumerate(sub[j:j + B])]
    p2 = (img_emb(ims) @ T.T).numpy().argmax(1)
    hij += p2.tolist()
hij = np.array(hij)
tgt = np.array(targets)
keep = float((hij == y[sub]).mean())
follow = float((hij == tgt).mean())
print("card test on %d photos it got right: still right %.1f%%, follows the card %.1f%%"
      % (len(sub), 100 * keep, 100 * follow))
# the followed photo with a card
fake = CLASSES.index("pomeranian")
imc = with_card(img, CLASSES[fake]); imc.save(os.path.join(OUT, "followed_card.jpg"))
prc = torch.softmax((img_emb([imc]) @ T.T)[0] * SCALE, 0).numpy()
card_top = [(CLASSES[k], float(prc[k])) for k in np.argsort(-prc)[:5]]
print("followed + card:", card_top)

# ---- 5. search by sentence
search = {}
for q in ("a dog in the snow", "a cat asleep in a box", "a puppy wearing a hat"):
    s = (E @ text_emb(["a photo of " + q + "."]).T).numpy()[:, 0]
    search[q] = [(test[k][0], float(s[k])) for k in np.argsort(-s)[:4]]
print(search)

json.dump(dict(n_test=len(test), classes=CLASSES, chance=1 / 37, acc=acc, acc_bare=acc_bare, acc_catdog=acc_catdog,
               photos_per_s=len(test) / secs, per_breed=per, worst_pairs=worst_pairs, logit_scale=SCALE,
               followed=dict(stem=stem, breed=CLASSES[cls], top5=top5, pair_sims=pair_sims, card=CLASSES[fake],
                             card_top5=card_top),
               card_test=dict(n=len(sub), still_right=keep, follows_card=follow), search=search),
          open(os.path.join(OUT, "results.json"), "w"), indent=1)
print("wrote out/results.json")
