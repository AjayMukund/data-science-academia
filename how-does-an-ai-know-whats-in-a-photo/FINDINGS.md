# How does an AI know what's in a photo? CLIP, measured (2026-11-22)

**Model:** openai/clip-vit-base-patch32 (MIT), on our laptop's CPU, 11.8 photos a second.
**Data:** Oxford-IIIT Pet (Parkhi et al. 2012, CC BY-SA 4.0), test split: 3,669 photos, 37 breeds (12 cat, 25 dog).
CLIP was never trained on these labels. Each breed becomes a sentence; the photo goes to the nearest sentence.
Published fact (not measured): CLIP was trained on 400 million photo-caption pairs from the web (Radford et al. 2021).

## Zero-shot, 37 breeds (chance: 1 in 37, 2.7%)
- "a photo of a {breed}, a type of pet.": **3,210 of 3,669 right, 87.5%**.
- Just the bare breed name: 2,941, **80.2%** (the wording of the sentence is worth 7 points).
- Cat or dog (2 sentences): **3,666 of 3,669, 99.9%**.
- Easiest: Shiba Inu 99 of 100, Sphynx 99%, Samoyed 98%. Hardest: **Bombay 14 of 88 (15.9%)**, black cats
  called Russian Blue (38) or Siamese (18); **Ragdoll 51 of 100, 42 of them called Birman**.

## One photo followed: shiba_inu_76 (a Shiba Inu puppy on a carpet)
Raw similarity (cosine, how closely the two lists of 512 numbers point the same way):
| sentence | score |
|---|---|
| a photo of a shiba inu. | **0.325** |
| a photo of a dog. | 0.275 |
| a photo of a fox. | 0.268 |
| a photo of a dog on a sofa. | 0.228 |
| a drawing of a dog. | 0.220 |
| a photo of a dog in the snow. | 0.219 |
| a photo of a cat. | 0.210 |
| a photo of a car. | 0.182 |
CLIP multiplies the scores by 100 before comparing (its learned scale, 100.0), so a 0.05 lead becomes near-certainty:
over the four sentences car / cat / dog / shiba inu, softmax gives **Shiba Inu 99.4%** (dog 0.6%).
Over the 37 breed sentences: **Shiba Inu 99.8%**, Chihuahua 0.07%, Pomeranian 0.04%.

## The catch: a word on a card
- The same photo with a white card reading "pomeranian" in the corner: **Pomeranian 62.1%, Shiba Inu 31.7%**.
- 500 photos it got right (seed 0), each given a card naming another breed of the same species:
  **383 still right (76.6%), 84 follow the card (16.8%)**.
Why (our reading, not measured here): in web captions, words in a picture often name what's in it (signs, labels,
packaging), so CLIP learned to read and to trust what it reads. We can't point to which of the 512 numbers carry it.

## Search the other way: a sentence finds the photo (top hit of 3,669)
- "a puppy wearing a hat" -> chihuahua_55, a chihuahua in a tiny bowler hat.
- "a cat asleep in a box" -> Abyssinian_76 (by a box in bubble wrap); 2nd Birman_52, a cat lying in a cardboard box.
- "a dog in the snow" -> saint_bernard_60, a Saint Bernard in a snowy garden.

Reproduce: `python experiment.py --model <clip folder> --data <oxford-pets folder>` (~5 min).
