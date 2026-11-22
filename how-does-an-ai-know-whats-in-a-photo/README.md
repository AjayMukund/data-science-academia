# How does an AI know what's in a photo? CLIP, measured

CLIP (openai/clip-vit-base-patch32, on a laptop CPU) sorts the Oxford-IIIT Pet test photos into 37 breeds without
ever being trained on them: each breed becomes one sentence, "a photo of a {breed}, a type of pet.", and each photo
goes to the sentence whose 512 numbers point the same way as its own. 3,210 of 3,669 come out right (87.5%; chance is
2.7%); bare breed names get 80.2%; cat or dog, 3,666 of 3,669. A Shiba Inu puppy (shiba_inu_76) is Shiba Inu at
99.8%; put a white card reading "pomeranian" in the corner and it says Pomeranian at 62.1%. On 500 photos it got
right, a card naming another breed makes 84 follow the card. A sentence also finds photos: "a puppy wearing a hat"
finds a Chihuahua in a tiny bowler hat.

```bash
pip install torch==2.5.1 transformers==4.46.3 numpy pillow
curl -LO https://thor.robots.ox.ac.uk/datasets/pets/images.tar.gz
curl -LO https://thor.robots.ox.ac.uk/datasets/pets/annotations.tar.gz
mkdir -p pets && tar xzf images.tar.gz -C pets && tar xzf annotations.tar.gz -C pets
python experiment.py --model openai/clip-vit-base-patch32 --data pets     # about 5 minutes on CPU
python pictures.py  --model openai/clip-vit-base-patch32 --data pets     # the photos and numbers the video shows
```

Model: openai/clip-vit-base-patch32 (MIT, about 600 MB, downloaded on first run). Photos: Oxford-IIIT Pet
(Parkhi et al. 2012, CC BY-SA 4.0). See `FINDINGS.md`. Things to try: your own pet; other sentence templates;
the card in the middle of the photo instead of the corner; a bigger CLIP.
