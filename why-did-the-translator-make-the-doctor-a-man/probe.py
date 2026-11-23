"""probe.py - first look: does the Turkish->English model give "o bir X" a gender?

  python probe.py --model C:/Users/DATASERVER1/models/opus-mt-tr-en
"""
import argparse
import torch
from transformers import MarianMTModel, MarianTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
a = ap.parse_args()
tok = MarianTokenizer.from_pretrained(a.model)
m = MarianMTModel.from_pretrained(a.model).eval()

S = ["O bir doktor.", "O bir hemşire.", "O bir mühendis.", "O bir öğretmen.", "O bir aşçı.",
     "O bir sekreter.", "O bir pilot.", "O bir temizlikçi.", "O bir asker.", "O bir dansçı.",
     "O bir doktor. O bir hemşire.", "O evli.", "O mutlu.", "O güzel.", "O güçlü."]
with torch.no_grad():
    b = tok(S, return_tensors="pt", padding=True)
    out = m.generate(**b, num_beams=4, max_new_tokens=30)
for s, o in zip(S, tok.batch_decode(out, skip_special_tokens=True)):
    print(f"{s:32s} -> {o}")
