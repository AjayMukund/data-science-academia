"""
experiment.py - why did the translator make the doctor a man?

Turkish "o" means he, she or it: "O bir doktor." says nothing about gender. An English translation has to pick one.
We ask a real, open Turkish->English model (Helsinki-NLP/opus-mt-tr-en, Apache-2.0, run on our laptop's CPU) to
translate "O bir <job>." for 45 jobs, and read how sure it is of "He" vs "She" as its first word. Then we line that
up against the real share of women in each job in the US (BLS, Current Population Survey 2025, Table 11).

  python experiment.py --model <folder with opus-mt-tr-en>      (writes out/results.json)
"""
import argparse, json, os
import numpy as np
import torch
from transformers import MarianMTModel, MarianTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
# (Turkish, English, BLS 2025 row, % women)
JOBS = [
    ("doktor", "doctor", "Other physicians", 41.7),
    ("cerrah", "surgeon", "Surgeons", 25.9),
    ("hemşire", "nurse", "Registered nurses", 87.3),
    ("eczacı", "pharmacist", "Pharmacists", 62.7),
    ("diş hekimi", "dentist", "Dentists", 38.6),
    ("veteriner", "vet", "Veterinarians", 70.9),
    ("psikolog", "psychologist", "Other psychologists", 71.3),
    ("kuaför", "hairdresser", "Hairdressers, hairstylists, and cosmetologists", 92.0),
    ("inşaat mühendisi", "civil engineer", "Civil engineers", 21.8),
    ("makine mühendisi", "mechanical engineer", "Mechanical engineers", 11.2),
    ("yazılımcı", "software developer", "Software developers", 20.3),
    ("mimar", "architect", "Architects, except landscape and naval", 29.8),
    ("kimyager", "chemist", "Chemists and materials scientists", 31.1),
    ("biyolog", "biologist", "Biological scientists", 47.8),
    ("öğretmen", "teacher", "Elementary and middle school teachers", 79.2),
    ("anaokulu öğretmeni", "kindergarten teacher", "Preschool and kindergarten teachers", 97.1),
    ("profesör", "professor", "Postsecondary teachers", 52.3),
    ("çocuk bakıcısı", "childminder", "Childcare workers", 93.2),
    ("kütüphaneci", "librarian", "Librarians and media collections specialists", 84.9),
    ("şef", "chef", "Chefs and head cooks", 26.4),
    ("aşçı", "cook", "Cooks", 43.7),
    ("garson", "waiter", "Waiters and waitresses", 69.8),
    ("fırıncı", "baker", "Bakers", 62.5),
    ("terzi", "tailor", "Tailors, dressmakers, and sewers", 72.8),
    ("sekreter", "secretary", "Secretaries and administrative assistants, except legal, medical, and executive", 91.9),
    ("resepsiyonist", "receptionist", "Receptionists and information clerks", 89.6),
    ("kasiyer", "cashier", "Cashiers", 68.5),
    ("veznedar", "bank teller", "Tellers", 76.6),
    ("postacı", "mail carrier", "Postal service mail carriers", 35.8),
    ("emlakçı", "estate agent", "Real estate brokers and sales agents", 55.8),
    ("muhasebeci", "accountant", "Accountants and auditors", 59.1),
    ("genel müdür", "chief executive", "Chief executives", 33.0),
    ("çiftçi", "farmer", "Farmers, ranchers, and other agricultural managers", 25.7),
    ("avukat", "lawyer", "Lawyers", 42.9),
    ("hakim", "judge", "Judges, magistrates, and other judicial workers", 50.5),
    ("sosyal hizmet uzmanı", "social worker", "Social workers, all other", 84.4),
    ("polis", "police officer", "Police officers", 16.4),
    ("itfaiyeci", "firefighter", "Firefighters", 5.1),
    ("güvenlik görevlisi", "security guard", "Security guards and gambling surveillance officers", 22.9),
    ("inşaat işçisi", "construction worker", "Construction laborers", 4.7),
    ("marangoz", "carpenter", "Carpenters", 3.1),
    ("elektrikçi", "electrician", "Electricians", 3.5),
    ("tesisatçı", "plumber", "Plumbers, pipefitters, and steamfitters", 3.1),
    ("kamyon şoförü", "truck driver", "Driver/sales workers and truck drivers", 7.7),
    ("taksi şoförü", "taxi driver", "Taxi drivers", 13.9),
    ("pilot", "pilot", "Aircraft pilots and flight engineers", 7.0),
    ("kabin memuru", "flight attendant", "Flight attendants", 74.1),
    ("yazar", "writer", "Writers and authors", 66.8),
    ("fotoğrafçı", "photographer", "Photographers", 50.7),
    ("müzisyen", "musician", "Musicians and singers", 34.1),
]
WORDS = ["mutlu", "güzel", "güçlü", "zeki", "evli", "yorgun", "ağlıyor", "çalışıyor", "yemek yapıyor",
         "futbol oynuyor", "alışveriş yapıyor", "araba kullanıyor", "dans ediyor", "bebek bakıyor"]

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--out", default=os.path.join(HERE, "out"))
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
tok = MarianTokenizer.from_pretrained(a.model)
m = MarianMTModel.from_pretrained(a.model).eval()
torch.manual_seed(0)
ids = {w: tok.convert_tokens_to_ids("▁" + w) for w in ("He", "She", "It", "They")}
print("token ids", ids)


def first_word(src):
    """translation (beam 4, as shipped), and the model's first-word probabilities."""
    with torch.no_grad():
        b = tok([src], return_tensors="pt")
        out = tok.decode(m.generate(**b, num_beams=4, max_new_tokens=40)[0], skip_special_tokens=True)
        start = torch.tensor([[m.config.decoder_start_token_id]])
        p = torch.softmax(m(**b, decoder_input_ids=start).logits[0, -1], -1)
    pr = {w: float(p[i]) for w, i in ids.items()}
    top = torch.topk(p, 5)
    return out, pr, [(tok.convert_ids_to_tokens(int(i)), float(v)) for v, i in zip(top.values, top.indices)]


rows = []
for tr, en, bls, women in JOBS:
    src = f"O bir {tr}."
    out, pr, top = first_word(src)
    she = pr["She"] / (pr["He"] + pr["She"])
    rows.append(dict(tr=tr, en=en, bls=bls, women=women, src=src, out=out, p=pr, she=she, top=top))
    print(f"{src:30s} {out:38s} he {pr['He']:.3f} she {pr['She']:.3f} -> she-share {she*100:5.1f}%  real {women:4.1f}%")

w = np.array([r["women"] for r in rows]); s = np.array([r["she"] * 100 for r in rows])
said_she = [r for r in rows if r["out"].lower().startswith("she")]
said_he = [r for r in rows if r["out"].lower().startswith("he")]
r = float(np.corrcoef(w, s)[0, 1])
print(f"\n{len(rows)} jobs: said She {len(said_she)}, He {len(said_he)}, other {len(rows)-len(said_she)-len(said_he)}")
print(f"correlation, real % women vs she-share: r = {r:.2f}")
maj_w = [x for x in rows if x["women"] > 50]
print(f"jobs mostly women in the US: {len(maj_w)}; translated She: {sum(x in said_she for x in maj_w)}")
maj_m = [x for x in rows if x["women"] < 50]
print(f"jobs mostly men in the US: {len(maj_m)}; translated He: {sum(x in said_he for x in maj_m)}")
# the flip point: where the model switches from He to She
for x in sorted(rows, key=lambda x: x["women"]):
    print(f"  {x['women']:5.1f}%  {x['en']:22s} {'SHE' if x in said_she else 'he '}  she-share {x['she']*100:5.1f}%")

print("\nnot jobs:")
extra = []
for wd in WORDS:
    out, pr, top = first_word(f"O {wd}.")
    she = pr["She"] / (pr["He"] + pr["She"])
    extra.append(dict(src=f"O {wd}.", out=out, p=pr, she=she))
    print(f"  O {wd:20s} {out:34s} she-share {she*100:5.1f}%")

pairs = []
for src in ["O bir doktor. O bir hemşire.", "O bir hemşire. O bir doktor.", "O bir doktor ve o bir anne.",
            "O bir doktor ve o hamile.", "O bir hemşire ve o bir baba.", "Kız kardeşim bir doktor.",
            "Erkek kardeşim bir hemşire."]:
    with torch.no_grad():
        out = tok.decode(m.generate(**tok([src], return_tensors="pt"), num_beams=4, max_new_tokens=40)[0],
                         skip_special_tokens=True)
    pairs.append(dict(src=src, out=out)); print(f"  {src:34s} -> {out}")

json.dump(dict(model="Helsinki-NLP/opus-mt-tr-en", decoding="beam 4", r=r, jobs=rows, words=extra, pairs=pairs),
          open(os.path.join(a.out, "results.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("wrote out/results.json")
