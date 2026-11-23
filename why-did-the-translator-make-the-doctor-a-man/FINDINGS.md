# Translation bias master (11-27): why did the translator make the doctor a man? — measured 2026-11-22

**Model:** Helsinki-NLP/opus-mt-tr-en (Apache-2.0), Turkish -> English, a transformer trained on the OPUS collection of
sentence pairs people had already translated (model card; weights dated 2020-01-16). Run on our laptop's CPU in the
research venv (torch 2.5.1, transformers 4.46.3), beam search 4, the decoding it ships with. `experiment.py` ->
`out/results.json`.

**The problem:** Turkish "o" means he, she (or it). "O bir doktor." = "[he/she] is a doctor": the gender is not in the
sentence. English must pick one, so the translator guesses.

**Real share of women in each job:** US BLS, Current Population Survey, Table 11, **2025 annual averages** (11-month
average, October missing: federal shutdown), https://www.bls.gov/cps/cpsaat11.htm, last modified 2026-02-20. All 50 rows
read verbatim from the page (two passes). Women are 47.1% of all employed. Physicians are split on the page; we use
"Other physicians" (41.7%). Public domain (US government work).

## The cold open (minimal pair)
- "O bir doktor." -> **"He's a doctor."**   "O bir hemşire." -> **"She's a nurse."** One word changed; so did the person.
- Google's own 2018 post uses the same sentence: Translate "historically translated the Turkish equivalent of 'He/she is
  a doctor' into the masculine form, and ... 'He/she is a nurse' into the feminine form" (Melvin Johnson, Google
  Research blog, 2018-12-10, "Providing gender-specific translations in Google Translate").

## 50 jobs (`O bir <job>.`)
- **He 42, She 8**, nothing else (never "they").
- **Mostly-men jobs (26): He 26 of 26.**  **Mostly-women jobs (24): She only 8 of 24.**
- She: nurse, receptionist, secretary, hairdresser, childminder ("babysitter"), kindergarten teacher, cashier, waiter
  ("She's a waitress").
- He, although most US workers are women: pharmacist (62.7% women), accountant 59.1, baker 62.5, writer 66.8,
  vet 70.9, psychologist 71.3, tailor 72.8, flight attendant 74.1 ("He's a cabin attendant"), bank teller 76.6,
  **teacher 79.2**, social worker 84.4, **librarian 84.9**.
- Its own words leak: "He's a fireman", "He's a mailman", "She's a waitress".

## How sure it is (first word: P(She) / (P(He) + P(She)))
- It does track reality: correlation with the real % of women **r = 0.76** across 50 jobs.
- But it leans to "he" all the way up: nurse 90.7% she (real 87.3), pharmacist 24.4% (real 62.7), professor 21.5%
  (real 52.3), librarian 44.7% (real 84.9).
- **The coin it always calls the same way: teacher, He 0.429 vs She 0.367 (she-share 46.1%).** Nearly a coin toss,
  but beam search takes the likelier word every time: 46% becomes "He", 100 times out of 100.
- Doctor: He 0.573, She 0.224 (she-share 28.1%); 41.7% of US doctors are women; translated "she" 0 times.
- Cook 48.8% she-share -> "He's a cook". Waiter 52.7% -> "She's a waitress". The flip is at 50%: a small lean becomes
  a certain answer.

## Not jobs (`O <word>.`)
- "O güzel" -> **"She's beautiful"** (she-share 88.7%). "O güçlü" -> "He's strong" (33.1%). "O zeki" -> "He's smart"
  (36.9%). "O bebek bakıyor" -> "She's babysitting" (72.8%). "O futbol oynuyor" -> "He plays football" (27.8%).
  "O mutlu" -> "She's happy" (50.7%, a coin flip that landed she).

## Context (one sentence, one person)
- "O bir doktor ve o bir anne." -> "She's a doctor and she's a mother." (told she's a mother, it gets it right)
- "Kız kardeşim bir doktor." -> "My sister's a doctor."  "Erkek kardeşim bir hemşire." -> "My brother's a nurse."
- **"O bir hemşire ve o bir baba." -> "She's a nurse and he's a father."** One person (a nurse who is a father)
  split into two people rather than write "he's a nurse".
- "O bir doktor. O bir hemşire." -> "She's a doctor, she's a nurse." (the second sentence pulls the first)

## The fix in practice
- Google (2018 post): for ambiguous Turkish->English sentences, show **both** translations ("She is a doctor" / "He is a
  doctor"). About 10% of Turkish queries were ambiguous; the model produced both forms "99% of the time" when asked;
  recall of the ambiguity detector "only around 60%". (Quoted; we did not measure Google.)
- Our own: when the sentence carries the gender (sister / mother), the model follows it.

## Not claimed
- We did not measure Google Translate or any commercial system today. The OPUS training text was not inspected; "it
  learned it from text people wrote" is the model card's description, not a count of doctors in its data.
