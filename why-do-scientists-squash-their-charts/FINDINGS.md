# Why do scientists squash their charts? (log scales, doubling time)

Reel 12-06. Run: `python experiment.py` (plain Python; downloads on first run) -> results.json.

## Data (both CC BY 4.0, via Our World in Data)
- Transistors per microprocessor, "World" series 1971-2021: Rupp (2022), "Microprocessor Trend Data", processed by OWID.
  https://ourworldindata.org/grapher/transistors-per-microprocessor
- Training computation of 536 notable AI models (petaFLOP), 1950 - Aug 2026: Epoch AI (2026), "Parameter, Compute and
  Data Trends in Machine Learning", processed by OWID. https://ourworldindata.org/grapher/artificial-intelligence-training-computation
  Compute for closed models is Epoch's ESTIMATE - say "estimated" for GPT-4 / Grok 4.

## Chips (the running example)
- **2,308** transistors in 1971 -> **58.2 billion** in 2021: **25 million times** more.
- On an ordinary (linear) chart the first decades vanish: 1990 is 0.002% of the 2021 peak, 2000 is 0.06%,
  2010 is 4%. Forty years sit on the floor, then a wall - it LOOKS like nothing happened until ~2015.
- On a log scale (each gridline x10) it is a straight line: least-squares fit on log2, **doubling every 1.98 years**,
  r² = 0.994. Decade by decade: 2.2 / 2.9 / 2.2 / 1.7 / 2.2 years - the same steady doubling throughout (Moore's law).

## AI training compute (the contrast, same chart trick)
- Every model since 2010: doubling every **5.7 months** (fit r² 0.65; the record-setters alone: 5.6 months).
- AlexNet (Sep 2012) 470 petaFLOP -> GPT-3 (May 2020) 314 million -> GPT-4 (Mar 2023) 21 billion (estimate)
  -> Grok 4 (Jul 2025) 500 billion (estimate), the largest in the data: about **a billion** times AlexNet
  (500,000,000,000 / 470 = 1.06 billion; corrected 2026-11-26 - the first draft said "a trillion", a misread caught by the scene's assert).
- On one log chart the two lines can be compared directly: the AI line is about four times steeper
  (5.7 months vs 24 months per doubling).

## Script cautions
- Don't say "Moore's law is dead/alive" - the fit only covers to 2021.
- "PetaFLOP" is jargon: say "computer operations" / "sums", and put the number on screen.
