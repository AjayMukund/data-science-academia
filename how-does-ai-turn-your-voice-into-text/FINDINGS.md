# Speech master (12-01): how does AI turn your voice into text? — measured (2026-11-22)

**Model:** openai/whisper-small (**241.7 million** numbers, MIT licence), English, greedy decoding, on our laptop's CPU.
Trained by OpenAI on **680,000 hours** of audio from the web (Radford et al. 2022, the Whisper paper) — about 77 years.
**Data:** LibriSpeech test-clean (Panayotov et al. 2015, CC BY 4.0): volunteers reading public-domain books aloud.
**Code:** `experiment.py` → `out/results.json`; `minimal_pair.py` → `out/pair.json` (+ `audio/`);
`pictures.py` → `out/pictures.npz`. Greedy, seeded: re-runs match.

## What the model receives
- Sound as numbers: **16,000 a second** (the "recognise speech" clip: 50,688 numbers, 3.2 s).
- Whisper turns them into a spectrogram: **80 pitch bands × 100 slices a second** (that clip: 80 × 317), always padded
  to 30 seconds (80 × 3,000). An encoder reads the picture; a decoder writes word-pieces one at a time, each a guess
  from the sound **and the words already written**.

## The minimal pair (synthetic voices: en-GB-RyanNeural, en-US-GuyNeural)
| Said | Whisper wrote (both voices) |
|---|---|
| This new phone can recognise speech. | This new phone can recognise / recognize speech. ✓ |
| **This new phone can wreck a nice beach.** | **This new phone can recognize speech.** ✗ |
| The big storm will wreck a nice beach. | The big storm will **recognize beach**. ✗ |
| The big storm will recognise speech. | The big storm will recognise / recognize speech. ✓ |
- Inside "wreck a nice beach" (Ryan): after "This new phone can", its guesses: **recognize 74.7%**, recognise 8.5%,
  reckon 5.1%. Then **speech 81.9%**, beach 6.2%.
- After "The big storm will": recognize 49.4%, recognise 47.4%; then **beach 52.6%**, speech 16.4%. The storm pulls it
  half way back; the first wrong word stays.
- With hiss at 0 dB: "recognize each", "recognize these", "recognize peace".
- Honesty: these are synthetic voices reading our sentences, chosen to be a hard pair; the 200 real recordings below
  are the fair test.

## 200 real recordings (every 13th of LibriSpeech test-clean's 2,620; 25.0 minutes, 4,022 words)
- **Word error rate 3.28%**: 132 words wrong in 4,022 (about **1 in 30**). **128 of 200** come out perfect.
- Speed: **0.85 seconds of compute per second of speech** on our CPU.
- Its slips are guesses a language would make: "MACDONALDS" (the clan) → **"McDonald's"**; "effected" → "affected";
  "though I am glad you ask me" → "the way I'm glad you asked me"; "Montalais" → "Montelet".
- With hiss as loud as the voice (0 dB): **WER 21.85%** (879 of 4,022); **30 of 200** perfect.

## Nothing to hear (the honest turn)
- 30 s of silence, 10 times: it writes **"you"** every time. 30 s of pure hiss, 10 times: **"you"** every time.
- It never says "I heard nothing": it always writes its likeliest guess.

## Cautions
- whisper-small is the second-smallest Whisper; larger ones make fewer mistakes. LibriSpeech is clean read speech;
  phone calls, accents and crowds are harder.
- WER is counted after Whisper's English normaliser (so "recognise"/"recognize", case and punctuation don't count).
