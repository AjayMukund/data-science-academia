"""
minimal_pair.py - "recognise speech" vs "wreck a nice beach": two sentences that sound almost the same.
Both are spoken by two synthetic voices (Microsoft neural TTS via edge-tts: en-GB-RyanNeural, en-US-GuyNeural),
then transcribed by whisper-small. Also the same pair with hiss at 0 dB. Writes out/pair.json.
  python minimal_pair.py --model <whisper-small folder>
Then, for each Ryan clip, the model's top 5 guesses (with probabilities) for every word-piece it writes.
"""
import argparse, asyncio, io, json, os, subprocess
import numpy as np, soundfile as sf, torch
ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); a = ap.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__)); AUD = os.path.join(HERE, "audio"); os.makedirs(AUD, exist_ok=True); os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
SENT = {"speech": "This new phone can recognise speech.", "beach": "This new phone can wreck a nice beach.",
        "storm_beach": "The big storm will wreck a nice beach.", "storm_speech": "The big storm will recognise speech."}
VOICES = ["en-GB-RyanNeural", "en-US-GuyNeural"]
import edge_tts


async def say(text, voice, path):
    await edge_tts.Communicate(text, voice).save(path)

for k, t in SENT.items():
    for v in VOICES:
        mp3 = os.path.join(AUD, "%s_%s.mp3" % (k, v.split("-")[1]))
        if not os.path.exists(mp3):
            asyncio.run(say(t, v, mp3))
        wav = mp3[:-4] + ".wav"
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", mp3, "-ar", "16000", "-ac", "1", wav], check=True)

torch.set_grad_enabled(False)
from transformers import WhisperProcessor, WhisperForConditionalGeneration
proc = WhisperProcessor.from_pretrained(a.model); model = WhisperForConditionalGeneration.from_pretrained(a.model).eval()
rng = np.random.default_rng(0)
res = []
for k in SENT:
    for v in VOICES:
        x, sr = sf.read(os.path.join(AUD, "%s_%s.wav" % (k, v.split("-")[1])), dtype="float32")
        for cond in ("clean", "noise0db"):
            y = x if cond == "clean" else x + rng.normal(0, 1, len(x)).astype(np.float32) * np.sqrt(np.mean(x ** 2))
            f = proc(y, sampling_rate=16000, return_tensors="pt").input_features
            out = model.generate(f, language="en", task="transcribe", num_beams=1, do_sample=False,
                                 output_scores=True, return_dict_in_generate=True)
            text = proc.batch_decode(out.sequences, skip_special_tokens=True)[0].strip()
            steps = []
            if cond == "clean" and "Ryan" in v:
                for sc, tid in zip(out.scores, out.sequences[0, -len(out.scores):]):
                    pr = torch.softmax(sc[0].float(), -1); top = pr.topk(5)
                    steps.append(dict(chosen=proc.tokenizer.decode([int(tid)]),
                                      top=[[proc.tokenizer.decode([int(i)]), round(float(p), 4)] for p, i in zip(top.values, top.indices)]))
            res.append(dict(said=SENT[k], voice=v, cond=cond, heard=text, secs=round(len(x) / sr, 2), steps=steps))
            print(cond, v, "|", SENT[k], "->", text, flush=True)
json.dump(res, open(os.path.join(HERE, "out", "pair.json"), "w", encoding="utf-8"), indent=1)
