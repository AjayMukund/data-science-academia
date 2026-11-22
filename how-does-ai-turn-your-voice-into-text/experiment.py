"""
experiment.py - how does AI turn your voice into text? Whisper, measured.

Model: openai/whisper-small (244M parameters, MIT), English, greedy decoding, on our laptop's CPU.
Data: LibriSpeech test-clean (Panayotov et al. 2015, CC BY 4.0): people reading public-domain audiobooks aloud.
https://www.openslr.org/resources/12/test-clean.tar.gz

  1. 200 recordings (every 13th of the 2,620, sorted by id): word error rate (WER = words wrong / words said), how
     many come out perfect, seconds of compute per second of speech.
  2. The same 200 with hiss added at 0 dB signal-to-noise (as loud as the voice).
  3. Nothing to hear: 10 clips of silence and 10 of pure hiss (30 s each). What does it write?
Text compared after Whisper's own English normaliser (case, punctuation, spelled-out numbers).
Writes out/results.json.   python experiment.py --model <whisper-small folder> --data <LibriSpeech/test-clean>
"""
import argparse, glob, json, os, time
import numpy as np
import soundfile as sf
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--data", required=True)
a = ap.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "out"); os.makedirs(OUT, exist_ok=True)
torch.set_grad_enabled(False); torch.set_num_threads(8)
from transformers import WhisperProcessor, WhisperForConditionalGeneration
proc = WhisperProcessor.from_pretrained(a.model)
model = WhisperForConditionalGeneration.from_pretrained(a.model).eval()
norm = proc.tokenizer.normalize

refs = {}
for f in glob.glob(os.path.join(a.data, "*", "*", "*.trans.txt")):
    for line in open(f, encoding="utf-8"):
        k, t = line.strip().split(" ", 1)
        refs[k] = (t, os.path.join(os.path.dirname(f), k + ".flac"))
ids = sorted(refs)[::13][:200]


def transcribe(audio):
    feats = proc(audio, sampling_rate=16000, return_tensors="pt").input_features
    out = model.generate(feats, language="en", task="transcribe", num_beams=1, do_sample=False, max_new_tokens=200)
    return proc.batch_decode(out, skip_special_tokens=True)[0].strip()


def edits(r, h):
    """word-level edit distance"""
    d = np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev, d[j] = d[j], cur
    return int(d[len(h)])


rng = np.random.default_rng(0)
R = {"model": "openai/whisper-small", "recordings": len(ids), "runs": {}}
for run in ("clean", "noise0db"):
    rows, errs, words, secs, audio_s = [], 0, 0, 0.0, 0.0
    for k in ids:
        text, path = refs[k]
        x, sr = sf.read(path, dtype="float32")
        if run == "noise0db":
            x = x + rng.normal(0, 1, len(x)).astype(np.float32) * np.sqrt(np.mean(x ** 2))
        t0 = time.time(); hyp = transcribe(x); secs += time.time() - t0; audio_s += len(x) / sr
        r, h = norm(text).split(), norm(hyp).split()
        e = edits(r, h); errs += e; words += len(r)
        rows.append(dict(id=k, ref=text, hyp=hyp, words=len(r), errors=e, secs=round(len(x) / sr, 2)))
    R["runs"][run] = dict(wer=errs / words, errors=errs, words=words, perfect=sum(r["errors"] == 0 for r in rows),
                          compute_per_audio_second=secs / audio_s, audio_minutes=audio_s / 60, rows=rows)
    print(run, "WER %.2f%%  perfect %d/%d  %.2f s compute per s of audio" % (100 * errs / words, R["runs"][run]["perfect"],
                                                                         len(rows), secs / audio_s), flush=True)
# 3. nothing to hear
R["nothing"] = {}
for name in ("silence", "hiss"):
    outs = []
    for i in range(10):
        x = np.zeros(16000 * 30, np.float32) if name == "silence" else (rng.normal(0, 0.05, 16000 * 30)).astype(np.float32)
        outs.append(transcribe(x))
    R["nothing"][name] = outs
    print(name, outs[:5], flush=True)
json.dump(R, open(os.path.join(OUT, "results.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
