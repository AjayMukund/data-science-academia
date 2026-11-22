"""
pictures.py - what the model actually receives, for the scene: the waveform (16,000 numbers a second) and Whisper's
log-mel spectrogram (80 pitch bands x 100 slices a second) of the two Ryan clips. Writes out/pictures.npz.
  python pictures.py --model <whisper-small folder>
"""
import argparse, os
import numpy as np, soundfile as sf
ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); a = ap.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__)); os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
from transformers import WhisperProcessor
proc = WhisperProcessor.from_pretrained(a.model)
out = {}
for k in ("speech", "beach"):
    x, sr = sf.read(os.path.join(HERE, "audio", "%s_GB.wav" % k), dtype="float32")
    mel = proc(x, sampling_rate=16000, return_tensors="np").input_features[0]
    n = int(len(x) / 160) + 1
    out[k + "_wave"] = x
    out[k + "_mel"] = mel[:, :n]
    print(k, len(x), "samples", mel.shape, "->", n, "slices")
np.savez_compressed(os.path.join(HERE, "out", "pictures.npz"), **out)
