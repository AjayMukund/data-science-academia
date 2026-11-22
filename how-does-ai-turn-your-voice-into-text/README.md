# How does AI turn your voice into text? Whisper, measured

Whisper-small (244M parameters) on a laptop CPU. On 200 LibriSpeech recordings it gets 132 of 4,022 words wrong
(word error rate 3.28%), and 128 recordings come out perfect. With hiss as loud as the voice, 21.85%. Given 30
seconds of silence or pure hiss, it writes "you" (10 of 10): it always writes something. The sound of "recognize
speech" and "wreck a nice beach" is nearly the same; `minimal_pair.py` asks Whisper for its top guesses word by word
("recognize" 74.7%, "speech" 81.9%) and shows how a sentence about a storm tips "speech" to "beach".

```bash
pip install torch==2.5.1 transformers==4.46.3 soundfile numpy edge-tts
python experiment.py --model path/to/whisper-small --data path/to/LibriSpeech/test-clean   # ~25 min on CPU
python minimal_pair.py --model path/to/whisper-small    # clips in audio/ (re-synthesised if missing)
python pictures.py --model path/to/whisper-small        # waveforms and spectrograms, out/pictures.npz
```

Model: openai/whisper-small (MIT). Data: LibriSpeech test-clean (Panayotov et al. 2015, CC BY 4.0),
https://www.openslr.org/resources/12/test-clean.tar.gz. See `FINDINGS.md`.
Things to try: whisper-tiny vs whisper-medium; your own voice; music instead of hiss.
