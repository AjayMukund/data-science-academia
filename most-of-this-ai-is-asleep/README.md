# Most of this AI is asleep: a mixture-of-experts model, measured

IBM's Granite 3.0 1B-A400M, a mixture-of-experts language model, run in plain NumPy (`granite_np.py`, no PyTorch).
Each of its 24 layers holds 32 experts; for every word piece a router wakes the best 8. Of its 1,334,625,280 numbers,
428,655,616 (32%) are used per word piece. On the opening of Alice in Wonderland, it guesses the next word piece
right 255 times in 299 with 8 experts, 11 with 2, and 126 with all 32. In layer 21, expert 18 is chosen for 96% of the
word pieces of the 1869 French translation and 7% of the English original. On our four texts, 34 of the 768 experts
are clear specialists.

```bash
pip install numpy tokenizers huggingface_hub
mkdir texts
curl -L -o texts/alice_en.txt   https://www.gutenberg.org/cache/epub/11/pg11.txt
curl -L -o texts/alice_fr.txt   https://www.gutenberg.org/cache/epub/55456/pg55456.txt
curl -L -o texts/lerouge_fr.txt https://www.gutenberg.org/cache/epub/798/pg798.txt
python experiment.py --models models    # downloads the model (5.5 GB, float32) the first time; out/results.json
python alice.py --models models         # the running example, English and French; out/alice.json
python specialists.py --models models   # how many experts specialise, and every layer's routing; out/specialists.json
```

Needs about 7 GB of free memory. Greedy decoding, so a re-run gives the same text and counts; the timings in
results.json depend on your machine. Model: ibm-granite/granite-3.0-1b-a400m-base (Apache-2.0). Texts: Project
Gutenberg #11, #55456 and #798, public domain. See `FINDINGS.md`.

Things to try: another layer's specialists, your own language's text, or `route_k` values the model wasn't trained for.
