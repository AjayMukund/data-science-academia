# Data Science Academia — the code behind the videos

Every number in our explainers comes from an experiment we actually ran.
This repository is those experiments, so you can run them yourself, change
them, and check our work.

Everything is plain Python and NumPy — no PyTorch, no GPU. Each folder
downloads its own data from the original source the first time you run it.

```bash
pip install -r requirements.txt
cd what-is-machine-learning
python experiment.py
```

| Folder | Video | What you can run |
|---|---|---|
| [`what-is-machine-learning`](what-is-machine-learning) | What machine learning actually is | Hand-written spam rules vs a filter that learns from 4,459 real text messages |
| [`one-penguin-two-questions`](one-penguin-two-questions) | One penguin, two questions | Regression vs classification on 344 real penguins |
| [`can-a-computer-sort-penguins`](can-a-computer-sort-penguins) | Can a computer sort penguins it was never told about? | k-means finds the species with no labels, and three ways it goes wrong |
| [`what-does-cleaning-data-do`](what-does-cleaning-data-do) | What does cleaning data actually do? | Titanic: delete the blanks and survival reads 67%; read them and the model reaches 83% |
| [`how-a-model-learns`](how-a-model-learns) | How a model learns | Train a tiny language model on *Alice in Wonderland* from random numbers, in about two minutes |
| [`inside-the-transformer`](inside-the-transformer) | How ChatGPT works inside, step by step | Run GPT-2 in NumPy and watch "Paris" form, layer by layer |
| [`how-chatgpt-picks-its-next-word`](how-chatgpt-picks-its-next-word) | The cat sat on the… | GPT-2's real next-word list, greedy loops, temperature, top-k and top-p |

Each folder has a `FINDINGS.md` with the exact numbers the video shows.

**Data and models** are fetched from their sources and are not stored here:
SMS Spam Collection (UCI, CC BY 4.0), *Alice's Adventures in Wonderland*
(Project Gutenberg, public domain), GPT-2 small (OpenAI, via Hugging Face,
MIT licence). The code in this repository is MIT licensed.

More folders arrive with each new video.
