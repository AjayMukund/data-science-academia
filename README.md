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
| [`women-and-children-first`](women-and-children-first) | Was it really "women and children first"? | Exploratory data analysis: split by sex, age, class, then class x group |
| [`never-grade-on-the-homework`](never-grade-on-the-homework) | Never grade on the homework | A memoriser scores 0 g on the penguins it learned from and misses new ones by 379 g |
| [`one-wrong-row`](one-wrong-row) | One wrong row | One typed extra zero bends a regression line from 49 to 35 g/mm |
| [`the-model-trusted-dice`](the-model-trusted-dice) | The model trusted dice (L1 vs L2) | Hide random noise among real inputs: L1 sets it to exactly zero, L2 never does |
| [`did-money-buy-a-lifeboat`](did-money-buy-a-lifeboat) | Did money buy a lifeboat? | Correlation vs causation: Titanic fare, class, sex, and fare per person |
| [`why-is-a-list-so-slow`](why-is-a-list-so-slow) | Why is a list so slow? | 1,000 lookups in a million: a list walks, a dictionary jumps (timed) |
| [`the-survey-that-only-reached-survivors`](the-survey-that-only-reached-survivors) | The survey that only reached survivors | Sampling bias: 342 survivors vs 1,000 random samples of 100 |
| [`same-average-different-penguins`](same-average-different-penguins) | Same average, different penguins | Standard deviation: two groups both average 4.7 kg, one spread 2.8x wider |
| [`etl-pipeline`](etl-pipeline) | The screen that lied | A real extract-transform-load pipeline into SQLite; rename one column and it loads 0 women without an error, until two checks stop it |
| [`how-a-model-learns`](how-a-model-learns) | How a model learns | Train a tiny language model on *Alice in Wonderland* from random numbers, in about two minutes |
| [`inside-the-transformer`](inside-the-transformer) | How ChatGPT works inside, step by step | Run GPT-2 in NumPy and watch "Paris" form, layer by layer |
| [`how-chatgpt-picks-its-next-word`](how-chatgpt-picks-its-next-word) | The cat sat on the… | GPT-2's real next-word list, greedy loops, temperature, top-k and top-p |

Each folder has a `FINDINGS.md` with the exact numbers the video shows.

**Data and models** are fetched from their sources and are not stored here:
SMS Spam Collection (UCI, CC BY 4.0), *Alice's Adventures in Wonderland*
(Project Gutenberg, public domain), Palmer Penguins (CC0), the Titanic
passenger list (a public historical record), GPT-2 small (OpenAI, via Hugging Face,
MIT licence). The code in this repository is MIT licensed.

More folders arrive with each new video.
