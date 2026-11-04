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
| [`king-minus-man-plus-woman`](king-minus-man-plus-woman) | King minus man plus woman equals... king? | The famous word-vector demo over all 400,000 GloVe words, with and without the hidden step |
| [`two-questions-decision-tree`](two-questions-decision-tree) | Two questions to name a penguin | A decision tree from scratch: how it picks its questions, and why depth 2 does almost all the work |
| [`why-accuracy-lies`](why-accuracy-lies) | Right 99.5% of the time, never warned once | Why accuracy lies: 26 years of California earthquakes; "always no" is 99.5% accurate and warns of none of 48 big-quake days |
| [`a-mile-in-no-time`](a-mile-in-no-time) | A mile in no time by 2562? | Extrapolation: a straight line through 32 mile world records, the 1954 line that called 1999, and 27 years without a record |
| [`two-numbers-481-sons`](two-numbers-481-sons) | Two numbers, 481 sons | The bell curve on Galton's 1880s family heights: two numbers predict 70 sons over six foot (83 counted), and a mixed group breaks it |
| [`never-off-by-a-dollar`](never-off-by-a-dollar) | Never off by a dollar | Data leakage on 2,925 Ames house sales: a price-per-foot column gives $0 errors in testing and $21,200 on sale day |
| [`where-do-you-draw-the-line`](where-do-you-draw-the-line) | Where do you draw the line? | Classification thresholds on 569 Wisconsin biopsies: 10 missed at 0.5, and 231 healthy flagged to catch every cancer |
| [`nine-numbers-find-every-crater`](nine-numbers-find-every-crater) | Nine numbers find every crater | Convolution: one 3x3 edge grid slid over NASA's Moon, worked out on real pixels |
| [`does-rain-empty-the-bike-racks`](does-rain-empty-the-bike-racks) | Does rain empty Seoul's bike racks? | Group-by in pandas on a year of Seoul's bikes: 766 vs 167 an hour, and a morning-evening flip in the rain |
| [`seven-measurements-two-ideas`](seven-measurements-two-ideas) | Seven measurements, two ideas | PCA on 210 wheat grains: size holds 72% of the spread, shape 17%, and two numbers name 195 of 210 |
| [`etl-pipeline`](etl-pipeline) | The screen that lied | A real extract-transform-load pipeline into SQLite; rename one column and it loads 0 women without an error, until two checks stop it |
| [`the-page-that-lost-the-day`](the-page-that-lost-the-day) | The page that lost the day | Big data on one laptop: 151.6 M Wikipedia pageview lines, too big to hold, streamed in 2:22, map-reduce on 8 workers in 53 s |
| [`do-new-yorkers-tip-13-or-22`](do-new-yorkers-tip-13-or-22) | Do New Yorkers tip 13% or 22%? | Python for data: 3.3 M NYC taxi trips in pandas; cash tips are never recorded, card riders tip 21.9% |
| [`is-the-difference-real`](is-the-difference-real) | Is the difference real? | A/B testing with shuffles, no formulas: a real 64,000-customer email experiment; 27 of 100,000 shuffles match the gap |
| [`how-do-you-know-your-model-works`](how-do-you-know-your-model-works) | How do you know your model works? | Cross-validation: one penguin model scores 88-100% depending on the test split; 5-fold CV 96.8% |
| [`does-red-mean-poisonous`](does-red-mean-poisonous) | Does red mean poisonous? | A decision tree on 8,124 field-guide mushrooms tries all 116 questions; smell beats colour, and a perfect test score turns out to be near-twins |
| [`when-will-old-faithful-erupt`](when-will-old-faithful-erupt) | When will Old Faithful erupt? | Linear regression from scratch on 272 eruptions: squared misses, gradient descent, a test, and two clumps that two plain guesses fit better |
| [`could-the-numbers-have-warned-them`](could-the-numbers-have-warned-them) | Could the numbers have warned them? | Logistic regression on the 23 shuttle flights before Challenger: no pattern in the damaged flights alone, 99.96% at 31°F with all of them |
| [`why-does-ai-make-things-up`](why-does-ai-make-things-up) | Why does AI make things up? | GPT-2 in NumPy asked the capital of 50 real and 50 invented countries: it answers every invented one, and its confidence only partly gives it away |
| [`can-you-just-hand-it-the-news`](can-you-just-hand-it-the-news) | Can you just hand an AI the news? (RAG) | A 2024 chat model and a small search model in NumPy, 12 questions about 2025-26: 0 right alone, 5 with a Wikipedia library, and 88% sure of the wrong answer when handed last time's page |
| [`why-doesnt-a-long-chat-take-forever`](why-doesnt-a-long-chat-take-forever) | Why doesn't a long chat take forever? (KV cache) | A chat model in NumPy through one real 48-message chat: 21 s per token re-reading vs 0.33 s with the KV cache, 12 KB of notes per token, and an invented dog's name when the first message is dropped |
| [`why-does-a-chatbot-answer-you`](why-does-a-chatbot-answer-you) | Why does a chatbot answer you? (fine-tuning) | A base model and its tuned twin in NumPy: 8 vs 20 of 20 capitals answered, and the tuned one agrees with 20 of 20 wrong claims while knowing better |
| [`how-does-a-big-ai-fit-on-a-laptop`](how-does-a-big-ai-fit-on-a-laptop) | How does a big AI fit on a laptop? (quantization) | A chat model rounded to 8, 4, 3 and 2 bits: 45, 44, 38, then 0 of 50 capitals, and "8.8.8.8" |
| [`how-a-model-learns`](how-a-model-learns) | How a model learns | Train a tiny language model on *Alice in Wonderland* from random numbers, in about two minutes |
| [`inside-the-transformer`](inside-the-transformer) | How ChatGPT works inside, step by step | Run GPT-2 in NumPy and watch "Paris" form, layer by layer |
| [`how-chatgpt-picks-its-next-word`](how-chatgpt-picks-its-next-word) | The cat sat on the… | GPT-2's real next-word list, greedy loops, temperature, top-k and top-p |

Each folder has a `FINDINGS.md` with the exact numbers the video shows.

**Data and models** are fetched from their sources and are not stored here:
SMS Spam Collection (UCI, CC BY 4.0), *Alice's Adventures in Wonderland*
(Project Gutenberg, public domain), Palmer Penguins (CC0), GloVe word vectors (PDDL), the Titanic
passenger list (a public historical record), GPT-2 small (OpenAI, via Hugging Face,
MIT licence), the USGS earthquake catalogue (public domain), the mile world-record progression
(Wikipedia / World Athletics), Galton's family heights (public domain, via HistData), the Ames housing data
(De Cock, Journal of Statistics Education), Breast Cancer Wisconsin, Seoul Bike Sharing Demand and
Seeds (UCI, CC BY 4.0), and NASA Goddard's "Full Moon" (LRO, public domain). The code in this repository is MIT licensed.

More folders arrive with each new video.
