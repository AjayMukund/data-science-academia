# "What machine learning actually is" master — measured findings (2026-10-19)

Data: **SMS Spam Collection** (UCI #228; Almeida & Gómez Hidalgo, 2011;
CC BY 4.0) - 5,574 real text messages, 747 spam. `download/SMSSpamCollection`.
`experiment.py` (plain numpy) writes `out/results.json` (all the numbers in the video).
Split: seeded shuffle, 4,459 training / **1,115 test (137 spam, 978 normal)**.

## Rules written by hand (scored on the 1,115 test messages)
| rules | spam caught | normal texts flagged | accuracy |
|---|---|---|---|
| 7 words: free win winner prize cash urgent claim | **85 / 137** | **23** | 93.3% |
| 16 words (+ txt, call now, £, mobile, reply, stop, won, offer, award) | 108 / 137 | **66** | 91.5% (worse) |

- Substring matching flags real texts: "Do u hav any frnd by name **ashwini**..."
  (contains "win"), "He... winds up...", "...move **free**ly...".

## The learned filter (multinomial Naive Bayes, 7,942 words learned)
- **133 / 137 spam caught, 16 normal texts flagged, 98.2%.**
- Word counts in the training messages (spam / normal): claim 94 / 0,
  prize 68 / 0, won 56 / 0, 150p 54 / 0, guaranteed 37 / 0; later 0 / 111,
  i'll 0 / 138, said 0 / 75. "free" leans spam (log-ratio +2.26) but is not decisive.
- The minimal pair (both real test messages, both flagged by the 7 rules):
  "Free any day but i finish at 6 on mon n thurs..." scores **-12.7** (normal: "free"
  +2.26, but nine of its twelve words lean normal - "i" -2.94, "but" -2.93, "finish" -2.58);
  "URGENT! You have won a 1 week FREE membership in our £100,000 Prize Jackpot!..." scores
  **+52.8** (spam: "claim" +5.56, "prize" +5.24, "won" +5.04).

## More examples, better (test accuracy by training-set size)
10 -> 56.2% · 30 -> 74.5% · 100 -> 91.9% · 300 -> 95.9% · 1,000 -> 97.4% ·
3,000 -> 98.1% · 4,459 -> 98.2%.

## On screen
- Mask phone numbers in spam texts (premium-rate numbers from 2011).
- Do not show the dataset's adult-content spam.
- The messages were collected around 2011-2012 ("over ten years old").
