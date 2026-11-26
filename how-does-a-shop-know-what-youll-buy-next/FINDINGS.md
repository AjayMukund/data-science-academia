# How does a shop know what you'll buy next? (market-basket analysis: support, confidence, lift)

Reel 12-05. Run: `python experiment.py` (plain Python; reads the .xlsx directly; downloads on first run) -> results.json.

## Data
UCI Online Retail (Daqing Chen, Sai Liang Sain, Kun Guo, 2012), **CC BY 4.0**,
https://archive.ics.uci.edu/dataset/352/online+retail - every order of a UK online gift shop, 1 Dec 2010 - 9 Dec 2011.
541,909 rows. Many customers are wholesalers (say "a UK online gift shop", not "shoppers like you").
Cancellations (invoice "C...") and postage/fee codes dropped -> **19,957 baskets**, 3,916 products, median 15 items a basket.
Pairs counted in the 18,465 baskets of 60 items or fewer.

## Best-sellers (share of all baskets)
White hanging heart T-light holder 11.0% · Jumbo bag red retrospot 10.5% · Regency cakestand 3 tier 10.0% · Party bunting 8.4%.

## Running example: someone adds the GREEN REGENCY TEACUP AND SAUCER (in 1,015 baskets, 5.1% of all)
| also in the basket | baskets with both | confidence (of teacup baskets) | normal share of all baskets | lift |
|---|---|---|---|---|
| Roses Regency teacup | **480** | **47.3%** | 5.3% | **8.9** |
| Pink Regency teacup | 409 | 40.3% | 3.8% | **10.5** (strongest) |
| Regency cakestand 3 tier | 312 | 30.7% | 10.0% | 3.1 |
| Party bunting | 79 | 7.8% | 8.4% | **0.9** (no link: as common as anywhere) |
| Jumbo bag red retrospot | 55 | 5.4% | 10.5% | **0.5** (teacup buyers take it LESS) |

- Confidence = of the baskets with the teacup, how many also have X. Lift = confidence / X's normal share.
  Lift 1 = no connection; 8.9 = almost nine times likelier than usual.
- The trap: the cakestand is the 3rd most common companion (312 baskets) partly because it is in 10% of all baskets;
  lift 3.1 still real but far below the matching teacups. Party bunting turns up in 79 teacup baskets - sounds like a
  pattern, but 7.8% vs 8.4% everywhere: lift 0.9, nothing.

## Strongest rules overall (at least 100 baskets)
- Coffee mug "pears" -> coffee mug "apples": 129 baskets, 60.8% confidence, lift 46.5.
- Herb markers (thyme -> rosemary 54.9%, lift 45.6; chives/parsley/mint the same): people buy the set.
- Top pair by count: Jumbo bag pink polkadot + Jumbo bag red retrospot, 584 baskets.

## Script cautions
- Say "baskets", not customers (one customer, many orders). Say "this shop", not "Amazon": real recommenders are more
  (the 11-20 recommenders master); this is the classic counting rule, still used for "frequently bought together".
