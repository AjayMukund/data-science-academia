# Findings: the model trusted dice

Inputs standardised on the training split (240 penguins, seed 0), so each
weight is grams per standard deviation. Test error = mean absolute error on
102 held-out penguins. The two penalties use different scales of lambda.

- Plain fit: bill length 8.2, bill depth 29.9, flipper 725.4, dice 36.5 /
  11.5 / 33.6; test error **299 g**. Dice 1 gets **4.4x** bill length's weight.
- L2 (ridge): no weight is ever zero. At lambda 3200 the dice still carry
  3.3 / 0.9 / 1.1 and the error has nearly doubled, **299 -> 592 g**. The
  large flipper weight shrinks while bill length's grows: the load spreads.
- L1 (lasso): at lambda 60 every weight except flipper is exactly 0 and the
  error is **313 g**. Once a weight reaches zero on the path it stays there.
- Five different splits, L1 at 60: all three dice columns are zero every time.
- Why: L2's budget for the weights is a circle, L1's a diamond whose corners
  lie on the axes; the best fit under the budget tends to land on a corner,
  where a weight is exactly zero.
