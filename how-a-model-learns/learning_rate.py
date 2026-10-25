"""
learning_rate.py - "One knob, three models" (reel, 2026-11-09).

The same model, the same book and the same 3,000 steps, at three learning
rates: 0.01 (creeps), 0.3 (just right) and 3 (overshoots: its error shoots
past 10, more than twice random guessing, before the run's schedule cuts the
step tenfold for the last fifth). Reuses experiment.py's train() unchanged.

    python learning_rate.py       -> out/learning_rate.json  (about a minute)
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import experiment as L

out = {"random_guessing": float(np.log(L.V)), "steps": 3000, "runs": {}}
print("random guessing: error %.2f\n" % out["random_guessing"])
for lr in (0.01, 0.3, 3.0):
    held_out = []
    hook = lambda P, t: held_out.append((t, float(L.loss_of(P, L.Xva[:4000], L.Yva[:4000]))))
    P, _ = L.train(lr, 3000, hooks={t: hook for t in range(0, 3001, 500)})
    final = float(L.loss_of(P, L.Xva[:20000], L.Yva[:20000]))
    text = L.sample(P, "Alice was ", seed=1)
    out["runs"][lr] = {"held_out_every_500_steps": held_out, "final_held_out_error": final, "writes": text}
    print("learning rate %-5g final error %.2f   every 500 steps: %s" % (lr, final, " ".join("%.2f" % v for _, v in held_out)))
    print("   writes: %s\n" % text)
json.dump(out, open(os.path.join(L.OUT, "learning_rate.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
