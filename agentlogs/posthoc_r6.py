#!/usr/bin/env python3
"""POST-HOC re-tabulation of round 6 (PLAN.md, 'Round 6 errata'). Descriptive, no bar; R6 verdict unchanged.
Same predictor as forecast_r6.py. Per turn: table MAE vs the FIXED k-only baseline and vs the group
median, separately (R6's per-turn table used min() of the two per point, an oracle)."""
import json, os
from collections import defaultdict
from traj_common import DATASETS, HERE, kb, xb, tq, load, split


def analyse(ds):
    rows = load(ds)
    fit, test, gfin, med, skipped = split(rows)
    cells, kcells = defaultdict(list), defaultdict(list)
    for g, _, s in fit:
        for k, c in enumerate(s, 1):
            cells[(kb(k), xb(c / med[g]))].append(s[-1])
            kcells[(g, kb(k))].append(s[-1])
    gm = {g: tq(v) for g, v in gfin.items()}
    PRED = {("x", k1, k2): tq(v) for (k1, k2), v in cells.items() if len(v) >= 50}
    PRED.update({("k", g, k1): tq(v) for (g, k1), v in kcells.items() if len(v) >= 50})
    KMED = {(g, k1): tq(v)[0] for (g, k1), v in kcells.items() if len(v) >= 50}

    def predict(g, k, c):
        return PRED.get(("x", kb(k), xb(c / med[g]))) or PRED.get(("k", g, kb(k))) or gm[g]

    turn = defaultdict(lambda: [0.0, 0.0, 0.0, 0])  # table, k-only, group median, n
    start = [0.0, 0.0, 0, 0]  # table err, group-median err, covered, n
    for g, _, s in test:
        fin = s[-1]
        for k, c in enumerate(s, 1):
            p, lo, hi = predict(g, k, c)
            if k == 1:
                start[0] += abs(p - fin); start[1] += abs(med[g] - fin); start[2] += lo <= fin <= hi; start[3] += 1
                continue
            if k < 3:
                continue
            pk = KMED.get((g, kb(k)), med[g])
            b = turn[min(k, 21)]
            b[0] += abs(p - fin); b[1] += abs(pk - fin); b[2] += abs(med[g] - fin); b[3] += 1
    out = {"skipped": skipped, "start": {"mae_table": start[0] / start[3], "mae_group_median": start[1] / start[3],
                                         "coverage": start[2] / start[3], "n": start[3]}, "turns": {}}
    print(f"{ds}: test runs {len(test):,} (skipped {skipped:,})")
    print(f"  task start k=1: table MAE {start[0]/start[3]:,.0f} | group median MAE {start[1]/start[3]:,.0f} "
          f"| table 80% interval covers {start[2]/start[3]:.1%}")
    tot = [0.0, 0.0, 0.0, 0]
    for k in sorted(turn):
        b = turn[k]
        for i in range(4):
            tot[i] += b[i]
        out["turns"]["21+" if k == 21 else str(k)] = {"n": b[3], "mae_table": b[0] / b[3], "mae_konly": b[1] / b[3],
                                                     "mae_group_median": b[2] / b[3]}
        print(f"    turn {'21+' if k == 21 else k:>3}: n={b[3]:8,} | table {b[0]/b[3]:8,.0f} | k-only {b[1]/b[3]:8,.0f} "
              f"(ratio {b[0]/b[1]:.2f}) | group median {b[2]/b[3]:8,.0f} (ratio {b[0]/b[2]:.2f})")
    print(f"  pooled k>=3 (check vs R6): table {tot[0]/tot[3]:,.0f} | k-only {tot[1]/tot[3]:,.0f} "
          f"(ratio {tot[0]/tot[1]:.3f}) | group median {tot[2]/tot[3]:,.0f}", flush=True)
    out["pooled"] = {"mae_table": tot[0] / tot[3], "mae_konly": tot[1] / tot[3], "mae_group_median": tot[2] / tot[3]}
    return out


res = {ds: analyse(ds) for ds in DATASETS}
json.dump(res, open(os.path.join(HERE, "posthoc_r6.json"), "w"), indent=1)
print("wrote posthoc_r6.json")
