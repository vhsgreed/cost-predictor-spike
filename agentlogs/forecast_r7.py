#!/usr/bin/env python3
"""Round 7: during-run forecast with the group kept. One look. Plan: PLAN.md 'Round 7 pre-registration'.
Forecaster: cell (group, turn bucket, x bucket), x = cumulative / fit-group median final;
fallback (group, turn bucket), then group. Reference baseline fixed in advance: k-only table."""
import json, os, random
from collections import defaultdict, Counter
from traj_common import DATASETS, HERE, kb, xb, tq, load, split


def analyse(ds):
    rows = load(ds)
    fit, test, gfin, med, skipped = split(rows)
    full, kc, pn = defaultdict(list), defaultdict(list), defaultdict(list)
    for g, _, s in fit:
        m = med[g]
        for k, c in enumerate(s, 1):
            kk, xx = kb(k), xb(c / m)
            full[(g, kk, xx)].append(s[-1])
            kc[(g, kk)].append(s[-1])
            pn[(kk, xx)].append(s[-1] / m)
    FULL = {key: tq(v) for key, v in full.items() if len(v) >= 50}
    KC = {key: tq(v) for key, v in kc.items() if len(v) >= 50}
    GM = {g: tq(v) for g, v in gfin.items()}
    PN = {key: tq(v) for key, v in pn.items() if len(v) >= 50}
    print(f"{ds}: {len(rows):,} runs, fit {len(fit):,} / test {len(test):,} (skipped {skipped:,}); "
          f"{len(FULL)} of {len(full)} full cells have >= 50 fit rows", flush=True)

    def ref(g, k):
        return KC.get((g, kb(k))) or GM[g]

    def r7(g, k, c):
        key = (g, kb(k), xb(c / med[g]))
        if key in FULL:
            return FULL[key], "cell"
        if (g, kb(k)) in KC:
            return KC[(g, kb(k))], "group x turn"
        return GM[g], "group"

    def varB(g, k, c):
        v = PN.get((kb(k), xb(c / med[g])))
        if v:
            return tuple(t * med[g] for t in v)
        return ref(g, k)

    per = defaultdict(lambda: [0.0, 0.0, 0.0, 0, 0, 0])  # e7, eref, eB, hit7, hitB, n
    turn = defaultdict(lambda: [0.0, 0.0, 0.0, 0])
    src = Counter()
    for g, inst, s in test:
        fin = s[-1]; acc = per[inst]
        for k, c in enumerate(s, 1):
            if k < 3:
                continue
            (p, lo, hi), how = r7(g, k, c)
            pr = ref(g, k)[0]
            pb, lob, hib = varB(g, k, c)
            e7, er, eB = abs(p - fin), abs(pr - fin), abs(pb - fin)
            acc[0] += e7; acc[1] += er; acc[2] += eB
            acc[3] += lo <= fin <= hi; acc[4] += lob <= fin <= hib; acc[5] += 1
            src[how] += 1
            b = turn[min(k, 21)]
            b[0] += e7; b[1] += er; b[2] += eB; b[3] += 1

    tot = [sum(v[i] for v in per.values()) for i in range(6)]
    n = tot[5]
    point, pointB = tot[0] / tot[1], tot[2] / tot[1]
    cov, covB = tot[3] / n, tot[4] / n
    rng = random.Random(20261015); ids = list(per)
    R, C, RB, CB = [], [], [], []
    for _ in range(200):
        a = [0.0] * 6
        for _ in ids:
            v = per[ids[rng.randrange(len(ids))]]
            for i in range(6):
                a[i] += v[i]
        R.append(a[0] / a[1]); C.append(a[3] / a[5]); RB.append(a[2] / a[1]); CB.append(a[4] / a[5])
    for L in (R, C, RB, CB):
        L.sort()
    ok = point <= .8 and R[189] < .8 and C[10] <= .8 <= C[189]
    print(f"  R7 update points k>=3: {n:,} | MAE R7 {tot[0]/n:,.0f} vs k-only {tot[1]/n:,.0f} | ratio {point:.3f} "
          f"[{R[10]:.3f},{R[189]:.3f}] | coverage {cov:.1%} [{C[10]:.1%},{C[189]:.1%}] | {'PASS' if ok else 'fail'}")
    print(f"  variant B (descriptive): MAE {tot[2]/n:,.0f} | ratio {pointB:.3f} [{RB[10]:.3f},{RB[189]:.3f}] "
          f"| coverage {covB:.1%} [{CB[10]:.1%},{CB[189]:.1%}]")
    print("  R7 answered by: " + ", ".join(f"{h} {src[h]/n:.1%}" for h in ("cell", "group x turn", "group")))
    turns = {}
    for k in sorted(turn):
        b = turn[k]
        turns["21+" if k == 21 else str(k)] = {"n": b[3], "mae_r7": b[0] / b[3], "mae_ref": b[1] / b[3], "mae_B": b[2] / b[3]}
        print(f"    turn {'21+' if k == 21 else k:>3}: n={b[3]:8,} | R7 {b[0]/b[3]:8,.0f} | k-only {b[1]/b[3]:8,.0f} "
              f"| ratio {b[0]/b[1]:.2f} | B ratio {b[2]/b[1]:.2f}", flush=True)
    return {"runs": len(rows), "test": len(test), "skipped": skipped, "update_points": n,
            "mae_r7": tot[0] / n, "mae_ref": tot[1] / n, "ratio": point, "ratio_ci": [R[10], R[189]],
            "coverage": cov, "coverage_ci": [C[10], C[189]], "pass": bool(ok),
            "variant_B": {"mae": tot[2] / n, "ratio": pointB, "ratio_ci": [RB[10], RB[189]],
                          "coverage": covB, "coverage_ci": [CB[10], CB[189]]},
            "source_share": {h: src[h] / n for h in src}, "turns": turns}


res = {ds: analyse(ds) for ds in DATASETS}
npass = sum(res[d]["pass"] for d in DATASETS)
res["verdict"] = "PASS" if npass >= 2 else "FAIL"
print(f"\nR7: passes in {npass}/3 datasets -> {res['verdict']} (need >= 2)")
json.dump(res, open(os.path.join(HERE, "forecast_r7.json"), "w"), indent=1)
print("wrote forecast_r7.json")
