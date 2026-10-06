#!/usr/bin/env python3
"""Re-score today's earlier claims on the clean dev set with the fixed method:
repo split, two-sided p10-p90, bootstrap CIs, per-model baseline, controls, tokens and $.
Exploratory (dev data). Does not count as confirmation."""
import json, math, random, statistics as st
from collections import defaultdict
import lab

R = [json.loads(l) for l in open("dev.jsonl")]
R = [r for r in R if r["calls"] > 3]
models = [m for m, _ in sorted(((m, sum(r["model"] == m for r in R)) for m in {r["model"] for r in R}), key=lambda x: -x[1])][:6]
onehot = lambda r: [1.0 if r["model"] == m else 0.0 for m in models[1:]]
bucket = lambda L: 0 if (L or 0) < 200 else 1 if L < 1500 else 2

for target_name, target, rows in (("TOKENS", lambda r: r["tokens"], R),
                                  ("DOLLARS", lambda r: r["cost"], [r for r in R if r["cost"]])):
    fit, cal, te = lab.split_by_group(rows, lambda r: r["repo"])
    print(f"\n== {target_name}: {len(rows)} sessions | fit {len(fit)} cal {len(cal)} test {len(te)} (split by repo)")
    specs = {
        "global average": lambda r: [1.0],
        "per-model baseline": lambda r: [1.0] + onehot(r),
        "model+event+promptlen": lambda r: [1.0] + onehot(r) + [1.0 if r["event"] == e else 0.0 for e in
                                         ("pull_request_comment", "issues_agent_assignment", "review-request", "pull_request_review")]
                                         + [1.0 if bucket(r["prompt_chars"]) == b else 0.0 for b in (1, 2)],
        "CEILING: true call count": lambda r: [1.0] + onehot(r) + [math.log(r["calls"])],
    }
    for name, f in specs.items():
        print(f"  {name:26} {lab.fmt(lab.evaluate(te, lab.fit_interval(fit, cal, f, target), target, boot=300))}")
    sh = [dict(r) for r in rows]; ys = [target(r) for r in sh]; random.Random(7).shuffle(ys)
    for r, y in zip(sh, ys): r["_y"] = y
    sf, sc, stt = lab.split_by_group(sh, lambda r: r["repo"])
    ctrl = lab.evaluate(stt, lab.fit_interval(sf, sc, specs["model+event+promptlen"], lambda r: r["_y"]), lambda r: r["_y"], boot=100)
    base = lab.evaluate(stt, lab.fit_interval(sf, sc, lambda r: [1.0], lambda r: r["_y"]), lambda r: r["_y"], boot=100)
    print(f"  CONTROL shuffled target: features width {ctrl['width'][0]:.2f}x vs global {base['width'][0]:.2f}x "
          f"-> {'OK' if ctrl['width'][0] >= base['width'][0] * 0.95 else 'LEAK/BUG'}")

    # archetype groups, per-group intervals vs per-model baseline (dev: exploratory)
    print("  archetype groups (>=100 train sessions):")
    key = lambda r: (r["model"], r["event"], bucket(r["prompt_chars"]))
    gtrain = defaultdict(list)
    for r in fit + cal: gtrain[key(r)].append(target(r))
    mtrain = defaultdict(list)
    for r in fit + cal: mtrain[r["model"]].append(target(r))
    for g, vals in sorted(gtrain.items(), key=lambda x: -len(x[1])):
        if len(vals) < 100: continue
        tg = [r for r in te if key(r) == g]
        if len(tg) < 30: continue
        glo, ghi = lab.quantile(vals, .1), lab.quantile(vals, .9)
        mlo, mhi = lab.quantile(mtrain[g[0]], .1), lab.quantile(mtrain[g[0]], .9)
        gr = lab.evaluate(tg, lambda r: (glo, ghi), target, boot=300)
        mr = lab.evaluate(tg, lambda r: (mlo, mhi), target, boot=300)
        print(f"    {str(g)[:58]:58} test n={len(tg):4} | group cov {gr['coverage'][0]:.0%} width {ghi/glo:4.1f}x "
              f"| model-baseline cov {mr['coverage'][0]:.0%} width {mhi/mlo:4.1f}x")

# loops, on dollars, with CI on the ratio, plus outcome (H3)
P = [r for r in R if r["cost"]]
P.sort(key=lambda r: -r["cost"]); k = len(P) // 10
flag = lambda r: r["loop_R"] or r["loop_S"]
rng = random.Random(0); ratios = []
for _ in range(500):
    s = [P[rng.randrange(len(P))] for _ in range(len(P))]; s.sort(key=lambda r: -r["cost"]); t, o = s[:k], s[k:]
    ratios.append((sum(map(flag, t)) / len(t)) / max(1e-9, sum(map(flag, o)) / len(o)))
ratios.sort()
top, rest = P[:k], P[k:]
ratio = (sum(map(flag, top)) / len(top)) / (sum(map(flag, rest)) / len(rest))
after = sum(r["cost_after_onset"] or 0 for r in top) / sum(r["cost"] for r in top)
print(f"\n== LOOPS ($, n={len(P)}): flagged top10% {sum(map(flag, top))/len(top):.0%} vs rest {sum(map(flag, rest))/len(rest):.0%} "
      f"| ratio {ratio:.1f}x [{ratios[25]:.1f},{ratios[475]:.1f}] | top-decile $ after onset {after:.0%}")
bad = lambda r: r["state"] in ("failed", "cancelled", "timed_out")
fl = [r for r in R if flag(r)]; cl = [r for r in R if not flag(r)]
print(f"   H3 not-completed rate: flagged {sum(map(bad, fl))/len(fl):.1%} (n={len(fl)}) vs clean {sum(map(bad, cl))/len(cl):.1%} (n={len(cl)})")
