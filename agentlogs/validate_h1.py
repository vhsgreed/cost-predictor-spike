#!/usr/bin/env python3
"""H1 validation (PLAN.md). One look. Train = all dev sessions; test = validation shards.
Groups = EVERY (model, event, prompt-length bucket) with >= 100 dev sessions (not only the
ones that looked good on dev). Interval = empirical p10-p90 of the training group; baseline
= empirical p10-p90 of the training model. Dollars.
Pass per group: group width <= 0.8 x model width, with the 90% CI (bootstrap over training
repos) of the group/model width ratio entirely below 0.8... relaxed per PLAN to: CI upper
bound of group width below the model width; AND validation coverage 90% CI contains 80%.
H1 passes if >= 3 groups pass."""
import json, random
from collections import defaultdict
import lab

dev = [r for r in map(json.loads, open("dev.jsonl")) if r["calls"] > 3 and r["cost"]]
val = [r for r in map(json.loads, open("val.jsonl")) if r["calls"] > 3 and r["cost"]]
assert not ({r["session"] for r in dev} & {r["session"] for r in val})
bucket = lambda L: 0 if (L or 0) < 200 else 1 if L < 1500 else 2
key = lambda r: (r["model"], r["event"], bucket(r["prompt_chars"]))
repos = defaultdict(list)
for r in dev: repos[r["repo"]].append(r)
rk = list(repos); rng = random.Random(20261006)

def widths(rows):
    g, m = defaultdict(list), defaultdict(list)
    for r in rows: g[key(r)].append(r["cost"]); m[r["model"]].append(r["cost"])
    q = lambda v: (lab.quantile(v, .1), lab.quantile(v, .9))
    return {k: q(v) for k, v in g.items() if len(v) >= 30}, {k: q(v) for k, v in m.items()}

G, M = widths(dev)
groups = [k for k in G if sum(key(r) == k for r in dev) >= 100]
boot = [widths([r for _ in rk for r in repos[rk[rng.randrange(len(rk))]]]) for _ in range(200)]

print(f"dev {len(dev)} sessions, val {len(val)} sessions, {len(groups)} pre-registered groups (>=100 dev sessions)")
passed = 0
for g in sorted(groups, key=lambda k: -sum(key(r) == k for r in dev)):
    gl, gh = G[g]; ml, mh = M[g[0]]
    gw, mw = gh / gl, mh / ml
    ups = sorted(b[0][g][1] / b[0][g][0] for b in boot if g in b[0])
    gw_hi = ups[int(.95 * len(ups))]
    tv = [r["cost"] for r in val if key(r) == g]
    if len(tv) < 30:
        print(f"  {str(g)[:60]:60} val n={len(tv):3}  too few to judge"); continue
    inside = [gl <= v <= gh for v in tv]
    covs = sorted(sum(inside[rng.randrange(len(inside))] for _ in inside) / len(inside) for _ in range(500))
    c_lo, c_hi = covs[25], covs[475]; cov = sum(inside) / len(inside)
    mcov = sum(ml <= v <= mh for v in tv) / len(tv)
    ok = gw <= 0.8 * mw and gw_hi < mw and c_lo <= 0.8 <= c_hi
    passed += ok
    print(f"  {str(g)[:60]:60} val n={len(tv):4} | group {gw:4.1f}x (CI hi {gw_hi:4.1f}) cov {cov:.0%} [{c_lo:.0%},{c_hi:.0%}]"
          f" | model {mw:4.1f}x cov {mcov:.0%} | {'PASS' if ok else 'fail'}")
print(f"H1: {passed} groups pass -> {'PASS' if passed >= 3 else 'FAIL'} (need >= 3)")
