#!/usr/bin/env python3
"""Supplementary analyses for article v0.2 (post-hoc, descriptive; no new hypothesis tests).
Same data and train/test as locked_test.py: train = dev.jsonl, test = test.jsonl (locked shards,
already used once). Reports: (1) floor of the H1 groups vs a model-only floor and a global floor;
(2) each H1 group's share of locked-test sessions and spend, with pass flag from locked_test.out;
(3) the excluded low end (sessions with <= 3 calls) on the full 424k run."""
import glob, json, os, re
from collections import defaultdict
import lab

q = lab.quantile
ROOT = os.path.dirname(os.path.abspath(__file__))
load = lambda f: [r for r in map(json.loads, open(os.path.join(ROOT, f))) if r["calls"] > 3 and r["cost"]]
dev, test = load("dev.jsonl"), load("test.jsonl")
b = lambda L: 0 if (L or 0) < 200 else 1 if L < 1500 else 2
key = lambda r: (r["model"], r["event"], b(r["prompt_chars"]))

passed = {}
for line in open(os.path.join(ROOT, "locked_test.out")):
    if " val n=" in line and line.rstrip().endswith(("PASS", "fail")):
        passed[line.split(" val n=")[0].strip()] = line.rstrip().endswith("PASS")

g, mdl = defaultdict(list), defaultdict(list)
for r in dev:
    g[key(r)].append(r["cost"]); mdl[r["model"]].append(r["cost"])
glob_floor = q([r["cost"] for r in dev], .1)
tg = defaultdict(list)
for r in test:
    tg[key(r)].append(r["cost"])
T_n, T_spend = len(test), sum(r["cost"] for r in test)

print(f"locked test: {T_n} priced sessions (>= 4 calls), ${T_spend:,.0f}; global dev floor ${glob_floor:.3f}\n")
print("H1 group                                              n   %sess  %spend | group floor  held | model floor  held | global held | floor/median | H1")
tot = {"g": [0, 0], "m": [0, 0], "x": [0, 0]}; cov_pass = [0, 0, 0.0]
for k in sorted(g, key=lambda k: -len(g[k])):
    if len(g[k]) < 100 or len(tg[k]) < 30:
        continue
    c = tg[k]; gf, mf = q(g[k], .1), q(mdl[k[0]], .1)
    hg, hm, hx = (sum(v >= f for v in c) for f in (gf, mf, glob_floor))
    for t, h in (("g", hg), ("m", hm), ("x", hx)):
        tot[t][0] += h; tot[t][1] += len(c)
    p = next((v for kk, v in passed.items() if str(k).startswith(kk)), None)
    if p:
        cov_pass[0] += len(c); cov_pass[1] += 1; cov_pass[2] += sum(c)
    print(f"{str(k)[:50]:50} {len(c):5} {len(c)/T_n:6.1%} {sum(c)/T_spend:6.1%} | ${gf:6.3f} {hg/len(c):5.1%} | "
          f"${mf:6.3f} {hm/len(c):5.1%} | {hx/len(c):5.1%} | {gf/q(g[k], .5):5.2f} | {'PASS' if p else 'fail' if p is not None else '?'}")
print(f"\nfloor held, pooled over these groups: group {tot['g'][0]/tot['g'][1]:.2%} | model-only {tot['m'][0]/tot['m'][1]:.2%} | global {tot['x'][0]/tot['x'][1]:.2%}")
print(f"H1 passing groups: {cov_pass[1]} groups = {cov_pass[0]/T_n:.1%} of locked-test sessions, {cov_pass[2]/T_spend:.1%} of spend")

# (3) the excluded low end, full run
small = big = 0; small_c = big_c = 0.0; unpriced = 0
for f in glob.glob(os.path.join(ROOT, "full_rows", "*.jsonl")):
    for line in open(f):
        r = json.loads(line)
        if not r["cost"]:
            unpriced += 1; continue
        if r["calls"] > 3:
            big += 1; big_c += r["cost"]
        else:
            small += 1; small_c += r["cost"]
print(f"\nfull run: priced sessions with <= 3 calls (excluded) {small:,} = {small/(small+big):.1%} of priced sessions, "
      f"${small_c:,.0f} = {small_c/(small_c+big_c):.1%} of priced spend; unpriced sessions {unpriced:,}")
