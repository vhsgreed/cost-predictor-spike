#!/usr/bin/env python3
"""Full run, phase 2: production cost tables (descriptive, no new hypothesis tests).
Per (model, event, prompt-length bucket) group with >= 100 sessions: n, p10/p50/p90,
mean, floor = p10 = "at least X" at 90% confidence. Output: tables_full.json + stdout.
Groups match H1's definition exactly (validate_h1.py: bucket <200/0<1500/>=1500 chars)."""
import json, os, glob
from collections import defaultdict
import lab

ROOT = os.path.dirname(os.path.abspath(__file__))
rows = []
for f in sorted(glob.glob(os.path.join(ROOT, "full_rows", "*.jsonl"))):
    for line in open(f):
        r = json.loads(line)
        if r["calls"] > 3 and r["cost"] and r["model"] and r["model"] != "unknown":
            rows.append(r)
print(f"priced sessions with >= 4 calls: {len(rows)}")

bucket = lambda L: 0 if (L or 0) < 200 else 1 if L < 1500 else 2
key = lambda r: (r["model"], r["event"], bucket(r["prompt_chars"]))
g, m = defaultdict(list), defaultdict(list)
for r in rows:
    g[key(r)].append(r["cost"]); m[r["model"]].append(r["cost"])
q = lab.quantile

out = {"groups": [], "models": {}}
for k, v in sorted(g.items(), key=lambda kv: -len(kv[1])):
    if len(v) < 100:
        continue
    model, event, b = k
    out["groups"].append({
        "model": model, "event": event, "prompt_bucket": b, "n": len(v),
        "floor_p10": round(q(v, .1), 3), "p50": round(q(v, .5), 3), "p90": round(q(v, .9), 3),
        "mean": round(sum(v) / len(v), 3),
        "width_p90_over_p10": round(q(v, .9) / q(v, .1), 1),
        "floor_share_of_median": round(q(v, .1) / q(v, .5), 2),
    })
for k, v in sorted(m.items(), key=lambda kv: -len(kv[1])):
    out["models"][k] = {"n": len(v), "floor_p10": round(q(v, .1), 3), "p50": round(q(v, .5), 3),
                        "p90": round(q(v, .9), 3), "mean": round(sum(v) / len(v), 3),
                        "width_p90_over_p10": round(q(v, .9) / q(v, .1), 1)}

with open(os.path.join(ROOT, "tables_full.json"), "w") as fh:
    json.dump(out, fh, indent=1)

print(f"\nmodels ({len(out['models'])}):")
for k, v in out["models"].items():
    print(f"  {k:26} n={v['n']:7} | at least ${v['floor_p10']:6.2f} | typical ${v['p50']:6.2f} | "
          f"p90 ${v['p90']:7.2f} | mean ${v['mean']:7.2f} | width {v['width_p90_over_p10']:5.1f}x")
print(f"\ngroups with >= 100 sessions ({len(out['groups'])}); top 15 by n:")
for gr in out["groups"][:15]:
    print(f"  {gr['model'][:18]:18} {gr['event'][:14]:14} b{gr['prompt_bucket']} n={gr['n']:6} | "
          f"at least ${gr['floor_p10']:6.2f} ({gr['floor_share_of_median']:.0%} of median) | "
          f"p50 ${gr['p50']:6.2f} | p90 ${gr['p90']:7.2f} | width {gr['width_p90_over_p10']:5.1f}x")
print("\nwrote tables_full.json")
