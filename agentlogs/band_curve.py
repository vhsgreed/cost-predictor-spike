#!/usr/bin/env python3
"""How wide is the cost interval at each confidence level? Dev data, dollars.
For coverage p, report the central interval [q((1-p)/2), q((1+p)/2)] expressed relative
to the median (e.g. -60% / +150%), for: all sessions, per-model, best archetype groups,
and the ceiling (true call count known). Intervals from training split, measured on
the repo-held-out test split."""
import json, math
from collections import defaultdict
import lab

R = [json.loads(l) for l in open("dev.jsonl")]
R = [r for r in R if r["calls"] > 3 and r["cost"]]
fit, cal, te = lab.split_by_group(R, lambda r: r["repo"])
train = fit + cal
LEVELS = (0.2, 0.5, 0.8, 0.9, 0.95)
bucket = lambda L: 0 if (L or 0) < 200 else 1 if L < 1500 else 2


def band(train_vals, test_vals, label):
    med = lab.quantile(train_vals, .5)
    cells = []
    for p in LEVELS:
        lo, hi = lab.quantile(train_vals, (1 - p) / 2), lab.quantile(train_vals, (1 + p) / 2)
        cov = sum(lo <= v <= hi for v in test_vals) / len(test_vals)
        cells.append(f"{p:.0%}: {lo/med-1:+.0%}/{hi/med-1:+.0%} (cov {cov:.0%})")
    print(f"{label:46} median ${med:.2f} n_test={len(test_vals):4} | " + " | ".join(cells))


band([r["cost"] for r in train], [r["cost"] for r in te], "ALL sessions")
for m in ("claude-sonnet-4.5", "claude-sonnet-4.6", "claude-sonnet-4"):
    band([r["cost"] for r in train if r["model"] == m], [r["cost"] for r in te if r["model"] == m], m)
key = lambda r: (r["model"], r["event"], bucket(r["prompt_chars"]))
for g in [("claude-sonnet-4.5", "pull_request_comment", 0), ("claude-sonnet-4.5", "issues_agent_assignment", 0),
          ("claude-sonnet-4.5", "pull_request_review", 0)]:
    band([r["cost"] for r in train if key(r) == g], [r["cost"] for r in te if key(r) == g], str(g)[:46])

# ceiling: residual spread after knowing model + true call count (log-linear)
onehot = lambda r: [1.0 if r["model"] == m else 0.0 for m in ("claude-sonnet-4.6", "claude-sonnet-4", "claude-opus-4.6", "gpt-5.3-codex")]
feat = lambda r: [1.0, math.log(r["calls"])] + onehot(r)
w = lab.lstsq([feat(r) for r in fit], [math.log(r["cost"]) for r in fit])
mu = lambda r: sum(a * b for a, b in zip(w, feat(r)))
res_cal = [math.log(r["cost"]) - mu(r) for r in cal]
res_te = [math.log(r["cost"]) - mu(r) for r in te]
cells = []
for p in LEVELS:
    lo, hi = lab.quantile(res_cal, (1 - p) / 2), lab.quantile(res_cal, (1 + p) / 2)
    cov = sum(lo <= x <= hi for x in res_te) / len(res_te)
    cells.append(f"{p:.0%}: {math.exp(lo)-1:+.0%}/{math.exp(hi)-1:+.0%} (cov {cov:.0%})")
print(f"{'CEILING: model + true call count':46} {'':22} | " + " | ".join(cells))
