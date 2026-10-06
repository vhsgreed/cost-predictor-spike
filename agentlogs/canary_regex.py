#!/usr/bin/env python3
"""Regex-parser canary: do prompt features (regex only, no LLM) tighten the p90
bound vs the plain average? Same shard, same split, same scoring as canary.py.

Decision rule, fixed before running: p90 width <= ~2.5x at ~90% coverage
(baseline 3.4x) => building a real parser is worth it. Otherwise not.
"""
import math, os, random, re, statistics as st
from collections import defaultdict
import pyarrow.parquet as pq
import pyarrow.compute as pc

t = pq.read_table("part_00000_of_00276.parquet", columns=["session", "data"])
sess = pc.struct_field(t["session"], "id").to_pylist()
u = pc.struct_field(t["data"], "usage")
pt = pc.struct_field(u, "prompt_tokens").to_pylist()
ct = pc.struct_field(u, "completion_tokens").to_pylist()
tot = defaultdict(int); n = defaultdict(int)
for s, p, c in zip(sess, pt, ct):
    if p:
        tot[s] += p + (c or 0); n[s] += 1
del t, u

S = pq.read_table("part_00000_of_00002.parquet", columns=["id", "model", "prompt", "state"])
meta = {i: (m or "unknown", p or "", st_) for i, m, p, st_ in
        zip(S["id"].to_pylist(), S["model"].to_pylist(), S["prompt"].to_pylist(), S["state"].to_pylist())}

VAGUE = re.compile(r"\b(improve|refactor|clean ?up|optimi[sz]e|better|enhance|overhaul|modernize|polish)\b", re.I)
TYPES = {
    "fix": re.compile(r"\b(fix|bug|error|crash|broken|fails?|issue)\b", re.I),
    "add": re.compile(r"\b(add|implement|create|build|support|new feature)\b", re.I),
    "test": re.compile(r"\b(tests?|coverage|unit test|spec)\b", re.I),
    "docs": re.compile(r"\b(docs?|readme|documentation|comment)\b", re.I),
    "refactor": re.compile(r"\b(refactor|rename|restructure|migrat\w+|upgrade)\b", re.I),
}
FILE = re.compile(r"[\w./-]+\.(py|js|ts|tsx|jsx|go|rs|java|rb|php|c|cpp|h|cs|md|json|ya?ml|toml|sh|css|html)\b")
SYMBOL = re.compile(r"`[^`\n]{1,60}`")
BULLET = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+", re.M)
URL = re.compile(r"https?://\S+")
REQ = re.compile(r"\b(must|should|need to|ensure|make sure)\b", re.I)


def pfeats(p):
    L = len(p)
    f = [1.0, math.log(L + 1), math.log(len(BULLET.findall(p)) + 1), math.log(len(FILE.findall(p)) + 1),
         math.log(len(SYMBOL.findall(p)) + 1), math.log(p.count("```") // 2 + 1), math.log(len(URL.findall(p)) + 1),
         math.log(len(REQ.findall(p)) + 1), math.log(len(VAGUE.findall(p)) + 1)]
    f += [1.0 if r.search(p) else 0.0 for r in TYPES.values()]
    return f


def lstsq(X, y, lam=1e-3):
    k = len(X[0]); A = [[sum(r[a]*r[b] for r in X) + (lam if a == b else 0) for b in range(k)] for a in range(k)]
    v = [sum(r[a]*yy for r, yy in zip(X, y)) for a in range(k)]
    for col in range(k):
        q = max(range(col, k), key=lambda r: abs(A[r][col])); A[col], A[q], v[col], v[q] = A[q], A[col], v[q], v[col]
        for r in range(k):
            if r != col:
                f = A[r][col]/A[col][col]; A[r] = [x - f*yv for x, yv in zip(A[r], A[col])]; v[r] -= f*v[col]
    return [v[i]/A[i][i] for i in range(k)]


def spearman(a, b):
    def rank(x):
        o = sorted(range(len(x)), key=lambda i: x[i]); r = [0]*len(x)
        for j, i in enumerate(o): r[i] = j
        return r
    ra, rb = rank(a), rank(b); m = len(a)
    return 1 - 6*sum((x-y)**2 for x, y in zip(ra, rb))/(m*(m*m-1))


def evaluate(name, rows, featfn):
    rows = rows[:]; random.Random(0).shuffle(rows); m = len(rows)
    if m < 60: print(f"{name}: n={m}, skipped"); return
    fit, cal, te = rows[:m*5//10], rows[m*5//10:m*7//10], rows[m*7//10:]
    w = lstsq([featfn(r) for r in fit], [math.log(r["y"]) for r in fit])
    pred = lambda r: sum(a*b for a, b in zip(w, featfn(r)))
    res = sorted(math.log(r["y"]) - pred(r) for r in cal)
    q = res[min(len(res)-1, math.ceil(.9*(len(res)+1))-1)]
    prior = sorted(r["y"] for r in fit + cal); p90 = prior[int(.9*len(prior))]
    y = [r["y"] for r in te]; pb = [math.exp(pred(r)) for r in te]
    cov_a = sum(v <= p90 for v in y)/len(y); cov_b = sum(math.log(r["y"]) <= pred(r)+q for r in te)/len(y)
    wa = st.median(p90/v for v in y); wb = st.median(math.exp(pred(r)+q)/r["y"] for r in te)
    print(f"{name:34} n={m:5} | average: cov {cov_a:.0%} width {wa:.1f}x | regex: cov {cov_b:.0%} width {wb:.1f}x spearman {spearman(pb, y):.2f}")


rows = []
for s, y in tot.items():
    if n[s] > 3 and s in meta and meta[s][1].strip():
        m, p, state = meta[s]
        rows.append({"y": y, "model": m, "p": p, "state": state})
print(f"sessions with prompt text and > 3 calls: {len(rows)}")
plen = [len(r["p"]) for r in rows]
print(f"prompt length chars: median {st.median(plen):.0f}, p90 {sorted(plen)[int(.9*len(plen))]}")

models = sorted({r["model"] for r in rows}, key=lambda m: -sum(r["model"] == m for r in rows))[:5]
onehot = lambda r: [1.0 if r["model"] == m else 0.0 for m in models[1:]]
evaluate("ALL, prompt features", rows, lambda r: pfeats(r["p"]))
evaluate("ALL, prompt + model", rows, lambda r: pfeats(r["p"]) + onehot(r))
evaluate("ALL, model only (control)", rows, lambda r: [1.0] + onehot(r))
for m in models:
    evaluate(m, [r for r in rows if r["model"] == m], lambda r: pfeats(r["p"]))
done = [r for r in rows if r["state"] == "completed"]
evaluate("completed only, prompt + model", done, lambda r: pfeats(r["p"]) + onehot(r))
