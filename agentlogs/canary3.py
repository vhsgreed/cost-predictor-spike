#!/usr/bin/env python3
"""Three canaries on shard 0. Numbers only. Pass bar (fixed before running):
p90 bound <= ~2.5x actual at ~90% coverage, clearly beating the matched baseline.

A. Larger K: predict TOTAL from first K calls, K in 3/10/20. Baseline = average total
   among sessions that reached K calls (fair: conditions on survival to step K).
B. Mid-run: at step K, predict REMAINING tokens. Baseline = average remaining among
   sessions that reached K. Bound width measured on remaining.
C. Grouping: average within groups known at start (model, trigger event type,
   prompt-length bucket). No regression, just per-group p90.
Also: concentration (share of tokens in top 10% sessions) and call-count shape.
"""
import math, random, statistics as st
from collections import defaultdict
import pyarrow.parquet as pq
import pyarrow.compute as pc

t = pq.read_table("part_00000_of_00276.parquet", columns=["session", "entry_index", "data"])
sess = pc.struct_field(t["session"], "id").to_pylist(); idx = t["entry_index"].to_pylist()
u = pc.struct_field(t["data"], "usage")
pt = pc.struct_field(u, "prompt_tokens").to_pylist(); ct = pc.struct_field(u, "completion_tokens").to_pylist()
del t, u
calls = defaultdict(list)
for s, i, p, c in zip(sess, idx, pt, ct):
    if p: calls[s].append((i, p + (c or 0), p))
for s in calls: calls[s].sort()

S = pq.read_table("part_00000_of_00002.parquet", columns=["id", "model", "prompt", "event"])
ev = pc.struct_field(S["event"], "type").to_pylist()
meta = {i: (m or "unknown", len(p or ""), e or "none") for i, m, p, e in
        zip(S["id"].to_pylist(), S["model"].to_pylist(), S["prompt"].to_pylist(), ev)}

def lstsq(X, y, lam=1e-3):
    k = len(X[0]); A = [[sum(r[a]*r[b] for r in X)+(lam if a == b else 0) for b in range(k)] for a in range(k)]
    v = [sum(r[a]*yy for r, yy in zip(X, y)) for a in range(k)]
    for col in range(k):
        q = max(range(col, k), key=lambda r: abs(A[r][col])); A[col], A[q], v[col], v[q] = A[q], A[col], v[q], v[col]
        for r in range(k):
            if r != col:
                f = A[r][col]/A[col][col]; A[r] = [x-f*yv for x, yv in zip(A[r], A[col])]; v[r] -= f*v[col]
    return [v[i]/A[i][i] for i in range(k)]

def spearman(a, b):
    def rank(x):
        o = sorted(range(len(x)), key=lambda i: x[i]); r = [0]*len(x)
        for j, i in enumerate(o): r[i] = j
        return r
    ra, rb = rank(a), rank(b); m = len(a)
    return 1-6*sum((x-y)**2 for x, y in zip(ra, rb))/(m*(m*m-1))

def score(name, rows, feat, target):
    rows = rows[:]; random.Random(0).shuffle(rows); m = len(rows)
    if m < 80: print(f"  {name:28} n={m} skipped"); return
    fit, cal, te = rows[:m//2], rows[m//2:m*7//10], rows[m*7//10:]
    w = lstsq([feat(r) for r in fit], [math.log(target(r)) for r in fit])
    pr = lambda r: sum(a*b for a, b in zip(w, feat(r)))
    res = sorted(math.log(target(r))-pr(r) for r in cal); q = res[min(len(res)-1, math.ceil(.9*(len(res)+1))-1)]
    base = sorted(target(r) for r in fit+cal); b90 = base[int(.9*len(base))]
    y = [target(r) for r in te]
    cb = sum(v <= b90 for v in y)/len(y); wb = st.median(b90/v for v in y)
    cm = sum(math.log(target(r)) <= pr(r)+q for r in te)/len(y); wm = st.median(math.exp(pr(r)+q)/target(r) for r in te)
    ok = "PASS" if (wm <= 2.5 and cm >= .85 and wm < wb*0.8) else "fail"
    print(f"  {name:28} n={m:5} | baseline cov {cb:.0%} {wb:4.1f}x | model cov {cm:.0%} {wm:4.1f}x "
          f"spearman {spearman([math.exp(pr(r)) for r in te], y):.2f}  {ok}")

def early(c, K):
    e = c[:K]; return [1.0, math.log(sum(x[1] for x in e)), math.log(e[-1][2]+1),
                       math.log(max(1, (e[-1][2]-e[0][2])/max(1, K-1)))]

S_ = list(calls.values())
print("A. predict TOTAL from first K calls")
for K in (3, 10, 20):
    rows = [c for c in S_ if len(c) > K]
    score(f"K={K}", rows, lambda c: early(c, K), lambda c: sum(x[1] for x in c))
print("B. at step K, predict REMAINING tokens")
for K in (3, 10, 20, 40):
    rows = [c for c in S_ if len(c) > K]
    score(f"K={K}", rows, lambda c: early(c, K), lambda c: sum(x[1] for x in c[K:]))

print("C. per-group average (groups known at start)")
def bucket(L): return "short<200" if L < 200 else "mid<1500" if L < 1500 else "long"
grp = defaultdict(list)
for s, c in calls.items():
    if s in meta and len(c) > 3:
        m, L, e = meta[s]; grp[(m, e, bucket(L))].append(sum(x[1] for x in c))
allv = sorted(v for vs in grp.values() for v in vs); a90 = allv[int(.9*len(allv))]
print(f"  overall: n={len(allv)} p90 width {st.median(a90/v for v in allv):.1f}x")
for g, vs in sorted(grp.items(), key=lambda x: -len(x[1]))[:8]:
    if len(vs) < 40: continue
    vs = sorted(vs); g90 = vs[int(.9*len(vs))]
    print(f"  {str(g)[:62]:62} n={len(vs):4} p90 width {st.median(g90/v for v in vs):.1f}x")

tots = sorted((sum(x[1] for x in c) for c in S_ if len(c) > 3), reverse=True)
print(f"\nconcentration: top 10% of sessions = {sum(tots[:len(tots)//10])/sum(tots):.0%} of tokens")
nc = sorted(len(c) for c in S_ if len(c) > 3)
print("calls/session quantiles p10/p25/p50/p75/p90/p99:", [nc[int(q*len(nc))] for q in (.1, .25, .5, .75, .9, .99)])
