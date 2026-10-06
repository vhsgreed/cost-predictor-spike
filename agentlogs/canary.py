#!/usr/bin/env python3
"""Canary: one AgentLogs log shard. Numbers only (no prompt/message text read).

Go/no-go criteria, fixed BEFORE looking at results:
  G1 parse:     >= 90% of entries that carry `usage` have prompt_tokens > 0
  G2 volume:    >= 300 sessions with > K model calls
  G3 join:      >= 30% of those sessions found in the session shard (model/state)
  G4 integrity: >= 80% of sessions have non-decreasing prompt tokens on most steps
                (context growth; catches mis-ordered or mixed-up sessions)
  G5 runtime:   whole canary < 10 min, peak RSS < 8 GB
Then the same spike as on Hermes logs: first-K-calls predictor vs prior-only.
"""
import math, os, resource, statistics as st, sys, time
from collections import defaultdict
import pyarrow.parquet as pq
import pyarrow.compute as pc

K = int(os.environ.get("K", "3"))
T0 = time.time()
LOG = "part_00000_of_00276.parquet"
SES = "part_00000_of_00002.parquet"

t = pq.read_table(LOG, columns=["session", "entry_index", "data"])
sess = pc.struct_field(t["session"], "id").to_pylist()
idx = t["entry_index"].to_pylist()
data = t["data"]
usage = pc.struct_field(data, "usage")
pt = pc.struct_field(usage, "prompt_tokens").to_pylist()
ct = pc.struct_field(usage, "completion_tokens").to_pylist()
cached = pc.struct_field(pc.struct_field(usage, "prompt_tokens_details"), "cached_tokens").to_pylist()
del t, data, usage

entries = len(sess)
with_usage = sum(p is not None for p in pt)
parsed_ok = sum(1 for p in pt if p is not None and p > 0)
calls = defaultdict(list)
for s, i, p, c, ca in zip(sess, idx, pt, ct, cached):
    if p is not None and p > 0:
        calls[s].append((i, p, c or 0, ca or 0))
for s in calls:
    calls[s].sort()

meta = {}
st_tab = pq.read_table(SES, columns=["id", "model", "state"])
for i, m, s in zip(st_tab["id"].to_pylist(), st_tab["model"].to_pylist(), st_tab["state"].to_pylist()):
    meta[i] = (m, s)

eligible = {s: c for s, c in calls.items() if len(c) > K}
joined = sum(s in meta for s in eligible)


def grows(c):
    steps = [b[1] >= a[1] for a, b in zip(c, c[1:])]
    return sum(steps) / len(steps) >= 0.8 if steps else True

integrity = sum(grows(c) for c in eligible.values()) / max(1, len(eligible))
rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
elapsed = time.time() - T0

g = {
    "G1 parse": (parsed_ok / max(1, with_usage), parsed_ok / max(1, with_usage) >= 0.9),
    "G2 volume": (len(eligible), len(eligible) >= 300),
    "G3 join": (joined / max(1, len(eligible)), joined / max(1, len(eligible)) >= 0.3),
    "G4 integrity": (integrity, integrity >= 0.8),
    "G5 runtime": (f"{elapsed:.0f}s, {rss_gb:.1f} GB", elapsed < 600 and rss_gb < 8),
}
print(f"entries {entries:,}; with usage {with_usage:,}; sessions with calls {len(calls):,}; with > {K} calls {len(eligible):,}")
for k, (v, ok) in g.items():
    print(f"  {'PASS' if ok else 'FAIL'}  {k}: {v if isinstance(v, (int, str)) else f'{v:.1%}'}")
if not all(ok for _, ok in g.values()):
    print("NO-GO: fix before any broader sweep"); sys.exit(1)

# ---- spike, same method as Hermes logs (log-log least squares + split conformal p90)
def total(c): return sum(p + o for _, p, o, _ in c)
def feats(c):
    e = c[:K]; early = sum(p + o for _, p, o, _ in e)
    growth = max(1, (e[-1][1] - e[0][1]) / max(1, len(e) - 1))
    return [1.0, math.log(early), math.log(e[0][1] + 1), math.log(growth)]

def lstsq(X, y):
    n = len(X[0]); A = [[sum(r[a]*r[b] for r in X) + (1e-6 if a == b else 0) for b in range(n)] for a in range(n)]
    v = [sum(r[a]*yy for r, yy in zip(X, y)) for a in range(n)]
    for col in range(n):
        p = max(range(col, n), key=lambda r: abs(A[r][col])); A[col], A[p], v[col], v[p] = A[p], A[col], v[p], v[col]
        for r in range(n):
            if r != col:
                f = A[r][col] / A[col][col]; A[r] = [x - f*yv for x, yv in zip(A[r], A[col])]; v[r] -= f*v[col]
    return [v[i] / A[i][i] for i in range(n)]

def spearman(a, b):
    def rank(x):
        o = sorted(range(len(x)), key=lambda i: x[i]); r = [0]*len(x)
        for k, i in enumerate(o): r[i] = k
        return r
    ra, rb = rank(a), rank(b); n = len(a)
    return 1 - 6*sum((x-y)**2 for x, y in zip(ra, rb)) / (n*(n*n-1))

def run(name, items):
    import random; random.Random(0).shuffle(items)
    n = len(items)
    if n < 60: print(f"{name}: n={n}, skipped (< 60)"); return
    fit, cal, te = items[: n*5//10], items[n*5//10 : n*7//10], items[n*7//10 :]
    w = lstsq([feats(c) for c in fit], [math.log(total(c)) for c in fit])
    pred = lambda c: sum(a*b for a, b in zip(w, feats(c)))
    res = sorted(math.log(total(c)) - pred(c) for c in cal)
    q = res[min(len(res)-1, math.ceil(0.9*(len(res)+1)) - 1)]
    prior = sorted(total(c) for c in fit + cal); pq90 = prior[int(0.9*len(prior))]
    y = [total(c) for c in te]; pb = [math.exp(pred(c)) for c in te]
    mae = lambda p: st.median(abs(a-b)/b for a, b in zip(p, y))*100
    cov_a = sum(v <= pq90 for v in y)/len(y); cov_b = sum(math.log(v) <= pred(c)+q for v, c in zip(y, te))/len(y)
    wa = st.median(pq90/v for v in y); wb = st.median(math.exp(pred(c)+q)/v for v, c in zip(y, te))
    print(f"{name}: n={n} (test {len(te)}) | prior p90 cov {cov_a:.0%} width {wa:.1f}x err {mae([st.median(prior)]*len(y)):.0f}% "
          f"| first-{K} p90 cov {cov_b:.0%} width {wb:.1f}x err {mae(pb):.0f}% spearman {spearman(pb, y):.2f}")

items = list(eligible.values())
tot = [total(c) for c in items]
print(f"\ntotal tokens/session: median {st.median(tot):,.0f}, p90 {sorted(tot)[int(.9*len(tot))]:,}, max {max(tot):,}")
print(f"input share: {st.median(sum(p for _, p, _, _ in c)/total(c) for c in items):.1%}; "
      f"cached share of input: {st.median(sum(ca for *_, ca in c)/max(1,sum(p for _, p, _, _ in c)) for c in items):.1%}")
run("ALL", items[:])
by = defaultdict(list)
for s, c in eligible.items():
    if s in meta: by[meta[s][0] or "unknown"].append(c)
for m, its in sorted(by.items(), key=lambda x: -len(x[1]))[:5]:
    run(m, its)
print(f"elapsed {time.time()-T0:.0f}s")
