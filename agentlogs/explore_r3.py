#!/usr/bin/env python3
"""Round 3 EXPLORE (PLAN.md commit 9638be8). Touches only the EXPLORE half of repos
(random.Random(20261007), first 50%). Free exploration of cutoffs for H6 (repo size,
language) and H7 (user history). Output informs a frozen-choices commit; CONFIRM not read."""
import json, glob, random, math
from collections import defaultdict
import pyarrow.parquet as pq, pyarrow.compute as pc, pyarrow as pa
import lab

q = lab.quantile
rows = [r for f in sorted(glob.glob("full_rows/*.jsonl")) for r in map(json.loads, open(f))
        if r["calls"] > 3 and r["cost"] and r["model"] != "unknown"]

# user id per session
uid = {}
for f in ("part_00000_of_00002.parquet", "sessions1.parquet"):
    t = pq.read_table(f, columns=["id", "user"])
    uid.update(zip(t["id"].to_pylist(), pc.struct_field(t["user"], "id").to_pylist()))
# repo metadata
R = pa.concat_tables([pq.read_table(f"repos/r{i}.parquet", columns=["full_name", "code_lines", "main_language"]) for i in range(5)])
rmeta = dict(zip(R["full_name"].to_pylist(), zip(R["code_lines"].to_pylist(), R["main_language"].to_pylist())))
del R

# user history: computed over ALL priced sessions in time order (a user's own past is known
# at prediction time regardless of repo split); only strictly earlier sessions count
for r in rows:
    r["user"] = uid.get(r["session"])
by_user = defaultdict(list)
for r in rows:
    by_user[r["user"]].append(r)
for us in by_user.values():
    us.sort(key=lambda r: r["created_at"] or "")
    past = []
    for r in us:
        r["prior"] = list(past[-20:])  # last 20 earlier costs
        if r["created_at"]:
            past.append(r["cost"])

# split
repos = sorted({r["repo"] or "?" for r in rows})
random.Random(20261007).shuffle(repos)
explore = set(repos[:len(repos) // 2])
E = [r for r in rows if (r["repo"] or "?") in explore]
er = sorted({r["repo"] or "?" for r in E}); random.Random(1).shuffle(er)
fit_repos = set(er[:int(.7 * len(er))])
print(f"all {len(rows)} | EXPLORE {len(E)} sessions, {len(er)} repos")

b = lambda L: 0 if (L or 0) < 200 else 1 if L < 1500 else 2
base = lambda r: (r["model"], r["event"], b(r["prompt_chars"]))


def evaluate(name, sub):
    """Refine base groups by sub(r); fit p10/p90 on fit repos, score on held-out explore repos."""
    fit, fitb = defaultdict(list), defaultdict(list)
    for r in E:
        if (r["repo"] or "?") in fit_repos:
            fit[(base(r), sub(r))].append(r["cost"]); fitb[base(r)].append(r["cost"])
    tests = defaultdict(list)
    for r in E:
        if (r["repo"] or "?") not in fit_repos:
            tests[base(r)].append(r)
    ratios, cov, n = [], [], 0
    for g, ts in tests.items():
        if len(ts) < 150 or len(fitb[g]) < 100:
            continue
        bw = q(fitb[g], .9) / q(fitb[g], .1)
        ws, wn = 0, 0
        for r in ts:
            v = fit.get((g, sub(r)))
            if not v or len(v) < 50:
                v = fitb[g]
            lo, hi = q(v, .1), q(v, .9)
            ws += hi / lo; wn += 1; cov.append(lo <= r["cost"] <= hi)
        ratios.append((ws / wn) / bw); n += 1
    rs = sorted(ratios)
    print(f"{name:34} groups {n:3} | median ratio {q(rs,.5):.2f} | <=0.8: {sum(x<=.8 for x in rs):3} | coverage {sum(cov)/len(cov):.1%}")


def size_bucket(cuts):
    def f(r):
        cl = (rmeta.get(r["repo"]) or (None, None))[0]
        if not cl:
            return "na"
        return sum(cl >= c for c in cuts)
    return f

lang = lambda r: (rmeta.get(r["repo"]) or (None, None))[1] or "na"
TOP = {"TypeScript", "Python", "JavaScript", "Go", "Rust", "C#", "Java"}
lang7 = lambda r: lang(r) if lang(r) in TOP else "other"

print("\nH6 candidates")
evaluate("baseline (no refinement)", lambda r: 0)
for cuts in ([10_000, 100_000], [5_000, 50_000, 500_000], [2_000, 20_000, 200_000, 1_000_000]):
    evaluate(f"size {cuts}", size_bucket(cuts))
evaluate("language top7", lang7)
evaluate("size 3 x lang7", lambda r: (size_bucket([5_000, 50_000, 500_000])(r), lang7(r)))


def hist(k, nb, w):
    def f(r):
        p = r["prior"][-w:]
        if len(p) < k:
            return "none"
        m = q(p, .5)
        edges = [0.25, 0.6, 1.2, 3.0][:nb - 1] if nb <= 5 else None
        return sum(m >= e for e in edges)
    return f

print("\nH7 candidates (history = user's own earlier sessions, median cost)")
for k in (3, 5, 10):
    for w in (10, 20):
        for nb in (3, 5):
            evaluate(f"k>={k} window {w} buckets {nb}", hist(k, nb, w))
cnt = defaultdict(int)
for r in E:
    cnt[min(len(r['prior']), 20)] += 1
print("\nprior-count distribution (explore):", {k: cnt[k] for k in (0, 1, 3, 5, 10, 20)})
