#!/usr/bin/env python3
"""Full run, phase 3: H5 FINAL verdict (bar pre-registered in PLAN.md 2026-10-06).
Prompt text -> nomic-embed-text -> k-means k=10 (seed 20261006, centroids fit on train
prompts ONLY) -> cost separation on full_sessions.jsonl, repo-held-out 70/30.

Per cluster (>= 30 train and >= 30 test sessions):
  1. train p10-p90 $ width <= 0.8 x dominant-model train width
  2. 90% CI of the width ratio (200 bootstrap resamples over training repos;
     upper bound = 95th percentile) entirely below 0.8
  3. test coverage 90% CI (500 membership resamples) contains 80%
H5 passes if >= 2 clusters pass. One look."""
import json, os, glob, random, urllib.request
from collections import defaultdict
import numpy as np
import pyarrow.parquet as pq
import lab

ROOT = os.path.dirname(os.path.abspath(__file__))
K, SEED, NORM = 10, 20261006, True
CACHE = os.path.join(ROOT, "h5_embeddings.npz")
q = lab.quantile

rows = []
for f in sorted(glob.glob(os.path.join(ROOT, "full_rows", "*.jsonl"))):
    for line in open(f):
        r = json.loads(line)
        if r["calls"] > 3 and r["cost"] and r["model"] and r["model"] != "unknown":
            rows.append(r)
prompt = {}
for f in ("part_00000_of_00002.parquet", "sessions1.parquet"):
    S = pq.read_table(os.path.join(ROOT, f), columns=["id", "prompt"])
    for sid, p in zip(S["id"].to_pylist(), S["prompt"].to_pylist()):
        if p and len(p.split()) >= 4:
            prompt[sid] = p[:400]
rows = [r for r in rows if r["session"] in prompt]
print(f"sessions with prompt + cost: {len(rows)}", flush=True)

# ---- embeddings (checkpointed) ----
ids = [r["session"] for r in rows]
texts = [prompt[s] for s in ids]
known = {}
if os.path.exists(CACHE):
    z = np.load(CACHE)
    known = dict(zip(z["ids"].tolist(), z["vecs"]))
print(f"embedding cache: {len(known)} known", flush=True)

def embed_batch(batch):
    req = urllib.request.Request("http://127.0.0.1:11434/api/embed",
                                 data=json.dumps({"model": "nomic-embed-text", "input": batch}).encode())
    return json.load(urllib.request.urlopen(req, timeout=600))["embeddings"]

missing = [(i, t) for i, t in zip(ids, texts) if i not in known]
for off in range(0, len(missing), 64):
    chunk = missing[off:off + 64]
    vecs = embed_batch([t for _, t in chunk])
    for (i, _), v in zip(chunk, vecs):
        known[i] = v
    if off % 640 == 0:
        np.savez(CACHE, ids=np.array(list(known)), vecs=np.array([known[i] for i in known], dtype=np.float32))
        print(f"  embedded {len(known)}/{len(ids)}", flush=True)
np.savez(CACHE, ids=np.array(ids), vecs=np.array([known[i] for i in ids], dtype=np.float32))
X = np.array([known[i] for i in ids], dtype=np.float32)
if NORM:
    X /= np.linalg.norm(X, axis=1, keepdims=True)
print(f"embeddings ready: {X.shape}", flush=True)

# ---- repo split (70/30, seed 20261006) ----
repos = defaultdict(list)
for i, r in enumerate(rows):
    repos[r["repo"] or "?"].append(i)
rl = sorted(repos)
random.Random(SEED).shuffle(rl)
train_idx = [i for g in rl[:int(.7 * len(rl))] for i in repos[g]]
test_idx = [i for g in rl[int(.7 * len(rl)):] for i in repos[g]]
print(f"repos {len(rl)}: train {len(train_idx)} / test {len(test_idx)}", flush=True)

# ---- k-means on TRAIN prompts only (k-means++ init, 40 iters, seed 20261006) ----
Xt = X[train_idx]
xt_sq = (Xt ** 2).sum(1)

def assign_to(X_, sq, C_):
    """argmin ||x - c||^2 = ||x||^2 + ||c||^2 - 2 x.c (plain k-means)."""
    return (sq[:, None] + (C_ ** 2).sum(1)[None, :] - 2 * (X_ @ C_.T)).argmin(1)

rng = np.random.default_rng(SEED)
centers = [Xt[rng.integers(len(Xt))]]
for _ in range(K - 1):
    d2 = np.full(len(Xt), np.inf)
    for c in centers:
        d2 = np.minimum(d2, ((Xt - c) ** 2).sum(1))
    centers.append(Xt[rng.choice(len(Xt), p=d2 / d2.sum())])
C = np.array(centers)
for _ in range(40):
    a = assign_to(Xt, xt_sq, C)
    for j in range(K):
        if (a == j).any():
            C[j] = Xt[a == j].mean(0)
assign = assign_to(X, (X ** 2).sum(1), C)  # all sessions -> nearest centroid

cost = [r["cost"] for r in rows]
model = [r["model"] for r in rows]
train_cl, train_md, test_cl = defaultdict(list), defaultdict(list), defaultdict(list)
for i in train_idx:
    train_cl[assign[i]].append(cost[i]); train_md[model[i]].append(cost[i])
for i in test_idx:
    test_cl[assign[i]].append(cost[i])

print("\nclusters (n, train median, shortest example):")
for c in range(K):
    members = [i for i in train_idx + test_idx if assign[i] == c]
    if not members:
        continue
    ex = min((texts[i] for i in members), key=len)
    print(f"  k{c}: n={len(members):6} median ${q([cost[i] for i in members], .5):5.2f} | {ex[:80].replace(chr(10), ' ')}")

# ---- evaluation per pre-registered bar ----
print("\nH5 final (repo-held-out; bar: width <= 0.8x model, ratio CI hi < 0.8, cov CI has 80%)")
# precompute per-repo cluster/model cost lists for the bootstrap
repo_cl, repo_md = defaultdict(lambda: defaultdict(list)), defaultdict(lambda: defaultdict(list))
for i in train_idx:
    g = rows[i]["repo"] or "?"
    repo_cl[g][assign[i]].append(cost[i])
    repo_md[g][model[i]].append(cost[i])
trepo = [g for g in rl[:int(.7 * len(rl))]]
dom = {c: max(train_md, key=lambda m: sum(model[i] == m for i in train_idx if assign[i] == c))
       for c in range(K) if len(train_cl[c]) >= 30}

ratio_ci = {c: [] for c in dom}
brng = random.Random(SEED)
for it in range(200):
    samp = [trepo[brng.randrange(len(trepo))] for _ in trepo]
    clv = defaultdict(list); mdv = defaultdict(list)
    for g in samp:
        for c, v in repo_cl[g].items():
            clv[c] += v
        for m, v in repo_md[g].items():
            mdv[m] += v
    for c in dom:
        gv, mv = clv[c], mdv[dom[c]]
        if len(gv) >= 30 and len(mv) >= 30:
            ratio_ci[c].append((q(gv, .9) / q(gv, .1)) / (q(mv, .9) / q(mv, .1)))
    if it % 40 == 39:
        print(f"  bootstrap {it + 1}/200", flush=True)

passed, results = 0, []
for c in sorted(dom):
    gl, gh = q(train_cl[c], .1), q(train_cl[c], .9)
    m = dom[c]
    ml, mh = q(train_md[m], .1), q(train_md[m], .9)
    gw, mw = gh / gl, mh / ml
    rc = sorted(ratio_ci[c])
    r_hi = rc[int(.95 * len(rc))] if rc else float("nan")
    tc = test_cl[c]
    if len(tc) < 30:
        print(f"  k{c}: n_test={len(tc):4} < 30 -> skip"); continue
    inside = [gl <= v <= gh for v in tc]
    cov = sum(inside) / len(inside)
    covs = sorted(sum(inside[brng.randrange(len(inside))] for _ in inside) / len(inside) for _ in range(500))
    ok = gw <= 0.8 * mw and r_hi < 0.8 and covs[25] <= .8 <= covs[475]
    passed += ok
    results.append({"cluster": c, "n_train": len(train_cl[c]), "n_test": len(tc), "width": round(gw, 1),
                    "model": m, "model_width": round(mw, 1), "ratio": round(gw / mw, 2),
                    "ratio_ci_hi": round(r_hi, 2), "coverage": round(cov, 2),
                    "cov_ci": [round(covs[25], 2), round(covs[475], 2)], "pass": bool(ok)})
    print(f"  k{c}: n_train={len(train_cl[c]):5} n_test={len(tc):4} | cluster {gw:5.1f}x (ratio {gw/mw:4.2f}, "
          f"CI hi {r_hi:4.2f}) cov {cov:.0%} [{covs[25]:.0%},{covs[475]:.0%}] | model {m[:14]} {mw:5.1f}x | "
          f"{'PASS' if ok else 'fail'}")

verdict = "PASS" if passed >= 2 else "FAIL"
print(f"\nH5 FINAL: {passed} clusters pass -> {verdict} (need >= 2)")

ev = defaultdict(list)
for i in train_idx:
    ev[(model[i], rows[i]["event"])].append(cost[i])
evw = [q(v, .9) / q(v, .1) for v in ev.values() if len(v) >= 30]
clw = [q(train_cl[c], .9) / q(train_cl[c], .1) for c in train_cl if len(train_cl[c]) >= 30]
print(f"informational: model x event width median {q(evw, .5):.1f}x (n={len(evw)}); "
      f"cluster width median {q(clw, .5):.1f}x (n={len(clw)})")

with open(os.path.join(ROOT, "h5_final.json"), "w") as fh:
    json.dump({"verdict": verdict, "clusters_passed": passed, "n_sessions": len(rows),
               "n_train": len(train_idx), "n_test": len(test_idx), "results": results}, fh, indent=1)
print("wrote h5_final.json")
