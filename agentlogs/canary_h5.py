#!/usr/bin/env python3
"""H5 canary (bar pre-registered in PLAN.md before this ran).
Prompt text -> nomic-embed-text -> k-means (k=10) -> cost separation on dev sessions,
repo-held-out. Compares cluster ranges to model-only ranges; model x event informational.
Prompts come from the session table's prompt column (same source as canary_regex.py)."""
import json, os, random, urllib.request
from collections import defaultdict
import numpy as np
import pyarrow.parquet as pq
import lab

ROOT = os.path.dirname(os.path.abspath(__file__))
K, SEED = 10, 20261006

prompt = {}
for f in ("part_00000_of_00002.parquet", "sessions1.parquet"):
    S = pq.read_table(os.path.join(ROOT, f), columns=["id", "prompt"])
    for sid, p in zip(S["id"].to_pylist(), S["prompt"].to_pylist()):
        if p and len(p.split()) >= 4:
            prompt[sid] = p[:400]
print(f"session prompts: {len(prompt)}")

dev = {r["session"]: r for r in map(json.loads, open(os.path.join(ROOT, "dev.jsonl")))}
rows = [(s, dev[s], prompt[s]) for s in prompt if s in dev and dev[s]["cost"] and dev[s]["calls"] > 3]
print(f"joined {len(rows)} dev sessions with prompt + cost")


def embed(texts, batch=64):
    vecs = []
    for i in range(0, len(texts), batch):
        req = urllib.request.Request("http://127.0.0.1:11434/api/embed",
                                     data=json.dumps({"model": "nomic-embed-text",
                                                      "input": texts[i:i + batch]}).encode())
        vecs += json.load(urllib.request.urlopen(req, timeout=300))["embeddings"]
    return np.array(vecs, dtype=np.float32)


def kmeans(X, k=K, iters=40):
    rng = np.random.default_rng(SEED)
    centers = [X[rng.integers(len(X))]]
    for _ in range(k - 1):
        d2 = np.full(len(X), np.inf)
        for c in centers:
            d2 = np.minimum(d2, ((X - c) ** 2).sum(1))
        centers.append(X[rng.choice(len(X), p=d2 / d2.sum())])
    C = np.array(centers)
    for _ in range(iters):
        a = ((X[:, None, :] - C[None, :, :]) ** 2).sum(2).argmin(1)
        for j in range(k):
            if (a == j).any():
                C[j] = X[a == j].mean(0)
    return ((X[:, None, :] - C[None, :, :]) ** 2).sum(2).argmin(1)


X = embed([t for _, _, t in rows])
X /= np.linalg.norm(X, axis=1, keepdims=True)
a = kmeans(X)
by_sid = {s: (r, t, int(c)) for (s, r, t), c in zip(rows, a)}

print("\nclusters (shortest example prompt each):")
for c in sorted(set(a)):
    members = [v for v in by_sid.values() if v[2] == c]
    med = lab.quantile([r["cost"] for r, t, c in members], .5)
    ex = min((t for r, t, c in members), key=len)
    print(f"  k{c}: n={len(members):4} median ${med:5.2f} | {ex[:90].replace(chr(10), ' ')}")

repos = defaultdict(list)
for s, r, t in rows:
    repos[r["repo"]].append(s)
rk = sorted(repos, key=lambda g: -len(repos[g]))
train_s = {s for g in rk[:int(.7 * len(rk))] for s in repos[g]}
test_s = {s for g in rk[int(.7 * len(rk)):] for s in repos[g]}
print(f"\nrepos {len(rk)}: train {len(train_s)} sessions / test {len(test_s)}")
q = lab.quantile

train_cl, train_md, test_cl = defaultdict(list), defaultdict(list), defaultdict(list)
for s in train_s:
    r, t, c = by_sid[s]; train_cl[c].append(r["cost"]); train_md[r["model"]].append(r["cost"])
for s in test_s:
    r, t, c = by_sid[s]; test_cl[c].append(r["cost"])
rng = random.Random(SEED)

print("H5 evaluation (repo-held-out; bar: width <= 0.8 x model-only AND ratio CI < 0.8 AND cov CI has 80%)")
passed = 0
for c in sorted(train_cl):
    if len(train_cl[c]) < 30 or len(test_cl[c]) < 30:
        continue
    gl, gh = q(train_cl[c], .1), q(train_cl[c], .9)
    mkey = max(train_md, key=lambda m: sum(by_sid[s][0]["model"] == m for s in train_s if by_sid[s][2] == c))
    ml, mh = q(train_md[mkey], .1), q(train_md[mkey], .9)
    gw, mw = gh / gl, mh / ml
    # bootstrap the width ratio over training repos (pre-registered in PLAN.md)
    ratios = []
    trepos = defaultdict(list)
    for s in train_s:
        trepos[by_sid[s][0]["repo"]].append(s)
    tl = sorted(trepos)
    for _ in range(200):
        samp = [s for _ in tl for s in trepos[tl[rng.randrange(len(tl))]]]
        gv = [by_sid[s][0]["cost"] for s in samp if by_sid[s][2] == c]
        mv = [by_sid[s][0]["cost"] for s in samp if by_sid[s][0]["model"] == mkey]
        if len(gv) >= 30 and len(mv) >= 30:
            ratios.append((q(gv, .9) / q(gv, .1)) / (q(mv, .9) / q(mv, .1)))
    ratios.sort()
    r_hi = ratios[int(.95 * len(ratios))] if ratios else float("nan")
    inside = [gl <= v <= gh for v in test_cl[c]]
    cov = sum(inside) / len(inside)
    covs = sorted(sum(inside[rng.randrange(len(inside))] for _ in inside) / len(inside) for _ in range(500))
    ok = gw <= 0.8 * mw and r_hi < 0.8 and covs[25] <= .8 <= covs[475]
    passed += ok
    print(f"  k{c}: n_test={len(test_cl[c]):4} | cluster {gw:5.1f}x (ratio CI hi {r_hi:4.2f}) cov {cov:.0%} [{covs[25]:.0%},{covs[475]:.0%}]"
          f" | model {mw:5.1f}x | {'PASS' if ok else 'fail'}")
print(f"H5: {passed} clusters pass -> {'PASS' if passed >= 2 else 'FAIL'} (need >= 2)")

ev = defaultdict(list)
for s in train_s:
    r, t, c = by_sid[s]; ev[(r["model"], r["event"])].append(r["cost"])
evw = [q(v, .9) / q(v, .1) for k, v in ev.items() if len(v) >= 30]
clw = [q(train_cl[c], .9) / q(train_cl[c], .1) for c in train_cl]
print(f"informational: model x event width median {q(evw, .5):.1f}x (n={len(evw)}); "
      f"cluster width median {q(clw, .5):.1f}x (n={len(clw)})")
