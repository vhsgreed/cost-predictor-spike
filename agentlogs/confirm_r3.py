#!/usr/bin/env python3
"""Round 3 CONFIRM, one look. Bar and choices: PLAN.md commits 9638be8 + freeze commit.
CONFIRM = repos not in the first 50% of random.Random(20261007) shuffle; inside it,
70% fit / 30% test by random.Random(20261008)."""
import json, glob, random
from collections import defaultdict
import pyarrow.parquet as pq, pyarrow.compute as pc, pyarrow as pa
import lab

q = lab.quantile
rows = [r for f in sorted(glob.glob("full_rows/*.jsonl")) for r in map(json.loads, open(f))
        if r["calls"] > 3 and r["cost"] and r["model"] != "unknown"]
uid = {}
for f in ("part_00000_of_00002.parquet", "sessions1.parquet"):
    t = pq.read_table(f, columns=["id", "user"])
    uid.update(zip(t["id"].to_pylist(), pc.struct_field(t["user"], "id").to_pylist()))
R = pa.concat_tables([pq.read_table(f"repos/r{i}.parquet", columns=["full_name", "main_language"]) for i in range(5)])
lang_of = dict(zip(R["full_name"].to_pylist(), R["main_language"].to_pylist())); del R

by_user = defaultdict(list)
for r in rows:
    by_user[uid.get(r["session"])].append(r)
for us in by_user.values():
    us.sort(key=lambda r: r["created_at"] or "")
    past = []
    for r in us:
        r["prior"] = past[-10:]
        if r["created_at"]:
            past = past + [r["cost"]]

TOP = {"TypeScript", "Python", "JavaScript", "Go", "Rust", "C#", "Java"}
def h6(r):
    l = lang_of.get(r["repo"])
    return "na" if not l else l if l in TOP else "other"
def h7(r):
    p = r["prior"]
    if len(p) < 3:
        return "none"
    m = q(p, .5)
    return sum(m >= e for e in (0.25, 0.6, 1.2, 3.0))
b = lambda L: 0 if (L or 0) < 200 else 1 if L < 1500 else 2
base = lambda r: (r["model"], r["event"], b(r["prompt_chars"]))

repos = sorted({r["repo"] or "?" for r in rows})
random.Random(20261007).shuffle(repos)
confirm = repos[len(repos) // 2:]
cr = sorted(confirm); random.Random(20261008).shuffle(cr)
fitset = set(cr[:int(.7 * len(cr))]); fitlist = sorted(fitset)
C = [r for r in rows if (r["repo"] or "?") in set(confirm)]
fit = [r for r in C if (r["repo"] or "?") in fitset]
test = [r for r in C if (r["repo"] or "?") not in fitset]
print(f"CONFIRM {len(C)} sessions, {len(confirm)} repos | fit {len(fit)} / test {len(test)}", flush=True)


def run(name, sub):
    tests = defaultdict(list)
    for r in test:
        tests[base(r)].append(r)
    elig = [g for g, ts in tests.items() if len(ts) >= 300]
    tcount = {g: defaultdict(int) for g in elig}
    for g in elig:
        for r in tests[g]:
            tcount[g][sub(r)] += 1
    by_repo = defaultdict(list)
    for r in fit:
        by_repo[r["repo"] or "?"].append((base(r), sub(r), r["cost"]))

    def fitq(repo_sample):
        cell, grp = defaultdict(list), defaultdict(list)
        for rp in repo_sample:
            for g, s, c in by_repo[rp]:
                if g in tcount:
                    cell[(g, s)].append(c); grp[g].append(c)
        return cell, grp

    def ratio(cell, grp, g):
        if len(grp[g]) < 100:
            return None
        bl, bh = q(grp[g], .1), q(grp[g], .9)
        ws = 0
        for s, n in tcount[g].items():
            v = cell.get((g, s))
            lo, hi = (q(v, .1), q(v, .9)) if v and len(v) >= 50 else (bl, bh)
            ws += n * hi / lo
        return (ws / sum(tcount[g].values())) / (bh / bl)

    cell, grp = fitq(fitlist)
    boots = defaultdict(list); rng = random.Random(20261009)
    for it in range(200):
        cb, gb = fitq([fitlist[rng.randrange(len(fitlist))] for _ in fitlist])
        for g in elig:
            x = ratio(cb, gb, g)
            if x is not None:
                boots[g].append(x)
        if it % 50 == 49:
            print(f"  {name} bootstrap {it+1}/200", flush=True)
    passed, out = 0, []
    for g in sorted(elig, key=lambda g: -len(tests[g])):
        rt = ratio(cell, grp, g)
        if rt is None:
            continue
        bs = sorted(boots[g]); hi = bs[int(.95 * len(bs))]
        inside = []
        for r in tests[g]:
            v = cell.get((g, sub(r)))
            lo, h = (q(v, .1), q(v, .9)) if v and len(v) >= 50 else (q(grp[g], .1), q(grp[g], .9))
            inside.append(lo <= r["cost"] <= h)
        covs = sorted(sum(inside[rng.randrange(len(inside))] for _ in inside) / len(inside) for _ in range(500))
        ok = rt <= .8 and hi < .8 and covs[25] <= .8 <= covs[475]
        passed += ok
        out.append({"group": list(map(str, g)), "n_test": len(tests[g]), "ratio": round(rt, 3), "ratio_ci_hi": round(hi, 3),
                    "coverage": round(sum(inside) / len(inside), 3), "cov_ci": [covs[25], covs[475]], "pass": ok})
        print(f"  {g[0][:17]:17} {str(g[1])[:16]:16} b{g[2]} n={len(tests[g]):6} | ratio {rt:.2f} CI hi {hi:.2f} | "
              f"cov {sum(inside)/len(inside):.0%} [{covs[25]:.0%},{covs[475]:.0%}] | {'PASS' if ok else 'fail'}", flush=True)
    n = len(out)
    verdict = "PASS" if passed * 3 >= n and n else "FAIL"
    print(f"{name}: {passed}/{n} groups pass -> {verdict} (need >= 1/3)\n", flush=True)
    return {"passed": passed, "eligible": n, "verdict": verdict, "groups": out}

res = {"H6": run("H6 language", h6), "H7": run("H7 user history", h7)}
json.dump(res, open("confirm_r3.json", "w"), indent=1)
print("wrote confirm_r3.json")
