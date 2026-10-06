#!/usr/bin/env python3
"""Round 4: external replication of H1 (range) and H4 (floor). One look.
Pre-registration: PLAN.md commits 2b8c3f7, e513bf5, 02f87ba.
E1 = Exgentic/agent-llm-traces-v2 (cost = agent_cost; group = model x benchmark[/subset] x harness;
     split by (benchmark, task hash) 70/30, random.Random(20261010); bootstrap unit = task).
E3 = MaxDevv/real-pi-coding-agent-traces-sessions (cost = sum usage.cost.total; group = dominant
     model x project; split chronological within project, first 70% fit; bootstrap unit = session).
Primary eligibility >= 100 fit / >= 30 test; secondary >= 50 fit / >= 20 test."""
import glob, hashlib, json, os, random, re
from collections import Counter, defaultdict
import pyarrow.parquet as pq
import lab

q = lab.quantile
EXT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "external")


def load_e1():
    rows = []
    for f in sorted(glob.glob(os.path.join(EXT, "Exgentic__agent-llm-traces-v2/data/**/*.parquet"), recursive=True)):
        for b in pq.ParquetFile(f).iter_batches(batch_size=64, columns=["benchmark", "benchmark_subset", "harness", "models", "agent_cost", "spans"]):
            for r in b.to_pylist():
                key = ""
                for s in r["spans"] or []:
                    im = (s["attributes"] or {}).get("gen_ai.input.messages")
                    if im:
                        m = json.loads(im)
                        u = [p.get("content", "") for x in m if x.get("role") == "user" for p in x.get("parts", []) if p.get("type") == "text"]
                        key = hashlib.sha1((u[0] if u else "").encode()).hexdigest()[:12]
                        break
                bench = r["benchmark"] + ("/" + r["benchmark_subset"] if r["benchmark_subset"] else "")
                rows.append({"model": str(r["models"]), "group": (str(r["models"]), bench, r["harness"]),
                             "unit": (r["benchmark"], key), "cost": r["agent_cost"]})
    return rows


def load_e3():
    rows = []
    for f in sorted(glob.glob(os.path.join(EXT, "MaxDevv__real-pi-coding-agent-traces-sessions/*.jsonl"))):
        name = os.path.basename(f)
        if name == "manifest.jsonl":
            continue
        m = re.match(r"(.+?)-(?:sessions-)?(20\d\d-\d\d-\d\dT[\d-]+Z)", name)
        proj, ts = (m.group(1), m.group(2)) if m else ("?", "")
        cost, ms = 0.0, Counter()
        for line in open(f):
            o = json.loads(line); msg = o.get("message") or {}
            u = msg.get("usage") if isinstance(msg, dict) else None
            if isinstance(u, dict) and isinstance(u.get("cost"), dict):
                cost += u["cost"].get("total") or 0
                ms[msg.get("model") or "?"] += 1
        if not ms:
            continue
        model = ms.most_common(1)[0][0]
        rows.append({"model": model, "group": (model, proj), "unit": name, "proj": proj, "ts": ts, "cost": cost})
    return rows


def split_e1(rows):
    units = sorted({r["unit"] for r in rows}); random.Random(20261010).shuffle(units)
    fit = set(units[:int(.7 * len(units))])
    return [r for r in rows if r["unit"] in fit], [r for r in rows if r["unit"] not in fit]


def split_e3(rows):
    byp = defaultdict(list)
    for r in rows:
        byp[r["proj"]].append(r)
    fit, test = [], []
    for v in byp.values():
        v.sort(key=lambda r: r["ts"]); k = int(.7 * len(v))
        fit += v[:k]; test += v[k:]
    return fit, test


def evaluate(name, fit, test, min_fit, min_test, rng):
    fg, fm, tg = defaultdict(list), defaultdict(list), defaultdict(list)
    for r in fit:
        fg[r["group"]].append(r["cost"]); fm[r["model"]].append(r["cost"])
    for r in test:
        tg[r["group"]].append(r["cost"])
    elig = [g for g in fg if len(fg[g]) >= min_fit and len(tg[g]) >= min_test]
    out = {"eligible": len(elig), "groups": []}
    if len(elig) < 3:
        out["H4"] = out["H1"] = "not evaluable"
        print(f"  {name}: {len(elig)} eligible groups -> not evaluable")
        return out
    # bootstrap over fit units for the width ratio
    by_unit = defaultdict(list)
    for r in fit:
        by_unit[r["unit"]].append(r)
    units = sorted(by_unit, key=str)
    boots = defaultdict(list)
    for _ in range(200):
        bg, bm = defaultdict(list), defaultdict(list)
        for _u in units:
            for r in by_unit[units[rng.randrange(len(units))]]:
                bg[r["group"]].append(r["cost"]); bm[r["model"]].append(r["cost"])
        for g in elig:
            if len(bg[g]) >= 10 and len(bm[g[0]]) >= 10 and q(bg[g], .1) > 0 and q(bm[g[0]], .1) > 0:
                boots[g].append((q(bg[g], .9) / q(bg[g], .1)) / (q(bm[g[0]], .9) / q(bm[g[0]], .1)))

    def ci(flags, n=500):
        s = sorted(sum(flags[rng.randrange(len(flags))] for _ in flags) / len(flags) for _ in range(n))
        return s[int(.05 * n)], s[int(.95 * n)]

    h1p = h4p = 0; pooled = []
    for g in sorted(elig, key=lambda g: -len(tg[g])):
        lo, hi = q(fg[g], .1), q(fg[g], .9); ml, mh = q(fm[g[0]], .1), q(fm[g[0]], .9)
        ratio = (hi / lo) / (mh / ml)
        bs = sorted(boots[g]); r_hi = bs[int(.95 * len(bs))] if bs else float("inf")
        inside = [lo <= c <= hi for c in tg[g]]; above = [c >= lo for c in tg[g]]
        pooled += above
        c80, c90 = ci(inside), ci(above)
        ok1 = ratio <= .8 and r_hi < .8 and c80[0] <= .8 <= c80[1]
        ok4 = c90[0] <= .9 <= c90[1]
        h1p += ok1; h4p += ok4
        out["groups"].append({"group": [str(x) for x in g], "n_fit": len(fg[g]), "n_test": len(tg[g]),
                              "floor": round(lo, 4), "median": round(q(fg[g], .5), 4), "p90": round(hi, 4),
                              "width": round(hi / lo, 2), "model_width": round(mh / ml, 2), "ratio": round(ratio, 3),
                              "ratio_ci_hi": round(r_hi, 3), "cov80": round(sum(inside) / len(inside), 3), "cov80_ci": c80,
                              "floor_cov": round(sum(above) / len(above), 3), "floor_cov_ci": c90, "H1": ok1, "H4": ok4})
        print(f"    {'/'.join(map(str, g))[:58]:58} fit {len(fg[g]):4} test {len(tg[g]):3} | floor ${lo:.3f} held "
              f"{sum(above)/len(above):.0%} [{c90[0]:.0%},{c90[1]:.0%}] {'ok' if ok4 else '--'} | width {hi/lo:5.1f}x vs "
              f"{mh/ml:5.1f}x ratio {ratio:.2f} hi {r_hi:.2f} cov {sum(inside)/len(inside):.0%} {'PASS' if ok1 else '--'}")
    pc = ci(pooled)
    h4 = pc[0] <= .9 <= pc[1] and h4p * 3 >= 2 * len(elig)
    h1 = h1p >= max(1, -(-len(elig) // 5))
    out.update({"H4": "PASS" if h4 else "FAIL", "H1": "PASS" if h1 else "FAIL", "h4_groups": h4p, "h1_groups": h1p,
                "pooled_floor_cov": round(sum(pooled) / len(pooled), 3), "pooled_floor_ci": pc})
    print(f"  {name}: R-H4 {out['H4']} (pooled floor held {sum(pooled)/len(pooled):.1%} [{pc[0]:.1%},{pc[1]:.1%}], "
          f"{h4p}/{len(elig)} groups) | R-H1 {out['H1']} ({h1p}/{len(elig)} groups, need {max(1, -(-len(elig)//5))})")
    return out


res = {}
for tag, loader, splitter in (("E1 Exgentic", load_e1, split_e1), ("E3 pi sessions", load_e3, split_e3)):
    rows = loader(); n0 = len(rows)
    rows = [r for r in rows if r["cost"] and r["cost"] > 0]
    fit, test = splitter(rows)
    print(f"\n{tag}: {n0} runs, {n0 - len(rows)} excluded (cost missing or 0), fit {len(fit)} / test {len(test)}", flush=True)
    res[tag] = {"n": len(rows), "excluded": n0 - len(rows)}
    for label, mf, mt in (("primary >=100/30", 100, 30), ("secondary >=50/20", 50, 20)):
        print(f"  [{label}]")
        res[tag][label] = evaluate(f"{tag} {label}", fit, test, mf, mt, random.Random(20261011))

print("\nREPLICATION VERDICT (claim needs pass in every evaluable dataset AND >= 2 evaluable)")
datasets = list(res)
for label in ("primary >=100/30", "secondary >=50/20"):
    for h in ("H4", "H1"):
        v = [res[t][label][h] for t in datasets]
        ev = [x for x in v if x != "not evaluable"]
        verdict = ("REPLICATED" if len(ev) >= 2 and all(x == "PASS" for x in ev)
                   else "NOT EVALUABLE" if len(ev) < 2 else "NOT REPLICATED")
        print(f"  {label:18} R-{h}: {dict(zip(datasets, v))} -> {verdict}")
        res.setdefault("verdict", {})[f"{label} {h}"] = verdict
json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "replicate_r4.json"), "w"), indent=1, default=list)
print("wrote replicate_r4.json")
