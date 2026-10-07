#!/usr/bin/env python3
"""Round 6: remaining-cost forecast during the run. One look. Plan: PLAN.md bb5d4dc.
Table forecast: at each assistant turn k, cell = (k bucket, cumulative/fit-median-final bucket);
predict FINAL estimated size from fit cell medians. Baselines: group median, k-only table."""
import glob, json, os, random
from collections import defaultdict
import pyarrow.parquet as pq
import lab

q = lab.quantile
EXT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "external")
KB = (1, 2, 3, 5, 10, 20, 40, 10 ** 9)
XB = (0.1, 0.25, 0.5, 0.75, 1.0, 1.5)
kb = lambda k: sum(k > x for x in KB[:-1])
xb = lambda x: sum(x >= t for t in XB)


def steps(msgs, role, text, assistant):
    cum, out = 0, []
    for m in msgs or []:
        cum += len(m.get(text) or "")
        if m.get(role) == assistant:
            out.append(cum / 4)
    return out


def load(ds):
    rows, seen = [], set()
    if ds == "SWE-smith":
        for f in sorted(glob.glob(f"{EXT}/SWE-bench__SWE-smith-trajectories/**/train-*.parquet", recursive=True)):
            for b in pq.ParquetFile(f).iter_batches(batch_size=256, columns=["messages", "instance_id", "resolved", "model", "traj_id"]):
                for r in b.to_pylist():
                    if r["traj_id"] in seen:
                        continue
                    seen.add(r["traj_id"])
                    s = steps(r["messages"], "role", "content", "assistant")
                    if s:
                        rows.append((r["model"], r["instance_id"], s))
    elif ds == "SWE-rebench-OH":
        for b in pq.ParquetFile(f"{EXT}/nebius__SWE-rebench-openhands-trajectories/trajectories.parquet").iter_batches(
                batch_size=128, columns=["trajectory", "instance_id", "resolved", "repo"]):
            for r in b.to_pylist():
                s = steps(r["trajectory"], "role", "content", "assistant")
                if s:
                    rows.append((r["repo"], r["instance_id"], s))
    else:
        for f in sorted(glob.glob(f"{EXT}/nebius__SWE-agent-trajectories/**/*.parquet", recursive=True)):
            for b in pq.ParquetFile(f).iter_batches(batch_size=256, columns=["trajectory", "instance_id", "target", "model_name"]):
                for r in b.to_pylist():
                    s = steps(r["trajectory"], "role", "text", "ai")
                    if s:
                        rows.append((r["model_name"], r["instance_id"], s))
    return rows


def analyse(ds):
    rows = load(ds)
    inst = sorted({r[1] for r in rows}); random.Random(20261014).shuffle(inst)
    fit_i = set(inst[:int(.7 * len(inst))])
    fit = [r for r in rows if r[1] in fit_i]
    test = [r for r in rows if r[1] not in fit_i]

    gfin = defaultdict(list)
    for g, _, s in fit:
        gfin[g].append(s[-1])
    med = {g: q(v, .5) for g, v in gfin.items()}
    cells, kcells = defaultdict(list), defaultdict(list)
    for g, _, s in fit:
        for k, c in enumerate(s, 1):
            cells[(kb(k), xb(c / med[g]))].append(s[-1])
            kcells[(g, kb(k))].append(s[-1])
    print(f"{ds}: {len(rows)} runs, fit {len(fit)} / test {len(test)}; "
          f"{sum(len(v) >= 50 for v in cells.values())} of {len(cells)} cells have >= 50 fit rows", flush=True)

    tq = lambda v: (q(v, .5), q(v, .1), q(v, .9))
    gm = {g: tq(v) for g, v in gfin.items()}
    PRED = {("x", k1, k2): tq(v) for (k1, k2), v in cells.items() if len(v) >= 50}
    PRED.update({("k", g, k1): tq(v) for (g, k1), v in kcells.items() if len(v) >= 50})
    KMED = {(g, k1): q(v, .5) for (g, k1), v in kcells.items() if len(v) >= 50}

    def predict(g, k, c):
        x = c / med[g]
        return PRED.get(("x", kb(k), xb(x))) or PRED.get(("k", g, kb(k))) or gm[g]

    per = defaultdict(lambda: [0.0, 0.0, 0.0, 0, 0, 0.0, 0.0])  # e_ours,e_a,e_b,hit,n,ape,best_ape
    kbuck = defaultdict(lambda: [0.0, 0.0, 0])  # ours err, best-baseline err, n (k>=3)
    start = [0.0, 0, 0]  # error, covered, n
    for g, inst_id, s in test:
        fin = s[-1]; ga = med[g]
        acc = per[inst_id]
        for k, c in enumerate(s, 1):
            if k == 1:
                _, lo, hi = predict(g, 1, c)
                start[0] += abs(ga - fin); start[1] += lo <= fin <= hi; start[2] += 1
            if k < 3:
                continue
            p, lo, hi = predict(g, k, c)
            e_o = abs(p - fin)
            e_a = abs(ga - fin)
            pm = KMED.get((g, kb(k)))
            e_b = abs(pm - fin) if pm is not None else e_a
            acc[0] += e_o; acc[1] += e_a; acc[2] += e_b
            acc[3] += lo <= fin <= hi; acc[4] += 1; acc[5] += e_o / fin; acc[6] += min(e_o, e_a, e_b) / fin
            b = kbuck[min(k, 21)]
            b[0] += e_o; b[1] += min(e_a, e_b); b[2] += 1

    def pooled(idx):
        return sum(v[idx] for v in per.values()), sum(v[4] for v in per.values())
    eo, n = pooled(0); ea, _ = pooled(1); eb, _ = pooled(2)
    ref = 1 if ea <= eb else 2
    base = "group median" if ref == 1 else "k-only table"; eb = ea if ref == 1 else eb
    rng = random.Random(20261015); ratios, covs = [], []
    ids = list(per)
    for _ in range(200):
        s_o = s_b = s_h = s_n = 0
        for _ in ids:
            v = per[ids[rng.randrange(len(ids))]]
            s_o += v[0]; s_b += v[ref]; s_h += v[3]; s_n += v[4]
        ratios.append(s_o / s_b); covs.append(s_h / s_n)
    ratios.sort(); covs.sort()
    point = eo / eb; cov = sum(v[3] for v in per.values()) / n
    ok = point <= .8 and ratios[189] < .8 and covs[10] <= .8 <= covs[189]
    print(f"  update points k>=3: {n:,} | MAE ours {eo/n:,.0f} tok vs best baseline ({base}) {eb/n:,.0f} "
          f"| ratio {point:.3f} [{ratios[10]:.3f},{ratios[189]:.3f}] | coverage {cov:.1%} "
          f"[{covs[10]:.1%},{covs[189]:.1%}] | {'PASS' if ok else 'fail'}", flush=True)
    print(f"  task start (k=1): MAE {start[0]/start[2]:,.0f} tok | 80% interval covers {start[1]/start[2]:.1%}")
    for k in sorted(kbuck):
        b = kbuck[k]
        print(f"    turns {'21+' if k == 21 else k:4}: n={b[2]:7,} | MAE ours {b[0]/b[2]:8,.0f} vs baseline {b[1]/b[2]:8,.0f} "
              f"| ratio {(b[0]/b[1]) if b[1] else float('nan'):.2f}")
    return {"runs": len(rows), "test": len(test), "update_points": n, "mae_ours": eo / n,
            "mae_baseline": eb / n, "baseline": base, "ratio": point,
            "ratio_ci": [ratios[10], ratios[189]], "coverage": cov,
            "coverage_ci": [covs[10], covs[189]], "pass": bool(ok)}


res = {}
for ds in ("SWE-smith", "SWE-rebench-OH", "SWE-agent"):
    res[ds] = analyse(ds)
np = sum(res[d]["pass"] for d in res)
verdict = "PASS" if np >= 2 else "FAIL"
print(f"\nR6: passes in {np}/3 datasets -> {verdict} (need >= 2)")
res["verdict"] = verdict
json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "forecast_r6.json"), "w"), indent=1)
print("wrote forecast_r6.json")
