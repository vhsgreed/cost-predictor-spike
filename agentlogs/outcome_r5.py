#!/usr/bin/env python3
"""Round 5: do expensive runs fail more? One look. Plan: PLAN.md 272c139 + b48647f.
Estimated tokens = sum over assistant turns of cumulative chars up to that turn / 4."""
import glob, json, os, random
from collections import defaultdict
import pyarrow.parquet as pq
import lab

q = lab.quantile
EXT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "external")


def est(msgs, role_key, text_key, assistant):
    cum, tot, turns = 0, 0, 0
    for m in msgs or []:
        cum += len(m.get(text_key) or "")
        if m.get(role_key) == assistant:
            tot += cum; turns += 1
    return tot / 4, turns


def load(ds):
    rows = []
    if ds == "D1 SWE-smith":
        for f in sorted(glob.glob(f"{EXT}/SWE-bench__SWE-smith-trajectories/**/train-*.parquet", recursive=True)):
            for b in pq.ParquetFile(f).iter_batches(batch_size=256, columns=["messages", "instance_id", "resolved", "model"]):
                for r in b.to_pylist():
                    t, n = est(r["messages"], "role", "content", "assistant")
                    rows.append((r["model"], r["instance_id"], bool(r["resolved"]), t, n))
    elif ds == "D2 SWE-rebench-OH":
        for b in pq.ParquetFile(f"{EXT}/nebius__SWE-rebench-openhands-trajectories/trajectories.parquet").iter_batches(
                batch_size=128, columns=["trajectory", "instance_id", "resolved", "repo"]):
            for r in b.to_pylist():
                t, n = est(r["trajectory"], "role", "content", "assistant")
                rows.append((r["repo"], r["instance_id"], bool(r["resolved"]), t, n))
    else:
        for f in sorted(glob.glob(f"{EXT}/nebius__SWE-agent-trajectories/**/*.parquet", recursive=True)):
            for b in pq.ParquetFile(f).iter_batches(batch_size=256, columns=["trajectory", "instance_id", "target", "model_name"]):
                for r in b.to_pylist():
                    t, n = est(r["trajectory"], "role", "text", "ai")
                    rows.append((r["model_name"], r["instance_id"], bool(r["target"]), t, n))
    return rows


def analyse(ds, rows, idx, label, rng):
    g = defaultdict(list)
    for r in rows:
        g[r[0]].append(r)
    g = {k: v for k, v in g.items() if len(v) >= 200}
    tagged = []  # (instance, part, resolved)
    for k, v in g.items():
        xs = [r[idx] for r in v]; p50, p90 = q(xs, .5), q(xs, .9)
        for r in v:
            part = "tail" if r[idx] > p90 else "body" if r[idx] <= p50 else None
            if part:
                tagged.append((r[1], part, r[2]))
    def rr(items):
        t = [x[2] for x in items if x[1] == "tail"]; b = [x[2] for x in items if x[1] == "body"]
        return (sum(t) / len(t)) / (sum(b) / len(b)) if t and b and sum(b) else float("nan"), t, b
    point, t, b = rr(tagged)
    by_inst = defaultdict(list)
    for x in tagged:
        by_inst[x[0]].append(x)
    insts = sorted(by_inst); boots = []
    for _ in range(1000):
        s = [x for _ in insts for x in by_inst[insts[rng.randrange(len(insts))]]]
        boots.append(rr(s)[0])
    boots.sort(); lo, hi = boots[50], boots[949]
    ok = point <= .8 and hi < .8
    print(f"  {label:9} groups {len(g):3} | tail resolve {sum(t)/len(t):.1%} (n={len(t)}) vs body {sum(b)/len(b):.1%} "
          f"(n={len(b)}) | RR {point:.2f} [{lo:.2f},{hi:.2f}] {'PASS' if ok else 'fail'}", flush=True)
    return {"groups": len(g), "tail_rate": sum(t) / len(t), "body_rate": sum(b) / len(b), "rr": point, "ci": [lo, hi], "pass": ok}, g


def backtest(g, idx):
    out = {}
    for pc in (.9, .95, .99):
        saved = tot = lost = res = 0
        for v in g.values():
            th = q([r[idx] for r in v], pc)
            for r in v:
                tot += r[idx]; res += r[2]
                if r[idx] > th:
                    saved += r[idx] - th; lost += r[2]
        out[f"p{int(pc*100)}"] = {"saved": saved / tot, "resolved_lost": lost / res}
        print(f"    stop at group p{int(pc*100)}: saves {saved/tot:.1%} of est. tokens, loses {lost/res:.1%} of resolved runs")
    return out


def deciles(g, idx):
    acc = defaultdict(lambda: [0, 0])
    for v in g.values():
        cut = [q([r[idx] for r in v], k / 10) for k in range(1, 10)]
        for r in v:
            d = sum(r[idx] > c for c in cut); acc[d][0] += r[2]; acc[d][1] += 1
    s = [round(acc[d][0] / acc[d][1], 3) for d in range(10)]
    print("    resolve rate by decile (cheap -> expensive):", " ".join(f"{x:.0%}" for x in s))
    return s


res = {}
for ds in ("D1 SWE-smith", "D2 SWE-rebench-OH", "D3 SWE-agent"):
    rows = load(ds)
    print(f"\n{ds}: {len(rows)} runs, overall resolve {sum(r[2] for r in rows)/len(rows):.1%}", flush=True)
    rng = random.Random(20261012)
    o, g = analyse(ds, rows, 3, "est.tok", rng)
    o["backtest"] = backtest(g, 3); o["deciles"] = deciles(g, 3)
    t, _ = analyse(ds, rows, 4, "turns", rng)
    res[ds] = {"n": len(rows), "O1_est_tokens": o, "turns": t}

ev = [d for d in res if res[d]["O1_est_tokens"]["groups"] >= 3]
npass = sum(res[d]["O1_est_tokens"]["pass"] for d in ev)
verdict = "PASS" if len(ev) >= 2 and npass >= 2 else "NOT EVALUABLE" if len(ev) < 2 else "FAIL"
print(f"\nO1: passes in {npass}/{len(ev)} evaluable datasets ({ev}) -> {verdict} (need >= 2)")
res["verdict"] = verdict
json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "outcome_r5.json"), "w"), indent=1)
print("wrote outcome_r5.json")
