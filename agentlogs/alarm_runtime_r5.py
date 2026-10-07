#!/usr/bin/env python3
"""Runtime alarm on SWE trajectories (post-hoc, exploratory; added for article v0.2).
Threshold per group = p90 of FINAL estimated-token size on fit instances (70%, split by
instance_id, random.Random(20261013)); evaluated on held-out instances (30%). The alarm fires at
the first assistant turn where the cumulative estimated size crosses the threshold.
Reports, per dataset: fire rate; fire rate among resolved runs (false alarms if one would stop);
resolve rate of fired vs not-fired runs; lead (share of the run's turns and tokens after firing);
estimated tokens per resolved run with and without stopping at the alarm."""
import glob, json, os, random
from collections import defaultdict
import pyarrow.parquet as pq
import lab

q = lab.quantile
EXT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "external")


def steps(msgs, role, text, assistant):
    cum, tot, out = 0, 0, []
    for m in msgs or []:
        cum += len(m.get(text) or "")
        if m.get(role) == assistant:
            tot += cum; out.append(tot / 4)
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
                    rows.append((r["model"], r["instance_id"], bool(r["resolved"]), steps(r["messages"], "role", "content", "assistant")))
    elif ds == "SWE-rebench-OH":
        for b in pq.ParquetFile(f"{EXT}/nebius__SWE-rebench-openhands-trajectories/trajectories.parquet").iter_batches(
                batch_size=128, columns=["trajectory", "instance_id", "resolved", "repo"]):
            for r in b.to_pylist():
                rows.append((r["repo"], r["instance_id"], bool(r["resolved"]), steps(r["trajectory"], "role", "content", "assistant")))
    else:
        for f in sorted(glob.glob(f"{EXT}/nebius__SWE-agent-trajectories/**/*.parquet", recursive=True)):
            for b in pq.ParquetFile(f).iter_batches(batch_size=256, columns=["trajectory", "instance_id", "target", "model_name"]):
                for r in b.to_pylist():
                    rows.append((r["model_name"], r["instance_id"], bool(r["target"]), steps(r["trajectory"], "role", "text", "ai")))
    return [r for r in rows if r[3]]


res = {}
for ds in ("SWE-smith", "SWE-rebench-OH", "SWE-agent"):
    rows = load(ds)
    inst = sorted({r[1] for r in rows}); random.Random(20261013).shuffle(inst)
    fit_i = set(inst[:int(.7 * len(inst))])
    fitg = defaultdict(list)
    for r in rows:
        if r[1] in fit_i:
            fitg[r[0]].append(r[3][-1])
    th = {g: q(v, .9) for g, v in fitg.items() if len(v) >= 140}
    test = [r for r in rows if r[1] not in fit_i and r[0] in th]
    fired, lead_t, lead_k = [], [], []
    tok_all = tok_stop = 0.0; res_all = res_stop = 0
    for g, _, ok, cum in test:
        t = th[g]; k = next((i for i, c in enumerate(cum) if c > t), None)
        tok_all += cum[-1]; res_all += ok
        if k is None:
            fired.append((False, ok)); tok_stop += cum[-1]; res_stop += ok
        else:
            fired.append((True, ok)); tok_stop += cum[k]
            lead_k.append(1 - (k + 1) / len(cum)); lead_t.append(1 - cum[k] / cum[-1])
    F = [ok for f, ok in fired if f]; N = [ok for f, ok in fired if not f]
    out = {
        "test_runs": len(test), "groups": len(th),
        "fire_rate": len(F) / len(test),
        "fire_rate_among_resolved": sum(F) / max(1, res_all),
        "resolve_fired": sum(F) / max(1, len(F)), "resolve_not_fired": sum(N) / max(1, len(N)),
        "median_turns_remaining_after_fire": q(lead_k, .5) if lead_k else None,
        "median_tokens_remaining_after_fire": q(lead_t, .5) if lead_t else None,
        "est_tokens_per_resolved_no_stop": tok_all / max(1, res_all),
        "est_tokens_per_resolved_stop_at_alarm": tok_stop / max(1, res_stop),
    }
    res[ds] = out
    print(f"\n{ds}: {len(test)} held-out runs, {len(th)} groups")
    print(f"  alarm fires on {out['fire_rate']:.1%} of runs; on {out['fire_rate_among_resolved']:.1%} of resolved runs (false alarms if stopping)")
    print(f"  resolve rate: fired {out['resolve_fired']:.1%} vs not fired {out['resolve_not_fired']:.1%}")
    print(f"  when it fires, median {out['median_turns_remaining_after_fire']:.0%} of turns and "
          f"{out['median_tokens_remaining_after_fire']:.0%} of est. tokens are still ahead")
    print(f"  est. tokens per resolved run: {out['est_tokens_per_resolved_no_stop']:,.0f} without stopping, "
          f"{out['est_tokens_per_resolved_stop_at_alarm']:,.0f} stopping at the alarm "
          f"({out['est_tokens_per_resolved_stop_at_alarm']/out['est_tokens_per_resolved_no_stop']-1:+.1%})")
json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "alarm_runtime_r5.json"), "w"), indent=1)
print("\nwrote alarm_runtime_r5.json")
