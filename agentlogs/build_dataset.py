#!/usr/bin/env python3
"""Build one row per session from log shards. Numbers and labels only.
Usage: build_dataset.py OUT.jsonl SHARD [SHARD ...]   (shard = 5-digit number)

Per session: completeness (P1), calls, tokens, dollar cost priced per call by the
call's own model (P6/P8), cached share, loop onset and tokens/cost after it (P10),
state (P7), repository and created_at (P2), session-model, trigger event type,
prompt length in chars (number only).
"""
import json, sys
from collections import Counter, defaultdict
import pyarrow.parquet as pq
import pyarrow.compute as pc
import lab

out_path, shards = sys.argv[1], sys.argv[2:]

# session + task metadata (both session shards)
meta = {}
for f in ("part_00000_of_00002.parquet", "sessions1.parquet"):
    S = pq.read_table(f, columns=["id", "task", "model", "state", "prompt", "event", "created_at"])
    for sid, tid, m, stt, p, ev, ca in zip(
            S["id"].to_pylist(), pc.struct_field(S["task"], "id").to_pylist(), S["model"].to_pylist(),
            S["state"].to_pylist(), S["prompt"].to_pylist(), pc.struct_field(S["event"], "type").to_pylist(),
            S["created_at"].to_pylist()):
        meta[sid] = {"task": tid, "session_model": lab.canon_model(m), "state": stt,
                     "prompt_chars": len(p or ""), "event": ev or "none", "created_at": ca.isoformat() if ca else None}
T = pq.read_table("tasks.parquet", columns=["id", "repository"])
repo_of = dict(zip(T["id"].to_pylist(), pc.struct_field(T["repository"], "full_name").to_pylist()))
del T

n_written = 0
with open(out_path, "w") as fh:
    for sh in shards:
        t = pq.read_table(f"part_{sh}_of_00276.parquet", columns=["session", "entry_index", "data"])
        sess = pc.struct_field(t["session"], "id").to_pylist(); idx = t["entry_index"].to_pylist()
        d = t["data"]; u = pc.struct_field(d, "usage")
        pt = pc.struct_field(u, "prompt_tokens").to_pylist(); ct = pc.struct_field(u, "completion_tokens").to_pylist()
        cached = pc.struct_field(pc.struct_field(u, "prompt_tokens_details"), "cached_tokens").to_pylist()
        cmodel = pc.struct_field(d, "model").to_pylist(); choices = pc.struct_field(d, "choices").to_pylist()
        del t, d, u
        entries = defaultdict(list); usage = defaultdict(list); toolchunks = defaultdict(list)
        for s, i, p, c, ca, m, ch in zip(sess, idx, pt, ct, cached, cmodel, choices):
            entries[s].append(i)
            if p:
                usage[s].append((i, p, ca or 0, c or 0, lab.canon_model(m)))
            tcs = []
            for x in ch or []:
                tcs += ((x.get("message") or {}).get("tool_calls") or []) + ((x.get("delta") or {}).get("tool_calls") or [])
            if tcs:
                toolchunks[s].append((i, tcs))
        del sess, idx, pt, ct, cached, cmodel, choices
        for s, us in usage.items():
            us.sort(); md = meta.get(s, {})
            fallback = md.get("session_model") or Counter(m for *_, m in us if m).most_common(1)[0][0] if any(m for *_, m in us) else md.get("session_model")
            costs = [lab.call_cost(m or fallback, p, ca, c) for _, p, ca, c, m in us]
            priced = all(x is not None for x in costs)
            tools = lab.reassemble(sorted(toolchunks.get(s, []), key=lambda e: e[0]))
            onset = lab.loop_onset(tools); first = min(onset.values()) if onset else None
            row = {
                "session": s, "shard": sh, "complete": lab.complete(entries[s]),
                "calls": len(us), "tool_calls": len(tools),
                "tokens": sum(p + c for _, p, _, c, _ in us),
                "cached_share": sum(ca for _, _, ca, _, _ in us) / max(1, sum(p for _, p, _, _, _ in us)),
                "cost": sum(costs) if priced else None,
                "model": fallback or "unknown",
                "model_source": "session" if md.get("session_model") else ("log" if fallback else "none"),
                "loop_R": "R" in onset, "loop_S": "S" in onset,
                "tokens_after_onset": sum(p + c for i, p, _, c, _ in us if first is not None and i >= first),
                "cost_after_onset": (sum(x for (i, *_), x in zip(us, costs) if first is not None and i >= first) if priced else None),
                "state": md.get("state"), "event": md.get("event"), "prompt_chars": md.get("prompt_chars"),
                "created_at": md.get("created_at"), "repo": repo_of.get(md.get("task")),
            }
            fh.write(json.dumps(row) + "\n"); n_written += 1
        print(f"shard {sh}: {len(usage)} sessions", flush=True)
print(f"wrote {n_written} rows -> {out_path}")
