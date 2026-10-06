#!/usr/bin/env python3
"""Full 56 GB run, phase 1: download all 276 step-log shards, parse each to
full_rows/{shard}.jsonl (schema identical to build_dataset.py rows), delete the parquet
after parsing. Resumable: a shard is done when its row file exists; safe to re-run
(no duplicated rows). Phase 2/3 read all files in full_rows/.

Dataset: risenlab/agentlogs on HuggingFace, sha 04013a44d3432c6654bca1dcd4a01218a9406b80
(run pre-registered in PLAN.md 2026-10-06 before data).
"""
import json, os, queue, sys, threading, time, urllib.request
from collections import Counter, defaultdict
import pyarrow.parquet as pq
import pyarrow.compute as pc
import lab

ROOT = os.path.dirname(os.path.abspath(__file__))
ROWDIR = os.path.join(ROOT, "full_rows")
os.makedirs(ROWDIR, exist_ok=True)
BASE = "https://huggingface.co/datasets/risenlab/agentlogs/resolve/main/agent_session_logs"
SHARDS = [f"{i:05d}" for i in range(276)]

# session + task metadata, loaded once (build_dataset.py reloaded per call; same fields)
t0 = time.time()
meta = {}
for f in ("part_00000_of_00002.parquet", "sessions1.parquet"):
    S = pq.read_table(os.path.join(ROOT, f), columns=["id", "task", "model", "state", "prompt", "event", "created_at"])
    for sid, tid, m, stt, p, ev, ca in zip(
            S["id"].to_pylist(), pc.struct_field(S["task"], "id").to_pylist(), S["model"].to_pylist(),
            S["state"].to_pylist(), S["prompt"].to_pylist(), pc.struct_field(S["event"], "type").to_pylist(),
            S["created_at"].to_pylist()):
        meta[sid] = {"task": tid, "session_model": lab.canon_model(m), "state": stt,
                     "prompt_chars": len(p or ""), "event": ev or "none", "created_at": ca.isoformat() if ca else None}
T = pq.read_table(os.path.join(ROOT, "tasks.parquet"), columns=["id", "repository"])
repo_of = dict(zip(T["id"].to_pylist(), pc.struct_field(T["repository"], "full_name").to_pylist()))
del T
print(f"meta: {len(meta)} sessions loaded in {time.time()-t0:.0f}s", flush=True)


def download(sh, path):
    """Fetch one shard to `path`, 3 attempts, atomic rename."""
    url = f"{BASE}/part_{sh}_of_00276.parquet"
    for att in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "agentlogs-research/1.0"})
            with urllib.request.urlopen(req, timeout=600) as r, open(path + ".part", "wb") as fh:
                while chunk := r.read(1 << 20):
                    fh.write(chunk)
            os.rename(path + ".part", path)
            return
        except Exception as e:
            if att == 2:
                raise
            print(f"  download {sh} attempt {att+1} failed: {e}", flush=True)
            time.sleep(10 * (att + 1))


def parse_shard(sh, path):
    """Same parsing as build_dataset.py, one row file per shard (atomic rename)."""
    out = os.path.join(ROWDIR, f"{sh}.jsonl")
    tmp = out + ".tmp"
    n = 0
    with open(tmp, "w") as fh:
        t = pq.read_table(path, columns=["session", "entry_index", "data"])
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
            fh.write(json.dumps(row) + "\n"); n += 1
    os.rename(tmp, out)
    return n


def main():
    done = {f[:-6] for f in os.listdir(ROWDIR) if f.endswith(".jsonl")}
    shards = sys.argv[1:] or SHARDS
    todo = [s for s in shards if s not in done]
    print(f"{len(done)} shards already done, {len(todo)} to go", flush=True)
    if not todo:
        print("nothing to do"); return

    q = queue.Queue(maxsize=2)

    def producer():
        for sh in todo:
            path = os.path.join(ROOT, f"part_{sh}_of_00276.parquet")
            if not os.path.exists(path):
                download(sh, path)
            q.put((sh, path))
        q.put(None)

    threading.Thread(target=producer, daemon=True).start()
    t_start = time.time(); finished = 0
    while True:
        item = q.get()
        if item is None:
            break
        sh, path = item
        n = parse_shard(sh, path)
        os.remove(path)
        finished += 1
        el = time.time() - t_start
        eta = el / finished * (len(todo) - finished)
        print(f"[{time.strftime('%H:%M:%S')}] {sh}: {n} rows | {finished}/{len(todo)} shards | "
              f"{el/60:.0f} min elapsed, ETA {eta/60:.0f} min", flush=True)
    print(f"DONE: {finished} shards in {(time.time()-t_start)/60:.0f} min -> {ROWDIR}", flush=True)


if __name__ == "__main__":
    main()
