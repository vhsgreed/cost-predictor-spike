#!/usr/bin/env python3
"""Loop canary: how much of the cost tail is loops? Shard 0.

Tool-call arguments are reassembled from streamed deltas and immediately hashed;
no argument or message text is printed or stored.

Loop signals (defined before running):
  R  repeat:      same (tool, args-hash) issued >= 3 times in a session
  S  no-progress: >= 25 consecutive tool calls with no file edit, after the first edit
Onset = the call where a signal first fires. "Tokens after onset" = model-call tokens
from that point on: the most a perfect live detector could have saved.
Pass bar: signals markedly more common in the top 10% by cost than in the rest
(>= 2x), and >= 20% of top-decile tokens fall after onset.
"""
import hashlib, statistics as st
from collections import defaultdict, Counter
import pyarrow.parquet as pq
import pyarrow.compute as pc

EDIT = {"edit", "str_replace_editor", "create", "apply_patch", "str_replace", "write", "write_file"}

t = pq.read_table("part_00000_of_00276.parquet", columns=["session", "entry_index", "data"])
sess = pc.struct_field(t["session"], "id").to_pylist(); idx = t["entry_index"].to_pylist()
d = t["data"]
u = pc.struct_field(d, "usage")
pt = pc.struct_field(u, "prompt_tokens").to_pylist(); ct = pc.struct_field(u, "completion_tokens").to_pylist()
choices = pc.struct_field(d, "choices").to_pylist()
del t, d, u

ev = defaultdict(list)                 # session -> [(entry_index, kind, payload)]
frag = defaultdict(lambda: ["", "", 0])  # (session, call id) -> [name, args, first idx]
last_id = {}
for s, i, p, c, ch in zip(sess, idx, pt, ct, choices):
    if p:
        ev[s].append((i, "usage", p + (c or 0)))
    for x in ch or []:
        tcs = ((x.get("message") or {}).get("tool_calls") or []) + ((x.get("delta") or {}).get("tool_calls") or [])
        for tc in tcs:
            cid = tc.get("id") or last_id.get((s, tc.get("index")))
            if not cid:
                continue
            last_id[(s, tc.get("index"))] = cid
            f = frag[(s, cid)]
            if not f[2]: f[2] = i
            f[0] = f[0] or tc.get("function_name") or tc.get("custom_name") or ""
            f[1] += tc.get("function_arguments") or tc.get("custom_input") or ""
for (s, cid), (name, args, i0) in frag.items():
    h = hashlib.sha1(f"{name}\0{args}".encode()).hexdigest()[:12]
    ev[s].append((i0, "tool", (name, h)))
del frag, choices

rows = []
for s, es in ev.items():
    es.sort(key=lambda e: e[0])
    usage = [(i, v) for i, k, v in es if k == "usage"]
    if len(usage) <= 3:
        continue
    total = sum(v for _, v in usage)
    seen = Counter(); onset = {}; streak = 0; edited = False; ntool = 0
    for i, k, v in es:
        if k != "tool": continue
        ntool += 1
        name, h = v
        seen[h] += 1
        if seen[h] == 3 and "R" not in onset: onset["R"] = i
        if name in EDIT: edited, streak = True, 0
        elif edited:
            streak += 1
            if streak == 25 and "S" not in onset: onset["S"] = i
    first = min(onset.values()) if onset else None
    after = sum(v for i, v in usage if first is not None and i >= first)
    rows.append({"total": total, "R": "R" in onset, "S": "S" in onset, "any": bool(onset),
                 "after": after, "ntool": ntool, "edited": edited})

rows.sort(key=lambda r: -r["total"])
top, rest = rows[: len(rows) // 10], rows[len(rows) // 10:]
print(f"sessions {len(rows)}; with tool calls parsed {sum(r['ntool'] > 0 for r in rows)}; "
      f"median tool calls {st.median(r['ntool'] for r in rows):.0f}")
for sig in ("R", "S", "any"):
    a = sum(r[sig] for r in top) / len(top); b = sum(r[sig] for r in rest) / len(rest)
    print(f"  signal {sig:3}: top 10% {a:5.0%} | rest {b:5.0%} | ratio {a / max(b, 1e-9):.1f}x")
share_top = sum(r["after"] for r in top) / sum(r["total"] for r in top)
share_all = sum(r["after"] for r in rows) / sum(r["total"] for r in rows)
print(f"  tokens after loop onset: top 10% {share_top:.0%} | all sessions {share_all:.0%}")
looped = [r for r in rows if r["any"]]; clean = [r for r in rows if not r["any"]]
print(f"  median total tokens: looped {st.median(r['total'] for r in looped):,.0f} (n={len(looped)}) | "
      f"clean {st.median(r['total'] for r in clean):,.0f} (n={len(clean)})")
ratio = (sum(r['any'] for r in top) / len(top)) / max(1e-9, sum(r['any'] for r in rest) / len(rest))
print("VERDICT:", "PASS" if ratio >= 2 and share_top >= 0.20 else "fail",
      f"(ratio {ratio:.1f}x, top-decile tokens after onset {share_top:.0%})")
