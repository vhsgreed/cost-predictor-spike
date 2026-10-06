#!/usr/bin/env python3
"""Blind hand-check of the loop detector (PLAN.md H2). Dev shards only.

Samples 30 flagged + 30 unflagged sessions (unflagged matched to >= 25 tool calls so
they aren't trivially short), shuffles them under random case IDs, and renders the
tool-call sequence only: tool name + a letter per distinct argument hash (same letter
= identical call repeated). No arguments, prompts, messages, models or costs shown.
Writes handcheck.html (for the reviewer) and handcheck_key.json (detector flags; open
only after answers are in).
"""
import html, json, random, string
from collections import defaultdict
import pyarrow.parquet as pq
import pyarrow.compute as pc
import lab

rng = random.Random(20261006)
R = [json.loads(l) for l in open("dev.jsonl")]
flag = lambda r: r["loop_R"] or r["loop_S"]
fl = [r for r in R if flag(r)]
cl = [r for r in R if not flag(r) and r["tool_calls"] >= 25]
pick = rng.sample(fl, 30) + rng.sample(cl, 30)
rng.shuffle(pick)
want = {r["session"]: r for r in pick}

tools = {}
for sh in sorted({r["shard"] for r in pick}):
    t = pq.read_table(f"part_{sh}_of_00276.parquet", columns=["session", "entry_index", "data"])
    sess = pc.struct_field(t["session"], "id").to_pylist(); idx = t["entry_index"].to_pylist()
    ch = pc.struct_field(t["data"], "choices").to_pylist(); del t
    chunks = defaultdict(list)
    for s, i, c in zip(sess, idx, ch):
        if s not in want: continue
        tcs = []
        for x in c or []:
            tcs += ((x.get("message") or {}).get("tool_calls") or []) + ((x.get("delta") or {}).get("tool_calls") or [])
        if tcs: chunks[s].append((i, tcs))
    for s, cs in chunks.items():
        tools[s] = lab.reassemble(sorted(cs, key=lambda e: e[0]))

def letters():
    for n in range(1, 4):
        for combo in __import__("itertools").product(string.ascii_uppercase, repeat=n):
            yield "".join(combo)

cases, key = [], {}
for n, r in enumerate(pick, 1):
    cid = f"C{n:02d}"
    seq = tools.get(r["session"], [])
    lab_of, gen, count = {}, letters(), defaultdict(int)
    cells = []
    for _, name, h in seq:
        if h not in lab_of: lab_of[h] = next(gen)
        count[h] += 1
        rep = count[h] > 1
        edit = name in lab.EDIT_TOOLS
        cls = "edit" if edit else ("rep" if rep else "")
        cells.append(f'<span class="{cls}">{html.escape(name)}·{lab_of[h]}</span>')
    rows = "".join(f'<div class="row"><b>{i:>4}</b> ' + " ".join(cells[i:i + 12]) + "</div>" for i in range(0, len(cells), 12))
    cases.append(f'''<section><h3>{cid} <small>{len(seq)} tool calls</small></h3>{rows}
<p>Stuck in a loop?
<label><input type="radio" name="{cid}" value="yes">yes</label>
<label><input type="radio" name="{cid}" value="no">no</label>
<label><input type="radio" name="{cid}" value="unsure">unsure</label></p></section>''')
    key[cid] = {"flagged": flag(r), "R": r["loop_R"], "S": r["loop_S"], "session": r["session"]}

page = f'''<!doctype html><meta charset="utf-8"><title>Loop hand-check (blind)</title>
<style>body{{font:14px system-ui;max-width:1100px;margin:2em auto;padding:0 1em}}
.row{{font:12px ui-monospace,monospace;margin:2px 0;white-space:nowrap;overflow-x:auto}}
.edit{{background:#c8f7c5}} .rep{{background:#ffd6d6}} section{{border-top:1px solid #ccc;padding:.5em 0}}
#out{{width:100%;height:6em}}</style>
<h1>Loop hand-check (blind, 60 sessions)</h1>
<p><b>Question:</b> is the agent stuck, repeating actions without making progress? Judge by the pattern only.</p>
<p><b>Legend:</b> each cell is <code>tool·LETTER</code>. Same letter = exactly the same call (same arguments) again.
<span class="rep">red</span> = a repeat of an earlier identical call. <span class="edit">green</span> = a file edit.
Numbers on the left are the call position. Long stretches of red, or long stretches with no green after work has
started, suggest a loop. Normal exploration (many different reads early on) is not a loop.</p>
{"".join(cases)}
<p><button onclick="go()">Export answers</button></p><textarea id="out"></textarea>
<script>function go(){{const a={{}};document.querySelectorAll('input:checked').forEach(e=>a[e.name]=e.value);
const t=JSON.stringify(a);document.getElementById('out').value=t;navigator.clipboard&&navigator.clipboard.writeText(t);}}</script>'''
open("handcheck.html", "w").write(page)
json.dump(key, open("handcheck_key.json", "w"), indent=1)
print(f"cases {len(cases)}; sessions with no parsed tools: {sum(1 for r in pick if r['session'] not in tools)}")
