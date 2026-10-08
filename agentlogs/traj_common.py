"""Shared loader for round 6 post-hoc and round 7. steps/load copied verbatim from forecast_r6.py
(that script runs on import). Rows are cached to traj_cache_<ds>.pkl (gitignored) to avoid
re-parsing 7 GB twice; delete the cache to re-parse."""
import glob, os, pickle, random
from collections import defaultdict
import pyarrow.parquet as pq
import lab

q = lab.quantile
HERE = os.path.dirname(os.path.abspath(__file__))
EXT = os.path.join(HERE, "..", "external")
KB = (1, 2, 3, 5, 10, 20, 40, 10 ** 9)
XB = (0.1, 0.25, 0.5, 0.75, 1.0, 1.5)
kb = lambda k: sum(k > x for x in KB[:-1])
xb = lambda x: sum(x >= t for t in XB)
tq = lambda v: (q(v, .5), q(v, .1), q(v, .9))
DATASETS = ("SWE-smith", "SWE-rebench-OH", "SWE-agent")


def steps(msgs, role, text, assistant):
    cum, out = 0, []
    for m in msgs or []:
        cum += len(m.get(text) or "")
        if m.get(role) == assistant:
            out.append(cum / 4)
    return out


def _load(ds):
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


def load(ds):
    p = os.path.join(HERE, f"traj_cache_{ds}.pkl")
    if os.path.exists(p):
        return pickle.load(open(p, "rb"))
    rows = _load(ds)
    pickle.dump(rows, open(p, "wb"))
    return rows


def split(rows):
    """Identical to forecast_r6.analyse: 70/30 by instance_id, seed 20261014, skip test groups < 50 fit."""
    inst = sorted({r[1] for r in rows}); random.Random(20261014).shuffle(inst)
    fit_i = set(inst[:int(.7 * len(inst))])
    fit = [r for r in rows if r[1] in fit_i]
    test = [r for r in rows if r[1] not in fit_i]
    gfin = defaultdict(list)
    for g, _, s in fit:
        gfin[g].append(s[-1])
    med = {g: q(v, .5) for g, v in gfin.items()}
    gfin = dict(gfin)  # no default: a lookup must not create empty groups
    skipped = sum(1 for g, _, _ in test if len(gfin.get(g, ())) < 50)
    test = [(g, i, s) for g, i, s in test if len(gfin.get(g, ())) >= 50]
    return fit, test, gfin, med, skipped
