"""Shared, tested core for AgentLogs canaries. Numbers only: no prompt or message text
is returned; tool arguments are hashed on read.

Fixes from the 2026-10-06 test review:
  P1  session completeness verified per session (entry 0 present, contiguous indices)
  P2  repo-grouped split (no repository on both sides) + time split
  P4  bootstrap confidence intervals
  P5  two-sided p10-p90 range, per-group coverage, pinball loss
  P6  dollar cost (cached / uncached input / output priced separately)
  P7  outcome state kept per session
  P8  model taken from per-call log field when the session field is empty
  P10 tool-call reassembly isolated in a pure function with unit tests
"""
from __future__ import annotations
import hashlib, math, random, re, statistics as st
from collections import Counter, defaultdict

# ---- P6 pricing, USD per token: (uncached input, cached input, output).
# Snapshot of OpenRouter list prices 2026-10-06. Missing models -> None (excluded from $).
PRICES = {
    "claude-sonnet-4": (3e-6, 3e-7, 15e-6), "claude-sonnet-4.5": (3e-6, 3e-7, 15e-6),
    "claude-sonnet-4.6": (3e-6, 3e-7, 15e-6), "claude-opus-4.5": (5e-6, 5e-7, 25e-6),
    "claude-opus-4.6": (5e-6, 5e-7, 25e-6), "claude-haiku-4.5": (1e-6, 1e-7, 5e-6),
    "gpt-5.3-codex": (1.75e-6, 1.75e-7, 14e-6), "gpt-5.2-codex": (1.75e-6, 1.75e-7, 14e-6),
    "gpt-5.1-codex-max": (1.25e-6, 1.25e-7, 10e-6),
}


def canon_model(raw: str | None) -> str | None:
    """'sweagent-capi:claude-sonnet-4.6', 'Claude Sonnet 4.5 (Preview)' -> 'claude-sonnet-4.6' / '-4.5'."""
    if not raw:
        return None
    s = raw.split(":")[-1].lower().strip()
    s = re.sub(r"\(.*?\)", "", s).strip().replace(" ", "-")
    return s or None


def call_cost(model: str | None, prompt: int, cached: int, completion: int) -> float | None:
    p = PRICES.get(model or "")
    if not p:
        return None
    cached = min(cached, prompt)
    return (prompt - cached) * p[0] + cached * p[1] + completion * p[2]


# ---- P1 completeness
def complete(indices: list[int]) -> bool:
    s = sorted(indices)
    return bool(s) and s[0] == 0 and s == list(range(s[-1] + 1))


# ---- P10 tool-call reassembly (pure, unit-tested)
EDIT_TOOLS = {"edit", "str_replace_editor", "create", "apply_patch", "str_replace", "write", "write_file"}


def reassemble(chunks):
    """chunks: iterable of (entry_index, [tool_call dicts]) in log order. Streamed deltas
    carry the id only on the first piece; later pieces carry `index`. Returns
    [(first_entry_index, name, args_hash)] sorted by first index."""
    calls: dict[str, list] = {}
    by_index: dict = {}
    for ei, tcs in chunks:
        for tc in tcs or []:
            cid = tc.get("id")
            if cid:
                by_index[tc.get("index")] = cid
            else:
                cid = by_index.get(tc.get("index"))
            if not cid:
                continue
            c = calls.setdefault(cid, [ei, "", ""])
            c[1] = c[1] or tc.get("function_name") or tc.get("custom_name") or ""
            c[2] += tc.get("function_arguments") or tc.get("custom_input") or ""
    out = [(ei, name, hashlib.sha1(f"{name}\0{args}".encode()).hexdigest()[:12]) for ei, name, args in calls.values()]
    return sorted(out)


def loop_onset(tools, repeat_n=3, idle_n=25):
    """tools: [(entry_index, name, hash)]. Returns {signal: entry_index of onset}."""
    seen = Counter(); onset = {}; streak = 0; edited = False
    for ei, name, h in tools:
        seen[h] += 1
        if seen[h] == repeat_n and "R" not in onset:
            onset["R"] = ei
        if name in EDIT_TOOLS:
            edited, streak = True, 0
        elif edited:
            streak += 1
            if streak == idle_n and "S" not in onset:
                onset["S"] = ei
    return onset


# ---- P2 splits
def split_by_group(rows, key, fracs=(0.5, 0.2, 0.3), seed=0):
    groups = sorted({key(r) for r in rows}); random.Random(seed).shuffle(groups)
    n = len(groups); a = int(n * fracs[0]); b = a + int(n * fracs[1])
    ga, gb = set(groups[:a]), set(groups[a:b])
    return ([r for r in rows if key(r) in ga], [r for r in rows if key(r) in gb],
            [r for r in rows if key(r) not in ga and key(r) not in gb])


def split_by_time(rows, key, fracs=(0.5, 0.2, 0.3)):
    rows = sorted(rows, key=key); n = len(rows); a = int(n * fracs[0]); b = a + int(n * fracs[1])
    return rows[:a], rows[a:b], rows[b:]


# ---- regression + P5 scoring
def lstsq(X, y, lam=1e-3):
    k = len(X[0]); A = [[sum(r[a] * r[b] for r in X) + (lam if a == b else 0) for b in range(k)] for a in range(k)]
    v = [sum(r[a] * yy for r, yy in zip(X, y)) for a in range(k)]
    for col in range(k):
        q = max(range(col, k), key=lambda r: abs(A[r][col])); A[col], A[q], v[col], v[q] = A[q], A[col], v[q], v[col]
        for r in range(k):
            if r != col:
                f = A[r][col] / A[col][col]; A[r] = [x - f * yv for x, yv in zip(A[r], A[col])]; v[r] -= f * v[col]
    return [v[i] / A[i][i] for i in range(k)]


def quantile(xs, q):
    s = sorted(xs); return s[min(len(s) - 1, max(0, math.ceil(q * (len(s) + 1)) - 1))]


def pinball(y, qhat, tau):
    d = y - qhat; return max(tau * d, (tau - 1) * d)


def fit_interval(fit, cal, feat, target, lo=0.1, hi=0.9):
    """Log-linear point model + split-conformal two-sided interval. Returns predict(r) -> (lo, hi)."""
    w = lstsq([feat(r) for r in fit], [math.log(target(r)) for r in fit])
    mu = lambda r: sum(a * b for a, b in zip(w, feat(r)))
    res = [math.log(target(r)) - mu(r) for r in cal]
    ql, qh = quantile(res, lo), quantile(res, hi)
    return lambda r: (math.exp(mu(r) + ql), math.exp(mu(r) + qh))


def evaluate(test, predict, target, group=None, boot=500, seed=0):
    """Coverage, median ratio hi/lo, pinball (log space), with bootstrap 90% CIs."""
    ys = [target(r) for r in test]; iv = [predict(r) for r in test]
    inside = [lo <= y <= hi for y, (lo, hi) in zip(ys, iv)]
    ratio = [hi / lo for lo, hi in iv]
    pin = [pinball(math.log(y), math.log(lo), .1) + pinball(math.log(y), math.log(hi), .9) for y, (lo, hi) in zip(ys, iv)]
    rng = random.Random(seed); n = len(ys)

    def ci(vals, f):
        bs = sorted(f([vals[rng.randrange(n)] for _ in range(n)]) for _ in range(boot))
        return f(vals), bs[int(.05 * boot)], bs[int(.95 * boot)]
    out = {"n": n, "coverage": ci(inside, lambda v: sum(v) / len(v)),
           "width": ci(ratio, st.median), "pinball": ci(pin, lambda v: sum(v) / len(v))}
    if group:
        g = defaultdict(list)
        for r, ok in zip(test, inside): g[group(r)].append(ok)
        out["group_coverage"] = {k: (sum(v) / len(v), len(v)) for k, v in g.items() if len(v) >= 20}
    return out


def refit_width_ci(fit, cal, test, feat, target, group_key, boot=100, seed=0):
    """Width CI that reflects estimation uncertainty: resample training repos (fit and
    cal separately, by group), refit, re-measure median width and coverage on the fixed
    test set. Fixes the degenerate CI when the interval width is constant across rows."""
    rng = random.Random(seed)

    def resample(rows):
        g = defaultdict(list)
        for r in rows: g[group_key(r)].append(r)
        keys = list(g)
        return [r for _ in keys for r in g[keys[rng.randrange(len(keys))]]]
    widths, covs = [], []
    for _ in range(boot):
        pred = fit_interval(resample(fit), resample(cal), feat, target)
        iv = [pred(r) for r in test]
        widths.append(st.median(hi / lo for lo, hi in iv))
        covs.append(sum(lo <= target(r) <= hi for r, (lo, hi) in zip(test, iv)) / len(test))
    widths.sort(); covs.sort()
    return {"width": (widths[int(.05 * boot)], widths[int(.95 * boot)]),
            "coverage": (covs[int(.05 * boot)], covs[int(.95 * boot)])}


def fmt(res):
    c, w, p = res["coverage"], res["width"], res["pinball"]
    return (f"n={res['n']:5} cov {c[0]:.0%} [{c[1]:.0%},{c[2]:.0%}] | p10-p90 width {w[0]:.1f}x "
            f"[{w[1]:.1f},{w[2]:.1f}] | pinball {p[0]:.3f} [{p[1]:.3f},{p[2]:.3f}]")
