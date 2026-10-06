"""Unit tests for lab.py. Synthetic data only. Run: .venv/bin/python -m pytest -q test_lab.py"""
import math, random
import lab


def tc(id=None, index=0, name=None, args=""):
    return {"id": id, "index": index, "function_name": name, "function_arguments": args}


def test_reassemble_streamed_pieces():
    out = lab.reassemble([(5, [tc("a", 0, "view", '{"path":')]), (6, [tc(None, 0, None, '"x.py"}')])])
    assert len(out) == 1 and out[0][0] == 5 and out[0][1] == "view"
    whole = lab.reassemble([(1, [tc("b", 0, "view", '{"path":"x.py"}')])])
    assert out[0][2] == whole[0][2]          # same name+args -> same hash regardless of chunking


def test_reassemble_parallel_calls_kept_apart():
    out = lab.reassemble([(1, [tc("a", 0, "view", "A"), tc("b", 1, "grep", "B")]),
                          (2, [tc(None, 0, None, "1"), tc(None, 1, None, "2")])])
    names = {n: h for _, n, h in out}
    assert set(names) == {"view", "grep"}
    assert names["view"] == lab.reassemble([(1, [tc("z", 0, "view", "A1")])])[0][2]


def test_reassemble_orphan_delta_ignored():
    assert lab.reassemble([(1, [tc(None, 3, None, "x")])]) == []


def calls(seq):
    return [(i, n, h) for i, (n, h) in enumerate(seq)]


def test_loop_repeat_fires_on_third():
    on = lab.loop_onset(calls([("view", "h1"), ("bash", "h2"), ("view", "h1"), ("view", "h1")]))
    assert on == {"R": 3}


def test_loop_idle_needs_prior_edit():
    assert "S" not in lab.loop_onset(calls([("view", f"v{i}") for i in range(40)]))   # pure exploration
    on = lab.loop_onset(calls([("edit", "e")] + [("view", f"v{i}") for i in range(30)]))
    assert on["S"] == 25


def test_loop_clean_run():
    seq = [("view", "a"), ("edit", "b"), ("bash", "c"), ("edit", "d"), ("bash", "e")]
    assert lab.loop_onset(calls(seq)) == {}


def test_complete():
    assert lab.complete([2, 0, 1]) and not lab.complete([1, 2]) and not lab.complete([0, 2]) and not lab.complete([])


def test_canon_model():
    assert lab.canon_model("sweagent-capi:claude-sonnet-4.6") == "claude-sonnet-4.6"
    assert lab.canon_model("Claude Sonnet 4.5 (Preview)") == "claude-sonnet-4.5"
    assert lab.canon_model(None) is None


def test_call_cost_cached_cheaper():
    full = lab.call_cost("claude-sonnet-4.5", 1000, 0, 100)
    cached = lab.call_cost("claude-sonnet-4.5", 1000, 900, 100)
    assert math.isclose(full, 1000 * 3e-6 + 100 * 15e-6)
    assert math.isclose(cached, 100 * 3e-6 + 900 * 3e-7 + 100 * 15e-6)
    assert lab.call_cost("claude-sonnet-4.5", 1000, 900, 0) < lab.call_cost("claude-sonnet-4.5", 1000, 0, 0) / 3
    assert lab.call_cost("claude-sonnet-4.5", 1000, 5000, 0) == lab.call_cost("claude-sonnet-4.5", 1000, 1000, 0)  # cached clamped
    assert lab.call_cost("unknown-model", 10, 0, 1) is None


def test_group_split_no_leak():
    rows = [{"repo": f"r{i % 37}", "y": i} for i in range(500)]
    a, b, c = lab.split_by_group(rows, lambda r: r["repo"])
    ra, rb, rc = ({r["repo"] for r in x} for x in (a, b, c))
    assert not (ra & rb or ra & rc or rb & rc) and len(a) + len(b) + len(c) == 500


def test_interval_calibrated_on_known_noise():
    rng = random.Random(1)
    rows = [{"x": rng.uniform(1, 5)} for _ in range(3000)]
    for r in rows: r["y"] = math.exp(r["x"] + rng.gauss(0, 0.5))
    fit, cal, te = rows[:1500], rows[1500:2100], rows[2100:]
    pred = lab.fit_interval(fit, cal, lambda r: [1.0, r["x"]], lambda r: r["y"])
    res = lab.evaluate(te, pred, lambda r: r["y"], boot=100)
    assert 0.76 <= res["coverage"][0] <= 0.84          # nominal 80% (p10-p90)
    assert 3.0 <= res["width"][0] <= 4.0               # exp(2*1.2816*0.5) = 3.6


def test_refit_width_ci_not_degenerate():
    rng = random.Random(3)
    rows = [{"x": rng.uniform(1, 5), "g": i % 200} for i in range(2000)]
    for r in rows: r["y"] = math.exp(r["x"] + rng.gauss(0, 0.5))
    fit, cal, te = lab.split_by_group(rows, lambda r: r["g"])
    ci = lab.refit_width_ci(fit, cal, te, lambda r: [1.0, r["x"]], lambda r: r["y"], lambda r: r["g"], boot=40)
    lo, hi = ci["width"]
    assert lo < hi and lo <= 3.6 * 1.15 and hi >= 3.6 * 0.85


def test_negative_control_shuffled_target():
    rng = random.Random(2)
    rows = [{"x": rng.uniform(1, 5)} for _ in range(3000)]
    ys = [math.exp(r["x"] + rng.gauss(0, .5)) for r in rows]; rng.shuffle(ys)
    for r, y in zip(rows, ys): r["y"] = y
    fit, cal, te = rows[:1500], rows[1500:2100], rows[2100:]
    good = lab.fit_interval(fit, cal, lambda r: [1.0, r["x"]], lambda r: r["y"])
    base = lab.fit_interval(fit, cal, lambda r: [1.0], lambda r: r["y"])
    g = lab.evaluate(te, good, lambda r: r["y"], boot=50); b = lab.evaluate(te, base, lambda r: r["y"], boot=50)
    assert g["width"][0] >= b["width"][0] * 0.95       # features must not "help" on shuffled targets
