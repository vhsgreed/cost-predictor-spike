"""Offline tests for run-cost. Run: python3 tests/test_run_cost.py (or pytest)."""
from __future__ import annotations

import importlib.util
import os
import sqlite3
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PLUGIN_DIR = os.path.join(HERE, "..")
sys.path.insert(0, PLUGIN_DIR)

import cost_core  # noqa: E402

SPEC = importlib.util.spec_from_file_location("run_cost", os.path.join(PLUGIN_DIR, "__init__.py"))
run_cost = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(run_cost)
run_cost._log = lambda entry: None  # keep test runs out of the real event log


def make_db(rows, messages=None) -> str:
    """rows: (id, model, source, estimated, actual, api_call_count); messages: (session_id, content)."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY, model TEXT, source TEXT, "
                "estimated_cost_usd REAL, actual_cost_usd REAL, api_call_count INTEGER)")
    con.execute("CREATE TABLE messages (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, "
                "role TEXT, content TEXT, active INTEGER DEFAULT 1)")
    con.executemany("INSERT INTO sessions VALUES (?,?,?,?,?,?)", rows)
    for sid, content in (messages or []):
        con.execute("INSERT INTO messages (session_id, role, content, active) VALUES (?, 'user', ?, 1)",
                    (sid, content))
    con.commit()
    con.close()
    return path


def test_prompt_bucket_edges():
    assert cost_core.prompt_bucket(None) == -1
    assert cost_core.prompt_bucket(0) == 0
    assert cost_core.prompt_bucket(199) == 0
    assert cost_core.prompt_bucket(200) == 1
    assert cost_core.prompt_bucket(1499) == 1
    assert cost_core.prompt_bucket(1500) == 2


def test_entry_point_mapping():
    assert cost_core.entry_point("oneshot") == "cli"
    assert cost_core.entry_point("tui") == "cli"
    assert cost_core.entry_point("subagent") == "delegated"
    assert cost_core.entry_point("desktop") == "desktop"
    assert cost_core.entry_point(None) == "unknown"


def test_quantile_known_data():
    v = list(range(1, 11))  # 1..10
    assert abs(cost_core.quantile(v, 0.5) - 5.5) < 1e-9
    assert abs(cost_core.quantile(v, 0.1) - 1.9) < 1e-9
    assert abs(cost_core.quantile(v, 0.9) - 9.1) < 1e-9
    assert cost_core.quantile([7.0], 0.9) == 7.0


def _rows(n, model="m1", source="desktop", base=1.0, msg="x" * 2000):
    rows = [(f"s{i}", model, source, base + i / 100.0, None, 3) for i in range(n)]
    return rows, [(f"s{i}", msg) for i in range(n)]


def test_cost_filter_excludes_zero_and_no_calls():
    base_rows, base_msgs = _rows(2)
    rows = base_rows + [
        ("z1", "m1", "desktop", 0.0, None, 3),      # zero cost
        ("z2", "m1", "desktop", None, None, 3),      # no cost
        ("z3", "m1", "desktop", 5.0, None, 0),       # no api calls
    ]
    path = make_db(rows, base_msgs)
    core = cost_core.RunCostCore(db_path=path)
    assert len(core.load(force=True)) == 2
    os.unlink(path)


def test_billed_cost_wins_over_estimate():
    path = make_db([("s1", "m1", "desktop", 9.99, 4.5, 3)])
    core = cost_core.RunCostCore(db_path=path)
    assert core.load(force=True)[0]["cost"] == 4.5
    assert core.session_cost("s1") == 4.5
    assert core.session_cost("missing") is None
    os.unlink(path)


def test_group_fallback_chain():
    # 20 in the exact cell, 25 with same model+entry, 35 with same model -> level 3
    base_rows, _ = _rows(20, base=1.0)
    rows = (base_rows
            + [(f"o{i}", "m1", "desktop", 2.0, None, 3) for i in range(5)]  # bucket -1 (no message)
            + [(f"p{i}", "m1", "cron", 3.0, None, 3) for i in range(10)])
    path = make_db(rows)
    core = cost_core.RunCostCore(db_path=path, min_n=30)
    level, key, costs = core.group("m1", "desktop", 2)
    assert level == 3, (level, key, len(costs))
    assert len(costs) == 35
    os.unlink(path)


def test_group_level_one_when_enough():
    rows, msgs = _rows(32)
    path = make_db(rows, msgs)
    core = cost_core.RunCostCore(db_path=path, min_n=30)
    level, key, costs = core.group("m1", "desktop", 2)
    assert level == 1 and len(costs) == 32
    os.unlink(path)


def test_collecting_below_min():
    rows, msgs = _rows(5)
    path = make_db(rows, msgs)
    core = cost_core.RunCostCore(db_path=path, min_n=30)
    card = core.card("m1", "desktop", 2)
    assert card["ok"] is False and card["n"] == 5 and card["need"] == 30
    assert "collecting (5/30)" in core.card_text(card)
    os.unlink(path)


def test_card_numbers_and_prompt_bucket_split():
    # 30 short-prompt runs at 1.00..1.29 and 30 long-prompt runs at 5.00..5.29
    rows = ([(f"a{i}", "m1", "desktop", 1.0 + i / 100, None, 3) for i in range(30)]
            + [(f"b{i}", "m1", "desktop", 5.0 + i / 100, None, 3) for i in range(30)])
    msgs = [(f"a{i}", "x" * 50) for i in range(30)] + [(f"b{i}", "x" * 2000) for i in range(30)]
    path = make_db(rows, msgs)
    core = cost_core.RunCostCore(db_path=path, min_n=30)
    short = core.card("m1", "desktop", 0)
    long_ = core.card("m1", "desktop", 2)
    assert short["ok"] and long_["ok"]
    assert abs(short["typical"] - 1.145) < 1e-9, short
    assert abs(long_["typical"] - 5.145) < 1e-9, long_
    assert short["p90"] < long_["at_least"]  # the buckets separate cleanly
    os.unlink(path)


def test_notice_board_once_per_level():
    board = cost_core.NoticeBoard()
    card = {"ok": True, "p90": 10.0}
    assert board.check("s", 5.0, card) == 0
    assert board.check("s", 12.0, card) == 1          # first crossing
    assert board.pop_pending("s")[0] == 1
    assert board.check("s", 15.0, card) == 0          # no repeat
    assert board.check("s", 25.0, card) == 2          # 2x crossing
    assert board.pop_pending("s")[0] == 2
    assert board.check("s", 40.0, card) == 0          # silence after both
    assert board.pop_pending("s") is None
    assert board.check("s2", 99.0, {"ok": False}) == 0  # no card, no alarm


def test_alarm_footer_mentions_spend():
    footer = cost_core.RunCostCore.alarm_footer(12.5, {"p90": 10.0}, 1)
    assert "$12.50" in footer and "$10.00" in footer
    footer2 = cost_core.RunCostCore.alarm_footer(25.0, {"p90": 10.0}, 2)
    assert "2x" in footer2


def test_hooks_never_inject_context_and_warn_once():
    rows, msgs = _rows(30, base=10.0, msg="hello world")  # p90 = 10 + 0.9*0.29
    path = make_db(rows, msgs)
    run_cost._core = cost_core.RunCostCore(db_path=path, min_n=30)
    run_cost._board = cost_core.NoticeBoard()
    out = run_cost._on_first_turn(is_first_turn=True, model="m1", platform="desktop",
                                  user_message="hello world", session_id="s0")
    assert out is None                                   # pre_llm_call must not inject
    assert run_cost._on_first_turn(is_first_turn=False) is None
    card = run_cost._state["card"]
    assert card["ok"] and card["level"] == 1

    # live session row with spend below then above p90
    con = sqlite3.connect(path)
    con.execute("INSERT INTO sessions VALUES ('live','m1','desktop',4.0,NULL,3)")
    con.commit()
    assert run_cost._on_api_request(session_id="live") is None
    assert run_cost._on_llm_output(session_id="live", response_text="answer") is None
    con.execute("UPDATE sessions SET estimated_cost_usd=11.0 WHERE id='live'")
    con.commit()
    con.close()
    run_cost._on_api_request(session_id="live")
    text = run_cost._on_llm_output(session_id="live", response_text="answer")
    assert text.startswith("answer") and "[run-cost]" in text
    assert run_cost._on_llm_output(session_id="live", response_text="answer2") is None
    os.unlink(path)


def test_alarm_mode_off():
    rows, msgs = _rows(30, base=10.0)
    path = make_db(rows, msgs)
    run_cost._core = cost_core.RunCostCore(db_path=path, min_n=30)
    run_cost._board = cost_core.NoticeBoard()
    run_cost.ALARM_MODE = "off"
    run_cost._state["card"] = {"ok": True, "p90": 10.0}
    con = sqlite3.connect(path)
    con.execute("INSERT INTO sessions VALUES ('live','m1','desktop',99.0,NULL,3)")
    con.commit()
    con.close()
    assert run_cost._on_api_request(session_id="live") is None
    run_cost.ALARM_MODE = "warn"
    os.unlink(path)


def test_overhead_under_5ms_per_check():
    rows, msgs = _rows(200, base=1.0)
    path = make_db(rows, msgs)
    core = cost_core.RunCostCore(db_path=path, min_n=30)
    board = cost_core.NoticeBoard()
    card = core.card("m1", "desktop", 2)
    assert card["ok"]
    t0 = time.time()
    n = 2000
    for i in range(n):
        board.check("s", 1.0 + (i % 50) / 10, card)
        core.card("m1", "desktop", 2)
    per_ms = (time.time() - t0) * 1000 / n
    assert per_ms < 5.0, f"{per_ms:.3f} ms per check"
    os.unlink(path)


if __name__ == "__main__":
    fns = sorted((k, v) for k, v in globals().items() if k.startswith("test_") and callable(v))
    failed = 0
    for name, fn in fns:
        try:
            fn()
            print(f"PASS {name}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {name}: {exc}")
    print(f"{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
