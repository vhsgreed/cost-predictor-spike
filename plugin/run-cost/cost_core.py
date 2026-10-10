"""run-cost: history, group fallback, card and p90 alarm logic. No Hermes imports.

History source: the session store at $HERMES_HOME/state.db (read-only). Cost per
session = COALESCE(actual_cost_usd, estimated_cost_usd) — billed cost when present,
otherwise Hermes' own token x list-price estimate (validated in the study's V1:
median billed/estimate 1.0000 on 1,493 OpenRouter generations).

Group (spec docs/plugin-spec.md section 3): (model, entry point, prompt-length
bucket) with fallbacks (model, entry) -> (model) -> all runs, first level with
>= MIN_N past runs wins; below that the card says "collecting". The prompt-length
bucket edges (<200 / <1500 chars of the session's first user message) are the
study's H1 edges.
"""
from __future__ import annotations

import os
import sqlite3
import time

MIN_N = 30
BUCKET_EDGES = (200, 1500)
REFRESH_S = 300.0
# state.db source / hook platform -> canonical entry point
ENTRY_MAP = {"oneshot": "cli", "tui": "cli", "subagent": "delegated"}
LEVEL_NAMES = {1: "model+entry+prompt", 2: "model+entry", 3: "model", 4: "all runs"}


def hermes_home() -> str:
    return os.environ.get("HERMES_HOME") or os.path.expanduser("~/.hermes")


def entry_point(raw: str | None) -> str:
    raw = (raw or "").strip().lower()
    return ENTRY_MAP.get(raw, raw or "unknown")


def prompt_bucket(chars) -> int:
    if chars is None:
        return -1
    return 0 if chars < BUCKET_EDGES[0] else 1 if chars < BUCKET_EDGES[1] else 2


def quantile(values, p: float) -> float:
    """Linear-interpolated quantile on a non-empty sequence (same math as lab.quantile)."""
    v = sorted(values)
    if not v:
        raise ValueError("quantile of empty data")
    if len(v) == 1:
        return float(v[0])
    x = p * (len(v) - 1)
    lo = int(x)
    hi = min(lo + 1, len(v) - 1)
    return float(v[lo] + (v[hi] - v[lo]) * (x - lo))


class RunCostCore:
    def __init__(self, db_path: str | None = None, min_n: int | None = None):
        self.db_path = db_path or os.path.join(hermes_home(), "state.db")
        self.min_n = int(os.environ.get("RUN_COST_MIN_N") or min_n or MIN_N)
        self.rows: list[dict] = []
        self._loaded_at = 0.0

    # ---------- history ----------
    def load(self, force: bool = False) -> list[dict]:
        now = time.time()
        if not force and self.rows and now - self._loaded_at < REFRESH_S:
            return self.rows
        rows = []
        con = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        try:
            cur = con.execute(
                """
                SELECT s.model, s.source,
                       COALESCE(s.actual_cost_usd, s.estimated_cost_usd) AS cost,
                       (SELECT m.content FROM messages m
                         WHERE m.session_id = s.id AND m.role = 'user' AND m.active = 1
                         ORDER BY m.id LIMIT 1) AS first_user
                FROM sessions s
                WHERE s.api_call_count > 0
                """
            )
            for model, source, cost, first_user in cur.fetchall():
                if cost is None or cost <= 0:
                    continue
                rows.append(
                    {
                        "model": model or "unknown",
                        "entry": entry_point(source),
                        "bucket": prompt_bucket(len(first_user) if first_user else None),
                        "cost": float(cost),
                    }
                )
        finally:
            con.close()
        self.rows = rows
        self._loaded_at = now
        return rows

    def session_cost(self, session_id: str) -> float | None:
        """Current cumulative cost of one live session, as the store knows it."""
        con = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        try:
            row = con.execute(
                "SELECT COALESCE(actual_cost_usd, estimated_cost_usd) FROM sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
        finally:
            con.close()
        return float(row[0]) if row and row[0] is not None else None

    # ---------- grouping ----------
    def _keys(self, model: str, entry: str, bucket: int):
        yield 1, (model, entry, bucket)
        yield 2, (model, entry)
        yield 3, (model,)
        yield 4, ()

    def group(self, model: str, entry: str, bucket: int):
        """First level with >= min_n past runs. Returns (level, key, costs) or (None, key, [])."""
        rows = self.load()
        for level, key in self._keys(model, entry, bucket):
            n = len(key)
            costs = [r["cost"] for r in rows if (r["model"], r["entry"], r["bucket"])[:n] == key]
            if len(costs) >= self.min_n:
                return level, key, costs
        _, key = next(self._keys(model, entry, bucket))
        return None, key, [r["cost"] for r in rows]

    # ---------- card + alarm ----------
    def card(self, model: str, entry: str, bucket: int) -> dict:
        level, key, costs = self.group(model, entry, bucket)
        if level is None:
            return {"ok": False, "n": len(costs), "need": self.min_n, "level": None}
        return {
            "ok": True,
            "level": level,
            "level_name": LEVEL_NAMES[level],
            "n": len(costs),
            "at_least": quantile(costs, 0.10),
            "typical": quantile(costs, 0.50),
            "under90": quantile(costs, 0.90),
            "p90": quantile(costs, 0.90),
        }

    @staticmethod
    def card_text(card: dict) -> str:
        if not card["ok"]:
            return (f"run-cost: collecting ({card['n']}/{card['need']}), no card yet "
                    f"(counted over all runs on record)")
        return (
            f"run-cost [{card['level_name']}, n={card['n']}]: at least ${card['at_least']:.2f} "
            f"(90% of similar runs cost more) / typical ${card['typical']:.2f} / 90% under ${card['under90']:.2f}"
        )

    @staticmethod
    def alarm_footer(spent: float, card: dict, times: int) -> str:
        """times = 1 (past p90) or 2 (past 2x p90)."""
        if times == 2:
            return (f"> [run-cost] ⚠ this run is at ${spent:.2f}, past 2x the group p90 "
                    f"(${card['p90']:.2f}); similar runs that got here rarely resolved.")
        return (f"> [run-cost] ⚠ this run is at ${spent:.2f}, past its group p90 "
                f"(${card['p90']:.2f}); 90% of similar runs ended below that.")


class NoticeBoard:
    """One p90 notice and one 2x-p90 notice per session, then silence."""

    def __init__(self):
        self._fired: dict[str, set] = {}
        self._pending: dict[str, tuple] = {}

    def check(self, session_id: str, spent: float, card: dict) -> int:
        fired = self._fired.setdefault(session_id, set())
        if not card.get("ok"):
            return 0
        times = 0
        if spent >= 2 * card["p90"] and 2 not in fired:
            times = 2
        elif spent >= card["p90"] and 1 not in fired:
            times = 1
        if times:
            fired.add(times)
            self._pending[session_id] = (times, spent, card)
        return times

    def pop_pending(self, session_id: str):
        return self._pending.pop(session_id, None)
