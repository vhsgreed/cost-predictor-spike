"""run-cost: pre-run cost card + in-run p90 alarm for Hermes.

Spec: docs/plugin-spec.md in https://github.com/vhsgreed/cost-predictor-spike (MIT).
Warn-only: never stops, never alters requests, never stores prompt text (only the
first user message's length bucket), no network calls.

Surfaces:
- card: one stderr line at a session's first turn, `/cost` in chat,
  `hermes run-cost card` in a terminal.
- alarm: when the session's cumulative cost (from the session store) crosses the
  group p90, the first assistant response after that carries one warning line;
  a second at 2x p90. Log: $HERMES_HOME/logs/run-cost.jsonl.
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # sibling module, loaded bare
from cost_core import NoticeBoard, RunCostCore, entry_point, hermes_home, prompt_bucket  # noqa: E402

ALARM_MODE = os.environ.get("RUN_COST_ALARM", "warn")  # "warn" | "off"

_core = RunCostCore()
_board = NoticeBoard()
_state: dict = {"card": None, "session_id": None, "spent": 0.0}


def _log(entry: dict) -> None:
    try:
        path = os.path.join(hermes_home(), "logs", "run-cost.jsonl")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"ts": time.time(), **entry}) + "\n")
    except OSError:
        pass


def _card_for(model: str, entry: str, bucket: int) -> dict:
    return _core.card(model or "unknown", entry, bucket)


def _on_first_turn(**kwargs):
    """Card at the session's first turn. Returns None: never injects context."""
    if not kwargs.get("is_first_turn"):
        return None
    model = kwargs.get("model") or ""
    entry = entry_point(kwargs.get("platform") or "")
    bucket = prompt_bucket(len(kwargs.get("user_message") or ""))
    card = _card_for(model, entry, bucket)
    _state.update(card=card, session_id=kwargs.get("session_id"), spent=0.0, model=model,
                  entry=entry, bucket=bucket)
    text = RunCostCore.card_text(card)
    print(f"[run-cost] {text}", file=sys.stderr)
    _log({"event": "card", "model": model, "entry": entry, "bucket": bucket,
          "text": text, "ok": card["ok"], "n": card["n"]})
    return None


def _on_api_request(**kwargs):
    """Track cumulative spend of the live session; queue a notice when it crosses p90."""
    if ALARM_MODE != "warn":
        return None
    session_id = kwargs.get("session_id") or ""
    if not session_id:
        return None
    spent = _core.session_cost(session_id)
    if spent is None:
        return None
    _state["spent"] = spent
    card = _state.get("card")
    if not card or not card.get("ok"):
        return None
    times = _board.check(session_id, spent, card)
    if times:
        _log({"event": "alarm", "times": times, "spent": spent, "p90": card.get("p90"),
              "session_id": session_id})
    return None


def _on_llm_output(**kwargs):
    """Append the queued warning line to the next assistant response. Else no change."""
    pending = _board.pop_pending(kwargs.get("session_id") or "")
    if not pending:
        return None
    times, spent, card = pending
    return (kwargs.get("response_text") or "").rstrip() + "\n\n" + RunCostCore.alarm_footer(spent, card, times)


def _handle_cost_command(raw_args: str) -> str:
    card = _state.get("card")
    if not card:
        return "run-cost: no card yet (no turn started in this session). Use `hermes run-cost card` for history."
    text = RunCostCore.card_text(card)
    spent = _state.get("spent") or 0.0
    extra = ""
    if card.get("ok") and spent > 0:
        ratio = spent / card["p90"]
        extra = f"\nthis run so far: ${spent:.2f} ({ratio:.1f}x the group p90)"
    return text + extra


def _cli_handler(args):
    sub = getattr(args, "run_cost_command", None)
    model = getattr(args, "model", None) or _state.get("model") or "unknown"
    entry = entry_point(getattr(args, "entry", None) or _state.get("entry") or "cli")
    bucket = (int(args.bucket) if getattr(args, "bucket", None) not in (None, "")
              else _state.get("bucket", -1))
    card = _card_for(model, entry, bucket)
    if sub == "alarm" and card.get("ok"):
        print(f"group p90 (alarm threshold): ${card['p90']:.2f} [{card['level_name']}, n={card['n']}]")
        return
    print(RunCostCore.card_text(card))
    if card["ok"]:
        print(f"group p90 (alarm threshold): ${card['p90']:.2f}; "
              f"spend so far this session: ${_state.get('spent') or 0.0:.2f}")


def _cli_setup(subparser):
    subs = subparser.add_subparsers(dest="run_cost_command")
    card = subs.add_parser("card", help="Show the cost card: card [model] [entry] [bucket]")
    card.add_argument("model", nargs="?", default=None)
    card.add_argument("entry", nargs="?", default=None)
    card.add_argument("bucket", nargs="?", default=None)
    alarm = subs.add_parser("alarm", help="Show the alarm threshold: alarm [model] [entry] [bucket]")
    alarm.add_argument("model", nargs="?", default=None)
    alarm.add_argument("entry", nargs="?", default=None)
    alarm.add_argument("bucket", nargs="?", default=None)
    subparser.set_defaults(func=_cli_handler)


def register(ctx):
    ctx.register_hook("pre_llm_call", _on_first_turn)
    ctx.register_hook("post_api_request", _on_api_request)
    ctx.register_hook("transform_llm_output", _on_llm_output)
    ctx.register_command("cost", handler=_handle_cost_command,
                         description="Cost card + spend so far for this run")
    ctx.register_cli_command(name="run-cost", help="run-cost card and p90 alarm",
                             setup_fn=_cli_setup, handler_fn=_cli_handler)
