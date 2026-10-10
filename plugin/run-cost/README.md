# run-cost (Hermes plugin, MIT)

Pre-run cost card and warn-only in-run p90 alarm, built from your own past runs in
`~/.hermes/state.db`. Spec and evidence table: `../../docs/plugin-spec.md`.
Tests: `python3 tests/test_run_cost.py` (14 tests, offline).

## What it shows

- **Card** (`/cost` in chat, `hermes run-cost card [model] [entry] [bucket]` in a
  terminal, one stderr line at a session's first turn):
  `run-cost [model+entry+prompt, n=30]: at least $X (90% of similar runs cost more) / typical $Y / 90% under $Z`
  The group falls back model+entry+prompt -> model+entry -> model -> all runs;
  below 30 runs on record it says `collecting (n/30)`.
- **Alarm**: when a session's cumulative cost crosses its group p90, the next
  assistant response carries one warning line; a second at 2x p90. Warn only:
  it never stops a run and never changes a request.

## Honest limits (read before trusting it)

- History is your own Hermes sessions (85 on this machine as of 2026-10-10).
  With 30-run groups, most cards show the `all runs` fallback for months.
- The "at least" (p10) line is calibrated on AgentLogs Copilot sessions (H4),
  not on own-history groups. The 90% figure has not been re-validated here.
- The alarm's "usually fail" claim comes from benchmark runs (O1); it is
  associational and untested on your own runs.
- In-run spend is read from the session store after each provider call; if the
  store lags, the alarm fires one call late.

## Privacy

No network calls. No prompt text stored (only the first message's length bucket).
Reads state.db read-only; writes one JSONL event log at
`$HERMES_HOME/logs/run-cost.jsonl` (cards + alarms, for the later blind check).
