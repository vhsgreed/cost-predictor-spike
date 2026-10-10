# Spec: run-cost card + p90 alarm (Hermes plugin, MIT)

Status: DRAFT 2026-10-10, not approved. No build until Karl approves.
Decision rule (STATUS.md, 2026-10-10): the learned forecast (section 5) is included only if R8 Part A passes. Otherwise this spec ships without it and the cost-predictor line closes.

## 1. What it does

1. **Pre-run card.** Before a task starts, show three numbers for runs like this one: "at least $X (90% of similar runs cost more) / typical $Y / 90% cost under $Z", plus n and the source.
2. **In-run alarm.** During the run, warn once when cumulative spend crosses the group's p90 (what 90% of similar runs ended below). Warn only; never stop or alter the run.

## 2. Evidence each number rests on (from PLAN.md, not restated from memory)

| Feature | Evidence | Claim allowed in the UI |
|---|---|---|
| "at least" (p10) | H4 PASS on the locked test: 90.0% held; 10/15 groups consistent | "90% of similar runs cost more than this" |
| typical (p50) / 90% under (p90) | Descriptive, `groups_full.csv`; H1 narrowing only in 5/15 groups | "Typical" and "range", with width shown. No accuracy claim. |
| p90 alarm | O1 PASS 3/3 (benchmarks, estimated size): tail runs resolve 2.3-7x less often | "Runs past this point usually fail on benchmarks." Associational; no "stop now" advice. |
| Remaining-cost forecast | R6 FAIL, R7 FAIL; R8 pending | Only if R8 Part A passes |

## 3. The hard problem: group matching (push-back)

`groups_full.csv` holds 175 groups of **GitHub Copilot coding agent** sessions: models claude-sonnet-4/4.5/4.6, claude-opus-4.5/4.6, gpt-5.x-codex; trigger events such as `pull_request_comment`, `agents_panel`, `api_call_received`. A Hermes run (Opus 5.5 via subscription, mimo v2.6 pro, local gemma) matches **none** of these groups on model, and Hermes has no trigger-event analog. Shipping the card as "from 424k sessions" to a Hermes user would show numbers for a different agent, different models and different prices.

Options, in order of honesty:

- **A. Own-history groups (default).** Build the card from the user's own past runs on the same machine: group = (model, entry point: cli / desktop / cron / delegated, prompt-length bucket <200 / <1500 / >=1500 chars), the same bucket edges as H1. Show a group only when it has >= 30 past runs; otherwise fall back to (model) then (all runs), and label which level was used. The public table is the fallback of last resort and is labelled "GitHub Copilot agent sessions, 2026 Q3, not your setup".
- **B. Public table only,** for users whose setup matches (Copilot coding agent). Outside Hermes; out of scope.

Caveat stated in the UI and README: H7 found that a user's own history did not narrow the range beyond model x trigger groups (0/37). Own history is used here because it is the only data that matches the setup, not because it is more accurate. Floor calibration (H4) has not been tested on own-history groups; the plugin logs every card and outcome locally so it can be (section 7).

## 4. Data and privacy

- Hermes already keeps per-session tokens, `estimated_cost_usd`, `actual_cost_usd`, `model`, `source` (cli/desktop/cron/oneshot/subagent) and `api_call_count` in `~/.hermes/state.db` `sessions`. The plugin reads that table instead of logging its own history. Reality check (counts only, 2026-10-10): 85 sessions on this machine since 2026-09-30, 80 with a cost estimate, split 33 desktop / 38 oneshot / 13 cron / 1 subagent. At the 30-run floor, only 1-2 groups would show a card today; most cards will fall back to model-only or "collecting" for months. That is the main reason to keep the build to ~1 day.
- Inputs: Hermes per-call usage already logged locally (model, input / cached / output tokens, cost where the provider bills it, session id, entry point). Cost = billed cost when present, else tokens x a pinned price table (the pricing model V1 validated: median billed/estimate 1.000 on 1,493 calls).
- Prompt text is never stored; only its length bucket.
- Everything stays on the machine. No network calls. No telemetry.
- Store: `~/.hermes/plugins/run-cost/runs.sqlite` (one row per finished run: group key, final cost, turns, resolved = unknown).

## 5. Learned forecast (conditional on R8 Part A PASS)

If R8 passes: ship the R8 LightGBM quantile model trained on public SWE data as an "estimated remaining" line ("likely ends between $A and $B"), only for groups whose shape matches (none of the Hermes groups do today), so in practice: re-fit on own history once >= 2,000 own runs exist, then re-validate on a held-out month before showing it. If R8 fails: this section is deleted.

## 6. Hermes integration

- Hook points: start of a user turn / task (card), after each LLM call (alarm check). Uses existing plugin middleware: `llm_request` (read-only) and post-call usage events. Never mutates requests.
- Card surfaces: desktop as a one-line notice above the run; CLI as one stderr line; cron/delegated runs log only.
- Alarm: one notice per run when cumulative cost > group p90; second notice at 2x p90. Config: `run_cost.alarm: warn | off` (default warn), `run_cost.min_group_n: 30`.
- Cold start: with < 30 own runs in total, no card; log only. Says "collecting (n/30)".

## 7. How we will know it works (numbers)

- Floor calibration on own runs: share of runs costing more than the shown "at least" value, target 90% (CI from bootstrap over days), checked after 100 cards.
- Alarm: share of alarmed runs Karl would have stopped (blind check of 30), reported as-is. No bar; this is product feedback, not a registered test.
- Overhead: card + alarm add < 5 ms per LLM call (measured in tests).

## 8. Out of scope

Auto-stop, budgets, cross-machine sync, any upload, Copilot integration, TokenCast code (AGPL; not used).

## 9. Build estimate

~1 day: table builder from local logs, two hooks, sqlite store, tests (group fallback, bucket edges, alarm fires once, no network). Plus a README with section 2's table.
