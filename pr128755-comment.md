Tested on a real setup: Claude subscription as primary, OpenRouter (`xiaomi/mimo-v2.6-pro`) as fallback. Linux, Python 3.14.

**Tests:** `tests/agent/test_usage_pricing.py`, `tests/agent/test_turn_usage_log_line.py`, `tests/providers/test_provider_profiles.py`
- main: 65 passed
- this PR: 69 passed
- this PR rebased onto current main (1,613 commits ahead): 69 passed, rebase clean

**Live response:** one real OpenRouter call with `usage: {include: true}` returned `cost: 1.653e-05` and `cost_details.upstream_inference_cost: 1.653e-05`. Feeding that usage object through `normalize_usage` + `estimate_usage_cost`:
- main: `status=estimated source=provider_models_api label=~$<0.0001`
- this PR: `status=actual source=provider_cost_api label=$<0.0001`

So the billed amount is picked up as actual, as described. Here the table estimate matched the billed amount exactly, so this call doesn't exercise a case where they differ (tiered pricing, provider routing).

Note for other testers: with a git worktree whose `.git` lives under `~/.hermes`, three tests in `test_turn_usage_log_line.py` trip `home_io_guard` (on `.git/worktrees/.../hermes-update-pull`). A plain clone outside the Hermes home is clean.
